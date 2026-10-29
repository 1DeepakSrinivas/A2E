"""Terminal UI: browse suites, run them, watch progress, read results, run test groups, check the machine.

Start with `a2e tui` (needs `uv sync --extra tui`). Keys: q quits, 1 to 5 switch tabs.
Everything the TUI does is also available from the `a2e` command line.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (Button, DataTable, Footer, Header, Input, Label, ListItem, ListView, ProgressBar,
                             RichLog, Static, TabbedContent, TabPane)

from a2e.settings import CONFIG_DIR, REPO, Settings, load_settings

SUMMARY_COLS = ["device", "model", "quant", "kv_quant", "task_kind", "family", "category", "grammar", "runs",
                "prefill_ms", "decode_ms", "tool_wait_ms", "decode_tok_s", "cache_hit", "success", "j_per_turn"]


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if v != v else f"{v:.4g}"
    return str(v)


class A2EApp(App):
    TITLE = "A2E"
    SUB_TITLE = "tool-calling profiling on Pi 5 and Mac"
    CSS = """
    #suite-list, #group-list { width: 32; }
    #suite-detail, #group-detail { padding: 0 1; }
    #run-controls, #results-controls, #test-controls { height: auto; }
    #run-limit { width: 18; }
    RichLog { border: round $primary; }
    DataTable { height: 1fr; }
    """
    BINDINGS = [("q", "quit", "Quit"), ("1", "tab('suites')", "Suites"), ("2", "tab('run')", "Run"),
                ("3", "tab('results')", "Results"), ("4", "tab('tests')", "Tests"), ("5", "tab('doctor')", "Doctor")]

    def __init__(self, settings: Settings | None = None, suites_dir: Path | None = None, groups=None,
                 runner_factory=None):
        super().__init__()
        self.settings = settings or load_settings()
        self.suites_dir = Path(suites_dir or CONFIG_DIR / "suites")
        self._groups = groups
        self.runner_factory = runner_factory
        self.selected_suite: Path | None = None
        self.selected_group: str | None = None
        self.stop_event = threading.Event()
        self.running = False
        self.last_result: dict | None = None

    # ------------------------------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="suites", id="tabs"):
            with TabPane("Suites", id="suites"):
                with Horizontal():
                    yield ListView(id="suite-list")
                    yield Static("Select a suite to see what it runs.", id="suite-detail")
            with TabPane("Run", id="run"):
                with Vertical():
                    yield Static("No suite selected. Pick one on the Suites tab.", id="run-target")
                    with Horizontal(id="run-controls"):
                        yield Input(placeholder="job limit", id="run-limit")
                        yield Button("Run / resume", id="run-start", variant="primary")
                        yield Button("Stop after this job", id="run-stop", variant="error", disabled=True)
                    yield ProgressBar(id="run-progress", show_eta=False)
                    yield RichLog(id="run-log", wrap=True, markup=False)
            with TabPane("Results", id="results"):
                with Vertical():
                    with Horizontal(id="results-controls"):
                        yield Button("Refresh", id="results-refresh")
                        yield Button("Write report for selected", id="results-report")
                    yield DataTable(id="exp-table", cursor_type="row")
                    yield Static("", id="results-note")
                    yield DataTable(id="summary-table")
            with TabPane("Tests", id="tests"):
                with Horizontal():
                    yield ListView(id="group-list")
                    with Vertical():
                        yield Static("Select a test group.", id="group-detail")
                        with Horizontal(id="test-controls"):
                            yield Button("Run group", id="test-start", variant="primary")
                        yield RichLog(id="test-log", wrap=True, markup=False)
            with TabPane("Doctor", id="doctor"):
                with Vertical():
                    yield Button("Check again", id="doctor-refresh")
                    yield DataTable(id="doctor-table")
        yield Footer()

    def on_mount(self) -> None:
        sl = self.query_one("#suite-list", ListView)
        for p in sorted(self.suites_dir.glob("*.toml")):
            sl.append(ListItem(Label(p.stem), name=str(p)))
        gl = self.query_one("#group-list", ListView)
        for name in self.groups:
            gl.append(ListItem(Label(name), name=name))
        self.query_one("#exp-table", DataTable).add_columns("exp", "name", "status", "started", "done", "pending",
                                                            "failed", "skipped", "commit")
        self.query_one("#doctor-table", DataTable).add_columns("result", "check", "detail")
        self.refresh_results()
        self.refresh_doctor()

    @property
    def groups(self):
        if self._groups is None:
            from a2e.testgroups import load_groups
            self._groups = load_groups()
        return self._groups

    def action_tab(self, tab: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab

    # ------------------------------------------------------------------------------------- suites
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id == "suite-list":
            self.select_suite(Path(event.item.name))
        elif event.list_view.id == "group-list":
            self.selected_group = event.item.name
            g = self.groups[event.item.name]
            extra = f"\nNeeds environment variables: {', '.join(g.env)}" if g.env else ""
            self.query_one("#group-detail", Static).update(f"{g.name}: {g.description}{extra}")

    def select_suite(self, path: Path) -> None:
        from a2e.suite import expand, load_suite, plan_summary
        self.selected_suite = path
        try:
            s = load_suite(path)
            p = plan_summary(expand(s, self.settings))
            text = (f"{s.suite.name}\n{s.suite.description}\n\n"
                    f"jobs: {p['jobs']} ({p['runnable']} runnable, {p['skipped']} skipped)\n"
                    f"by kind: {p['by_kind']}\nby device: {p['by_device']}\n"
                    f"model loads: {p['server_loads']}\nruns (warm-up + measured): {p['runs']}\n"
                    f"repeats: {s.repeats.warmup} warm-up + {s.repeats.measured} measured\nsuite hash: {s.hash}")
        except Exception as e:  # noqa: BLE001
            text = f"{path.name}: {e}"
        self.query_one("#suite-detail", Static).update(text)
        self.query_one("#run-target", Static).update(f"Suite: {path.stem}  (database {self.settings.db})")

    # ------------------------------------------------------------------------------------- run
    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "run-start":
            self.start_run()
        elif bid == "run-stop":
            self.stop_event.set()
            self.query_one("#run-log", RichLog).write("stop requested: the current job finishes first")
        elif bid == "results-refresh":
            self.refresh_results()
        elif bid == "results-report":
            self.write_report()
        elif bid == "test-start":
            self.start_tests()
        elif bid == "doctor-refresh":
            self.refresh_doctor()

    def start_run(self) -> None:
        log = self.query_one("#run-log", RichLog)
        if self.running:
            log.write("a run is already in progress")
            return
        if not self.selected_suite:
            log.write("pick a suite on the Suites tab first")
            return
        raw = self.query_one("#run-limit", Input).value.strip()
        limit = int(raw) if raw.isdigit() else None
        self.stop_event.clear()
        self.running = True
        self.query_one("#run-start", Button).disabled = True
        self.query_one("#run-stop", Button).disabled = False
        self.run_worker_thread(self.selected_suite, limit)

    @work(thread=True, exclusive=True, group="run")
    def run_worker_thread(self, suite: Path, limit: int | None) -> None:
        from a2e.cli import format_event
        from a2e.runner import Runner

        def log(e: dict) -> None:
            self.call_from_thread(self.on_runner_event, e, format_event(e))

        try:
            make = self.runner_factory or (lambda path, log, stop: Runner(path, self.settings, log=log, stop=stop))
            out = make(suite, log, self.stop_event).run(limit=limit)
        except Exception as e:  # noqa: BLE001
            out = {"error": f"{type(e).__name__}: {e}"}
        self.call_from_thread(self.on_run_finished, out)

    def on_runner_event(self, e: dict, text: str | None) -> None:
        if text:
            self.query_one("#run-log", RichLog).write(text)
        bar = self.query_one("#run-progress", ProgressBar)
        if e.get("event") == "plan":
            bar.update(total=max(e["todo"], 1), progress=0)
        elif e.get("event") in ("job_done", "job_failed", "job_skipped"):
            bar.advance(1)

    def on_run_finished(self, out: dict) -> None:
        self.running = False
        self.last_result = out
        self.query_one("#run-start", Button).disabled = False
        self.query_one("#run-stop", Button).disabled = True
        if "error" in out:
            self.query_one("#run-log", RichLog).write(f"error: {out['error']}")
        self.refresh_results()

    # ------------------------------------------------------------------------------------- results
    def _store(self):
        from a2e.store import Store
        return Store(self.settings.db)

    def refresh_results(self) -> None:
        from a2e import results as R
        table = self.query_one("#exp-table", DataTable)
        table.clear()
        try:
            df = R.experiments(self._store())
        except Exception as e:  # noqa: BLE001
            self.query_one("#results-note", Static).update(f"cannot read {self.settings.db}: {e}")
            return
        for _, r in df.iterrows():
            table.add_row(*[_fmt(r.get(c)) for c in ("exp_id", "name", "status", "started", "done", "pending",
                                                     "failed", "skipped", "git_commit")], key=str(r["exp_id"]))
        self.query_one("#results-note", Static).update(
            f"{len(df)} experiments in {self.settings.db}. Select one to see its summary." if len(df) else
            f"No experiments yet in {self.settings.db}.")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "exp-table":
            self.show_summary(int(event.row_key.value))

    def show_summary(self, exp_id: int) -> None:
        from a2e import results as R
        self.selected_exp = exp_id
        t = self.query_one("#summary-table", DataTable)
        t.clear(columns=True)
        df = R.summary(self._store(), exp_id)
        if df.empty:
            self.query_one("#results-note", Static).update(f"Experiment {exp_id} has no measured runs yet.")
            return
        cols = [c for c in SUMMARY_COLS if c in df.columns]
        t.add_columns(*cols)
        for _, r in df.iterrows():
            t.add_row(*[_fmt(r[c]) for c in cols])
        self.query_one("#results-note", Static).update(f"Experiment {exp_id}: {len(df)} configurations "
                                                       "(measured repeats only).")

    def write_report(self) -> None:
        from a2e import results as R
        exp = getattr(self, "selected_exp", None)
        note = self.query_one("#results-note", Static)
        if exp is None:
            note.update("Select an experiment first.")
            return
        out = Path(self.settings.db).parent / f"report_exp{exp}.md"
        out.write_text(R.report_markdown(self._store(), exp))
        note.update(f"Wrote {out}")

    # ------------------------------------------------------------------------------------- tests
    def start_tests(self) -> None:
        log = self.query_one("#test-log", RichLog)
        if not self.selected_group:
            log.write("pick a test group first")
            return
        g = self.groups[self.selected_group]
        log.clear()
        if g.missing_env():
            log.write(f"note: {', '.join(g.missing_env())} not set, so these tests will skip")
        self.query_one("#test-start", Button).disabled = True
        self.run_tests_thread(g.command())

    @work(thread=True, exclusive=True, group="tests")
    def run_tests_thread(self, cmd: list[str]) -> None:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            self.call_from_thread(self.query_one("#test-log", RichLog).write, line.rstrip())
        code = proc.wait()
        self.call_from_thread(self.on_tests_finished, code)

    def on_tests_finished(self, code: int) -> None:
        self.query_one("#test-log", RichLog).write(f"exit code {code}")
        self.query_one("#test-start", Button).disabled = False
        self.last_test_exit = code

    # ------------------------------------------------------------------------------------- doctor
    def refresh_doctor(self) -> None:
        from a2e.doctor import run_checks
        t = self.query_one("#doctor-table", DataTable)
        t.clear()
        for c in run_checks(self.settings):
            t.add_row("ok" if c.ok else ("FAIL" if c.critical else "warn"), c.name, c.detail)


def main() -> None:
    A2EApp().run()


if __name__ == "__main__":
    main()
