"""Headless TUI tests: the real app driven by Textual's pilot, against the fake-server rig."""
import asyncio
import sys

import pytest

pytest.importorskip("textual")
pytest.importorskip("mpmath")

from textual.widgets import DataTable, ListView, RichLog, Static  # noqa: E402

from a2e.runner import Runner, RunnerHooks  # noqa: E402
from a2e.store import Store  # noqa: E402
from a2e.testgroups import Group  # noqa: E402
from a2e.tui import A2EApp  # noqa: E402
from tests.rig import SMALL, instruments, make_settings, write_suite  # noqa: E402


def rig(tmp_path):
    s = make_settings(tmp_path)
    suites = tmp_path / "suites"
    suites.mkdir()
    write_suite(suites, SMALL, name="tiny")
    groups = {"echo": Group(name="echo", description="prints hello")}
    groups["echo"].command = lambda extra=None: [sys.executable, "-c", "print('hello from tests')"]

    def factory(path, log, stop):
        return Runner(path, s, store=Store(s.db), log=log, stop=stop, hooks=RunnerHooks(instruments=instruments()))

    return A2EApp(settings=s, suites_dir=suites, groups=groups, runner_factory=factory), s


async def wait_for(pilot, cond, timeout=120):
    for _ in range(int(timeout * 10)):
        if cond():
            return
        await pilot.pause(0.1)
    raise AssertionError("timed out")


def log_text(app, wid):
    return "\n".join(str(line.text) for line in app.query_one(wid, RichLog).lines)


def test_suites_tab_lists_and_describes_suites(tmp_path):
    app, _ = rig(tmp_path)

    async def go():
        async with app.run_test(size=(140, 45)) as pilot:
            lv = app.query_one("#suite-list", ListView)
            assert len(lv.children) == 1
            app.select_suite(app.suites_dir / "tiny.toml")
            await pilot.pause()
            detail = str(app.query_one("#suite-detail", Static).render())
            assert "jobs: 8" in detail and "runnable" in detail
            assert "tiny" in str(app.query_one("#run-target", Static).render())

    asyncio.run(go())


def test_run_tab_runs_a_suite_and_results_tab_shows_it(tmp_path):
    app, s = rig(tmp_path)

    async def go():
        async with app.run_test(size=(160, 50)) as pilot:
            app.select_suite(app.suites_dir / "tiny.toml")
            await pilot.press("2")
            await pilot.click("#run-start")
            await wait_for(pilot, lambda: app.last_result is not None)
            assert app.last_result["counts"] == {"done": 4}
            text = log_text(app, "#run-log")
            assert "new experiment 1" in text and "finished experiment 1" in text
            assert app.query_one("#run-progress").progress == 4
            await pilot.press("3")
            assert app.query_one("#exp-table", DataTable).row_count == 1
            app.show_summary(1)
            await pilot.pause()
            assert app.query_one("#summary-table", DataTable).row_count == 4
            app.write_report()
            assert (tmp_path / "report_exp1.md").exists()

    asyncio.run(go())


def test_stop_button_stops_after_the_current_job(tmp_path):
    app, _ = rig(tmp_path)

    async def go():
        async with app.run_test(size=(160, 50)) as pilot:
            app.select_suite(app.suites_dir / "tiny.toml")
            app.start_run()
            app.stop_event.set()  # same as pressing the stop button right away
            await wait_for(pilot, lambda: app.last_result is not None)
            assert app.last_result["stopped"] and app.last_result["counts"].get("pending", 0) >= 3

    asyncio.run(go())


def test_run_without_a_suite_explains_what_to_do(tmp_path):
    app, _ = rig(tmp_path)

    async def go():
        async with app.run_test() as pilot:
            await pilot.press("2")  # RichLog renders once its tab has a size
            app.start_run()
            await pilot.pause()
            assert "pick a suite" in log_text(app, "#run-log")

    asyncio.run(go())


def test_tests_tab_runs_a_group_and_streams_output(tmp_path):
    app, _ = rig(tmp_path)

    async def go():
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.press("4")
            app.selected_group = "echo"
            app.start_tests()
            await wait_for(pilot, lambda: hasattr(app, "last_test_exit"))
            await pilot.pause()
            assert app.last_test_exit == 0 and "hello from tests" in log_text(app, "#test-log")

    asyncio.run(go())


def test_doctor_tab_shows_checks(tmp_path):
    app, _ = rig(tmp_path)

    async def go():
        async with app.run_test() as pilot:
            await pilot.press("5")
            assert app.query_one("#doctor-table", DataTable).row_count > 3

    asyncio.run(go())
