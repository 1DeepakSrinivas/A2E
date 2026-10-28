import time
from pathlib import Path

import typer

from a2e.client import Client
from a2e.config import RunConfig
from a2e.store import Store

app = typer.Typer(help="A2E: tool-calling profiling harness")
bfcl_app = typer.Typer(help="BFCL v4 (vendored official checker)")
app.add_typer(bfcl_app, name="bfcl")


@app.command()
def baseline(model: str, quant: str, device: str) -> None:
    """1 warm-up + 3 measured runs (implemented in Phase 5)."""
    raise typer.Exit("not implemented yet")


@bfcl_app.command("list")
def bfcl_list() -> None:
    """Categories in scope with their case counts."""
    from a2e.bfcl.load import ALL_CATEGORIES, load_cases
    for c in ALL_CATEGORIES:
        typer.echo(f"{c:24s} {len(load_cases(c)):5d}")


@bfcl_app.command("run")
def bfcl_run(
    url: str = typer.Option("http://127.0.0.1:8080", help="running llama-server"),
    category: list[str] = typer.Option(..., help="repeat for several categories"),
    model: str = typer.Option(...), quant: str = typer.Option(...), device: str = typer.Option(...),
    kv_quant: str = "f16", n: int = typer.Option(None, help="cases per category (fixed-seed stratified); all if unset"),
    seed: int = 0, grammar: bool = False, no_prefix_cache: bool = False, n_predict: int = 256,
    db: Path = Path("a2e.db"), raw: Path = Path("bfcl_raw.jsonl"),
) -> None:
    """Run BFCL categories against a running llama-server and store results (cases, turns, raw outputs)."""
    from a2e.bfcl.runner import run_category, safe_token_counter
    cfg = RunConfig(model=model, quant=quant, kv_quant=kv_quant, device=device, grammar=grammar,
                    prefix_cache=not no_prefix_cache)
    client, store = Client(url), Store(db)
    count = safe_token_counter(client)
    for cat in category:
        run_id = store.insert("runs", {"config_json": cfg.model_dump_json(), "device": device,
                                       "phase": f"bfcl:{cat}", "started_at": time.time()})
        outs = run_category(client, cat, store, run_id, n=n, seed=seed, raw_path=raw, grammar=grammar,
                            cache_prompt=not no_prefix_cache, n_predict=n_predict, count_tokens=count)
        ok = sum(o.result.valid for o in outs)
        typer.echo(f"{cat}: {ok}/{len(outs)} valid ({100 * ok / max(len(outs), 1):.1f}%)  run_id={run_id}")


@bfcl_app.command("rescore")
def bfcl_rescore(raw: Path = Path("bfcl_raw.jsonl")) -> None:
    """Re-score recorded raw outputs with the vendored checkers (no model needed)."""
    from collections import defaultdict

    from a2e.bfcl.runner import rescore
    tally = defaultdict(lambda: [0, 0])
    for r in rescore(raw):
        tally[r.category][0] += r.valid
        tally[r.category][1] += 1
    for cat, (ok, total) in tally.items():
        typer.echo(f"{cat}: {ok}/{total} valid ({100 * ok / total:.1f}%)")


if __name__ == "__main__":
    app()
