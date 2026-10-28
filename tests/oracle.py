"""Oracle model: emits exactly the expected call for each step (resolving dependent args from real tool output)."""
import json
from pathlib import Path

from a2e.tools.builtin import build_registry, make_corpus


def make_env(tmp: Path, name="env"):
    wd = tmp / name
    meta = make_corpus(wd, n=12, seed=1)
    return build_registry(wd), meta


def oracle_outputs(task, tmp: Path, name="oracle") -> list[str]:
    reg, _ = make_env(tmp, name)
    results, outs = [], []
    for step in task.steps:
        args = step.resolve(results)
        outs.append(json.dumps([{"name": step.name, "arguments": args}]))
        results.append(reg.execute(step.name, args).output)
    outs.append(json.dumps([{"name": "final_answer", "arguments": {"answer": "done"}}]))
    return outs
