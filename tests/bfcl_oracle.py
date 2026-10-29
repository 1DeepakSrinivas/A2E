"""Oracle and perturbation helpers for BFCL: turn ground truth into model outputs."""
from __future__ import annotations

import ast
import copy
import json

from a2e.bfcl.load import Case


def concretize(v):
    """Ground truth encodes dict-valued answers as {key: [acceptable values]}: pick one value per key."""
    if isinstance(v, dict):
        out = {}
        for k, vals in v.items():
            pick = next((x for x in vals if x != ""), None)
            if pick is not None or vals == [None]:
                out[k] = concretize(pick)
        return out
    if isinstance(v, list):
        return [concretize(x) for x in v]
    return v


def accepted_to_args(accepted: dict, doc: dict | None = None) -> dict:
    """{param: [accepted values...]} -> one minimal concrete call.

    "" in the list marks an optional param. It is omitted unless the function doc lists it as required (a few
    upstream docs disagree with their ground truth), and params the ground truth always requires are included.
    """
    required = set((doc or {}).get("parameters", {}).get("required", []))
    args = {}
    for p, vals in accepted.items():
        if "" in vals and p not in required:
            continue
        pick = next((v for v in vals if v != ""), None)
        if pick is not None or vals == [None]:
            args[p] = concretize(pick)
    return args


def single_calls(case: Case) -> list[dict]:
    docs = {d["name"]: d for d in case.functions}
    out = []
    for item in case.ground_truth:
        (name, accepted), = item.items()
        out.append({"name": name, "arguments": accepted_to_args(accepted, docs.get(name))})
    return out


def single_output(case: Case) -> str:
    return json.dumps(single_calls(case))


def parse_gt_string(s: str, docs: dict) -> dict:
    """'f(a=1, 2)' -> {"name": "f", "arguments": {...}} (positional args mapped via the doc's property order)."""
    call = ast.parse(s, mode="eval").body
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name), s
    name = call.func.id
    order = list(docs[name]["parameters"]["properties"])
    args = {order[i]: ast.literal_eval(a) for i, a in enumerate(call.args)}
    args.update({k.arg: ast.literal_eval(k.value) for k in call.keywords})
    return {"name": name, "arguments": args}


def multi_outputs(case: Case) -> list[str]:
    """Scripted model outputs: per user turn, all ground-truth calls in one step, then [] to end the turn."""
    docs = {d["name"]: d for d in case.functions + [d for ds in case.held_out.values() for d in ds]}
    outs = []
    for turn_gt in case.ground_truth:
        if turn_gt:
            outs.append(json.dumps([parse_gt_string(s, docs) for s in turn_gt]))
        outs.append("[]")
    return outs


def perturb_value(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + 123457.5
    if isinstance(v, str):
        return v + " zzzqqq"
    if isinstance(v, list):
        return v + ["zzzqqq"]
    if isinstance(v, dict):
        return {**v, "zzzqqq": 1}
    return "zzzqqq"


def perturbations(case: Case) -> dict[str, str]:
    """Named wrong outputs for an AST case; each must be scored invalid."""
    calls = single_calls(case)
    out = {"empty": "[]"}
    bad = copy.deepcopy(calls)
    bad[0]["name"] = "no_such_function_xyz"
    out["wrong_name"] = json.dumps(bad)
    if len(calls) > 1:
        out["dropped_call"] = json.dumps(calls[1:])
    out["duplicated_call"] = json.dumps(calls + calls[:1])
    bad = copy.deepcopy(calls)
    target = next(((i, p) for i, c in enumerate(bad) for p in c["arguments"]), None)
    if target:
        i, p = target
        bad[i]["arguments"][p] = perturb_value(bad[i]["arguments"][p])
        out["wrong_value"] = json.dumps(bad)
    return out
