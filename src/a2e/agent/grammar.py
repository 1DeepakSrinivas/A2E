from __future__ import annotations

# JSON Schema keywords the tool parameter schemas may use. llama-server's json-schema-to-grammar
# converter supports these; anything else risks being ignored or rejected, so tests enforce this subset.
SUPPORTED_KEYWORDS = {"type", "description", "properties", "required", "enum", "items", "default",
                      "minimum", "maximum"}


def call_schema(specs: list[dict], max_calls: int = 4) -> dict:
    """JSON Schema of a valid response: an array of 1..max_calls calls, each one of the given tools.

    Sent as `json_schema` to llama-server, which converts it to a grammar (grammar-on arm of the ablation).
    """
    one_of = [{"type": "object",
               "properties": {"name": {"const": s["name"]}, "arguments": s["parameters"]},
               "required": ["name", "arguments"]} for s in specs]
    return {"type": "array", "items": {"anyOf": one_of}, "minItems": 1, "maxItems": max_calls}
