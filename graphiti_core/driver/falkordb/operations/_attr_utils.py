"""FalkorDB attribute sanitizer (Yim 2026-07-08, graphiti-extraction-model swap).

FalkorDB property values must be primitives or arrays of primitives. Some LLM
extractions (open models) return nested dict / list-of-dict attribute values,
which hard-fail the whole episode. Serialize non-primitive values to JSON strings
so the fact still lands (searchable as text) instead of dropping the episode."""
import json
from typing import Any

_PRIMITIVE = (str, int, float, bool)


def _is_primitive_array(v: Any) -> bool:
    return isinstance(v, list) and all(x is None or isinstance(x, _PRIMITIVE) for x in v)


def falkor_safe_attributes(attrs: dict[str, Any] | None) -> dict[str, Any]:
    if not attrs:
        return {}
    safe: dict[str, Any] = {}
    for k, v in attrs.items():
        if v is None or isinstance(v, _PRIMITIVE) or _is_primitive_array(v):
            safe[k] = v
        else:
            safe[k] = json.dumps(v, ensure_ascii=False, default=str)
    return safe
