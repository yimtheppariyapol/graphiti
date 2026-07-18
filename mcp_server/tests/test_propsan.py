"""sanitize_graph_property: nothing non-primitive may reach a FalkorDB property.

Proven failure mode (traceback 2026-07-17): an LLM attribute containing a nested dict rode
node.attributes into the bulk-save Cypher params and FalkorDB rejected the WHOLE transaction —
"Property values can only be of primitive types or arrays of primitive types" — losing every
entity and edge of that episode, 2-12 times a day. None is deliberately NOT coerced: every
entity edge carries expired_at=None and those writes succeed daily.
"""

import json
from datetime import datetime, timezone

from graphiti_core.utils.bulk_utils import _spread_sanitized_attributes, sanitize_graph_property


def test_primitives_pass_untouched():
    for v in ('ข้อความ', 42, 3.14, True, False):
        out, coerced = sanitize_graph_property(v)
        assert out == v and coerced is False


def test_none_and_datetime_pass_untouched():
    out, coerced = sanitize_graph_property(None)
    assert out is None and coerced is False   # expired_at=None saves fine today — keep it that way
    now = datetime.now(timezone.utc)
    out, coerced = sanitize_graph_property(now)
    assert out is now and coerced is False    # the driver serializes datetimes itself


def test_primitive_arrays_pass():
    out, coerced = sanitize_graph_property(['a', 'b'])
    assert out == ['a', 'b'] and coerced is False
    out, coerced = sanitize_graph_property((1, 2, 3))
    assert out == [1, 2, 3] and coerced is False
    out, coerced = sanitize_graph_property([0.1, None, 'x'])  # sparse primitive list stays a list
    assert out == [0.1, None, 'x'] and coerced is False


def test_dict_is_json_encoded():
    """THE production shape: a bilingual/nested LLM answer instead of a flat string."""
    v = {'th': 'นักพัฒนา', 'en': 'developer'}
    out, coerced = sanitize_graph_property(v)
    assert coerced is True
    assert isinstance(out, str)
    assert json.loads(out) == v, 'the data must survive as queryable JSON text, not be dropped'


def test_list_of_dicts_is_json_encoded():
    v = [{'id': 1}, {'id': 2}]
    out, coerced = sanitize_graph_property(v)
    assert coerced is True and isinstance(out, str)
    assert json.loads(out) == v


def test_unserializable_object_still_becomes_a_string():
    class Weird:
        def __str__(self):
            return 'weird'

    out, coerced = sanitize_graph_property({'obj': Weird()})
    assert coerced is True and isinstance(out, str)  # default=str — never raises mid-save


def test_spread_never_overwrites_explicit_fields():
    target = {'uuid': 'u1', 'name': 'real-name'}
    _spread_sanitized_attributes(
        target, {'name': 'stale-name', 'role': {'th': 'x'}}, uuid='u1', kind='node'
    )
    assert target['name'] == 'real-name', 'explicit save fields must win over attributes'
    assert isinstance(target['role'], str) and json.loads(target['role']) == {'th': 'x'}
