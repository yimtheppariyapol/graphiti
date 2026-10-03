"""Guards fleet.19: extraction responses shaped as str/list must not dead-letter an episode.

Remove the coercion in node_operations._as_list_field_mapping and test_list_* / test_str_* fail with
"argument after ** must be a mapping" — the exact error that dead-lettered two episodes on 2026-09-29.
"""
import pytest
from graphiti_core.prompts.extract_nodes import ExtractedEntities, SummarizedEntities
from graphiti_core.utils.maintenance.node_operations import _as_list_field_mapping

ENT = [{"name": "Yim", "entity_type_id": 0}]
SUM = [{"name": "Yim", "summary": "Owner of the fleet."}]

def test_dict_passthrough_unchanged():
    d = {"extracted_entities": ENT}
    assert _as_list_field_mapping(d, "extracted_entities") is d

def test_list_is_wrapped_for_extracted_entities():
    assert ExtractedEntities(**_as_list_field_mapping(ENT, "extracted_entities")).extracted_entities[0].name == "Yim"

def test_str_json_is_parsed_for_summaries():
    import json
    r = SummarizedEntities(**_as_list_field_mapping(json.dumps({"summaries": SUM}), "summaries"))
    assert r.summaries[0].summary == "Owner of the fleet."

def test_str_json_list_is_parsed_and_wrapped():
    import json
    assert SummarizedEntities(**_as_list_field_mapping(json.dumps(SUM), "summaries")).summaries[0].name == "Yim"

def test_garbage_still_fails_loudly():
    with pytest.raises(TypeError):
        _as_list_field_mapping("not json at all", "summaries")
    with pytest.raises(TypeError):
        _as_list_field_mapping(42, "summaries")
