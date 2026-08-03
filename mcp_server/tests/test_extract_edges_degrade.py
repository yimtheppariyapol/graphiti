"""extract_edges must degrade, not drop.

The edge-extraction LLM (qwen-family via OpenRouter) intermittently answers with a bare edges
ARRAY rather than the {'edges': [...]} wrapper. `ExtractedEdges(**list)` is a TypeError inside
add_episode and the episode was lost with no retry — caught live by memory-canary 2026-08-04
(episode "memory-canary 20260803T195714Z" never reached the graph). Losing one extraction round
is recoverable; losing the episode is not. Mirrors test_noderes_degrade.py.
"""

from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.utils.maintenance.edge_operations import _normalize_edges_response

EDGE = {
    'relation_type': 'CHECKS',
    'source_entity_name': 'Dobby',
    'target_entity_name': 'fleet-memory host',
    'fact': 'Dobby checks the fleet-memory host',
    'valid_at': None,
    'invalid_at': None,
}


def test_bare_list_response_is_wrapped():
    """The exact shape that killed the canary episode in production."""
    normalized = _normalize_edges_response([EDGE])
    assert isinstance(normalized, dict)
    edges = ExtractedEdges(**normalized).edges
    assert len(edges) == 1
    assert edges[0].source_entity_name == 'Dobby'


def test_non_mapping_non_list_degrades_to_no_edges():
    for junk in ('a string', 42, None):
        normalized = _normalize_edges_response(junk)
        assert ExtractedEdges(**normalized).edges == []


def test_missing_edges_key_degrades_to_empty():
    assert ExtractedEdges().edges == []
    assert ExtractedEdges(**{}).edges == []


def test_well_formed_response_still_parses():
    """Relaxing the boundary must not weaken the happy path."""
    normalized = _normalize_edges_response({'edges': [EDGE]})
    edges = ExtractedEdges(**normalized).edges
    assert len(edges) == 1
    assert edges[0].fact == 'Dobby checks the fleet-memory host'
