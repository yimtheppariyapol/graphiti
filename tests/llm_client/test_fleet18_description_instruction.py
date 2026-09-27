"""fleet.18: attribute-extraction calls carry the one-sentence description instruction; others do not."""

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message

KEY = 'For any `description` field: write ONE sentence of at most 250 characters'


class _C(LLMClient):
    async def _generate_response(self, *a, **k):  # pragma: no cover - not called
        raise NotImplementedError


def _msgs():
    return [Message(role='system', content='sys'), Message(role='user', content='u')]


def test_instruction_added_for_attribute_extraction():
    c = _C(LLMConfig(), cache=False)
    m = _msgs()
    c._apply_attribute_extraction_preamble(m, True)
    assert KEY in m[0].content
    assert 'Never list events, decisions or facts' in m[0].content


def test_instruction_absent_for_other_prompts():
    c = _C(LLMConfig(), cache=False)
    m = _msgs()
    c._apply_attribute_extraction_preamble(m, False)
    assert KEY not in m[0].content


def test_instruction_added_once():
    c = _C(LLMConfig(), cache=False)
    m = _msgs()
    c._apply_attribute_extraction_preamble(m, True)
    c._apply_attribute_extraction_preamble(m, True)
    assert m[0].content.count(KEY) == 1
