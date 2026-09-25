from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from graphiti_core.prompts.extract_nodes import EntitySummary, SummarizedEntity
from graphiti_core.prompts.summarize_nodes import Summary, SummaryDescription
from graphiti_core.utils.maintenance.community_operations import generate_summary_description


@pytest.mark.asyncio
async def test_summary_description_uses_bounded_cap_and_rejects_looping_text():
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={'description': 'x' * 8193})

    with pytest.raises(ValidationError, match='String should have at most 8192 characters'):
        await generate_summary_description(llm_client, 'short summary')

    assert llm_client.generate_response.await_args.kwargs['max_tokens'] == 2304


@pytest.mark.parametrize('model', [EntitySummary, SummarizedEntity, Summary, SummaryDescription])
def test_two_thousand_character_thai_summary_validates(model):
    thai_summary = 'ก' * 2000
    payload = {'summary': thai_summary}
    if model is SummarizedEntity:
        payload['name'] = 'entity'
    elif model is SummaryDescription:
        payload = {'description': thai_summary}

    assert model(**payload)


@pytest.mark.parametrize('model', [EntitySummary, SummarizedEntity, Summary, SummaryDescription])
def test_runaway_sized_summary_fails_validation(model):
    runaway = 'ก' * 38_450
    payload = {'summary': runaway}
    if model is SummarizedEntity:
        payload['name'] = 'entity'
    elif model is SummaryDescription:
        payload = {'description': runaway}

    with pytest.raises(ValidationError, match='String should have at most 8192 characters'):
        model(**payload)
