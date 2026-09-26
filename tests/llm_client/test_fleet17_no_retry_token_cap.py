"""fleet.17: a token cap is not retried; other transient failures still are."""

import json

import pytest

from graphiti_core.llm_client.client import LLMClient, is_server_or_retry_error
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError, TokenCapError


class _CountingClient(LLMClient):
    def __init__(self, exc):
        super().__init__(LLMConfig(), cache=False)
        self.exc = exc
        self.calls = 0

    async def _generate_response(self, messages, response_model=None, max_tokens=0, model_size=None):
        self.calls += 1
        raise self.exc


def test_token_cap_is_not_retryable():
    assert is_server_or_retry_error(TokenCapError('LLM response hit the token cap (x)')) is False


@pytest.mark.parametrize(
    'exc',
    [
        EmptyResponseError('LLM returned an empty response'),
        json.decoder.JSONDecodeError('Unterminated string', '{"a', 2),
        RateLimitError(),
    ],
)
def test_other_transient_errors_still_retry(exc):
    assert is_server_or_retry_error(exc) is True


def test_token_cap_stays_an_empty_response_error():
    # the fleet.16 per-entity catch and the queue POISON regex both depend on this
    e = TokenCapError('LLM response hit the token cap (finish_reason=length, 10 chars)')
    assert isinstance(e, EmptyResponseError)
    assert 'hit the token cap' in str(e)


@pytest.mark.asyncio
async def test_token_cap_is_raised_after_exactly_one_call():
    client = _CountingClient(TokenCapError('LLM response hit the token cap (finish_reason=length, 9 chars)'))
    with pytest.raises(TokenCapError):
        await client._generate_response_with_retry([], None, 768)
    assert client.calls == 1
