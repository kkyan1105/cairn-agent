import asyncio

import pytest

from agents import agent


class ApiError(Exception):
    def __init__(self, message="", status_code=None):
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def instant(_seconds):
        return None

    monkeypatch.setattr(agent.asyncio, "sleep", instant)
    monkeypatch.setattr(agent, "print_retry", lambda *args: None)


@pytest.mark.parametrize("error", [ApiError(status_code=429), ApiError(status_code=529), ApiError("server overloaded"), ApiError("ECONNRESET")])
def test_transient_errors_are_retryable(error):
    assert agent._is_retryable(error)


@pytest.mark.parametrize("error", [ApiError(status_code=400), ApiError(status_code=401), ValueError("bad input")])
def test_client_errors_are_not_retryable(error):
    assert not agent._is_retryable(error)


def test_retries_until_success():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ApiError(status_code=503)
        return "ok"

    assert asyncio.run(agent._with_retry(flaky)) == "ok"
    assert len(calls) == 3


def test_gives_up_after_max_retries():
    calls = []

    async def always_busy():
        calls.append(1)
        raise ApiError(status_code=429)

    with pytest.raises(ApiError):
        asyncio.run(agent._with_retry(always_busy, max_retries=2))
    assert len(calls) == 3


def test_non_retryable_error_fails_immediately():
    calls = []

    async def broken():
        calls.append(1)
        raise ApiError(status_code=400)

    with pytest.raises(ApiError):
        asyncio.run(agent._with_retry(broken))
    assert len(calls) == 1
