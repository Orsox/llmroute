import asyncio
import logging

import llmrouter_router_service.backends.transwarp_external_openai_backend as external_backend
from llmrouter_router_service.backends import (
    BackendChatRequestNode,
    ExternalOpenAICompatibleBackendAdapter,
    ExternalOpenAICompatibleModelNode,
)


class _MockResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_external_openai_backend_probe_uses_five_minute_cache(monkeypatch) -> None:
    request_count = {"get": 0}

    class MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str) -> _MockResponse:
            request_count["get"] += 1
            return _MockResponse({"data": [{"id": "collective-alpha"}]})

    monkeypatch.setattr(external_backend.httpx, "AsyncClient", MockAsyncClient)

    adapter = ExternalOpenAICompatibleBackendAdapter(
        backend_id="node-0",
        backend_kind="openai-compatible",
        base_url="http://127.0.0.1:1234",
        models=[
            ExternalOpenAICompatibleModelNode(
                model_id="collective-alpha",
                context_window_tokens=8192,
                priority=10,
            )
        ],
    )

    first_probe = asyncio.run(adapter.probe_backend())
    second_probe = asyncio.run(adapter.probe_backend())

    assert first_probe.health.message == second_probe.health.message
    assert request_count["get"] == 1


def test_external_openai_backend_generate_chat_completion_preserves_tool_calls(
    monkeypatch,
) -> None:
    observed_payloads: list[dict[str, object]] = []

    class MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> _MockResponse:
            observed_payloads.append(json)
            return _MockResponse(
                {
                    "id": "resp_123",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "type": "function",
                                        "function": {
                                            "name": "inspect_cube",
                                            "arguments": "{\"target\":\"alpha\"}",
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 4,
                        "total_tokens": 15,
                    },
                }
            )

    monkeypatch.setattr(external_backend.httpx, "AsyncClient", MockAsyncClient)

    adapter = ExternalOpenAICompatibleBackendAdapter(
        backend_id="node-0",
        backend_kind="openai-compatible",
        base_url="http://127.0.0.1:1234",
        models=[
            ExternalOpenAICompatibleModelNode(
                model_id="collective-alpha",
                context_window_tokens=8192,
                priority=10,
            )
        ],
    )

    response = asyncio.run(
        adapter.generate_chat_completion(
            BackendChatRequestNode.model_validate(
                {
                    "backend_id": "node-0",
                    "model_id": "collective-alpha",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Inspect the cube.",
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "inspect_cube",
                                "parameters": {"type": "object"},
                            },
                        }
                    ],
                    "tool_choice": "auto",
                }
            )
        )
    )

    assert observed_payloads[0]["tool_choice"] == "auto"
    assert observed_payloads[0]["tools"][0]["function"]["name"] == "inspect_cube"
    assert response.output_text is None
    assert response.tool_calls[0]["function"]["name"] == "inspect_cube"
    assert response.finish_reason.value == "tool_calls"


def test_lmstudio_backend_disables_thinking_on_openai_compat_path_when_needed(
    monkeypatch,
) -> None:
    observed_payloads: list[dict[str, object]] = []

    class MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> _MockResponse:
            observed_payloads.append(json)
            return _MockResponse(
                {
                    "id": "resp_123",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Direct answer.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                }
            )

    monkeypatch.setattr(external_backend.httpx, "AsyncClient", MockAsyncClient)

    adapter = ExternalOpenAICompatibleBackendAdapter(
        backend_id="node-0",
        backend_kind="lmstudio",
        base_url="http://127.0.0.1:1234",
        models=[
            ExternalOpenAICompatibleModelNode(
                model_id="collective-alpha",
                context_window_tokens=8192,
                priority=10,
            )
        ],
    )

    asyncio.run(
        adapter.generate_chat_completion(
            BackendChatRequestNode.model_validate(
                    {
                        "backend_id": "node-0",
                        "model_id": "collective-alpha",
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "Hello."}],
                            }
                        ],
                        "metadata": {"lmstudio.enable_thinking": "false"},
                    }
                )
            )
        )

    assert observed_payloads[0]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_lmstudio_backend_uses_native_chat_api_for_simple_non_tool_requests(
    monkeypatch,
) -> None:
    observed_requests: list[tuple[str, dict[str, object]]] = []

    class MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> _MockResponse:
            observed_requests.append((url, json))
            return _MockResponse(
                {
                    "response_id": "resp_native_123",
                    "output": [
                        {
                            "type": "reasoning",
                            "content": "hidden chain of thought",
                        },
                        {
                            "type": "message",
                            "content": "Direct answer.",
                        },
                    ],
                    "stats": {
                        "input_tokens": 3,
                        "total_output_tokens": 2,
                    },
                }
            )

    monkeypatch.setattr(external_backend.httpx, "AsyncClient", MockAsyncClient)

    adapter = ExternalOpenAICompatibleBackendAdapter(
        backend_id="node-0",
        backend_kind="lmstudio",
        base_url="http://127.0.0.1:1234",
        models=[
            ExternalOpenAICompatibleModelNode(
                model_id="collective-alpha",
                context_window_tokens=8192,
                priority=10,
            )
        ],
    )

    response = asyncio.run(
        adapter.generate_chat_completion(
            BackendChatRequestNode.model_validate(
                {
                    "backend_id": "node-0",
                    "model_id": "collective-alpha",
                    "messages": [{"role": "user", "content": "Hello."}],
                    "metadata": {"lmstudio.enable_thinking": "false"},
                }
            )
        )
    )

    assert observed_requests[0][0] == "http://127.0.0.1:1234/api/v1/chat"
    assert observed_requests[0][1]["reasoning"] == "off"
    assert observed_requests[0][1]["input"] == [
        {"type": "message", "role": "user", "content": "Hello."}
    ]
    assert response.output_text == "Direct answer."
    assert response.backend_response_id == "resp_native_123"


def test_external_openai_backend_ignores_reasoning_content_without_answer_text(
    monkeypatch,
) -> None:
    class MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> _MockResponse:
            return _MockResponse(
                {
                    "id": "resp_789",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_content": "hidden chain of thought",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                }
            )

    monkeypatch.setattr(external_backend.httpx, "AsyncClient", MockAsyncClient)

    adapter = ExternalOpenAICompatibleBackendAdapter(
        backend_id="node-0",
        backend_kind="openai-compatible",
        base_url="http://127.0.0.1:1234",
        models=[
            ExternalOpenAICompatibleModelNode(
                model_id="collective-alpha",
                context_window_tokens=8192,
                priority=10,
            )
        ],
    )

    try:
        asyncio.run(
            adapter.generate_chat_completion(
                BackendChatRequestNode.model_validate(
                    {
                        "backend_id": "node-0",
                        "model_id": "collective-alpha",
                        "messages": [{"role": "user", "content": "Hello."}],
                    }
                )
            )
        )
    except RuntimeError as exc:
        assert "no usable assistant text" in str(exc)
    else:
        raise AssertionError("Expected reasoning-only payload to be rejected")


def test_external_openai_backend_logs_request_and_response_correlation(
    monkeypatch,
    caplog,
) -> None:
    class MockAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> _MockResponse:
            return _MockResponse(
                {
                    "id": "resp_456",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Direct answer.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                }
            )

    monkeypatch.setattr(external_backend.httpx, "AsyncClient", MockAsyncClient)

    adapter = ExternalOpenAICompatibleBackendAdapter(
        backend_id="node-0",
        backend_kind="openai-compatible",
        base_url="http://127.0.0.1:1234",
        models=[
            ExternalOpenAICompatibleModelNode(
                model_id="collective-alpha",
                context_window_tokens=8192,
                priority=10,
            )
        ],
    )

    with caplog.at_level(logging.INFO, logger="llmrouter.router_service"):
        asyncio.run(
            adapter.generate_chat_completion(
                BackendChatRequestNode.model_validate(
                    {
                        "backend_id": "node-0",
                        "model_id": "collective-alpha",
                        "messages": [{"role": "user", "content": "Hello."}],
                        "metadata": {
                            "llmrouter.request_id": "chatcmpl-test-123",
                            "llmrouter.exact_request_hash": "a" * 64,
                        },
                    }
                )
            )
        )

    assert "chat_backend_dispatch request_id=chatcmpl-test-123" in caplog.text
    assert "chat_backend_response request_id=chatcmpl-test-123" in caplog.text
    assert "exact_hash=" + ("a" * 64) in caplog.text
