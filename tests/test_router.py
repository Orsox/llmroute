import json
import logging
import sqlite3
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from llmrouter.app import (
    DEFAULT_TOOLUSE_SYSTEM_HINT,
    LMStudioClient,
    LMStudioSettings,
    RouteDecision,
    RouterService,
    UpstreamError,
    UnifiedRequest,
    anthropic_to_openai_payload,
    create_app,
    normalize_anthropic_messages,
    normalize_openai_chat,
)


def _write_config(
    path: Path,
    token: str | None = None,
    small_context: int = 32996,
    default_temperature: float | None = None,
    repetition_similarity_threshold: float = 0.92,
    require_session_id: bool = True,
) -> None:
    data = {
        "server": {
            "host": "0.0.0.0",
            "port": 12345,
        },
        "upstreams": {
            "local": {
                "provider": "lm_studio",
                "base_url": "http://localhost:1234",
                "timeout_seconds": 30,
                "api_key": None,
                "api_key_env": "OPENAI_API_KEY",
                "organization": None,
                "project": None,
            },
            "deep": {
                "provider": "lm_studio",
                "base_url": "http://localhost:1234",
                "timeout_seconds": 30,
                "api_key": None,
                "api_key_env": "DEEP_API_KEY",
                "organization": None,
                "project": None,
            },
        },
        "security": {
            "shared_bearer_token": token,
        },
        "routing": {
            "judge_timeout_seconds": 5,
            "fallback_enabled": True,
            "hybrid_client_model_override": True,
            "default_temperature": default_temperature,
            "analytics_enabled": True,
            "analytics_sqlite_path": str(path.parent / "router_analytics.sqlite"),
            "heuristics": {
                "large_prompt_token_threshold": 1200,
                "large_max_tokens_threshold": 700,
                "judge_temperature": 0.0,
                "judge_max_tokens": 32,
                "judge_prompt_context_chars": 1200,
                "lightweight_max_tokens_cap": 384,
                "suspect_default_max_tokens_threshold": 2048,
            },
            "session_memory": {
                "enabled": True,
                "require_session_id": require_session_id,
                "max_sessions": 64,
                "max_entries_per_session": 16,
            },
            "repetition_escalation": {
                "enabled": True,
                "history_limit": 6,
                "min_streak": 1,
                "similarity_threshold": repetition_similarity_threshold,
            },
        },
        "router_identity": {
            "exposed_model_name": "borg-cpu",
            "publish_underlying_models": False,
        },
        "models": {
            "small": {
                "model_id": "qwen/qwen3-vl-8b",
                "context_window": small_context,
                "capabilities": ["chat", "completions", "vision", "tooluse"],
                "upstream_ref": "local",
                "relative_speed": 3.0,
                "suitable_for": "small",
            },
            "large": {
                "model_id": "qwen/qwen3.5-35b-a3b",
                "context_window": 262144,
                "capabilities": ["chat", "completions", "tooluse"],
                "upstream_ref": "local",
                "relative_speed": 1.0,
                "suitable_for": "large",
            },
            "deep": {
                "model_id": "gpt-4.1",
                "context_window": 200000,
                "capabilities": ["chat", "completions", "tooluse"],
                "upstream_ref": "deep",
                "relative_speed": 0.5,
                "suitable_for": "deep",
            },
            "backup": {
                "model_id": "gpt-4o-mini",
                "context_window": 128000,
                "capabilities": ["chat", "completions", "tooluse"],
                "upstream_ref": "deep",
                "relative_speed": 2.0,
                "suitable_for": "backup",
            },
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


class FakeLMClient:
    def __init__(self, fail_first_small: bool = False):
        self.calls: list[tuple[str, str]] = []
        self.fail_first_small = fail_first_small
        self.failed_once = False
        self.last_judge_payload: dict | None = None

    async def post_json(self, settings: LMStudioSettings, path: str, payload: dict):
        model = payload.get("model", "")
        self.calls.append((path, model))

        messages = payload.get("messages") or []
        if messages and isinstance(messages, list):
            first = messages[0]
            if isinstance(first, dict) and "router judge" in str(first.get("content", "")).lower():
                self.last_judge_payload = dict(payload)
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"route":"small","reason_code":"simple"}'
                            }
                        }
                    ]
                }

        if self.fail_first_small and model == "qwen/qwen3-vl-8b" and not self.failed_once:
            self.failed_once = True
            raise UpstreamError(500, "small model temporary failure")

        if path == "/v1/completions":
            return {
                "id": "cmpl_1",
                "choices": [{"text": "completion-ok", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            }

        return {
            "id": "chatcmpl_1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"response-from-{model}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }

    async def stream_openai(self, settings: LMStudioSettings, path: str, payload: dict):
        model = payload.get("model", "")
        self.calls.append((path + ":stream", model))
        if self.fail_first_small and model == "qwen/qwen3-vl-8b" and not self.failed_once:
            self.failed_once = True
            raise UpstreamError(500, "small stream failure")

        yield b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}' + b"\n\n"
        yield b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}' + b"\n\n"
        yield b"data: [DONE]\n\n"


class CapturePayloadLMClient(FakeLMClient):
    def __init__(self):
        super().__init__()
        self.last_payload: dict | None = None

    async def post_json(self, settings: LMStudioSettings, path: str, payload: dict):
        self.last_payload = dict(payload)
        return await super().post_json(settings, path, payload)


class ToolCallLMClient(FakeLMClient):
    async def post_json(self, settings: LMStudioSettings, path: str, payload: dict):
        model = payload.get("model", "")
        self.calls.append((path, model))
        return {
            "id": "chatcmpl_tool",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "toolu_weather_1",
                                "type": "function",
                                "function": {
                                    "name": "weather_lookup",
                                    "arguments": '{"city":"Berlin"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        }

    async def stream_openai(self, settings: LMStudioSettings, path: str, payload: dict):
        model = payload.get("model", "")
        self.calls.append((path + ":stream", model))
        yield (
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"toolu_weather_1","type":"function","function":{"name":"weather_lookup","arguments":"{\\"city\\":\\""}}]},"finish_reason":null}]}\n\n'
        )
        yield (
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"Berlin\\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":8}}\n\n'
        )
        yield b"data: [DONE]\n\n"


class EmptyJudgeLMClient(FakeLMClient):
    async def post_json(self, settings: LMStudioSettings, path: str, payload: dict):
        messages = payload.get("messages") or []
        if messages and isinstance(messages, list):
            first = messages[0]
            if isinstance(first, dict) and "router judge" in str(first.get("content", "")).lower():
                return {"choices": [{"message": {"content": ""}}]}
        return await super().post_json(settings, path, payload)


class EmptyAnthropicSmallThenLargeTextLMClient(FakeLMClient):
    async def stream_openai(self, settings: LMStudioSettings, path: str, payload: dict):
        model = payload.get("model", "")
        self.calls.append((path + ":stream", model))
        if model == "qwen/qwen3-vl-8b":
            # Simulate a stream that finishes without text or tool calls.
            yield b'data: {"choices":[{"delta":{},"finish_reason":null}]}\n\n'
            yield (
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":1}}\n\n'
            )
            yield b"data: [DONE]\n\n"
            return

        yield b'data: {"choices":[{"delta":{"content":"fallback works"},"finish_reason":null}]}\n\n'
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":3}}\n\n'
        yield b"data: [DONE]\n\n"


class ModelCatalogLMClient(FakeLMClient):
    def __init__(self, items: list[dict]):
        super().__init__()
        self.items = items
        self.list_calls = 0

    async def list_models(self, settings: LMStudioSettings):
        self.list_calls += 1
        return "/v1/models", self.items


@pytest.fixture
def cfg_file(tmp_path: Path) -> Path:
    cfg = tmp_path / "router_config.yaml"
    _write_config(cfg)
    return cfg


@pytest.fixture(autouse=True)
def _default_deep_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEP_ENABLED", "false")


def test_normalize_openai_chat_detects_vision_and_tooluse() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ],
        "tools": [{"type": "function", "function": {"name": "x"}}],
        "max_tokens": 100,
    }
    req = normalize_openai_chat(payload)
    assert req.needs_vision is True
    assert req.needs_tooluse is True
    assert req.required_capabilities == {"chat", "vision", "tooluse"}


def test_normalize_anthropic_messages_strips_wrapper_noise_for_routing() -> None:
    payload = {
        "model": "borg-cpu",
        "max_tokens": 32000,
        "tools": [{"name": "weather_lookup", "input_schema": {"type": "object", "properties": {}}}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<local-command-caveat>Caveat</local-command-caveat>\n"
                            "<command-name>/model</command-name>\n"
                            "<local-command-stdout>Set model to borg-cpu</local-command-stdout>\n"
                            "hallo"
                        ),
                    }
                ],
            }
        ],
    }
    req = normalize_anthropic_messages(payload)
    assert req.latest_user_prompt_text.endswith("hallo")
    assert req.routing_latest_user_prompt_text == "hallo"
    assert req.has_wrapper_noise is True
    assert req.tool_loop_context is False


@pytest.mark.asyncio
async def test_choose_route_large_when_small_context_is_not_enough(cfg_file: Path) -> None:
    _write_config(cfg_file, small_context=500)
    service = RouterService(config_store=create_app(config_path=cfg_file).state.config_store, lm_client=FakeLMClient())
    cfg = service.config_store.get_config()
    req = UnifiedRequest(
        source_api="openai_chat",
        requested_model=None,
        stream=False,
        max_tokens=1200,
        prompt_text="x" * 6000,
        estimated_input_tokens=1600,
        needs_vision=False,
        needs_tooluse=False,
        required_base_capability="chat",
    )
    decision: RouteDecision = await service.choose_route(cfg, req)
    assert decision.selected_alias == "large"
    assert decision.reason == "heuristic_fallback"


@pytest.mark.asyncio
async def test_choose_route_uses_judge_result_for_coding_prompt(cfg_file: Path) -> None:
    service = RouterService(config_store=create_app(config_path=cfg_file).state.config_store, lm_client=FakeLMClient())
    cfg = service.config_store.get_config()
    req = UnifiedRequest(
        source_api="anthropic_messages",
        requested_model="borg-cpu",
        stream=True,
        max_tokens=300,
        prompt_text="Schreibe Python-Code fuer eine FastAPI Route mit Validierung.",
        estimated_input_tokens=40,
        needs_vision=False,
        needs_tooluse=False,
        required_base_capability="chat",
    )
    decision: RouteDecision = await service.choose_route(cfg, req)
    assert decision.selected_alias == "small"
    assert decision.reason == "judge_small"


@pytest.mark.asyncio
async def test_choose_route_prefers_small_for_non_coding_prompt(cfg_file: Path) -> None:
    service = RouterService(config_store=create_app(config_path=cfg_file).state.config_store, lm_client=FakeLMClient())
    cfg = service.config_store.get_config()
    req = UnifiedRequest(
        source_api="openai_chat",
        requested_model="borg-cpu",
        stream=False,
        max_tokens=200,
        prompt_text="Erklaere mir in einfachen Worten, wie Photosynthese funktioniert.",
        estimated_input_tokens=30,
        needs_vision=False,
        needs_tooluse=False,
        required_base_capability="chat",
    )
    decision: RouteDecision = await service.choose_route(cfg, req)
    assert decision.selected_alias == "small"
    assert decision.reason == "judge_small"


@pytest.mark.asyncio
async def test_client_large_model_is_ignored_for_non_coding_prompt(cfg_file: Path) -> None:
    service = RouterService(config_store=create_app(config_path=cfg_file).state.config_store, lm_client=FakeLMClient())
    cfg = service.config_store.get_config()
    req = UnifiedRequest(
        source_api="openai_chat",
        requested_model="qwen/qwen3.5-35b-a3b",
        stream=False,
        max_tokens=200,
        prompt_text="Fasse bitte die Kernideen aus dem Text zusammen.",
        estimated_input_tokens=24,
        needs_vision=False,
        needs_tooluse=False,
        required_base_capability="chat",
    )
    decision: RouteDecision = await service.choose_route(cfg, req)
    assert decision.selected_alias == "small"
    assert decision.reason == "judge_small"


@pytest.mark.asyncio
async def test_judge_empty_defaults_to_small_even_when_coding_like(cfg_file: Path) -> None:
    service = RouterService(
        config_store=create_app(config_path=cfg_file).state.config_store,
        lm_client=EmptyJudgeLMClient(),
    )
    cfg = service.config_store.get_config()
    req = UnifiedRequest(
        source_api="openai_chat",
        requested_model="borg-cpu",
        stream=False,
        max_tokens=32000,
        prompt_text="Bitte schreibe eine Python-Funktion.",
        user_prompt_text="Bitte schreibe eine Python-Funktion.",
        latest_user_prompt_text="Bitte schreibe eine Python-Funktion.",
        estimated_input_tokens=20,
        needs_vision=False,
        needs_tooluse=False,
        required_base_capability="chat",
    )
    decision: RouteDecision = await service.choose_route(cfg, req)
    # candidates are ['large', 'backup'] because small limit is exceeded
    assert decision.selected_alias == "large"
    assert decision.reason == "heuristic_fallback"


@pytest.mark.asyncio
async def test_choose_route_uses_small_model_as_judge_for_multi_candidate(cfg_file: Path) -> None:
    lm = FakeLMClient()
    service = RouterService(config_store=create_app(config_path=cfg_file).state.config_store, lm_client=lm)
    cfg = service.config_store.get_config()
    req = UnifiedRequest(
        source_api="openai_chat",
        requested_model="borg-cpu",
        stream=False,
        max_tokens=120,
        prompt_text="hallo",
        user_prompt_text="hallo",
        estimated_input_tokens=2,
        needs_vision=False,
        needs_tooluse=False,
        required_base_capability="chat",
    )
    _ = await service.choose_route(cfg, req)
    assert lm.calls
    assert lm.calls[0] == ("/v1/chat/completions", "qwen/qwen3-vl-8b")


@pytest.mark.asyncio
async def test_choose_route_prefers_small_for_light_anthropic_tool_request_with_wrapper_noise(cfg_file: Path) -> None:
    service = RouterService(config_store=create_app(config_path=cfg_file).state.config_store, lm_client=FakeLMClient())
    cfg = service.config_store.get_config()
    req = normalize_anthropic_messages(
        {
            "model": "borg-cpu",
            "stream": True,
            "max_tokens": 32000,
            "tools": [{"name": "weather_lookup", "input_schema": {"type": "object", "properties": {}}}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<local-command-caveat>Caveat</local-command-caveat>\n"
                                "<command-name>/model</command-name>\n"
                                "<local-command-stdout>Set model to borg-cpu</local-command-stdout>\n"
                                "hallo"
                            ),
                        }
                    ],
                }
            ],
        }
    )
    decision = await service.choose_route(cfg, req)
    assert decision.selected_alias == "small"
    assert decision.routing_max_tokens_budget == 384
    assert decision.routing_latest_user_prompt_text == "hallo"


@pytest.mark.asyncio
async def test_judge_prompt_uses_sanitized_latest_user_text(cfg_file: Path) -> None:
    lm = FakeLMClient()
    service = RouterService(config_store=create_app(config_path=cfg_file).state.config_store, lm_client=lm)
    cfg = service.config_store.get_config()
    req = normalize_anthropic_messages(
        {
            "model": "borg-cpu",
            "stream": True,
            "max_tokens": 32000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<local-command-caveat>Caveat</local-command-caveat>\n"
                                "<command-name>/model</command-name>\n"
                                "<local-command-stdout>Set model to borg-cpu</local-command-stdout>\n"
                                "hallo"
                            ),
                        }
                    ],
                }
            ],
        }
    )
    _ = await service.choose_route(cfg, req)
    assert lm.last_judge_payload is not None
    judge_body = json.loads(lm.last_judge_payload["messages"][1]["content"])
    assert judge_body["latest_user_prompt_excerpt"] == "hallo"


@pytest.mark.asyncio
async def test_judge_prompt_includes_recent_request_memory(cfg_file: Path) -> None:
    lm = FakeLMClient()
    service = RouterService(config_store=create_app(config_path=cfg_file).state.config_store, lm_client=lm)
    cfg = service.config_store.get_config()

    first_req = UnifiedRequest(
        source_api="openai_chat",
        session_id="sess-a",
        requested_model="borg-cpu",
        stream=False,
        max_tokens=120,
        prompt_text="Bitte erklaere Quantencomputing kurz in einfachen Worten.",
        user_prompt_text="Bitte erklaere Quantencomputing kurz in einfachen Worten.",
        latest_user_prompt_text="Bitte erklaere Quantencomputing kurz in einfachen Worten.",
        estimated_input_tokens=20,
        needs_vision=False,
        needs_tooluse=False,
        required_base_capability="chat",
    )
    second_req = UnifiedRequest(
        source_api="openai_chat",
        session_id="sess-a",
        requested_model="borg-cpu",
        stream=False,
        max_tokens=120,
        prompt_text="Bitte erklaere Quantencomputing bitte kurz in sehr einfachen Worten.",
        user_prompt_text="Bitte erklaere Quantencomputing bitte kurz in sehr einfachen Worten.",
        latest_user_prompt_text="Bitte erklaere Quantencomputing bitte kurz in sehr einfachen Worten.",
        estimated_input_tokens=24,
        needs_vision=False,
        needs_tooluse=False,
        required_base_capability="chat",
    )

    _ = await service.choose_route(cfg, first_req)
    _ = await service.choose_route(cfg, second_req)

    assert lm.last_judge_payload is not None
    judge_body = json.loads(lm.last_judge_payload["messages"][1]["content"])
    memory = judge_body["features"]["recent_request_memory"]
    assert memory["previous_request"] is not None
    assert memory["previous_request"]["selected_alias"] == "small"
    assert memory["previous_request_similarity"] > 0.8
    assert "Quantencomputing" in memory["previous_request"]["prompt_excerpt"]


@pytest.mark.asyncio
async def test_choose_route_keeps_large_for_real_tool_loop_context(cfg_file: Path) -> None:
    service = RouterService(config_store=create_app(config_path=cfg_file).state.config_store, lm_client=FakeLMClient())
    cfg = service.config_store.get_config()
    req = normalize_anthropic_messages(
        {
            "model": "borg-cpu",
            "stream": True,
            "max_tokens": 32000,
            "tools": [{"name": "file_read", "input_schema": {"type": "object", "properties": {}}}],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "file_read",
                            "input": {"path": "app.py"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "def main(): pass"}],
                        },
                        {"type": "text", "text": "Bitte erklaere mir den Code."},
                    ],
                },
            ],
        }
    )
    decision = await service.choose_route(cfg, req)
    assert decision.selected_alias == "large"
    assert decision.tool_loop_context is True


@pytest.mark.asyncio
async def test_choose_route_can_select_deep_when_enabled_and_judge_unavailable(
    cfg_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEP_ENABLED", "true")
    service = RouterService(
        config_store=create_app(config_path=cfg_file).state.config_store,
        lm_client=EmptyJudgeLMClient(),
    )
    cfg = service.config_store.get_config()
    req = UnifiedRequest(
        source_api="openai_chat",
        requested_model="borg-cpu",
        stream=False,
        max_tokens=600,
        prompt_text="Bitte bewerte Architektur Trade-off und Compliance-Risiken fuer den Rollout.",
        user_prompt_text="Bitte bewerte Architektur Trade-off und Compliance-Risiken fuer den Rollout.",
        latest_user_prompt_text="Bitte bewerte Architektur Trade-off und Compliance-Risiken fuer den Rollout.",
        estimated_input_tokens=50000,
        needs_vision=False,
        needs_tooluse=False,
        required_base_capability="chat",
    )
    decision: RouteDecision = await service.choose_route(cfg, req)
    assert decision.selected_alias == "deep"
    assert decision.reason == "policy_deep_reasoning_or_websearch"


@pytest.mark.asyncio
async def test_choose_route_prefers_explicit_deep_model_when_judge_unavailable(
    cfg_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEP_ENABLED", "true")
    service = RouterService(
        config_store=create_app(config_path=cfg_file).state.config_store,
        lm_client=EmptyJudgeLMClient(),
    )
    cfg = service.config_store.get_config()
    req = UnifiedRequest(
        source_api="openai_chat",
        requested_model="gpt-4.1",
        stream=False,
        max_tokens=300,
        prompt_text="Kurze Frage ohne Coding.",
        user_prompt_text="Kurze Frage ohne Coding.",
        latest_user_prompt_text="Kurze Frage ohne Coding.",
        estimated_input_tokens=20,
        needs_vision=False,
        needs_tooluse=False,
        required_base_capability="chat",
    )
    decision: RouteDecision = await service.choose_route(cfg, req)
    assert decision.selected_alias == "deep"
    assert decision.reason == "client_model_preference_judge_unavailable"


def test_auth_enforced_for_api(tmp_path: Path) -> None:
    cfg = tmp_path / "router_config.yaml"
    _write_config(cfg, token="secret-token")
    app = create_app(config_path=cfg, lm_client=FakeLMClient())
    client = TestClient(app)

    payload = {"messages": [{"role": "user", "content": "hello"}]}
    unauthorized = client.post("/v1/chat/completions", json=payload)
    assert unauthorized.status_code == 401

    authorized = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer secret-token"},
    )
    assert authorized.status_code == 200
    assert authorized.json().get("model") == "borg-cpu"
    assert authorized.headers["x-router-selected-model"] in {
        "qwen/qwen3-vl-8b",
        "qwen/qwen3.5-35b-a3b",
    }


def test_default_temperature_from_yaml_is_applied_when_request_omits_temperature(tmp_path: Path) -> None:
    cfg = tmp_path / "router_config.yaml"
    _write_config(cfg, default_temperature=0.35)
    lm = CapturePayloadLMClient()
    app = create_app(config_path=cfg, lm_client=lm)
    client = TestClient(app)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Beschreibe das Bild"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ],
        "max_tokens": 100,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert lm.last_payload is not None
    assert lm.last_payload.get("temperature") == 0.35


def test_request_temperature_overrides_yaml_default_temperature(tmp_path: Path) -> None:
    cfg = tmp_path / "router_config.yaml"
    _write_config(cfg, default_temperature=0.35)
    lm = CapturePayloadLMClient()
    app = create_app(config_path=cfg, lm_client=lm)
    client = TestClient(app)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Beschreibe das Bild"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ],
        "temperature": 0.9,
        "max_tokens": 100,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert lm.last_payload is not None
    assert lm.last_payload.get("temperature") == 0.9


def test_non_coding_request_does_not_fallback_to_large_when_small_fails(cfg_file: Path) -> None:
    app = create_app(config_path=cfg_file, lm_client=FakeLMClient(fail_first_small=True))
    client = TestClient(app)
    payload = {
        "messages": [{"role": "user", "content": "kurze Frage"}],
        "max_tokens": 100,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 502


def test_repeated_similar_requests_escalate_from_small_to_large(cfg_file: Path) -> None:
    app = create_app(config_path=cfg_file, lm_client=FakeLMClient())
    client = TestClient(app)
    headers = {"x-router-session-id": "sess-a"}
    payload = {
        "model": "borg-cpu",
        "messages": [{"role": "user", "content": "Bitte erklaere Quantencomputing kurz in einfachen Worten."}],
        "max_tokens": 120,
    }

    first = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert first.status_code == 200
    assert first.headers["x-router-selected-model"] == "qwen/qwen3-vl-8b"
    assert first.headers["x-router-session-id"] == "sess-a"

    second = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert second.status_code == 200
    assert second.headers["x-router-selected-model"] == "qwen/qwen3.5-35b-a3b"
    assert second.headers["x-router-reason"] == "repetition_escalation_small_to_large"


def test_repeated_requests_can_escalate_from_large_to_deep(cfg_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEP_ENABLED", "true")
    _write_config(cfg_file, repetition_similarity_threshold=0.84)
    app = create_app(config_path=cfg_file, lm_client=FakeLMClient())
    client = TestClient(app)
    headers = {"x-router-session-id": "sess-a"}

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "borg-cpu",
            "messages": [{"role": "user", "content": "Bitte erklaere Quantencomputing kurz in einfachen Worten."}],
            "max_tokens": 120,
        },
        headers=headers,
    )
    assert first.status_code == 200
    assert first.headers["x-router-selected-model"] == "qwen/qwen3-vl-8b"

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "borg-cpu",
            "messages": [{"role": "user", "content": "Bitte erklaere Quantencomputing bitte kurz in sehr einfachen Worten."}],
            "max_tokens": 120,
        },
        headers=headers,
    )
    assert second.status_code == 200
    assert second.headers["x-router-selected-model"] == "qwen/qwen3.5-35b-a3b"
    assert second.headers["x-router-reason"] == "repetition_escalation_small_to_large"

    third = client.post(
        "/v1/chat/completions",
        json={
            "model": "borg-cpu",
            "messages": [{"role": "user", "content": "Bitte erklaere Quantencomputing bitte kurz in sehr einfachen Worten."}],
            "max_tokens": 120,
        },
        headers=headers,
    )
    assert third.status_code == 200
    assert third.headers["x-router-selected-model"] == "gpt-4.1"
    assert third.headers["x-router-reason"] == "repetition_escalation_large_to_deep"


def test_similar_requests_in_different_sessions_do_not_mix(cfg_file: Path) -> None:
    app = create_app(config_path=cfg_file, lm_client=FakeLMClient())
    client = TestClient(app)
    payload = {
        "model": "borg-cpu",
        "messages": [{"role": "user", "content": "Bitte erklaere Quantencomputing kurz in einfachen Worten."}],
        "max_tokens": 120,
    }

    first = client.post("/v1/chat/completions", json=payload, headers={"x-router-session-id": "sess-a"})
    assert first.status_code == 200
    assert first.headers["x-router-selected-model"] == "qwen/qwen3-vl-8b"

    second = client.post("/v1/chat/completions", json=payload, headers={"x-router-session-id": "sess-b"})
    assert second.status_code == 200
    assert second.headers["x-router-selected-model"] == "qwen/qwen3-vl-8b"
    assert second.headers["x-router-reason"] == "judge_small"


def test_missing_session_id_disables_memory_when_required(cfg_file: Path) -> None:
    app = create_app(config_path=cfg_file, lm_client=FakeLMClient())
    client = TestClient(app)
    payload = {
        "model": "borg-cpu",
        "messages": [{"role": "user", "content": "Bitte erklaere Quantencomputing kurz in einfachen Worten."}],
        "max_tokens": 120,
    }

    first = client.post("/v1/chat/completions", json=payload)
    second = client.post("/v1/chat/completions", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers.get("x-router-session-id", "") == ""
    assert second.headers["x-router-selected-model"] == "qwen/qwen3-vl-8b"
    assert second.headers["x-router-reason"] == "judge_small"


def test_anthropic_endpoint_returns_mvp_shape(cfg_file: Path) -> None:
    app = create_app(config_path=cfg_file, lm_client=FakeLMClient())
    client = TestClient(app)
    payload = {
        "max_tokens": 120,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Schreibe einen Satz."}],
            }
        ],
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "borg-cpu"
    assert isinstance(body["content"], list)
    assert body["content"][0]["type"] == "text"


def test_openai_stream_endpoint_proxies_sse(cfg_file: Path) -> None:
    app = create_app(config_path=cfg_file, lm_client=FakeLMClient())
    client = TestClient(app)
    payload = {
        "messages": [{"role": "user", "content": "stream me"}],
        "stream": True,
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        text = "".join(list(resp.iter_text()))
        assert "data:" in text
        assert '"model": "borg-cpu"' in text or '"model":"borg-cpu"' in text
        assert "[DONE]" in text


def test_models_endpoint_exposes_router_model(cfg_file: Path) -> None:
    app = create_app(config_path=cfg_file, lm_client=FakeLMClient())
    client = TestClient(app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "borg-cpu"


def test_router_alias_can_be_used_in_request_model(cfg_file: Path) -> None:
    app = create_app(config_path=cfg_file, lm_client=FakeLMClient())
    client = TestClient(app)
    payload = {
        "model": "borg-cpu",
        "messages": [{"role": "user", "content": "kurze Frage"}],
        "max_tokens": 120,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert resp.json().get("model") == "borg-cpu"
    assert resp.headers["x-router-reason"] in {
        "constraint_single_candidate",
        "judge_small",
        "judge_unavailable_default_small",
        "client_model_preference_judge_unavailable",
        "heuristic_fallback",
    }


def test_admin_config_reports_server_bind(cfg_file: Path) -> None:
    app = create_app(config_path=cfg_file, lm_client=FakeLMClient())
    client = TestClient(app)
    resp = client.put("/admin/config", content=Path(cfg_file).read_text(encoding="utf-8"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["server"]["port"] == 12345


def test_admin_status_page_is_human_readable(cfg_file: Path) -> None:
    app = create_app(config_path=cfg_file, lm_client=FakeLMClient())
    client = TestClient(app)
    resp = client.get("/admin/status")
    assert resp.status_code == 200
    assert "Router Status" in resp.text
    assert "/admin/model-availability" in resp.text


def test_anthropic_to_openai_translates_tool_result_to_tool_role() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_weather_1",
                        "content": [{"type": "text", "text": "15C und sonnig"}],
                    }
                ],
            }
        ]
    }
    out = anthropic_to_openai_payload(payload)
    msgs = out["messages"]
    assert any(
        msg.get("role") == "tool" and msg.get("tool_call_id") == "toolu_weather_1"
        for msg in msgs
    )


def test_anthropic_to_openai_injects_tool_hint_system_message() -> None:
    payload = {
        "tools": [
            {
                "name": "weather_lookup",
                "description": "Wetter",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
    }
    out = anthropic_to_openai_payload(payload)
    assert out["messages"]
    first = out["messages"][0]
    assert first.get("role") == "system"
    assert DEFAULT_TOOLUSE_SYSTEM_HINT in str(first.get("content", ""))


def test_anthropic_non_stream_returns_tool_use_block(cfg_file: Path) -> None:
    app = create_app(config_path=cfg_file, lm_client=ToolCallLMClient())
    client = TestClient(app)
    payload = {
        "max_tokens": 120,
        "tools": [
            {
                "name": "weather_lookup",
                "description": "Wetter nachschlagen",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Wie ist das Wetter?"}]}],
    }
    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    assert any(block.get("type") == "tool_use" for block in body["content"])


def test_anthropic_stream_emits_tool_use_events(cfg_file: Path) -> None:
    app = create_app(config_path=cfg_file, lm_client=ToolCallLMClient())
    client = TestClient(app)
    payload = {
        "max_tokens": 120,
        "stream": True,
        "tools": [
            {
                "name": "weather_lookup",
                "description": "Wetter nachschlagen",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Wie ist das Wetter?"}]}],
    }
    with client.stream("POST", "/v1/messages", json=payload) as resp:
        assert resp.status_code == 200
        text = "".join(list(resp.iter_text()))
    assert "event: content_block_start" in text
    assert '"type": "tool_use"' in text or '"type":"tool_use"' in text
    assert '"stop_reason": "tool_use"' in text or '"stop_reason":"tool_use"' in text


def test_anthropic_stream_retries_large_when_small_stream_is_semantically_empty(cfg_file: Path) -> None:
    app = create_app(config_path=cfg_file, lm_client=EmptyAnthropicSmallThenLargeTextLMClient())
    client = TestClient(app)
    payload = {
        "model": "borg-cpu",
        "max_tokens": 120,
        "stream": True,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hallo"}]}],
    }
    with client.stream("POST", "/v1/messages", json=payload) as resp:
        assert resp.status_code == 200
        text = "".join(list(resp.iter_text()))
    assert "fallback works" in text
    assert resp.headers["x-router-selected-model"] == "qwen/qwen3.5-35b-a3b"
    assert resp.headers["x-router-fallback"] == "1"


def test_openai_provider_headers_include_auth_and_optional_org_project() -> None:
    settings = LMStudioSettings(
        provider="openai",
        base_url="https://api.openai.com",
        api_key="sk-test-direct",
        organization="org_123",
        project="proj_123",
    )
    headers = LMStudioClient._upstream_headers(settings)
    assert headers["Authorization"] == "Bearer sk-test-direct"
    assert headers["OpenAI-Organization"] == "org_123"
    assert headers["OpenAI-Project"] == "proj_123"


def test_openai_provider_headers_read_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-env")
    settings = LMStudioSettings(
        provider="openai",
        base_url="https://api.openai.com",
        api_key=None,
        api_key_env="OPENAI_API_KEY",
    )
    headers = LMStudioClient._upstream_headers(settings)
    assert headers["Authorization"] == "Bearer sk-test-env"


def test_openai_provider_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = LMStudioSettings(
        provider="openai",
        base_url="https://api.openai.com",
        api_key=None,
        api_key_env="OPENAI_API_KEY",
    )
    with pytest.raises(UpstreamError) as exc:
        LMStudioClient._upstream_headers(settings)
    assert exc.value.status_code == 500
    assert "OpenAI API key missing" in exc.value.body


def test_openai_chat_payload_uses_max_completion_tokens_for_openai_provider() -> None:
    payload = {
        "model": "gpt-5-mini",
        "max_tokens": 77,
        "messages": [{"role": "user", "content": "hello"}],
    }
    settings = LMStudioSettings(provider="openai", base_url="https://api.openai.com")
    out = RouterService._normalize_openai_chat_token_param(settings, "/v1/chat/completions", payload)
    assert "max_tokens" not in out
    assert out["max_completion_tokens"] == 77


def test_model_availability_endpoint_reports_models_loaded(cfg_file: Path) -> None:
    lm = ModelCatalogLMClient(
        [
            {"id": "qwen/qwen3-vl-8b", "loaded": True},
            {"id": "qwen/qwen3.5-35b-a3b", "loaded": True},
            {"id": "gpt-4.1", "loaded": True},
            {"id": "gpt-4o-mini", "loaded": True},
        ]
    )
    app = create_app(config_path=cfg_file, lm_client=lm)
    with TestClient(app) as client:
        resp = client.get("/admin/model-availability")
        assert resp.status_code == 200
        body = resp.json()
        assert body["all_available"] is True
        assert body["all_loaded"] is True
        assert body["error"] is None
        assert lm.list_calls >= 1


def test_model_availability_endpoint_flags_missing_models(cfg_file: Path) -> None:
    lm = ModelCatalogLMClient([{"id": "qwen/qwen3-vl-8b", "loaded": True}])
    app = create_app(config_path=cfg_file, lm_client=lm)
    with TestClient(app) as client:
        resp = client.get("/admin/model-availability")
        assert resp.status_code == 200
        body = resp.json()
        assert body["all_available"] is False
        assert body["all_loaded"] is False
        large = next(item for item in body["models"] if item["alias"] == "large")
        assert large["available"] is False
        assert large["loaded"] is False


def test_route_analytics_logs_prompt_fields(cfg_file: Path, caplog: pytest.LogCaptureFixture) -> None:
    app = create_app(config_path=cfg_file, lm_client=FakeLMClient())
    client = TestClient(app)
    payload = {
        "messages": [{"role": "user", "content": "Bitte analysiere das Routing fuer diesen Prompt."}],
        "max_tokens": 120,
    }

    app_logger = logging.getLogger("llm-router")
    original_propagate = app_logger.propagate
    app_logger.propagate = True
    try:
        caplog.set_level(logging.INFO, logger="llm-router")
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200

        # The logs are captured by caplog. If empty, maybe the logger name in the app is different or not propagating.
        # But we see them in "Captured stderr call".
        # Let's try to find them in caplog.records again, maybe filter differently.
        route_logs = [r.message for r in caplog.records if "route_analytics" in r.message]
        assert route_logs
        analytics = json.loads(route_logs[-1].split(" ", 1)[1])
        assert analytics["request_id"]
        assert analytics["prompt_text"] == "Bitte analysiere das Routing fuer diesen Prompt."
        assert analytics["user_prompt_text"] == "Bitte analysiere das Routing fuer diesen Prompt."
        assert analytics["latest_user_prompt_text"] == "Bitte analysiere das Routing fuer diesen Prompt."
        assert analytics["routing_latest_user_prompt_text"] == "Bitte analysiere das Routing fuer diesen Prompt."
    finally:
        app_logger.propagate = original_propagate


def test_route_analytics_writes_sqlite_record(cfg_file: Path) -> None:
    app = create_app(config_path=cfg_file, lm_client=FakeLMClient())
    client = TestClient(app)
    payload = {
        "model": "borg-cpu",
        "stream": True,
        "max_tokens": 32000,
        "tools": [{"name": "weather_lookup", "input_schema": {"type": "object", "properties": {}}}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<local-command-caveat>Caveat</local-command-caveat>\n"
                            "<local-command-stdout>Set model to borg-cpu</local-command-stdout>\n"
                            "hallo"
                        ),
                    }
                ],
            }
        ],
    }

    with client.stream("POST", "/v1/messages", json=payload) as resp:
        assert resp.status_code == 200
        _ = "".join(resp.iter_text())
        request_id = resp.headers["x-request-id"]

    db_path = cfg_file.parent / "router_analytics.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM routing_runs WHERE request_id = ?", (request_id,)).fetchone()[0]
        row = conn.execute(
            """
            SELECT selected_alias, expected_route_class, routing_efficiency_label,
                   routing_latest_user_text, routing_input_tokens, full_input_tokens,
                   output_text_chars, latency_ms
            FROM routing_runs
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()
    finally:
        conn.close()

    assert count == 1
    assert row is not None
    assert row[0] == "small"
    assert row[1] == "small"
    assert row[2] == "good_fit"
    assert row[3] == "hallo"
    assert row[4] < row[5]
    assert row[6] is not None
    assert row[7] is not None


def test_route_analytics_marks_oversized_route_when_large_handles_greeting(cfg_file: Path) -> None:
    _write_config(cfg_file, small_context=10)
    app = create_app(config_path=cfg_file, lm_client=FakeLMClient())
    client = TestClient(app)
    payload = {
        "model": "borg-cpu",
        "max_tokens": 120,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hallo"}]}],
    }

    resp = client.post("/v1/messages", json=payload)
    assert resp.status_code == 200
    request_id = resp.headers["x-request-id"]

    conn = sqlite3.connect(cfg_file.parent / "router_analytics.sqlite")
    try:
        row = conn.execute(
            "SELECT selected_alias, routing_efficiency_label, routing_efficiency_score FROM routing_runs WHERE request_id = ?",
            (request_id,),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == "large"
    assert row[1] == "oversized_route"
    assert row[2] < 100
