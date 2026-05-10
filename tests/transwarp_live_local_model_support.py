from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from llmrouter_router_service.backends import (
    BackendCapabilityNode,
    BackendChatRequestNode,
    BackendChatResponseNode,
    BackendFailureKind,
    BackendFailureNode,
    BackendHealthReportNode,
    BackendProbeResultNode,
)
from llmrouter_shared_contracts.registry import BackendHealthState, ModelCapability


LOCAL_MODEL_MATRIX_JSON_ENV = "LLMROUTER_LOCAL_MODEL_MATRIX_JSON"
LOCAL_MODEL_MATRIX_PATH_ENV = "LLMROUTER_LOCAL_MODEL_MATRIX_PATH"
LOCAL_ENV_FILE_ENV = "LLMROUTER_ENV_FILE"
LMSTUDIO_BASE_URL_ENV = "LLMROUTER_LMSTUDIO_BASE_URL"
LMSTUDIO_BACKEND_ID_ENV = "LLMROUTER_LMSTUDIO_BACKEND_ID"
LMSTUDIO_BACKEND_KIND_ENV = "LLMROUTER_LMSTUDIO_BACKEND_KIND"
LMSTUDIO_SHORT_MODEL_ID_ENV = "LLMROUTER_LMSTUDIO_SHORT_MODEL_ID"
LMSTUDIO_SHORT_CONTEXT_TOKENS_ENV = "LLMROUTER_LMSTUDIO_SHORT_CONTEXT_TOKENS"
LMSTUDIO_LONG_MODEL_ID_ENV = "LLMROUTER_LMSTUDIO_LONG_MODEL_ID"
LMSTUDIO_LONG_CONTEXT_TOKENS_ENV = "LLMROUTER_LMSTUDIO_LONG_CONTEXT_TOKENS"
LMSTUDIO_TOOL_MODEL_ID_ENV = "LLMROUTER_LMSTUDIO_TOOL_MODEL_ID"
LMSTUDIO_TOOL_CONTEXT_TOKENS_ENV = "LLMROUTER_LMSTUDIO_TOOL_CONTEXT_TOKENS"
LMSTUDIO_VISION_MODEL_ID_ENV = "LLMROUTER_LMSTUDIO_VISION_MODEL_ID"
LMSTUDIO_VISION_CONTEXT_TOKENS_ENV = "LLMROUTER_LMSTUDIO_VISION_CONTEXT_TOKENS"


@dataclass(frozen=True)
class ExternalModelNode:
    model_id: str
    context_window_tokens: int
    priority: int
    capabilities: tuple[ModelCapability, ...] = ()
    gpu_assignment: str | None = None


@dataclass(frozen=True)
class ExternalBackendNode:
    backend_id: str
    backend_kind: str
    base_url: str
    models: tuple[ExternalModelNode, ...]


@dataclass(frozen=True)
class LocalModelScenarioNode:
    name: str
    payload: dict[str, object]
    expected_model_id: str
    expected_backend_id: str
    minimum_text_characters: int


@dataclass(frozen=True)
class LocalModelMatrixNode:
    backends: tuple[ExternalBackendNode, ...]
    scenarios: tuple[LocalModelScenarioNode, ...]


def load_local_model_matrix_from_environment() -> LocalModelMatrixNode | None:
    environment = _load_effective_environment()

    raw_json = environment.get(LOCAL_MODEL_MATRIX_JSON_ENV)
    if raw_json:
        return _parse_local_model_matrix(json.loads(raw_json))

    matrix_path = environment.get(LOCAL_MODEL_MATRIX_PATH_ENV)
    if matrix_path:
        with open(matrix_path, encoding="utf-8") as handle:
            return _parse_local_model_matrix(json.load(handle))

    return _build_lmstudio_matrix_from_environment(environment)


class ExternalOpenAICompatibleBackendAdapter:
    def __init__(self, backend: ExternalBackendNode) -> None:
        self._backend = backend

    @property
    def backend_id(self) -> str:
        return self._backend.backend_id

    @property
    def backend_kind(self) -> str:
        return self._backend.backend_kind

    async def describe_capabilities(self) -> list[BackendCapabilityNode]:
        return [
            BackendCapabilityNode(
                backend_id=self.backend_id,
                backend_kind=self.backend_kind,
                model_id=model.model_id,
                capabilities=list(model.capabilities),
                context_window_tokens=model.context_window_tokens,
                metadata={"endpoint": self._backend.base_url},
            )
            for model in self._backend.models
        ]

    async def probe_backend(self) -> BackendProbeResultNode:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(_openai_endpoint(self._backend.base_url, "models"))
            response.raise_for_status()

        payload = response.json()
        available_model_ids = {
            str(model_payload["id"]) for model_payload in payload.get("data", [])
        }
        configured_model_ids = {model.model_id for model in self._backend.models}
        missing_model_ids = configured_model_ids - available_model_ids
        if missing_model_ids:
            raise RuntimeError(
                "Configured external models are not exposed by the running backend: "
                + ", ".join(sorted(missing_model_ids))
            )

        return BackendProbeResultNode(
            health=BackendHealthReportNode(
                backend_id=self.backend_id,
                backend_kind=self.backend_kind,
                state=BackendHealthState.AVAILABLE,
                reachable=True,
                message="External local model backend responded to /v1/models.",
            ),
            capability_nodes=await self.describe_capabilities(),
        )

    async def generate_chat_completion(
        self,
        request: BackendChatRequestNode,
    ) -> BackendChatResponseNode:
        payload = {
            "model": request.model_id,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in request.messages
            ],
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream,
        }
        async with httpx.AsyncClient(timeout=request.request_timeout_seconds or 120.0) as client:
            response = await client.post(
                _openai_endpoint(self._backend.base_url, "chat/completions"),
                json=payload,
            )
            response.raise_for_status()

        response_payload = response.json()
        choice = response_payload["choices"][0]
        message = choice["message"]
        content = _extract_assistant_text(message)
        if content is None:
            raise RuntimeError(
                "External backend returned no usable assistant text in "
                "`content`, `reasoning_content`, or structured content parts."
            )

        usage = response_payload.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))

        return BackendChatResponseNode.model_validate(
            {
                "backend_id": self.backend_id,
                "model_id": request.model_id,
                "output_text": content,
                "finish_reason": choice.get("finish_reason") or "stop",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": int(
                        usage.get(
                            "total_tokens",
                            prompt_tokens + completion_tokens,
                        )
                    ),
                },
                "backend_latency_ms": 0.0,
                "backend_response_id": response_payload.get("id"),
                "metadata": {"endpoint": self._backend.base_url},
            }
        )

    def classify_exception(
        self,
        exception: Exception,
        request: BackendChatRequestNode | None = None,
    ) -> BackendFailureNode:
        if isinstance(exception, httpx.TimeoutException):
            failure_kind = BackendFailureKind.TIMEOUT
            suggested_health_state = BackendHealthState.DEGRADED
            status_code = 504
        elif isinstance(exception, httpx.HTTPStatusError):
            failure_kind = BackendFailureKind.PROTOCOL_ERROR
            suggested_health_state = BackendHealthState.DEGRADED
            status_code = exception.response.status_code
        elif isinstance(exception, httpx.HTTPError):
            failure_kind = BackendFailureKind.OFFLINE
            suggested_health_state = BackendHealthState.UNAVAILABLE
            status_code = 503
        else:
            failure_kind = BackendFailureKind.INTERNAL_ERROR
            suggested_health_state = BackendHealthState.DEGRADED
            status_code = 500

        return BackendFailureNode(
            backend_id=self.backend_id,
            backend_kind=self.backend_kind,
            model_id=request.model_id if request is not None else None,
            failure_kind=failure_kind,
            message=str(exception) or "External local model backend call failed.",
            retryable=failure_kind in {BackendFailureKind.TIMEOUT, BackendFailureKind.OFFLINE},
            suggested_health_state=suggested_health_state,
            status_code=status_code,
            exception_type=type(exception).__name__,
        )


def _parse_local_model_matrix(payload: dict[str, Any]) -> LocalModelMatrixNode:
    backends = tuple(_parse_backend_node(backend_payload) for backend_payload in payload["backends"])
    scenarios = tuple(
        _parse_scenario_node(scenario_payload) for scenario_payload in payload["scenarios"]
    )
    return LocalModelMatrixNode(backends=backends, scenarios=scenarios)


def _parse_backend_node(payload: dict[str, Any]) -> ExternalBackendNode:
    return ExternalBackendNode(
        backend_id=str(payload["backend_id"]),
        backend_kind=str(payload["backend_kind"]),
        base_url=str(payload["base_url"]).rstrip("/"),
        models=tuple(_parse_model_node(model_payload) for model_payload in payload["models"]),
    )


def _parse_model_node(payload: dict[str, Any]) -> ExternalModelNode:
    return ExternalModelNode(
        model_id=str(payload["model_id"]),
        context_window_tokens=int(payload["context_window_tokens"]),
        priority=int(payload["priority"]),
        capabilities=tuple(ModelCapability(capability) for capability in payload.get("capabilities", [])),
        gpu_assignment=(
            None if payload.get("gpu_assignment") is None else str(payload["gpu_assignment"])
        ),
    )


def _parse_scenario_node(payload: dict[str, Any]) -> LocalModelScenarioNode:
    return LocalModelScenarioNode(
        name=str(payload["name"]),
        payload=dict(payload["payload"]),
        expected_model_id=str(payload["expected_model_id"]),
        expected_backend_id=str(payload["expected_backend_id"]),
        minimum_text_characters=int(payload["minimum_text_characters"]),
    )


def _openai_endpoint(base_url: str, suffix: str) -> str:
    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/v1"):
        return f"{normalized_base_url}/{suffix}"
    return f"{normalized_base_url}/v1/{suffix}"


def _extract_assistant_text(message: dict[str, Any]) -> str | None:
    direct_content = message.get("content")
    if isinstance(direct_content, str) and direct_content.strip():
        return direct_content

    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content

    if isinstance(direct_content, list):
        text_parts = []
        for part in direct_content:
            if not isinstance(part, dict):
                continue
            text_value = part.get("text")
            if isinstance(text_value, str) and text_value.strip():
                text_parts.append(text_value)
        if text_parts:
            return "\n".join(text_parts)

    return None


def _build_lmstudio_matrix_from_environment(
    environment: dict[str, str],
) -> LocalModelMatrixNode | None:
    base_url = _get_non_empty_environment_value(environment, LMSTUDIO_BASE_URL_ENV)
    if base_url is None:
        return None

    short_model_id = _get_non_empty_environment_value(
        environment,
        LMSTUDIO_SHORT_MODEL_ID_ENV,
    )
    long_model_id = _get_non_empty_environment_value(
        environment,
        LMSTUDIO_LONG_MODEL_ID_ENV,
    )
    if short_model_id is None and long_model_id is None:
        discovered_model_id = _discover_first_openai_model_id(base_url)
        short_model_id = discovered_model_id
        long_model_id = discovered_model_id
    elif short_model_id is None:
        short_model_id = long_model_id
    elif long_model_id is None:
        long_model_id = short_model_id

    if short_model_id is None or long_model_id is None:
        raise ValueError(
            "LM Studio local-model integration requires at least one routable model id "
            "or one discoverable model from GET /v1/models."
        )

    backend_id = (
        _get_non_empty_environment_value(environment, LMSTUDIO_BACKEND_ID_ENV)
        or "lmstudio-local"
    )
    backend_kind = (
        _get_non_empty_environment_value(environment, LMSTUDIO_BACKEND_KIND_ENV)
        or "lmstudio"
    )

    models_by_id: dict[str, ExternalModelNode] = {}
    _upsert_external_model(
        models_by_id,
        ExternalModelNode(
            model_id=short_model_id,
            context_window_tokens=_get_int_environment_value(
                LMSTUDIO_SHORT_CONTEXT_TOKENS_ENV,
                default=8192,
                environment=environment,
            ),
            priority=90,
        ),
    )
    _upsert_external_model(
        models_by_id,
        ExternalModelNode(
            model_id=long_model_id,
            context_window_tokens=_get_int_environment_value(
                LMSTUDIO_LONG_CONTEXT_TOKENS_ENV,
                default=32768,
                environment=environment,
            ),
            priority=70,
        ),
    )
    long_prompt_text = " ".join(["collective-context-window"] * 2200)
    scenarios = [
        LocalModelScenarioNode(
            name="short-context",
            payload={
                "model": "borg-cpu",
                "messages": [
                    {
                        "role": "user",
                        "content": "Summarize the current router status in one short paragraph.",
                    }
                ],
                "max_tokens": 96,
            },
            expected_model_id=short_model_id,
            expected_backend_id=backend_id,
            minimum_text_characters=24,
        ),
    ]

    tool_model_id = _get_non_empty_environment_value(
        environment,
        LMSTUDIO_TOOL_MODEL_ID_ENV,
    )
    if tool_model_id is not None:
        _upsert_external_model(
            models_by_id,
            ExternalModelNode(
                model_id=tool_model_id,
                context_window_tokens=_get_int_environment_value(
                    LMSTUDIO_TOOL_CONTEXT_TOKENS_ENV,
                    default=8192,
                    environment=environment,
                ),
                priority=80,
                capabilities=(ModelCapability.TOOL_USE,),
            ),
        )
        scenarios.append(
            LocalModelScenarioNode(
                name="tool-variation",
                payload={
                    "model": "borg-cpu",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Call the inspect_cube function before answering.",
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
                    "max_tokens": 128,
                },
                expected_model_id=tool_model_id,
                expected_backend_id=backend_id,
                minimum_text_characters=24,
            )
        )

    vision_model_id = _get_non_empty_environment_value(
        environment,
        LMSTUDIO_VISION_MODEL_ID_ENV,
    )
    if vision_model_id is not None:
        _upsert_external_model(
            models_by_id,
            ExternalModelNode(
                model_id=vision_model_id,
                context_window_tokens=_get_int_environment_value(
                    LMSTUDIO_VISION_CONTEXT_TOKENS_ENV,
                    default=8192,
                    environment=environment,
                ),
                priority=80,
                capabilities=(ModelCapability.VISION,),
            ),
        )
        scenarios.append(
            LocalModelScenarioNode(
                name="vision-variation",
                payload={
                    "model": "borg-cpu",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe the image and report the anomaly.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://example.test/cube.png"},
                                },
                            ],
                        }
                    ],
                    "max_tokens": 128,
                },
                expected_model_id=vision_model_id,
                expected_backend_id=backend_id,
                minimum_text_characters=20,
            )
        )

    scenarios.insert(
        1,
        LocalModelScenarioNode(
            name="long-context",
            payload={
                "model": "borg-cpu",
                "messages": [
                    {
                        "role": "user",
                        "content": long_prompt_text,
                    }
                ],
                "max_tokens": _build_long_context_max_tokens(
                    models_by_id=models_by_id,
                    expected_long_model_id=long_model_id,
                    prompt_text=long_prompt_text,
                ),
            },
            expected_model_id=long_model_id,
            expected_backend_id=backend_id,
            minimum_text_characters=20000,
        ),
    )

    return LocalModelMatrixNode(
        backends=(
            ExternalBackendNode(
                backend_id=backend_id,
                backend_kind=backend_kind,
                base_url=base_url,
                models=tuple(models_by_id.values()),
            ),
        ),
        scenarios=tuple(scenarios),
    )


def _get_non_empty_environment_value(
    environment: dict[str, str],
    name: str,
) -> str | None:
    raw_value = environment.get(name)
    if raw_value is None:
        return None
    normalized = raw_value.strip()
    return normalized or None


def _get_int_environment_value(
    name: str,
    *,
    default: int,
    environment: dict[str, str],
) -> int:
    raw_value = _get_non_empty_environment_value(environment, name)
    if raw_value is None:
        return default
    return int(raw_value)


def _load_effective_environment() -> dict[str, str]:
    explicit_environment = dict(os.environ)
    dotenv_environment = _load_dotenv_environment(explicit_environment)
    dotenv_environment.update(explicit_environment)
    return dotenv_environment


def _load_dotenv_environment(explicit_environment: dict[str, str]) -> dict[str, str]:
    dotenv_path = Path(explicit_environment.get(LOCAL_ENV_FILE_ENV, ".env"))
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return {}

    dotenv_environment: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            continue
        dotenv_environment[normalized_key] = _strip_wrapping_quotes(value.strip())
    return dotenv_environment


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _discover_first_openai_model_id(base_url: str) -> str | None:
    with httpx.Client(timeout=20.0) as client:
        response = client.get(_openai_endpoint(base_url, "models"))
        response.raise_for_status()
    payload = response.json()
    models = payload.get("data", [])
    if not models:
        return None
    model_id = models[0].get("id")
    if not isinstance(model_id, str):
        return None
    normalized_model_id = model_id.strip()
    return normalized_model_id or None


def _upsert_external_model(
    models_by_id: dict[str, ExternalModelNode],
    model: ExternalModelNode,
) -> None:
    existing_model = models_by_id.get(model.model_id)
    if existing_model is None:
        models_by_id[model.model_id] = model
        return

    merged_capabilities = tuple(
        sorted(
            {*(existing_model.capabilities), *(model.capabilities)},
            key=lambda capability: capability.value,
        )
    )
    models_by_id[model.model_id] = ExternalModelNode(
        model_id=model.model_id,
        context_window_tokens=max(
            existing_model.context_window_tokens,
            model.context_window_tokens,
        ),
        priority=max(existing_model.priority, model.priority),
        capabilities=merged_capabilities,
        gpu_assignment=existing_model.gpu_assignment or model.gpu_assignment,
    )


def _build_long_context_max_tokens(
    *,
    models_by_id: dict[str, ExternalModelNode],
    expected_long_model_id: str,
    prompt_text: str,
) -> int:
    expected_long_model = models_by_id[expected_long_model_id]
    competing_context_windows = [
        model.context_window_tokens
        for model_id, model in models_by_id.items()
        if model_id != expected_long_model_id
    ]
    if not competing_context_windows:
        return 256

    input_token_estimate = max((len(prompt_text) + 3) // 4, 1)
    safety_margin_tokens = (input_token_estimate * 35 + 99) // 100
    required_context_floor = max(competing_context_windows) + 1024
    target_max_tokens = required_context_floor - input_token_estimate - safety_margin_tokens
    maximum_safe_tokens = (
        expected_long_model.context_window_tokens
        - input_token_estimate
        - safety_margin_tokens
        - 1024
    )
    if maximum_safe_tokens < 256:
        return 256
    return max(256, min(target_max_tokens, maximum_safe_tokens))


__all__ = [
    "ExternalBackendNode",
    "ExternalModelNode",
    "ExternalOpenAICompatibleBackendAdapter",
    "LOCAL_ENV_FILE_ENV",
    "LMSTUDIO_BACKEND_ID_ENV",
    "LMSTUDIO_BACKEND_KIND_ENV",
    "LMSTUDIO_BASE_URL_ENV",
    "LMSTUDIO_LONG_CONTEXT_TOKENS_ENV",
    "LMSTUDIO_LONG_MODEL_ID_ENV",
    "LMSTUDIO_SHORT_CONTEXT_TOKENS_ENV",
    "LMSTUDIO_SHORT_MODEL_ID_ENV",
    "LMSTUDIO_TOOL_CONTEXT_TOKENS_ENV",
    "LMSTUDIO_TOOL_MODEL_ID_ENV",
    "LMSTUDIO_VISION_CONTEXT_TOKENS_ENV",
    "LMSTUDIO_VISION_MODEL_ID_ENV",
    "LOCAL_MODEL_MATRIX_JSON_ENV",
    "LOCAL_MODEL_MATRIX_PATH_ENV",
    "LocalModelMatrixNode",
    "LocalModelScenarioNode",
    "load_local_model_matrix_from_environment",
]
