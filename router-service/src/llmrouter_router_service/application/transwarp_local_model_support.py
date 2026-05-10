"""Local-model bootstrap helpers for OpenAI-compatible external backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import httpx

from llmrouter_router_service.backends import TranswarpBackendAdapter
from llmrouter_router_service.backends.transwarp_external_openai_backend import (
    ExternalOpenAICompatibleBackendAdapter,
    ExternalOpenAICompatibleModelNode,
    build_openai_endpoint,
)
from llmrouter_shared_contracts.registry import ConfiguredModelNode, ModelCapability


LOCAL_MODEL_MATRIX_JSON_ENV = "LLMROUTER_LOCAL_MODEL_MATRIX_JSON"
LOCAL_MODEL_MATRIX_PATH_ENV = "LLMROUTER_LOCAL_MODEL_MATRIX_PATH"
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
    """One externally reachable model exposed through an OpenAI-compatible backend."""

    model_id: str
    context_window_tokens: int
    priority: int
    capabilities: tuple[ModelCapability, ...] = ()
    gpu_assignment: str | None = None


@dataclass(frozen=True)
class ExternalBackendNode:
    """One OpenAI-compatible backend endpoint and its exposed models."""

    backend_id: str
    backend_kind: str
    base_url: str
    models: tuple[ExternalModelNode, ...]


@dataclass(frozen=True)
class LocalModelScenarioNode:
    """One integration scenario used by the live local-model matrix tests."""

    name: str
    payload: dict[str, object]
    expected_model_id: str
    expected_backend_id: str
    minimum_text_characters: int


@dataclass(frozen=True)
class LocalModelMatrixNode:
    """Validated set of externally reachable local backends and test scenarios."""

    backends: tuple[ExternalBackendNode, ...]
    scenarios: tuple[LocalModelScenarioNode, ...]


def load_local_model_matrix_from_environment(
    environment: Mapping[str, str],
) -> LocalModelMatrixNode | None:
    """Load an explicit or LM Studio derived local-model matrix from environment."""

    raw_json = environment.get(LOCAL_MODEL_MATRIX_JSON_ENV)
    if raw_json:
        return _parse_local_model_matrix(json.loads(raw_json))

    matrix_path = environment.get(LOCAL_MODEL_MATRIX_PATH_ENV)
    if matrix_path:
        with Path(matrix_path).open(encoding="utf-8") as handle:
            return _parse_local_model_matrix(json.load(handle))

    return _build_lmstudio_matrix_from_environment(environment)


def build_local_model_runtime_components(
    environment: Mapping[str, str],
) -> tuple[list[ConfiguredModelNode], list[TranswarpBackendAdapter]]:
    """Build configured models and adapters for local OpenAI-compatible backends."""

    matrix = load_local_model_matrix_from_environment(environment)
    if matrix is None:
        return [], []

    configured_models = [
        ConfiguredModelNode(
            model_id=model.model_id,
            backend_id=backend.backend_id,
            backend_kind=backend.backend_kind,
            capabilities=list(model.capabilities),
            context_window_tokens=model.context_window_tokens,
            gpu_assignment=model.gpu_assignment,
            backend_address=backend.base_url,
            priority=model.priority,
        )
        for backend in matrix.backends
        for model in backend.models
    ]
    backend_adapters: list[TranswarpBackendAdapter] = [
        ExternalOpenAICompatibleBackendAdapter(
            backend_id=backend.backend_id,
            backend_kind=backend.backend_kind,
            base_url=backend.base_url,
            models=[
                ExternalOpenAICompatibleModelNode(
                    model_id=model.model_id,
                    context_window_tokens=model.context_window_tokens,
                    priority=model.priority,
                    capabilities=model.capabilities,
                    gpu_assignment=model.gpu_assignment,
                )
                for model in backend.models
            ],
        )
        for backend in matrix.backends
    ]
    return configured_models, backend_adapters


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
        capabilities=tuple(
            ModelCapability(capability) for capability in payload.get("capabilities", [])
        ),
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


def _build_lmstudio_matrix_from_environment(
    environment: Mapping[str, str],
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
    environment: Mapping[str, str],
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
    environment: Mapping[str, str],
) -> int:
    raw_value = _get_non_empty_environment_value(environment, name)
    if raw_value is None:
        return default
    return int(raw_value)


def _discover_first_openai_model_id(base_url: str) -> str | None:
    with httpx.Client(timeout=20.0) as client:
        response = client.get(build_openai_endpoint(base_url, "models"))
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
