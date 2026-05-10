from __future__ import annotations

import os

import pytest

import llmrouter_router_service.application.transwarp_local_model_support as live_support
from llmrouter_router_service.application import (
    LMSTUDIO_BASE_URL_ENV,
    LMSTUDIO_LONG_CONTEXT_TOKENS_ENV,
    LMSTUDIO_LONG_MODEL_ID_ENV,
    LMSTUDIO_SHORT_CONTEXT_TOKENS_ENV,
    LMSTUDIO_SHORT_MODEL_ID_ENV,
    LMSTUDIO_TOOL_MODEL_ID_ENV,
    LMSTUDIO_VISION_MODEL_ID_ENV,
    load_local_model_matrix_from_environment,
)

LOCAL_ENV_FILE_ENV = "LLMROUTER_ENV_FILE"


def test_load_local_model_matrix_from_lmstudio_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(LOCAL_ENV_FILE_ENV, str(tmp_path / "isolated.env"))
    monkeypatch.setenv(LMSTUDIO_BASE_URL_ENV, "http://192.168.178.2:1234")
    monkeypatch.setenv(LMSTUDIO_SHORT_MODEL_ID_ENV, "collective-short")
    monkeypatch.setenv(LMSTUDIO_SHORT_CONTEXT_TOKENS_ENV, "8192")
    monkeypatch.setenv(LMSTUDIO_LONG_MODEL_ID_ENV, "collective-long")
    monkeypatch.setenv(LMSTUDIO_LONG_CONTEXT_TOKENS_ENV, "32768")
    monkeypatch.setenv(LMSTUDIO_TOOL_MODEL_ID_ENV, "collective-tool")
    monkeypatch.setenv(LMSTUDIO_VISION_MODEL_ID_ENV, "collective-vision")

    matrix = load_local_model_matrix_from_environment(dict(os.environ))

    assert matrix is not None
    assert len(matrix.backends) == 1
    assert matrix.backends[0].base_url == "http://192.168.178.2:1234"
    assert tuple(model.model_id for model in matrix.backends[0].models) == (
        "collective-short",
        "collective-long",
        "collective-tool",
        "collective-vision",
    )
    assert tuple(scenario.name for scenario in matrix.scenarios) == (
        "short-context",
        "long-context",
        "tool-variation",
        "vision-variation",
    )


def test_load_local_model_matrix_requires_short_and_long_models_for_lmstudio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv(LOCAL_ENV_FILE_ENV, str(tmp_path / "isolated.env"))
    monkeypatch.setenv(LMSTUDIO_BASE_URL_ENV, "http://192.168.178.2:1234")
    monkeypatch.delenv(LMSTUDIO_SHORT_MODEL_ID_ENV, raising=False)
    monkeypatch.delenv(LMSTUDIO_LONG_MODEL_ID_ENV, raising=False)
    monkeypatch.setattr(
        live_support,
        "_discover_first_openai_model_id",
        lambda base_url: "collective-discovered",
    )

    matrix = load_local_model_matrix_from_environment(
        {
            LOCAL_ENV_FILE_ENV: str(tmp_path / "isolated.env"),
            LMSTUDIO_BASE_URL_ENV: "http://192.168.178.2:1234",
        }
    )

    assert matrix is not None
    assert tuple(model.model_id for model in matrix.backends[0].models) == (
        "collective-discovered",
    )
    assert matrix.scenarios[0].expected_model_id == "collective-discovered"
    assert matrix.scenarios[1].expected_model_id == "collective-discovered"
