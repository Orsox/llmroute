from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llmrouter_router_service.api import create_collective_router_app
from llmrouter_router_service.application import (
    LOCAL_MODEL_MATRIX_JSON_ENV,
    LOCAL_MODEL_MATRIX_PATH_ENV,
    LocalModelMatrixNode,
    RouterServiceBootstrapConfig,
    load_local_model_matrix_from_environment,
)
from llmrouter_router_service.backends import (
    ExternalOpenAICompatibleBackendAdapter,
    ExternalOpenAICompatibleModelNode,
)
from llmrouter_router_service.observability import RoutingLogCollective
from llmrouter_shared_contracts.api import OPENAI_ROUTER_MODEL_ID
from llmrouter_shared_contracts.registry import ConfiguredModelNode


@pytest.fixture(name="external_local_model_matrix")
def fixture_external_local_model_matrix() -> LocalModelMatrixNode:
    matrix = load_local_model_matrix_from_environment(dict(os.environ))
    if matrix is None:
        pytest.skip(
            "Real local-model integration tests require "
            f"`{LOCAL_MODEL_MATRIX_JSON_ENV}` or `{LOCAL_MODEL_MATRIX_PATH_ENV}`."
        )
    return matrix


@pytest.mark.local_model
@pytest.mark.integration
def test_local_model_matrix_routes_real_external_backends(
    external_local_model_matrix: LocalModelMatrixNode,
    tmp_path: Path,
) -> None:
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
        for backend in external_local_model_matrix.backends
        for model in backend.models
    ]
    backend_adapters = [
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
        for backend in external_local_model_matrix.backends
    ]
    bootstrap_config = RouterServiceBootstrapConfig(
        configured_models=configured_models,
        backend_adapters=backend_adapters,
        routing_log_collective=RoutingLogCollective(
            storage_path=tmp_path / "routing-log.json"
        ),
    )
    application = create_collective_router_app(bootstrap_config=bootstrap_config)

    with TestClient(application) as client:
        for scenario in external_local_model_matrix.scenarios:
            response = client.post("/v1/chat/completions", json=scenario.payload)
            recent_logs_response = client.get("/logs/recent", params={"limit": 1})

            assert response.status_code == 200
            response_payload = response.json()
            assert response_payload["model"] == OPENAI_ROUTER_MODEL_ID
            assert response_payload["choices"][0]["message"]["content"]

            routing_record = bootstrap_config.routing_log_collective.list_records(limit=1)[0]
            assert routing_record.selected_model is not None
            assert routing_record.selected_model.model_id == scenario.expected_model_id
            assert routing_record.selected_model.backend_id == scenario.expected_backend_id
            assert routing_record.error is None
            assert routing_record.prompt_diagnostics is not None
            assert (
                routing_record.prompt_diagnostics.text_character_count
                >= scenario.minimum_text_characters
            )

            recent_logs_payload = recent_logs_response.json()
            assert recent_logs_response.status_code == 200
            assert recent_logs_payload["metadata"]["returned_entries"] == 1
            assert recent_logs_payload["entries"][0]["attributes"]["selected_model"] == (
                scenario.expected_model_id
            )
            assert recent_logs_payload["entries"][0]["attributes"][
                "selected_backend"
            ] == scenario.expected_backend_id
