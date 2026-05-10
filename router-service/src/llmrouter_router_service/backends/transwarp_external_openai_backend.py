"""OpenAI-compatible backend adapter for externally running local model services."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from llmrouter_router_service.backends.transwarp_backend_adapter import (
    BackendCapabilityNode,
    BackendChatMessageNode,
    BackendChatRequestNode,
    BackendChatResponseNode,
    BackendFailureKind,
    BackendFailureNode,
    BackendHealthReportNode,
    BackendProbeResultNode,
)
from llmrouter_shared_contracts.registry import BackendHealthState, ModelCapability


LOGGER = logging.getLogger("llmrouter.router_service")


class ExternalOpenAICompatibleBackendAdapter:
    """Backend adapter that calls a remote OpenAI-compatible local model service."""

    def __init__(
        self,
        *,
        backend_id: str,
        backend_kind: str,
        base_url: str,
        models: list["ExternalOpenAICompatibleModelNode"],
        probe_cache_ttl_seconds: float = 300.0,
    ) -> None:
        self._backend_id = backend_id
        self._backend_kind = backend_kind
        self._base_url = base_url.rstrip("/")
        self._models = list(models)
        self._probe_cache_ttl_seconds = probe_cache_ttl_seconds
        self._cached_probe_result: BackendProbeResultNode | None = None
        self._probe_cache_expires_at = 0.0

    @property
    def backend_id(self) -> str:
        return self._backend_id

    @property
    def backend_kind(self) -> str:
        return self._backend_kind

    async def describe_capabilities(self) -> list[BackendCapabilityNode]:
        return [
            BackendCapabilityNode(
                backend_id=self.backend_id,
                backend_kind=self.backend_kind,
                model_id=model.model_id,
                capabilities=list(model.capabilities),
                context_window_tokens=model.context_window_tokens,
                metadata={"endpoint": self._base_url},
            )
            for model in self._models
        ]

    async def probe_backend(self) -> BackendProbeResultNode:
        if (
            self._cached_probe_result is not None
            and time.monotonic() < self._probe_cache_expires_at
        ):
            return self._cached_probe_result

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(build_openai_endpoint(self._base_url, "models"))
            response.raise_for_status()

        payload = response.json()
        available_model_ids = {
            str(model_payload["id"]) for model_payload in payload.get("data", [])
        }
        configured_model_ids = {model.model_id for model in self._models}
        missing_model_ids = configured_model_ids - available_model_ids
        if missing_model_ids:
            raise RuntimeError(
                "Configured external models are not exposed by the running backend: "
                + ", ".join(sorted(missing_model_ids))
            )

        probe_result = BackendProbeResultNode(
            health=BackendHealthReportNode(
                backend_id=self.backend_id,
                backend_kind=self.backend_kind,
                state=BackendHealthState.AVAILABLE,
                reachable=True,
                message="External local model backend responded to /v1/models.",
            ),
            capability_nodes=await self.describe_capabilities(),
        )
        self._cached_probe_result = probe_result
        self._probe_cache_expires_at = time.monotonic() + self._probe_cache_ttl_seconds
        return probe_result

    def invalidate_probe_cache(self) -> None:
        """Force the next probe call to fetch fresh backend state."""

        self._cached_probe_result = None
        self._probe_cache_expires_at = 0.0

    async def generate_chat_completion(
        self,
        request: BackendChatRequestNode,
    ) -> BackendChatResponseNode:
        request_id = request.metadata.get("llmrouter.request_id", "<missing>")
        exact_request_hash = request.metadata.get(
            "llmrouter.exact_request_hash",
            "<missing>",
        )
        if self._should_use_lmstudio_native_chat(request):
            return await self._generate_lmstudio_native_chat_completion(
                request,
                request_id=request_id,
                exact_request_hash=exact_request_hash,
            )

        payload = {
            "model": request.model_id,
            "messages": [
                _build_openai_chat_message_payload(message)
                for message in request.messages
            ],
            "stream": request.stream,
        }
        if request.tools:
            payload["tools"] = list(request.tools)
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if (
            self.backend_kind == "lmstudio"
            and request.metadata.get("lmstudio.enable_thinking") == "false"
        ):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        endpoint_url = build_openai_endpoint(self._base_url, "chat/completions")
        LOGGER.info(
            "chat_backend_dispatch request_id=%s backend=%s model=%s endpoint=%s "
            "messages=%s tools=%s stream=%s exact_hash=%s",
            request_id,
            self.backend_id,
            request.model_id,
            endpoint_url,
            len(request.messages),
            len(request.tools),
            str(request.stream).lower(),
            exact_request_hash,
        )
        async with httpx.AsyncClient(
            timeout=request.request_timeout_seconds or 120.0
        ) as client:
            response = await client.post(
                endpoint_url,
                json=payload,
            )
            response.raise_for_status()

        response_payload = response.json()
        choice = response_payload["choices"][0]
        message = choice["message"]
        assistant_payload = extract_assistant_payload(message)
        if assistant_payload["content"] is None and not assistant_payload["tool_calls"]:
            raise RuntimeError(
                "External backend returned no usable assistant text in "
                "`content`, structured content parts, or `tool_calls`."
            )

        usage = response_payload.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        finish_reason = choice.get("finish_reason") or "stop"
        LOGGER.info(
            "chat_backend_response request_id=%s backend=%s model=%s response_id=%s "
            "finish_reason=%s prompt_tokens=%s completion_tokens=%s tool_calls=%s exact_hash=%s",
            request_id,
            self.backend_id,
            request.model_id,
            response_payload.get("id") or "<missing>",
            finish_reason,
            prompt_tokens,
            completion_tokens,
            len(assistant_payload["tool_calls"]),
            exact_request_hash,
        )

        return BackendChatResponseNode.model_validate(
            {
                "backend_id": self.backend_id,
                "model_id": request.model_id,
                "output_text": assistant_payload["content"],
                "tool_calls": assistant_payload["tool_calls"],
                "finish_reason": finish_reason,
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
                "metadata": {"endpoint": self._base_url},
            }
        )

    def _should_use_lmstudio_native_chat(
        self,
        request: BackendChatRequestNode,
    ) -> bool:
        if self.backend_kind != "lmstudio":
            return False
        if request.metadata.get("lmstudio.enable_thinking") != "false":
            return False
        if request.tools or request.tool_choice is not None:
            return False
        return all(
            isinstance(message.content, str) and not message.tool_calls
            for message in request.messages
        )

    async def _generate_lmstudio_native_chat_completion(
        self,
        request: BackendChatRequestNode,
        *,
        request_id: str,
        exact_request_hash: str,
    ) -> BackendChatResponseNode:
        payload: dict[str, Any] = {
            "model": request.model_id,
            "input": [
                {
                    "type": "message",
                    "role": message.role.value,
                    "content": message.content,
                }
                for message in request.messages
            ],
            "stream": False,
            "reasoning": "off",
        }
        if request.max_output_tokens is not None:
            payload["max_output_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        endpoint_url = build_lmstudio_native_chat_endpoint(self._base_url)
        LOGGER.info(
            "chat_backend_dispatch request_id=%s backend=%s model=%s endpoint=%s "
            "messages=%s tools=%s stream=%s exact_hash=%s",
            request_id,
            self.backend_id,
            request.model_id,
            endpoint_url,
            len(request.messages),
            len(request.tools),
            "false",
            exact_request_hash,
        )
        async with httpx.AsyncClient(
            timeout=request.request_timeout_seconds or 120.0
        ) as client:
            response = await client.post(
                endpoint_url,
                json=payload,
            )
            response.raise_for_status()

        response_payload = response.json()
        assistant_content = _extract_lmstudio_native_message_text(
            response_payload.get("output")
        )
        if assistant_content is None:
            raise RuntimeError(
                "LM Studio native chat returned no usable assistant message content."
            )

        stats = response_payload.get("stats", {})
        prompt_tokens = int(stats.get("input_tokens", 0))
        completion_tokens = int(
            stats.get("total_output_tokens", stats.get("output_tokens", 0))
        )
        LOGGER.info(
            "chat_backend_response request_id=%s backend=%s model=%s response_id=%s "
            "finish_reason=%s prompt_tokens=%s completion_tokens=%s tool_calls=%s exact_hash=%s",
            request_id,
            self.backend_id,
            request.model_id,
            response_payload.get("response_id") or "<missing>",
            "stop",
            prompt_tokens,
            completion_tokens,
            0,
            exact_request_hash,
        )
        return BackendChatResponseNode.model_validate(
            {
                "backend_id": self.backend_id,
                "model_id": request.model_id,
                "output_text": assistant_content,
                "tool_calls": (),
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "backend_latency_ms": 0.0,
                "backend_response_id": response_payload.get("response_id"),
                "metadata": {"endpoint": self._base_url},
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
            retryable=failure_kind
            in {BackendFailureKind.TIMEOUT, BackendFailureKind.OFFLINE},
            suggested_health_state=suggested_health_state,
            status_code=status_code,
            exception_type=type(exception).__name__,
        )


class ExternalOpenAICompatibleModelNode:
    """One model exposed by an external OpenAI-compatible backend."""

    def __init__(
        self,
        *,
        model_id: str,
        context_window_tokens: int,
        priority: int,
        capabilities: tuple[ModelCapability, ...] = (),
        gpu_assignment: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.context_window_tokens = context_window_tokens
        self.priority = priority
        self.capabilities = capabilities
        self.gpu_assignment = gpu_assignment


def build_openai_endpoint(base_url: str, suffix: str) -> str:
    """Build one OpenAI-compatible endpoint URL from a base URL."""

    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/v1"):
        return f"{normalized_base_url}/{suffix}"
    return f"{normalized_base_url}/v1/{suffix}"


def build_lmstudio_native_chat_endpoint(base_url: str) -> str:
    """Build the LM Studio native chat endpoint URL from a base URL."""

    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/v1"):
        normalized_base_url = normalized_base_url[: -len("/v1")]
    return f"{normalized_base_url}/api/v1/chat"


def extract_assistant_text(message: dict[str, Any]) -> str | None:
    """Extract assistant text from direct or structured content fields."""

    content = extract_assistant_payload(message)["content"]
    return content if isinstance(content, str) else None


def extract_assistant_payload(
    message: dict[str, Any],
) -> dict[str, str | tuple[dict[str, Any], ...] | None]:
    """Extract assistant text and tool calls from one OpenAI-compatible message."""

    direct_content = message.get("content")
    tool_calls = _extract_tool_calls(message.get("tool_calls"))
    if isinstance(direct_content, str) and direct_content.strip():
        return {"content": direct_content, "tool_calls": tool_calls}

    if isinstance(direct_content, list):
        text_parts: list[str] = []
        for part in direct_content:
            if not isinstance(part, dict):
                continue
            text_value = part.get("text")
            if isinstance(text_value, str) and text_value.strip():
                text_parts.append(text_value)
        if text_parts:
            return {"content": "\n".join(text_parts), "tool_calls": tool_calls}

    return {"content": None, "tool_calls": tool_calls}


def _extract_lmstudio_native_message_text(output_items: Any) -> str | None:
    if not isinstance(output_items, list):
        return None
    message_parts: list[str] = []
    for item in output_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            message_parts.append(content)
    if not message_parts:
        return None
    return "\n".join(message_parts)


def _extract_tool_calls(tool_calls: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(tool_calls, list):
        return ()
    normalized_tool_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            normalized_tool_calls.append(tool_call)
    return tuple(normalized_tool_calls)


def _build_openai_chat_message_payload(message: BackendChatMessageNode) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.name is not None:
        payload["name"] = message.name
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = list(message.tool_calls)
    return payload
