from __future__ import annotations

import contextlib
import json
from collections import OrderedDict, deque
from difflib import SequenceMatcher

from .shared import *
from .shared import (
    _clip_for_log,
    _current_request_latency_ms,
    _env_flag,
    _extract_assistant_text,
    _extract_text_and_vision,
    _extract_openai_tool_call_count,
    _hash_text,
    _log_local_llm_traffic,
    _log_text_max_chars,
    _payload_summary,
    _request_id_ctx,
    _routing_efficiency,
    _stream_chunk_thinking_hint,
    _thinking_debug_enabled,
    _thinking_payload_probe,
    _utc_now_iso,
)
from .settings import *
from .requests import *
from .protocols import *
from .protocols import (
    _apply_public_model_name_to_openai_response,
    _is_meaningful_anthropic_event,
    _log_output_analytics,
    _parse_sse_event,
    anthropic_to_openai_payload,
    openai_to_anthropic_response,
    rewrite_openai_stream_model_name,
    translate_openai_stream_to_anthropic,
)

class UpstreamError(Exception):
    def __init__(self, status_code: int, body: str):
        super().__init__(f"Upstream request failed with status {status_code}")
        self.status_code = status_code
        self.body = body


class LMStudioClient:
    @staticmethod
    def _upstream_headers(settings: LMStudioSettings) -> dict[str, str]:
        if settings.provider != "openai":
            return {}

        api_key = settings.resolve_api_key()
        if not api_key:
            env_name = (settings.api_key_env or "OPENAI_API_KEY").strip() or "OPENAI_API_KEY"
            raise UpstreamError(
                500,
                f"OpenAI API key missing. Configure upstream.api_key or set env var {env_name}.",
            )

        headers = {"Authorization": f"Bearer {api_key}"}
        organization = (settings.organization or "").strip()
        if organization:
            headers["OpenAI-Organization"] = organization
        project = (settings.project or "").strip()
        if project:
            headers["OpenAI-Project"] = project
        return headers

    @staticmethod
    def _native_rest_preferred(settings: LMStudioSettings) -> bool:
        return settings.provider == "lm_studio" and bool(settings.prefer_native_rest_api)

    @staticmethod
    def _extract_lmstudio_thinking_flag(payload: dict[str, Any]) -> Optional[bool]:
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, str):
            normalized = reasoning.strip().lower()
            if normalized in {"off", "none", "false", "0"}:
                return False
            if normalized in {"on", "low", "medium", "high", "true", "1"}:
                return True
        if isinstance(reasoning, bool):
            return reasoning

        for container_key, nested_key in (
            ("chat_template_kwargs", "enable_thinking"),
            ("extra_body", "thinking"),
            ("options", "thinking"),
        ):
            container = payload.get(container_key)
            if isinstance(container, dict) and isinstance(container.get(nested_key), bool):
                return bool(container.get(nested_key))

        thinking = payload.get("thinking")
        if isinstance(thinking, bool):
            return thinking
        return None

    @classmethod
    def _lmstudio_native_reasoning(cls, payload: dict[str, Any]) -> Optional[str]:
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, dict):
            effort = str(reasoning.get("effort") or "").strip().lower()
            if effort in {"low", "medium", "high"}:
                return effort
        elif isinstance(reasoning, str):
            normalized = reasoning.strip().lower()
            if normalized in {"off", "low", "medium", "high", "on"}:
                return normalized

        thinking = cls._extract_lmstudio_thinking_flag(payload)
        if thinking is True:
            return "on"
        if thinking is False:
            return "off"
        return None

    @staticmethod
    def _content_to_lmstudio_native_input(content: Any) -> Optional[Any]:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return None

        items: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                if part:
                    items.append({"type": "message", "content": part})
                continue
            if not isinstance(part, dict):
                return None

            part_type = str(part.get("type") or "").strip().lower()
            if part_type in {"text", "input_text"}:
                text = str(part.get("text") or "")
                if text:
                    items.append({"type": "message", "content": text})
                continue
            if part_type in {"image_url", "input_image", "image"}:
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    url = str(image_url.get("url") or "").strip()
                else:
                    url = str(part.get("url") or "").strip()
                if not url:
                    source = part.get("source")
                    if isinstance(source, dict):
                        media_type = str(source.get("media_type") or "image/png")
                        if source.get("type") == "base64":
                            data = str(source.get("data") or "")
                            if data:
                                url = f"data:{media_type};base64,{data}"
                        else:
                            url = str(source.get("url") or "").strip()
                if not url:
                    return None
                items.append({"type": "image", "data_url": url})
                continue
            return None

        if not items:
            return ""
        if len(items) == 1 and items[0]["type"] == "message":
            return items[0]["content"]
        return items

    @classmethod
    def _lmstudio_native_chat_request(cls, payload: dict[str, Any]) -> tuple[Optional[dict[str, Any]], str]:
        if payload.get("tools") or payload.get("tool_choice"):
            return None, "tools_not_supported_by_native_chat"

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return None, "messages_missing"

        system_prompt = ""
        user_input: Optional[Any] = None
        user_messages = 0
        for msg in messages:
            if not isinstance(msg, dict):
                return None, "message_not_object"
            role = str(msg.get("role") or "").strip().lower()
            content = msg.get("content")
            if role == "system":
                text, has_vision = _extract_text_and_vision(content)
                if has_vision:
                    return None, "system_vision_not_supported"
                if not text.strip():
                    continue
                if system_prompt:
                    system_prompt = f"{system_prompt}\n\n{text}".strip()
                else:
                    system_prompt = text.strip()
                continue
            if role != "user":
                return None, f"role_{role or 'unknown'}_requires_openai_compat"
            user_messages += 1
            if user_messages > 1:
                return None, "multiple_user_messages_require_openai_compat"
            user_input = cls._content_to_lmstudio_native_input(content)
            if user_input is None:
                return None, "user_content_not_supported"

        if user_messages != 1 or user_input is None:
            return None, "single_user_message_required"

        native_payload: dict[str, Any] = {
            "model": payload.get("model"),
            "input": user_input,
            "store": False,
        }
        if system_prompt:
            native_payload["system_prompt"] = system_prompt
        if "stream" in payload:
            native_payload["stream"] = bool(payload.get("stream"))
        if "temperature" in payload:
            native_payload["temperature"] = payload.get("temperature")
        if "top_p" in payload:
            native_payload["top_p"] = payload.get("top_p")
        if "max_completion_tokens" in payload:
            native_payload["max_output_tokens"] = payload.get("max_completion_tokens")
        elif "max_tokens" in payload:
            native_payload["max_output_tokens"] = payload.get("max_tokens")

        reasoning = cls._lmstudio_native_reasoning(payload)
        if reasoning:
            native_payload["reasoning"] = reasoning
        return native_payload, "native_chat_ready"

    @staticmethod
    def _lmstudio_native_output_items(response: dict[str, Any]) -> list[dict[str, Any]]:
        output = response.get("output")
        if isinstance(output, list):
            return [item for item in output if isinstance(item, dict)]
        result = response.get("result")
        if isinstance(result, dict):
            output = result.get("output")
            if isinstance(output, list):
                return [item for item in output if isinstance(item, dict)]
        return []

    @classmethod
    def _lmstudio_native_response_to_openai(
        cls,
        request_payload: dict[str, Any],
        response: dict[str, Any],
    ) -> dict[str, Any]:
        output_items = cls._lmstudio_native_output_items(response)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in output_items:
            item_type = str(item.get("type") or "").strip().lower()
            if item_type == "message":
                content = item.get("content")
                if isinstance(content, str):
                    if content:
                        text_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and str(part.get("type") or "").strip().lower() in {"text", "output_text"}:
                            text = str(part.get("text") or "")
                            if text:
                                text_parts.append(text)
            elif item_type == "reasoning":
                reasoning = str(item.get("content") or item.get("text") or "")
                if reasoning:
                    reasoning_parts.append(reasoning)
            elif item_type == "tool_call":
                tool_calls.append(
                    {
                        "id": str(item.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": str(item.get("name") or "tool"),
                            "arguments": json.dumps(item.get("arguments") or {}, ensure_ascii=False),
                        },
                    }
                )

        stats = response.get("stats")
        if not isinstance(stats, dict):
            stats = {}
        prompt_tokens = int(stats.get("input_tokens") or 0)
        completion_tokens = int(stats.get("output_tokens") or stats.get("total_output_tokens") or 0)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_parts),
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls

        finish_reason = "tool_calls" if tool_calls else "stop"
        created = int(time.time())
        response_id = str(response.get("response_id") or response.get("id") or f"chatcmpl_{uuid.uuid4().hex}")
        model_id = str(response.get("model") or request_payload.get("model") or "")
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @classmethod
    def _resolve_request_target(
        cls,
        settings: LMStudioSettings,
        path: str,
        payload: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str, str]:
        if not cls._native_rest_preferred(settings) or path != "/v1/chat/completions":
            transport = "openai_compat" if settings.provider == "lm_studio" else settings.provider
            return path, payload, transport, "native_rest_not_selected"

        native_payload, note = cls._lmstudio_native_chat_request(payload)
        if native_payload is None:
            return path, payload, "openai_compat", note
        return "/api/v1/chat", native_payload, "lm_studio_rest", note

    async def post_json(
        self,
        settings: LMStudioSettings,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        actual_path, actual_payload, transport, note = self._resolve_request_target(settings, path, payload)
        url = settings.base_url.rstrip("/") + actual_path
        headers = self._upstream_headers(settings)
        timeout = httpx.Timeout(settings.timeout_seconds)
        start = time.perf_counter()
        logger.info(
            "upstream_post_start provider=%s requested_path=%s actual_path=%s transport=%s %s",
            settings.provider,
            path,
            actual_path,
            transport,
            _payload_summary(actual_payload),
        )
        _log_local_llm_traffic(
            "request_json",
            provider=settings.provider,
            base_url=settings.base_url,
            requested_path=path,
            actual_path=actual_path,
            transport=transport,
            payload=actual_payload,
            note=note,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=actual_payload, headers=headers or None)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                if response.status_code >= 400:
                    logger.warning(
                        "upstream_post_failed requested_path=%s actual_path=%s status=%s duration_ms=%s body=%s",
                        path,
                        actual_path,
                        response.status_code,
                        elapsed_ms,
                        response.text[:300],
                    )
                    _log_local_llm_traffic(
                        "response_error",
                        provider=settings.provider,
                        base_url=settings.base_url,
                        requested_path=path,
                        actual_path=actual_path,
                        transport=transport,
                        status_code=response.status_code,
                        duration_ms=elapsed_ms,
                        response={"body": response.text},
                        note=note,
                    )
                    if transport == "lm_studio_rest" and response.status_code in {404, 405, 501}:
                        logger.warning(
                            "upstream_post_native_fallback requested_path=%s status=%s reason=%s",
                            path,
                            response.status_code,
                            note,
                        )
                        return await self.post_json(
                            settings.model_copy(update={"prefer_native_rest_api": False}),
                            path,
                            payload,
                        )
                    raise UpstreamError(response.status_code, response.text)
                logger.info(
                    "upstream_post_ok requested_path=%s actual_path=%s status=%s duration_ms=%s transport=%s",
                    path,
                    actual_path,
                    response.status_code,
                    elapsed_ms,
                    transport,
                )
                try:
                    body = response.json()
                    if transport == "lm_studio_rest":
                        body = self._lmstudio_native_response_to_openai(actual_payload, body)
                    _log_local_llm_traffic(
                        "response_json",
                        provider=settings.provider,
                        base_url=settings.base_url,
                        requested_path=path,
                        actual_path=actual_path,
                        transport=transport,
                        status_code=response.status_code,
                        duration_ms=elapsed_ms,
                        response=body,
                        note=note,
                    )
                    return body
                except ValueError as exc:
                    body = response.text[:300]
                    logger.warning("upstream_post_invalid_json requested_path=%s actual_path=%s body=%s", path, actual_path, body)
                    _log_local_llm_traffic(
                        "response_invalid_json",
                        provider=settings.provider,
                        base_url=settings.base_url,
                        requested_path=path,
                        actual_path=actual_path,
                        transport=transport,
                        status_code=response.status_code,
                        duration_ms=elapsed_ms,
                        response={"body": response.text},
                        note=note,
                    )
                    raise UpstreamError(502, f"Invalid JSON from upstream: {body}") from exc
        except httpx.TimeoutException as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "upstream_post_timeout requested_path=%s actual_path=%s duration_ms=%s timeout_s=%s error=%s",
                path,
                actual_path,
                elapsed_ms,
                settings.timeout_seconds,
                exc,
            )
            _log_local_llm_traffic(
                "response_timeout",
                provider=settings.provider,
                base_url=settings.base_url,
                requested_path=path,
                actual_path=actual_path,
                transport=transport,
                duration_ms=elapsed_ms,
                response={"error": str(exc)},
                note=note,
            )
            raise UpstreamError(504, f"Upstream timeout after {settings.timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "upstream_post_http_error requested_path=%s actual_path=%s duration_ms=%s error=%s",
                path,
                actual_path,
                elapsed_ms,
                exc,
            )
            _log_local_llm_traffic(
                "response_http_error",
                provider=settings.provider,
                base_url=settings.base_url,
                requested_path=path,
                actual_path=actual_path,
                transport=transport,
                duration_ms=elapsed_ms,
                response={"error": str(exc)},
                note=note,
            )
            raise UpstreamError(502, f"Upstream HTTP error: {exc}") from exc

    async def get_json(
        self,
        settings: LMStudioSettings,
        path: str,
    ) -> Any:
        url = settings.base_url.rstrip("/") + path
        headers = self._upstream_headers(settings)
        timeout = httpx.Timeout(settings.timeout_seconds)
        start = time.perf_counter()
        logger.info("upstream_get_start provider=%s path=%s", settings.provider, path)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=headers or None)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                if response.status_code >= 400:
                    logger.warning(
                        "upstream_get_failed path=%s status=%s duration_ms=%s body=%s",
                        path,
                        response.status_code,
                        elapsed_ms,
                        response.text[:300],
                    )
                    raise UpstreamError(response.status_code, response.text)
                logger.info(
                    "upstream_get_ok path=%s status=%s duration_ms=%s",
                    path,
                    response.status_code,
                    elapsed_ms,
                )
                try:
                    return response.json()
                except ValueError as exc:
                    body = response.text[:300]
                    logger.warning("upstream_get_invalid_json path=%s body=%s", path, body)
                    raise UpstreamError(502, f"Invalid JSON from upstream: {body}") from exc
        except httpx.TimeoutException as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "upstream_get_timeout path=%s duration_ms=%s timeout_s=%s error=%s",
                path,
                elapsed_ms,
                settings.timeout_seconds,
                exc,
            )
            raise UpstreamError(504, f"Upstream timeout after {settings.timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("upstream_get_http_error path=%s duration_ms=%s error=%s", path, elapsed_ms, exc)
            raise UpstreamError(502, f"Upstream HTTP error: {exc}") from exc

    @staticmethod
    def _parse_model_items(payload: Any) -> list[dict[str, Any]]:
        raw_items: Any
        if isinstance(payload, list):
            raw_items = payload
        elif isinstance(payload, dict):
            raw_items = payload.get("data")
            if raw_items is None:
                raw_items = payload.get("models")
        else:
            raw_items = None

        if not isinstance(raw_items, list):
            return []

        items: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_items):
            if isinstance(item, dict):
                items.append(item)
            else:
                items.append({"id": str(item), "_raw_index": idx})
        return items

    async def list_models(
        self,
        settings: LMStudioSettings,
    ) -> tuple[str, list[dict[str, Any]]]:
        if settings.provider == "lm_studio":
            candidate_paths = ["/api/v1/models", "/v1/models", "/api/v0/models"]
        else:
            candidate_paths = ["/v1/models"]

        last_error: Optional[UpstreamError] = None
        for path in candidate_paths:
            try:
                payload = await self.get_json(settings, path)
                return path, self._parse_model_items(payload)
            except UpstreamError as exc:
                last_error = exc
                logger.warning("upstream_list_models_failed path=%s status=%s", path, exc.status_code)

        if last_error:
            raise last_error
        raise UpstreamError(502, "Unable to read model list from upstream.")

    @classmethod
    async def _lmstudio_native_stream_to_openai(
        cls,
        response: httpx.Response,
        request_payload: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        response_id = f"chatcmpl_{uuid.uuid4().hex}"
        model_id = str(request_payload.get("model") or "")
        created = int(time.time())
        pending_tool_calls = False

        async for raw_line in response.aiter_lines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data_line = line[5:].strip()
            if not data_line:
                continue
            if data_line == "[DONE]":
                break
            try:
                parsed = json.loads(data_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue

            event_type = str(parsed.get("type") or "").strip().lower()
            if event_type == "reasoning.delta":
                delta = str(parsed.get("delta") or "")
                if delta:
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"reasoning_content": delta}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                continue

            if event_type == "message.delta":
                delta = str(parsed.get("delta") or "")
                if delta:
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                continue

            if event_type == "tool_call":
                pending_tool_calls = True
                tool_call = {
                    "index": 0,
                    "id": str(parsed.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": str(parsed.get("name") or "tool"),
                        "arguments": json.dumps(parsed.get("arguments") or {}, ensure_ascii=False),
                    },
                }
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"tool_calls": [tool_call]}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                continue

            if event_type != "chat.end":
                continue

            result = parsed.get("result")
            if not isinstance(result, dict):
                result = {}
            stop_reason = "tool_calls" if pending_tool_calls else "stop"
            usage = {}
            stats = result.get("stats")
            if isinstance(stats, dict):
                prompt_tokens = int(stats.get("input_tokens") or 0)
                completion_tokens = int(stats.get("output_tokens") or stats.get("total_output_tokens") or 0)
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            chunk = {
                "id": str(result.get("response_id") or response_id),
                "object": "chat.completion.chunk",
                "created": created,
                "model": str(result.get("model") or model_id),
                "choices": [{"index": 0, "delta": {}, "finish_reason": stop_reason}],
            }
            if usage:
                chunk["usage"] = usage
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")

        yield b"data: [DONE]\n\n"

    async def stream_openai(
        self,
        settings: LMStudioSettings,
        path: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        actual_path, actual_payload, transport, note = self._resolve_request_target(settings, path, payload)
        url = settings.base_url.rstrip("/") + actual_path
        headers = self._upstream_headers(settings)
        timeout = httpx.Timeout(settings.timeout_seconds)
        start = time.perf_counter()
        logger.info(
            "upstream_stream_start provider=%s requested_path=%s actual_path=%s transport=%s %s",
            settings.provider,
            path,
            actual_path,
            transport,
            _payload_summary(actual_payload),
        )
        _log_local_llm_traffic(
            "request_stream",
            provider=settings.provider,
            base_url=settings.base_url,
            requested_path=path,
            actual_path=actual_path,
            transport=transport,
            payload=actual_payload,
            note=note,
        )
        chunk_count = 0
        byte_count = 0
        raw_chunks: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=actual_payload, headers=headers or None) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", errors="replace")
                        elapsed_ms = int((time.perf_counter() - start) * 1000)
                        logger.warning(
                            "upstream_stream_failed requested_path=%s actual_path=%s status=%s duration_ms=%s body=%s",
                            path,
                            actual_path,
                            response.status_code,
                            elapsed_ms,
                            body[:300],
                        )
                        _log_local_llm_traffic(
                            "response_stream_error",
                            provider=settings.provider,
                            base_url=settings.base_url,
                            requested_path=path,
                            actual_path=actual_path,
                            transport=transport,
                            status_code=response.status_code,
                            duration_ms=elapsed_ms,
                            response={"body": body},
                            note=note,
                        )
                        if transport == "lm_studio_rest" and response.status_code in {404, 405, 501}:
                            logger.warning(
                                "upstream_stream_native_fallback requested_path=%s status=%s reason=%s",
                                path,
                                response.status_code,
                                note,
                            )
                            async for chunk in self.stream_openai(
                                settings.model_copy(update={"prefer_native_rest_api": False}),
                                path,
                                payload,
                            ):
                                yield chunk
                            return
                        raise UpstreamError(response.status_code, body)

                    stream_iter: AsyncIterator[bytes]
                    if transport == "lm_studio_rest":
                        stream_iter = self._lmstudio_native_stream_to_openai(response, actual_payload)
                    else:
                        stream_iter = response.aiter_bytes()

                    async for chunk in stream_iter:
                        if chunk:
                            chunk_count += 1
                            byte_count += len(chunk)
                            if len("".join(raw_chunks)) < _log_text_max_chars():
                                raw_chunks.append(chunk.decode("utf-8", errors="replace"))
                            if chunk_count == 1:
                                logger.info(
                                    "upstream_stream_first_chunk requested_path=%s actual_path=%s first_chunk_bytes=%s",
                                    path,
                                    actual_path,
                                    len(chunk),
                                )
                            yield chunk

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "upstream_stream_done requested_path=%s actual_path=%s chunks=%s bytes=%s duration_ms=%s transport=%s",
                path,
                actual_path,
                chunk_count,
                byte_count,
                elapsed_ms,
                transport,
            )
            _log_local_llm_traffic(
                "response_stream",
                provider=settings.provider,
                base_url=settings.base_url,
                requested_path=path,
                actual_path=actual_path,
                transport=transport,
                status_code=200,
                duration_ms=elapsed_ms,
                response={"raw_sse_excerpt": _clip_for_log("".join(raw_chunks), _log_text_max_chars())},
                note=note,
            )
        except httpx.TimeoutException as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "upstream_stream_timeout requested_path=%s actual_path=%s duration_ms=%s timeout_s=%s error=%s",
                path,
                actual_path,
                elapsed_ms,
                settings.timeout_seconds,
                exc,
            )
            _log_local_llm_traffic(
                "response_stream_timeout",
                provider=settings.provider,
                base_url=settings.base_url,
                requested_path=path,
                actual_path=actual_path,
                transport=transport,
                duration_ms=elapsed_ms,
                response={"error": str(exc)},
                note=note,
            )
            raise UpstreamError(504, f"Upstream stream timeout after {settings.timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "upstream_stream_http_error requested_path=%s actual_path=%s duration_ms=%s error=%s",
                path,
                actual_path,
                elapsed_ms,
                exc,
            )
            _log_local_llm_traffic(
                "response_stream_http_error",
                provider=settings.provider,
                base_url=settings.base_url,
                requested_path=path,
                actual_path=actual_path,
                transport=transport,
                duration_ms=elapsed_ms,
                response={"error": str(exc)},
                note=note,
            )
            raise UpstreamError(502, f"Upstream stream HTTP error: {exc}") from exc


class ModelAvailabilityMonitor:
    def __init__(self, config_store: ConfigStore, lm_client: Any, check_interval_seconds: float = 60.0):
        self.config_store = config_store
        self.lm_client = lm_client
        self.check_interval_seconds = max(5.0, float(check_interval_seconds))
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._run_lock = asyncio.Lock()
        self._status_lock = asyncio.Lock()
        self._status: dict[str, Any] = {
            "last_checked_at": None,
            "provider": None,
            "base_url": None,
            "catalog_path": None,
            "all_available": False,
            "all_loaded": False,
            "upstreams": [],
            "models": [],
            "error": "not_checked_yet",
            "check_interval_seconds": self.check_interval_seconds,
        }

    @staticmethod
    def _utc_now_iso() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @staticmethod
    def _extract_model_id(item: dict[str, Any]) -> str:
        for key in ("id", "key", "model_id", "model", "name", "display_name"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                raw = value.strip()
                # If it looks like a Windows or absolute path, take only the basename.
                # But if it's a "repo/model" style ID, keep it.
                # Windows paths: C:\... or \\server\...
                # Absolute Unix paths: /...
                import os
                if (":" in raw and "\\" in raw) or raw.startswith("/") or raw.startswith("\\\\"):
                    normalized = raw.replace("\\", "/")
                    return os.path.basename(normalized)
                return raw
        return ""

    @staticmethod
    def _normalize_model_id(value: str) -> str:
        return value.strip().lower()

    @staticmethod
    def _should_poll_upstream(upstream_ref: str) -> bool:
        return (upstream_ref or "").strip().lower() != "deep"

    @classmethod
    def _model_id_matches(cls, expected: str, actual: str) -> bool:
        return cls._normalize_model_id(expected) == cls._normalize_model_id(actual)

    @staticmethod
    def _extract_loaded_state(item: dict[str, Any]) -> Optional[bool]:
        for key in ("loaded", "is_loaded"):
            value = item.get(key)
            if isinstance(value, bool):
                return value

        for key in ("state", "status", "load_state"):
            value = item.get(key)
            if isinstance(value, str):
                normalized = value.strip().lower()
                # "ready" or "running" are common in LM-Studio or other local backends
                if normalized in {"loaded", "ready", "running", "active", "available", "on"}:
                    return True
                if normalized in {"unloaded", "not_loaded", "stopped", "inactive", "error", "failed", "off"}:
                    return False
        return None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run_loop(), name="model-availability-monitor")

    async def stop(self) -> None:
        self._stop_event.set()
        task = self._task
        self._task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.check_interval_seconds)
                break
            except asyncio.TimeoutError:
                await self.run_check_once()
            except asyncio.CancelledError:
                break

    async def run_check_once(self) -> None:
        async with self._run_lock:
            cfg = self.config_store.get_config()
            checked_at = self._utc_now_iso()
            models_status: list[dict[str, Any]] = []
            catalog_path: Optional[str] = None
            all_available = False
            all_loaded = False
            error: Optional[str] = None
            upstream_status: list[dict[str, Any]] = []

            list_models_fn = getattr(self.lm_client, "list_models", None)
            if not callable(list_models_fn):
                error = "lm_client_does_not_support_list_models"
                logger.warning("model_availability_check_failed error=%s", error)
            else:
                try:
                    upstream_catalogs: dict[str, dict[str, Any]] = {}
                    for upstream_ref, upstream_settings in cfg.upstreams.items():
                        if self._should_poll_upstream(upstream_ref):
                            try:
                                path, items = await list_models_fn(upstream_settings)
                                upstream_catalogs[upstream_ref] = {
                                    "path": path,
                                    "items": items,
                                    "error": None,
                                    "skipped": False,
                                }
                            except Exception as upstream_exc:  # noqa: BLE001
                                upstream_catalogs[upstream_ref] = {
                                    "path": None,
                                    "items": [],
                                    "error": str(upstream_exc),
                                    "skipped": False,
                                }
                        else:
                            upstream_catalogs[upstream_ref] = {
                                "path": None,
                                "items": [],
                                "error": None,
                                "skipped": True,
                            }
                        upstream_status.append(
                            {
                                "upstream_ref": upstream_ref,
                                "provider": upstream_settings.provider,
                                "base_url": upstream_settings.base_url,
                                "catalog_path": upstream_catalogs[upstream_ref]["path"],
                                "error": upstream_catalogs[upstream_ref]["error"],
                                "skipped": upstream_catalogs[upstream_ref]["skipped"],
                            }
                        )

                    expected_models = [(alias, profile.model_id) for alias, profile in cfg.models.items()]
                    for alias, expected_model_id in expected_models:
                        profile = cfg.models[alias]
                        upstream_ref = (profile.upstream_ref or "").strip() or "local"
                        upstream_data = upstream_catalogs.get(upstream_ref, {"items": [], "error": "unknown_upstream_ref"})
                        if upstream_data.get("skipped"):
                            models_status.append(
                                {
                                    "alias": alias,
                                    "enabled": profile.enabled,
                                    "model_id": expected_model_id,
                                    "upstream_ref": upstream_ref,
                                    "matched_upstream_id": None,
                                    "available": True,
                                    "loaded": True,
                                    "loaded_inferred": False,
                                    "upstream_error": None,
                                    "poll_skipped": True,
                                }
                            )
                            continue
                        items = upstream_data.get("items") or []

                        matched_item: Optional[dict[str, Any]] = None
                        matched_id = ""
                        for item in items:
                            model_id = self._extract_model_id(item)
                            if model_id and self._model_id_matches(expected_model_id, model_id):
                                matched_item = item
                                matched_id = model_id
                                break

                        available = matched_item is not None
                        loaded_inferred = False
                        if not available:
                            loaded = False
                        else:
                            explicit_loaded = self._extract_loaded_state(matched_item or {})
                            if explicit_loaded is None:
                                loaded = True
                                loaded_inferred = True
                            else:
                                loaded = explicit_loaded

                        models_status.append(
                            {
                                "alias": alias,
                                "enabled": profile.enabled,
                                "model_id": expected_model_id,
                                "upstream_ref": upstream_ref,
                                "matched_upstream_id": matched_id or None,
                                "available": available,
                                "loaded": loaded,
                                "loaded_inferred": loaded_inferred,
                                "upstream_error": upstream_data.get("error"),
                                "poll_skipped": False,
                            }
                        )

                    enabled_models = [item for item in models_status if item["enabled"]]
                    all_available = bool(enabled_models) and all(item["available"] for item in enabled_models)
                    all_loaded = bool(enabled_models) and all(item["loaded"] for item in enabled_models)
                    catalog_paths = sorted(
                        {
                            str(item.get("catalog_path"))
                            for item in upstream_status
                            if item.get("catalog_path")
                        }
                    )
                    catalog_path = ", ".join(catalog_paths) if catalog_paths else None

                    if all_available and all_loaded:
                        logger.info(
                            "model_availability_check_ok upstreams=%s path=%s models=%s",
                            [f"{item['upstream_ref']}:{item['provider']}" for item in upstream_status],
                            catalog_paths,
                            [f"{item['alias']}:{item['model_id']}" for item in models_status],
                        )
                    else:
                        logger.warning(
                            "model_availability_check_problem upstreams=%s path=%s all_available=%s all_loaded=%s models=%s",
                            [f"{item['upstream_ref']}:{item['provider']}" for item in upstream_status],
                            catalog_paths,
                            all_available,
                            all_loaded,
                            models_status,
                        )
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)
                    logger.warning("model_availability_check_failed error=%r", exc)

            status = {
                "last_checked_at": checked_at,
                "provider": cfg.default_upstream().provider,
                "base_url": cfg.default_upstream().base_url,
                "catalog_path": catalog_path,
                "all_available": all_available,
                "all_loaded": all_loaded,
                "upstreams": upstream_status,
                "models": models_status,
                "error": error,
                "check_interval_seconds": self.check_interval_seconds,
            }
            async with self._status_lock:
                self._status = status

    async def get_status(self) -> dict[str, Any]:
        async with self._status_lock:
            return dict(self._status)


class ModelAutoConfigurator:
    """Fetches available models from upstreams and auto-assigns them to
    router categories (small, large, deep, backup) based on a priority
    list stored in ``config/model_priorities.json``.

    Unknown models (not in the priority list) are classified by asking
    an available LLM to self-categorise via a structured prompt."""

    # DEFAULT_PRIORITIES_PATH = PROJECT_ROOT / "config" / "model_priorities.json"
    DEFAULT_CLASSIFY_PROMPT_PATH = PROJECT_ROOT / "config" / "model_classify_prompt.yaml"

    VALID_CATEGORIES = {"small", "large", "deep", "backup"}

    def __init__(
        self,
        config_store: ConfigStore,
        lm_client: Any,
        priorities_path: Optional[Path] = None,
        classify_prompt_path: Optional[Path] = None,
    ):
        self.config_store = config_store
        self.lm_client = lm_client
        self.priorities_path = priorities_path or self.DEFAULT_PRIORITIES_PATH
        self.classify_prompt_path = classify_prompt_path or self.DEFAULT_CLASSIFY_PROMPT_PATH
        self._last_result: dict[str, Any] = {}
        self._classify_failed: set[str] = set()  # models that failed classification – skip on retry

    # ------------------------------------------------------------------
    # Priority file helpers
    # ------------------------------------------------------------------

    def _load_priorities(self) -> dict[str, Any]:
        if not self.priorities_path.exists():
            logger.warning("model_auto_config priorities file not found path=%s", self.priorities_path)
            return {}
        with open(self.priorities_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Strip meta keys
        return {k: v for k, v in data.items() if not k.startswith("_")}

    @staticmethod
    def _match_priority(available_models: list[dict[str, Any]], priority_patterns: list[str]) -> Optional[str]:
        """Return the first available model id that matches a priority pattern (substring, case-insensitive).
        
        Within multiple models matching the same pattern, prefer those that are already loaded.
        """
        for pattern in priority_patterns:
            pat_lower = pattern.strip().lower()
            matching_models: list[tuple[str, dict[str, Any]]] = []
            
            for model in available_models:
                mid = ModelAvailabilityMonitor._extract_model_id(model)
                
                # Check BOTH the extracted ID AND all possible raw name keys
                match_candidates = {mid.lower()}
                for key in ("id", "model_id", "model", "name"):
                    val = model.get(key)
                    if isinstance(val, str):
                        raw_val = val.strip().lower().replace("\\", "/")
                        match_candidates.add(raw_val)
                        # Also add the basename just in case pattern is only the filename
                        import os
                        match_candidates.add(os.path.basename(raw_val))

                if any(pat_lower in c for c in match_candidates):
                    matching_models.append((mid, model))
            
            if matching_models:
                # Prefer loaded ones
                for mid, model in matching_models:
                    if ModelAvailabilityMonitor._extract_loaded_state(model) is True:
                        return mid
                
                # Fallback to first matching (even if not loaded)
                return matching_models[0][0]
        return None

    # ------------------------------------------------------------------
    # Unknown-model classification via LLM
    # ------------------------------------------------------------------

    def _load_classify_prompt(self) -> dict[str, str]:
        """Load the classification prompt templates from YAML."""
        if not self.classify_prompt_path.exists():
            logger.warning("model_classify_prompt not found path=%s", self.classify_prompt_path)
            return {}
        with open(self.classify_prompt_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _collect_known_patterns(self, priorities: dict[str, Any]) -> set[str]:
        """Return the set of all priority patterns (lowered) across all categories."""
        known: set[str] = set()
        for prio_cfg in priorities.values():
            for pat in prio_cfg.get("priority", []):
                known.add(pat.strip().lower())
        return known

    def _find_unknown_models(
        self, upstream_models: dict[str, list[dict[str, Any]]], known_patterns: set[str]
    ) -> list[tuple[str, str]]:
        """Return ``[(model_id, upstream_ref), ...]`` for models not matched by any known pattern."""
        unknown: list[tuple[str, str]] = []
        for upstream_ref, model_items in upstream_models.items():
            for item in model_items:
                model_id = ModelAvailabilityMonitor._extract_model_id(item)
                if not model_id:
                    continue
                mid_lower = model_id.strip().lower()
                if not any(pat in mid_lower for pat in known_patterns):
                    unknown.append((model_id, upstream_ref))
        return unknown

    async def _classify_unknown_model(
        self,
        model_id: str,
        upstream_ref: str,
        cfg: "RouterConfig",
        prompt_templates: dict[str, str],
        upstream_models: Optional[dict[str, list[dict[str, Any]]]] = None,
    ) -> Optional[dict[str, Any]]:
        """Ask an available LLM to classify *model_id* into a category.

        Returns the parsed JSON response or ``None`` on failure.
        Uses the *actually loaded* model (first in the upstream's model list)
        rather than a configured-but-potentially-unloaded model.
        """
        system_prompt = prompt_templates.get("system", "")
        user_template = prompt_templates.get("user_template", "")
        if not system_prompt or not user_template:
            return None

        user_msg = user_template.format(model_id=model_id, upstream_ref=upstream_ref)

        # Use the *loaded* model (first model in each upstream's list, per
        # LM-Studio convention) for the classification request.  Fall back to
        # configured models only if no loaded model can be determined.
        target_upstream_ref: Optional[str] = None
        target_model_id: Optional[str] = None

        if upstream_models:
            for uref, model_items in upstream_models.items():
                if model_items and cfg.upstreams.get(uref):
                    # First model = currently loaded in LM-Studio
                    candidate = ModelAvailabilityMonitor._extract_model_id(model_items[0])
                    if not candidate:
                        continue
                    # Don't use the model we're trying to classify
                    if candidate != model_id:
                        target_upstream_ref = uref
                        target_model_id = candidate
                        break

        # Fallback: try configured categories
        if not target_upstream_ref or not target_model_id:
            for cat in ("small", "large", "deep", "backup"):
                cat_cfg = cfg.models.get(cat)
                if cat_cfg and cfg.upstreams.get(cat_cfg.upstream_ref):
                    target_upstream_ref = cat_cfg.upstream_ref
                    target_model_id = cat_cfg.model_id
                    break

        if not target_upstream_ref or not target_model_id:
            logger.warning("model_classify no configured model available for classification of %s", model_id)
            return None

        upstream_settings = cfg.upstreams[target_upstream_ref]

        payload = {
            "model": target_model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.1,
            "max_tokens": 256,
        }

        post_fn = getattr(self.lm_client, "post_json", None)
        if not callable(post_fn):
            return None

        try:
            response = await post_fn(upstream_settings, "/v1/chat/completions", payload)
            content = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            import re
            content_cleaned = re.sub(r"<(thought|thinking)>.*?</\1>", "", content, flags=re.DOTALL | re.IGNORECASE)
            # If there's an unclosed tag at the beginning/middle (common in streaming or truncated responses)
            content_cleaned = re.sub(r"<(thought|thinking)>.*", "", content_cleaned, flags=re.DOTALL | re.IGNORECASE)

            # Strip markdown fences if present
            text = content_cleaned.strip()
            # Also try to extract JSON by finding first { and last }
            # which might be after or before other text
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
            if not text:
                logger.warning("model_classify empty response for model=%s", model_id)
                return None
            result = json.loads(text)
            category = result.get("category", "").lower()
            if category not in ("small", "large"):
                # Fallback per user request: only small or large allowed, default to small
                category = "small"
            
            result["category"] = category
            logger.info(
                "model_classify model=%s -> category=%s confidence=%.2f reasoning=%s",
                model_id,
                category,
                result.get("confidence", 0),
                result.get("reasoning", ""),
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("model_classify failed model=%s error=%s", model_id, exc)
            return None

    def _add_to_priorities_file(
        self, model_id: str, category: str, classification: dict[str, Any]
    ) -> None:
        """Append *model_id* to the priority list of *category* in the JSON file."""
        try:
            with open(self.priorities_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if category in data and "priority" in data[category]:
                prio_list: list[str] = data[category]["priority"]
                # Append at end (lowest priority within category)
                if model_id not in prio_list:
                    prio_list.append(model_id)
                    with open(self.priorities_path, "w", encoding="utf-8") as fh:
                        json.dump(data, fh, indent=2, ensure_ascii=False)
                        fh.write("\n")
                    logger.info(
                        "model_classify persisted model=%s in category=%s (confidence=%.2f)",
                        model_id,
                        category,
                        classification.get("confidence", 0),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("model_classify persist_failed model=%s error=%s", model_id, exc)

    # ------------------------------------------------------------------
    # Upstream fetching
    # ------------------------------------------------------------------

    async def _fetch_upstream_models(self, cfg: RouterConfig) -> dict[str, list[dict[str, Any]]]:
        """Return ``{upstream_ref: [model_item_dict, ...]}`` for every upstream."""
        list_models_fn = getattr(self.lm_client, "list_models", None)
        if not callable(list_models_fn):
            return {}

        result: dict[str, list[dict[str, Any]]] = {}
        for upstream_ref, upstream_settings in cfg.upstreams.items():
            try:
                _, items = await list_models_fn(upstream_settings)
                
                # Ensure each item is a dict and has an 'id' key for backward compatibility
                processed_items: list[dict[str, Any]] = []
                for item in items:
                    if isinstance(item, dict):
                        record = dict(item)
                        # If no standard key exists, inject a usable id placeholder.
                        if not any(k in record for k in ("id", "key", "model_id", "model", "name", "display_name")):
                            record["id"] = str(item)
                        processed_items.append(record)
                    else:
                        processed_items.append({"id": str(item)})
                
                result[upstream_ref] = processed_items
                model_ids = [ModelAvailabilityMonitor._extract_model_id(item) for item in processed_items]
                logger.info(
                    "model_auto_config fetched upstream=%s models=%s",
                    upstream_ref,
                    model_ids,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("model_auto_config fetch_failed upstream=%s error=%s", upstream_ref, exc)
                result[upstream_ref] = []
        return result

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        """Fetch models, match priorities, update config in-memory + on disk.

        Returns a summary dict of what was changed.
        """
        cfg = self.config_store.get_config()
        priorities = self._load_priorities()
        if not priorities:
            logger.info("model_auto_config skipped – no priorities loaded")
            self._last_result = {"skipped": True, "reason": "no_priorities"}
            return self._last_result

        upstream_models = await self._fetch_upstream_models(cfg)
        if not any(upstream_models.values()):
            logger.warning("model_auto_config skipped – no models fetched from any upstream")
            self._last_result = {"skipped": True, "reason": "no_upstream_models"}
            return self._last_result

        changes: dict[str, dict[str, str]] = {}

        # Build updated YAML data from current config
        current_data = cfg.model_dump(mode="python")
        current_data.pop("lm_studio", None)

        for alias, prio_cfg in priorities.items():
            if alias not in current_data.get("models", {}):
                continue
            prio_list = prio_cfg.get("priority", [])
            upstream_ref = prio_cfg.get("upstream_ref", "local")
            defaults = prio_cfg.get("defaults", {})

            available_models = upstream_models.get(upstream_ref, [])
            if not available_models:
                logger.info("model_auto_config alias=%s upstream=%s has no models – skipping", alias, upstream_ref)
                continue

            matched_id = self._match_priority(available_models, prio_list)
            if not matched_id:
                logger.info(
                    "model_auto_config alias=%s no priority match in %d available models – keeping current",
                    alias,
                    len(available_models),
                )
                continue

            old_model_id = current_data["models"][alias].get("model_id", "")
            if old_model_id == matched_id:
                logger.info("model_auto_config alias=%s model_id unchanged (%s)", alias, matched_id)
                continue

            # Apply change
            current_data["models"][alias]["model_id"] = matched_id
            current_data["models"][alias]["upstream_ref"] = upstream_ref
            # Apply defaults from priority file where not explicitly set in config
            for key, value in defaults.items():
                current_data["models"][alias][key] = value

            changes[alias] = {"old": old_model_id, "new": matched_id}
            logger.info(
                "model_auto_config alias=%s model_id changed %s -> %s",
                alias,
                old_model_id,
                matched_id,
            )

        if changes:
            yaml_text = yaml.safe_dump(current_data, sort_keys=False, allow_unicode=False)
            await self.config_store.update_from_yaml(yaml_text)
            logger.info("model_auto_config config updated with %d change(s): %s", len(changes), changes)
        else:
            logger.info("model_auto_config no changes needed – all models already optimal")

        # --- Classify unknown models via LLM ---
        # The first model in each upstream's response is the currently loaded
        # model (LM-Studio convention).  If it is unknown we classify it and
        # immediately activate it for the matching router category.
        # User requested: only ask about models that are NOT in the list AND are LOADED.
        classifications: dict[str, dict[str, Any]] = {}
        known_patterns = self._collect_known_patterns(priorities)
        
        # Filter unknown models to only include those that are currently loaded (first in list)
        loaded_unknown_models: list[tuple[str, str]] = []
        for upstream_ref, model_items in upstream_models.items():
            if model_items:
                # The first model in the list is by LM-Studio convention the currently loaded one.
                first_model_item = model_items[0]
                first_model_id = ModelAvailabilityMonitor._extract_model_id(first_model_item)
                if not first_model_id:
                    continue
                mid_lower = first_model_id.strip().lower()
                if not any(pat in mid_lower for pat in known_patterns):
                    loaded_unknown_models.append((first_model_id, upstream_ref))

        activated: dict[str, dict[str, str]] = {}  # category -> {model_id, upstream_ref}

        if loaded_unknown_models:
            prompt_templates = self._load_classify_prompt()
            if prompt_templates:
                # Re-read config in case it was updated above
                cfg = self.config_store.get_config()
                for model_id, upstream_ref in loaded_unknown_models:
                    if model_id in self._classify_failed:
                        logger.debug("model_classify skipping previously failed model=%s", model_id)
                        continue
                    result = await self._classify_unknown_model(
                        model_id, upstream_ref, cfg, prompt_templates,
                        upstream_models=upstream_models,
                    )
                    if result:
                        classifications[model_id] = result
                        self._add_to_priorities_file(model_id, result["category"], result)

                        category = result["category"]
                        if category in ("small", "large"):
                            activated[category] = {
                                "model_id": model_id,
                                "upstream_ref": upstream_ref,
                            }
                    else:
                        self._classify_failed.add(model_id)
            if classifications:
                logger.info(
                    "model_auto_config classified %d unknown model(s): %s",
                    len(classifications),
                    {m: c["category"] for m, c in classifications.items()},
                )

        # --- Activate loaded-but-unknown models for their category ---
        if activated:
            cfg = self.config_store.get_config()
            current_data = cfg.model_dump(mode="python")
            current_data.pop("lm_studio", None)
            for category, info in activated.items():
                if category in current_data.get("models", {}):
                    old_id = current_data["models"][category].get("model_id", "")
                    current_data["models"][category]["model_id"] = info["model_id"]
                    current_data["models"][category]["upstream_ref"] = info["upstream_ref"]
                    # Apply defaults from priority file if available
                    prio_cfg = priorities.get(category, {})
                    for key, value in prio_cfg.get("defaults", {}).items():
                        current_data["models"][category][key] = value
                    changes[category] = {"old": old_id, "new": info["model_id"], "reason": "loaded_model_classified"}
                    logger.info(
                        "model_auto_config activated loaded model=%s for category=%s (was %s)",
                        info["model_id"],
                        category,
                        old_id,
                    )
            yaml_text = yaml.safe_dump(current_data, sort_keys=False, allow_unicode=False)
            await self.config_store.update_from_yaml(yaml_text)
            logger.info("model_auto_config config updated after activating classified models")

        # --- Fallback: share model between small and large if one is missing ---
        cfg = self.config_store.get_config()
        current_data = cfg.model_dump(mode="python")
        current_data.pop("lm_studio", None)
        models_cfg = current_data.get("models", {})
        shared_fallback = False

        for src, dst in [("small", "large"), ("large", "small")]:
            src_model = models_cfg.get(src, {}).get("model_id", "")
            dst_model = models_cfg.get(dst, {}).get("model_id", "")
            if src_model and not dst_model and dst in models_cfg:
                src_upstream = models_cfg[src].get("upstream_ref", "local")
                # Verify the source model is actually available upstream
                available_model_ids = [
                    ModelAvailabilityMonitor._extract_model_id(item)
                    for ms in upstream_models.values()
                    for item in ms
                ]
                if src_model in available_model_ids:
                    models_cfg[dst]["model_id"] = src_model
                    models_cfg[dst]["upstream_ref"] = src_upstream
                    changes[dst] = {"old": dst_model, "new": src_model, "reason": "shared_from_" + src}
                    logger.info(
                        "model_auto_config shared model=%s from %s -> %s (no dedicated model available)",
                        src_model, src, dst,
                    )
                    shared_fallback = True

        if shared_fallback:
            yaml_text = yaml.safe_dump(current_data, sort_keys=False, allow_unicode=False)
            await self.config_store.update_from_yaml(yaml_text)
            logger.info("model_auto_config config updated after sharing models between categories")

        self._last_result = {
            "skipped": False,
            "changes": changes,
            "upstream_models": upstream_models,
            "classifications": classifications,
            "activated": activated,
        }
        return self._last_result

    def get_last_result(self) -> dict[str, Any]:
        return dict(self._last_result)


class AnalyticsStore:
    def __init__(self, config_store: ConfigStore):
        self.config_store = config_store
        self._lock = threading.Lock()
        self._initialized_path: Optional[Path] = None

    def _db_path(self) -> Path:
        cfg = self.config_store.get_config()
        path_value = (cfg.routing.analytics_sqlite_path or "").strip() or "logs/router_analytics.sqlite"
        return (PROJECT_ROOT / path_value).resolve()

    def _enabled(self) -> bool:
        return self.config_store.get_config().routing.analytics_enabled

    @staticmethod
    def _has_routing_runs_table(conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'routing_runs' LIMIT 1"
        ).fetchone()
        return row is not None

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS routing_runs (
                request_id TEXT PRIMARY KEY,
                session_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                route_logged_at TEXT,
                output_logged_at TEXT,
                source TEXT,
                requested_model TEXT,
                initial_alias TEXT,
                selected_alias TEXT,
                selected_model TEXT,
                reason TEXT,
                effective_reason TEXT,
                fallback_used INTEGER,
                stream INTEGER,
                candidate_aliases_json TEXT,
                required_capabilities_json TEXT,
                context_signature TEXT,
                complexity TEXT,
                full_input_tokens INTEGER,
                full_estimated_total_tokens INTEGER,
                routing_input_tokens INTEGER,
                routing_estimated_total_tokens INTEGER,
                max_tokens INTEGER,
                routing_max_tokens_budget INTEGER,
                needs_vision INTEGER,
                needs_tooluse INTEGER,
                is_coding INTEGER,
                has_wrapper_noise INTEGER,
                tool_loop_context INTEGER,
                repetition_key TEXT,
                prompt_text_hash TEXT,
                user_prompt_text_hash TEXT,
                latest_user_text_hash TEXT,
                routing_prompt_text_hash TEXT,
                routing_user_text_hash TEXT,
                routing_latest_user_text TEXT,
                thinking_requested INTEGER,
                thinking_applied INTEGER,
                expected_route_class TEXT,
                routing_efficiency_label TEXT,
                routing_efficiency_score INTEGER,
                output_text_chars INTEGER,
                output_text_excerpt TEXT,
                output_tokens INTEGER,
                input_tokens INTEGER,
                tool_calls INTEGER,
                stop_reason TEXT,
                latency_ms INTEGER
            )
            """
        )
        existing_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(routing_runs)").fetchall()}
        if "session_id" not in existing_columns:
            conn.execute("ALTER TABLE routing_runs ADD COLUMN session_id TEXT")
        conn.commit()

    def _connect(self) -> sqlite3.Connection:
        db_path = self._db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        if self._initialized_path != db_path or not self._has_routing_runs_table(conn):
            self._ensure_schema(conn)
            self._initialized_path = db_path
        return conn

    def write_route(self, payload: dict[str, Any]) -> None:
        if not self._enabled():
            return
        request_id = str(payload.get("request_id") or "").strip()
        if not request_id:
            return
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO routing_runs (
                        request_id, session_id, created_at, updated_at, route_logged_at, source, requested_model,
                        initial_alias, selected_alias, selected_model, reason, effective_reason,
                        fallback_used, stream, candidate_aliases_json, required_capabilities_json,
                        context_signature, complexity, full_input_tokens, full_estimated_total_tokens,
                        routing_input_tokens, routing_estimated_total_tokens, max_tokens,
                        routing_max_tokens_budget, needs_vision, needs_tooluse, is_coding,
                        has_wrapper_noise, tool_loop_context, repetition_key, prompt_text_hash,
                        user_prompt_text_hash, latest_user_text_hash, routing_prompt_text_hash,
                        routing_user_text_hash, routing_latest_user_text, thinking_requested,
                        thinking_applied, expected_route_class, routing_efficiency_label,
                        routing_efficiency_score, latency_ms
                    ) VALUES (
                        :request_id, :session_id, :created_at, :updated_at, :route_logged_at, :source, :requested_model,
                        :initial_alias, :selected_alias, :selected_model, :reason, :effective_reason,
                        :fallback_used, :stream, :candidate_aliases_json, :required_capabilities_json,
                        :context_signature, :complexity, :full_input_tokens, :full_estimated_total_tokens,
                        :routing_input_tokens, :routing_estimated_total_tokens, :max_tokens,
                        :routing_max_tokens_budget, :needs_vision, :needs_tooluse, :is_coding,
                        :has_wrapper_noise, :tool_loop_context, :repetition_key, :prompt_text_hash,
                        :user_prompt_text_hash, :latest_user_text_hash, :routing_prompt_text_hash,
                        :routing_user_text_hash, :routing_latest_user_text, :thinking_requested,
                        :thinking_applied, :expected_route_class, :routing_efficiency_label,
                        :routing_efficiency_score, :latency_ms
                    )
                    ON CONFLICT(request_id) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        route_logged_at=excluded.route_logged_at,
                        session_id=excluded.session_id,
                        source=excluded.source,
                        requested_model=excluded.requested_model,
                        initial_alias=excluded.initial_alias,
                        selected_alias=excluded.selected_alias,
                        selected_model=excluded.selected_model,
                        reason=excluded.reason,
                        effective_reason=excluded.effective_reason,
                        fallback_used=excluded.fallback_used,
                        stream=excluded.stream,
                        candidate_aliases_json=excluded.candidate_aliases_json,
                        required_capabilities_json=excluded.required_capabilities_json,
                        context_signature=excluded.context_signature,
                        complexity=excluded.complexity,
                        full_input_tokens=excluded.full_input_tokens,
                        full_estimated_total_tokens=excluded.full_estimated_total_tokens,
                        routing_input_tokens=excluded.routing_input_tokens,
                        routing_estimated_total_tokens=excluded.routing_estimated_total_tokens,
                        max_tokens=excluded.max_tokens,
                        routing_max_tokens_budget=excluded.routing_max_tokens_budget,
                        needs_vision=excluded.needs_vision,
                        needs_tooluse=excluded.needs_tooluse,
                        is_coding=excluded.is_coding,
                        has_wrapper_noise=excluded.has_wrapper_noise,
                        tool_loop_context=excluded.tool_loop_context,
                        repetition_key=excluded.repetition_key,
                        prompt_text_hash=excluded.prompt_text_hash,
                        user_prompt_text_hash=excluded.user_prompt_text_hash,
                        latest_user_text_hash=excluded.latest_user_text_hash,
                        routing_prompt_text_hash=excluded.routing_prompt_text_hash,
                        routing_user_text_hash=excluded.routing_user_text_hash,
                        routing_latest_user_text=excluded.routing_latest_user_text,
                        thinking_requested=excluded.thinking_requested,
                        thinking_applied=excluded.thinking_applied,
                        expected_route_class=excluded.expected_route_class,
                        routing_efficiency_label=excluded.routing_efficiency_label,
                        routing_efficiency_score=excluded.routing_efficiency_score,
                        latency_ms=COALESCE(excluded.latency_ms, routing_runs.latency_ms)
                    """,
                    {
                        "request_id": request_id,
                        "session_id": payload.get("session_id"),
                        "created_at": now,
                        "updated_at": now,
                        "route_logged_at": now,
                        "source": payload.get("source"),
                        "requested_model": payload.get("requested_model"),
                        "initial_alias": payload.get("initial_alias"),
                        "selected_alias": payload.get("selected_alias"),
                        "selected_model": payload.get("selected_model"),
                        "reason": payload.get("reason"),
                        "effective_reason": payload.get("effective_reason"),
                        "fallback_used": int(bool(payload.get("fallback_used"))),
                        "stream": int(bool(payload.get("stream"))),
                        "candidate_aliases_json": json.dumps(payload.get("candidate_aliases") or []),
                        "required_capabilities_json": json.dumps(payload.get("required_capabilities") or []),
                        "context_signature": payload.get("context_signature"),
                        "complexity": payload.get("complexity"),
                        "full_input_tokens": payload.get("full_input_tokens"),
                        "full_estimated_total_tokens": payload.get("full_estimated_total_tokens"),
                        "routing_input_tokens": payload.get("routing_input_tokens"),
                        "routing_estimated_total_tokens": payload.get("routing_estimated_total_tokens"),
                        "max_tokens": payload.get("max_tokens"),
                        "routing_max_tokens_budget": payload.get("routing_max_tokens_budget"),
                        "needs_vision": int(bool(payload.get("needs_vision"))),
                        "needs_tooluse": int(bool(payload.get("needs_tooluse"))),
                        "is_coding": int(bool(payload.get("is_coding"))),
                        "has_wrapper_noise": int(bool(payload.get("has_wrapper_noise"))),
                        "tool_loop_context": int(bool(payload.get("tool_loop_context"))),
                        "repetition_key": payload.get("repetition_key"),
                        "prompt_text_hash": payload.get("prompt_text_hash"),
                        "user_prompt_text_hash": payload.get("user_prompt_text_hash"),
                        "latest_user_text_hash": payload.get("latest_user_text_hash"),
                        "routing_prompt_text_hash": payload.get("routing_prompt_text_hash"),
                        "routing_user_text_hash": payload.get("routing_user_text_hash"),
                        "routing_latest_user_text": payload.get("routing_latest_user_text"),
                        "thinking_requested": int(bool(payload.get("thinking_requested"))),
                        "thinking_applied": int(bool(payload.get("thinking_applied"))),
                        "expected_route_class": payload.get("expected_route_class"),
                        "routing_efficiency_label": payload.get("routing_efficiency_label"),
                        "routing_efficiency_score": payload.get("routing_efficiency_score"),
                        "latency_ms": payload.get("latency_ms"),
                    },
                )
                conn.commit()
            finally:
                conn.close()

    def write_output(self, payload: dict[str, Any]) -> None:
        if not self._enabled():
            return
        request_id = str(payload.get("request_id") or "").strip()
        if not request_id:
            return
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO routing_runs (
                        request_id, session_id, created_at, updated_at, output_logged_at, source,
                        selected_alias, selected_model, reason, fallback_used, stream,
                        output_text_chars, output_text_excerpt, output_tokens, input_tokens,
                        tool_calls, stop_reason, routing_efficiency_label,
                        routing_efficiency_score, latency_ms
                    ) VALUES (
                        :request_id, :session_id, :created_at, :updated_at, :output_logged_at, :source,
                        :selected_alias, :selected_model, :reason, :fallback_used, :stream,
                        :output_text_chars, :output_text_excerpt, :output_tokens, :input_tokens,
                        :tool_calls, :stop_reason, :routing_efficiency_label,
                        :routing_efficiency_score, :latency_ms
                    )
                    ON CONFLICT(request_id) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        output_logged_at=excluded.output_logged_at,
                        session_id=COALESCE(excluded.session_id, routing_runs.session_id),
                        source=COALESCE(excluded.source, routing_runs.source),
                        selected_alias=COALESCE(excluded.selected_alias, routing_runs.selected_alias),
                        selected_model=COALESCE(excluded.selected_model, routing_runs.selected_model),
                        reason=COALESCE(excluded.reason, routing_runs.reason),
                        fallback_used=COALESCE(excluded.fallback_used, routing_runs.fallback_used),
                        stream=COALESCE(excluded.stream, routing_runs.stream),
                        output_text_chars=excluded.output_text_chars,
                        output_text_excerpt=excluded.output_text_excerpt,
                        output_tokens=excluded.output_tokens,
                        input_tokens=excluded.input_tokens,
                        tool_calls=excluded.tool_calls,
                        stop_reason=excluded.stop_reason,
                        routing_efficiency_label=COALESCE(excluded.routing_efficiency_label, routing_runs.routing_efficiency_label),
                        routing_efficiency_score=COALESCE(excluded.routing_efficiency_score, routing_runs.routing_efficiency_score),
                        latency_ms=COALESCE(excluded.latency_ms, routing_runs.latency_ms)
                    """,
                    {
                        "request_id": request_id,
                        "session_id": payload.get("session_id"),
                        "created_at": now,
                        "updated_at": now,
                        "output_logged_at": now,
                        "source": payload.get("source"),
                        "selected_alias": payload.get("selected_alias"),
                        "selected_model": payload.get("selected_model"),
                        "reason": payload.get("reason"),
                        "fallback_used": int(bool(payload.get("fallback_used"))),
                        "stream": int(bool(payload.get("stream"))),
                        "output_text_chars": payload.get("output_text_chars"),
                        "output_text_excerpt": payload.get("output_text_excerpt"),
                        "output_tokens": payload.get("output_tokens"),
                        "input_tokens": payload.get("input_tokens"),
                        "tool_calls": payload.get("tool_calls"),
                        "stop_reason": payload.get("stop_reason"),
                        "routing_efficiency_label": payload.get("routing_efficiency_label"),
                        "routing_efficiency_score": payload.get("routing_efficiency_score"),
                        "latency_ms": payload.get("latency_ms"),
                    },
                )
                conn.commit()
            finally:
                conn.close()

    def recent_routes(self, *, source: str, limit: int) -> list[dict[str, Any]]:
        if not self._enabled():
            return []
        safe_limit = max(1, int(limit))
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    SELECT request_id, source, selected_alias, reason, repetition_key,
                           routing_latest_user_text, needs_vision, needs_tooluse, is_coding
                    FROM routing_runs
                    WHERE route_logged_at IS NOT NULL
                      AND source = ?
                    ORDER BY COALESCE(route_logged_at, created_at) DESC, rowid DESC
                    LIMIT ?
                    """,
                    (source, safe_limit),
                )
                columns = [col[0] for col in cursor.description or []]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            except sqlite3.Error as exc:
                logger.warning("analytics_recent_routes_failed source=%s limit=%s error=%s", source, safe_limit, exc)
                return []
            finally:
                conn.close()


class RequestMemoryStore:
    def __init__(self, max_sessions: int = 128, max_entries_per_session: int = 32):
        self._max_sessions = max(8, int(max_sessions))
        self._max_entries_per_session = max(4, int(max_entries_per_session))
        self._entries_by_session: OrderedDict[str, deque[dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def remember(self, session_id: str, entry: dict[str, Any]) -> None:
        if not session_id:
            return
        with self._lock:
            bucket = self._entries_by_session.get(session_id)
            if bucket is None:
                if len(self._entries_by_session) >= self._max_sessions:
                    self._entries_by_session.popitem(last=False)
                bucket = deque(maxlen=self._max_entries_per_session)
                self._entries_by_session[session_id] = bucket
            else:
                self._entries_by_session.move_to_end(session_id)
            bucket.appendleft(dict(entry))

    def recent_entries(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        if not session_id:
            return []
        safe_limit = max(1, int(limit))
        with self._lock:
            bucket = self._entries_by_session.get(session_id)
            if bucket is None:
                return []
            self._entries_by_session.move_to_end(session_id)
            return [dict(item) for item in list(bucket)[:safe_limit]]


class RouterService:
    def __init__(self, config_store: ConfigStore, lm_client: Optional[LMStudioClient] = None):
        self.config_store = config_store
        self.lm_client = lm_client or LMStudioClient()
        self.analytics_store = AnalyticsStore(config_store)
        cfg = config_store.get_config()
        self.request_memory = RequestMemoryStore(
            max_sessions=cfg.routing.session_memory.max_sessions,
            max_entries_per_session=cfg.routing.session_memory.max_entries_per_session,
        )
        self._judge_inflight_lock = asyncio.Lock()
        self._judge_inflight: dict[str, asyncio.Future[tuple[Optional[str], Optional[bool]]]] = {}

    @staticmethod
    def _is_deep_reasoning_request(req: UnifiedRequest) -> bool:
        text = (
            req.routing_latest_user_prompt_text
            or req.routing_user_prompt_text
            or req.routing_prompt_text
            or req.latest_user_prompt_text
            or req.user_prompt_text
            or req.prompt_text
            or ""
        ).strip()
        if not text:
            return False
        return bool(DEEP_REASONING_RE.search(text) or WEBSEARCH_RE.search(text))

    @staticmethod
    def _is_deep_enabled(cfg: RouterConfig) -> bool:
        if not cfg.is_alias_enabled("deep"):
            return False
        if not _env_flag("DEEP_ENABLED", default=False):
            return False
        try:
            deep_upstream = cfg.upstream_for_alias("deep")
        except Exception:  # noqa: BLE001
            return False
        if deep_upstream.provider == "openai" and not deep_upstream.resolve_api_key():
            logger.warning("deep_route_disabled reason=missing_api_key env=%s", deep_upstream.api_key_env)
            return False
        return True

    @staticmethod
    def _is_agentic_request(req: UnifiedRequest) -> bool:
        return bool(req.needs_tooluse or req.tool_loop_context)

    @staticmethod
    def _judge_model_alias(cfg: RouterConfig) -> Optional[str]:
        # Wir bevorzugen small für den Judge, falls aktiviert und verfügbar.
        # Aber falls wir für Agenten nur large/deep erlauben wollen, sollte der Judge vielleicht auch darauf basieren?
        # Die Anforderung sagt "Wenn agenten genutzt werden soll beim llm nur large oder deep genutzt werden."
        # Das bezieht sich primär auf das finale Modell. Der Judge selbst kann klein bleiben,
        # solange er die Regel kennt (was er durch den System Prompt tut).
        for alias in ("small", "large", "deep", "backup"):
            if cfg.is_alias_enabled(alias):
                if alias == "deep" and not RouterService._is_deep_enabled(cfg):
                    continue
                return alias
        return None

    @classmethod
    def _judge_model_id(cls, cfg: RouterConfig) -> Optional[str]:
        alias = cls._judge_model_alias(cfg)
        if not alias:
            return None
        return cfg.models[alias].model_id

    @staticmethod
    def _fallback_alias(cfg: RouterConfig, req: UnifiedRequest) -> Optional[str]:
        is_agentic = RouterService._is_agentic_request(req)
        preferred = ("large", "deep") if is_agentic else ("large", "deep", "small")
        for alias in preferred:
            if cfg.is_alias_enabled(alias):
                if alias == "deep" and not RouterService._is_deep_enabled(cfg):
                    continue
                return alias
        if cfg.is_alias_enabled("backup"):
            return "backup"
        return None

    @staticmethod
    def _is_lightweight_anthropic_request(cfg: RouterConfig, req: UnifiedRequest, is_coding: bool) -> bool:
        if req.source_api != "anthropic_messages":
            return False
        if req.needs_vision or is_coding:
            return False
        if req.tool_loop_context:
            return False
        latest = (req.routing_latest_user_prompt_text or req.routing_user_prompt_text or "").strip()
        if not latest:
            return False
        if len(latest) > 160:
            return False
        if req.routing_input_tokens > 600:
            return False
        requested = req.max_tokens or 0
        suspect_threshold = cfg.routing.heuristics.suspect_default_max_tokens_threshold
        if requested and requested < suspect_threshold:
            return False
        return bool(req.has_wrapper_noise or req.needs_tooluse or LIGHTWEIGHT_TASK_RE.match(latest))

    @staticmethod
    def _is_lightweight_tool_scaffold_request(req: UnifiedRequest, is_coding: bool) -> bool:
        latest = (req.routing_latest_user_prompt_text or req.routing_user_prompt_text or "").strip()
        if not latest:
            return False
        if req.needs_vision or req.tool_loop_context or is_coding:
            return False
        if req.routing_input_tokens > 800:
            return False
        if not req.needs_tooluse:
            return False
        return bool(LIGHTWEIGHT_TASK_RE.match(latest))

    @staticmethod
    def _is_client_meta_request(req: UnifiedRequest) -> bool:
        text = (
            req.routing_latest_user_prompt_text
            or req.routing_user_prompt_text
            or req.routing_prompt_text
            or req.latest_user_prompt_text
            or req.user_prompt_text
            or req.prompt_text
            or ""
        ).strip()
        if not text:
            return False
        return bool(CLIENT_META_TASK_RE.search(text))

    def _prefer_small_shortcut(self, cfg: RouterConfig, req: UnifiedRequest, is_coding: bool) -> Optional[str]:
        latest = (
            req.routing_latest_user_prompt_text
            or req.routing_user_prompt_text
            or req.latest_user_prompt_text
            or req.user_prompt_text
            or ""
        ).strip()
        if not latest or req.needs_vision or req.tool_loop_context:
            return None
        if self._is_deep_reasoning_request(req):
            return None
        if self._is_client_meta_request(req):
            return "client_meta_request_prefer_small"
        if self._is_lightweight_tool_scaffold_request(req, is_coding):
            return "lightweight_tool_scaffold_prefer_small"
        if not is_coding and LIGHTWEIGHT_TASK_RE.match(latest):
            return "lightweight_greeting_prefer_small"
        return None

    @staticmethod
    def _apply_routing_budget(cfg: RouterConfig, req: UnifiedRequest, is_coding: bool) -> None:
        if req.routing_max_tokens_budget is not None:
            return
        if RouterService._is_lightweight_anthropic_request(cfg, req, is_coding):
            req.routing_max_tokens_budget = cfg.routing.heuristics.lightweight_max_tokens_cap

    @staticmethod
    def _expected_route_class(req: UnifiedRequest, is_coding: bool) -> str:
        latest = (req.routing_latest_user_prompt_text or req.routing_user_prompt_text or req.routing_prompt_text or "").strip()
        if RouterService._is_deep_reasoning_request(req):
            return "deep"
        if req.needs_vision or is_coding or req.tool_loop_context:
            return "large"
        if req.needs_tooluse and not latest:
            return "large"
        if req.routing_estimated_total_tokens >= 12000:
            return "large"
        if req.needs_tooluse and latest and len(latest) > 240:
            return "large"
        return "small"

    @staticmethod
    def _routing_efficiency(
        expected_route_class: str,
        final_alias: str,
        *,
        initial_alias: str,
        used_fallback: bool,
        stop_reason: Optional[str] = None,
    ) -> tuple[str, int]:
        return _routing_efficiency(
            expected_route_class,
            final_alias,
            initial_alias=initial_alias,
            used_fallback=used_fallback,
            stop_reason=stop_reason,
        )

    @staticmethod
    def _complexity_bucket(req: UnifiedRequest, is_coding: bool) -> str:
        total_tokens = req.routing_estimated_total_tokens
        if req.needs_vision or req.needs_tooluse:
            return "high"
        if is_coding and total_tokens >= 12000:
            return "high"
        if total_tokens >= 16000:
            return "high"
        if total_tokens >= 5000 or is_coding:
            return "medium"
        return "low"

    @staticmethod
    def _context_signature(req: UnifiedRequest, is_coding: bool) -> str:
        parts = [req.source_api, f"caps={','.join(sorted(req.required_capabilities))}"]
        if req.needs_vision:
            parts.append("vision")
        if req.needs_tooluse:
            parts.append("tooluse")
        if is_coding:
            parts.append("coding")
        if req.stream:
            parts.append("stream")
        return "|".join(parts)

    @staticmethod
    def _repetition_key(req: UnifiedRequest) -> str:
        base = (
            req.routing_latest_user_prompt_text
            or req.routing_user_prompt_text
            or req.routing_prompt_text
            or req.latest_user_prompt_text
            or req.user_prompt_text
            or req.prompt_text
            or ""
        ).strip().lower()
        normalized = re.sub(r"\s+", " ", base)[:2000]
        material = f"{req.source_api}|{req.required_base_capability}|{normalized}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _small_coding_context_limit_tokens() -> int:
        return max(2048, int(os.getenv("ROUTER_SMALL_CODING_MAX_TOTAL_TOKENS", "32000")))

    @staticmethod
    def _small_coding_task_limit_tokens() -> int:
        return max(1024, int(os.getenv("ROUTER_SMALL_CODING_TASK_MAX_TOTAL_TOKENS", "8000")))

    @staticmethod
    def _normalized_repetition_text(req: UnifiedRequest) -> str:
        base = (
            req.routing_latest_user_prompt_text
            or req.routing_user_prompt_text
            or req.routing_prompt_text
            or req.latest_user_prompt_text
            or req.user_prompt_text
            or req.prompt_text
            or ""
        ).strip().lower()
        return re.sub(r"\s+", " ", base)[:2000]

    @staticmethod
    def _route_tier_index(alias: str) -> int:
        route_tiers = {"small": 0, "large": 1, "deep": 2}
        return route_tiers.get(alias, -1)

    @staticmethod
    def _effective_session_id(cfg: RouterConfig, req: UnifiedRequest) -> str:
        session_cfg = cfg.routing.session_memory
        if not session_cfg.enabled:
            return ""
        session_id = (req.session_id or "").strip()
        if session_id:
            return session_id
        if session_cfg.require_session_id:
            return ""
        return "default"

    @classmethod
    def _similarity_score(
        cls,
        current_text: str,
        past_text: str,
        *,
        current_key: str,
        past_key: str,
    ) -> float:
        if current_key and past_key and current_key == past_key:
            return 1.0
        if not current_text or not past_text:
            return 0.0
        return SequenceMatcher(None, current_text, past_text).ratio()

    def _recent_request_memory_for_judge(
        self,
        cfg: RouterConfig,
        req: UnifiedRequest,
        is_coding: bool,
    ) -> dict[str, Any]:
        settings = cfg.routing.repetition_escalation
        current_text = self._normalized_repetition_text(req)
        session_id = self._effective_session_id(cfg, req)
        recent_entries = self.request_memory.recent_entries(session_id, limit=min(settings.history_limit, 3))
        summarized_entries: list[dict[str, Any]] = []

        for entry in recent_entries:
            similarity = self._similarity_score(
                current_text,
                str(entry.get("normalized_text") or ""),
                current_key=self._repetition_key(req),
                past_key=str(entry.get("repetition_key") or ""),
            )
            summarized_entries.append(
                {
                    "request_id": entry.get("request_id"),
                    "source_api": entry.get("source_api"),
                    "selected_alias": entry.get("selected_alias"),
                    "reason": entry.get("reason"),
                    "is_coding": bool(entry.get("is_coding")),
                    "needs_tooluse": bool(entry.get("needs_tooluse")),
                    "needs_vision": bool(entry.get("needs_vision")),
                    "similarity_to_current": round(similarity, 3),
                    "prompt_excerpt": str(entry.get("prompt_excerpt") or "")[:240],
                }
            )

        previous_request = summarized_entries[0] if summarized_entries else None
        previous_is_compatible = bool(
            previous_request
            and previous_request["is_coding"] == is_coding
            and previous_request["needs_tooluse"] == req.needs_tooluse
            and previous_request["needs_vision"] == req.needs_vision
        )
        previous_similarity = float(previous_request["similarity_to_current"]) if previous_request else 0.0

        return {
            "previous_request": previous_request,
            "recent_requests": summarized_entries,
            "previous_request_similarity": round(previous_similarity, 3),
            "previous_request_compatible": previous_is_compatible,
            "loop_risk": bool(
                previous_request
                and previous_is_compatible
                and previous_similarity >= settings.similarity_threshold
            ),
            "similarity_threshold": settings.similarity_threshold,
            "session_id": session_id or None,
        }

    def _judge_request_key(
        self,
        cfg: RouterConfig,
        req: UnifiedRequest,
        candidates: list[str],
        *,
        judge_model: str,
        is_deep_reasoning: bool,
        is_websearch: bool,
        is_commit_task: bool,
        is_file_search: bool,
        recent_request_memory: dict[str, Any],
    ) -> str:
        prompt_text = (
            req.routing_latest_user_prompt_text
            or req.routing_user_prompt_text
            or req.routing_prompt_text
            or req.latest_user_prompt_text
            or req.user_prompt_text
            or req.prompt_text
            or ""
        )
        memory_fingerprint = [
            (
                entry.get("request_id"),
                entry.get("selected_alias"),
                entry.get("reason"),
                entry.get("similarity_to_current"),
            )
            for entry in recent_request_memory.get("recent_requests") or []
            if isinstance(entry, dict)
        ]
        key_payload = {
            "judge_model": judge_model,
            "session_id": self._effective_session_id(cfg, req),
            "source_api": req.source_api,
            "requested_model": req.requested_model,
            "stream": req.stream,
            "candidates": candidates,
            "repetition_key": self._repetition_key(req),
            "prompt_hash": _hash_text(prompt_text),
            "routing_input_tokens": req.routing_input_tokens,
            "routing_estimated_total_tokens": req.routing_estimated_total_tokens,
            "full_input_tokens": req.full_input_tokens,
            "full_estimated_total_tokens": req.full_estimated_total_tokens,
            "max_tokens": req.max_tokens,
            "routing_max_tokens_budget": req.routing_max_tokens_budget,
            "needs_vision": req.needs_vision,
            "needs_tooluse": req.needs_tooluse,
            "has_wrapper_noise": req.has_wrapper_noise,
            "tool_loop_context": req.tool_loop_context,
            "is_deep_reasoning": is_deep_reasoning,
            "is_websearch": is_websearch,
            "is_commit_task": is_commit_task,
            "is_file_search": is_file_search,
            "recent_request_memory": memory_fingerprint,
        }
        return _hash_text(json.dumps(key_payload, sort_keys=True, ensure_ascii=False))

    def _find_repetition_escalation_alias(
        self,
        cfg: RouterConfig,
        req: UnifiedRequest,
        candidates: list[str],
        selected_alias: str,
        is_coding: bool,
    ) -> tuple[Optional[str], Optional[str], int, float]:
        settings = cfg.routing.repetition_escalation
        if not settings.enabled or len(candidates) <= 1:
            return None, None, 0, 0.0

        current_text = self._normalized_repetition_text(req)
        if not current_text:
            return None, None, 0, 0.0

        session_id = self._effective_session_id(cfg, req)
        if not session_id:
            return None, None, 0, 0.0

        current_key = self._repetition_key(req)
        recent_rows = self.request_memory.recent_entries(session_id, limit=settings.history_limit)
        if not recent_rows:
            return None, None, 0, 0.0

        streak = 0
        best_similarity = 0.0
        most_recent_similar_alias: Optional[str] = None
        for row in recent_rows:
            if str(row.get("source_api") or "") != req.source_api:
                break
            if bool(row.get("needs_vision")) != req.needs_vision:
                break
            if bool(row.get("needs_tooluse")) != req.needs_tooluse:
                break
            if bool(row.get("is_coding")) != is_coding:
                break

            past_text = str(row.get("normalized_text") or "")
            similarity = self._similarity_score(
                current_text,
                past_text,
                current_key=current_key,
                past_key=str(row.get("repetition_key") or ""),
            )
            if similarity < settings.similarity_threshold:
                break

            streak += 1
            best_similarity = max(best_similarity, similarity)
            if most_recent_similar_alias is None:
                most_recent_similar_alias = str(row.get("selected_alias") or "").strip()

        if streak < settings.min_streak or not most_recent_similar_alias:
            return None, None, streak, best_similarity

        start_tier = max(
            self._route_tier_index(selected_alias),
            self._route_tier_index(most_recent_similar_alias),
        )
        if start_tier < 0:
            return None, None, streak, best_similarity

        escalation_source = selected_alias
        if self._route_tier_index(most_recent_similar_alias) > self._route_tier_index(selected_alias):
            escalation_source = most_recent_similar_alias

        for alias in ("small", "large", "deep"):
            tier = self._route_tier_index(alias)
            if tier <= start_tier:
                continue
            if alias in candidates:
                return alias, escalation_source, streak, best_similarity
        return None, None, streak, best_similarity

    def _build_decision(
        self,
        cfg: RouterConfig,
        req: UnifiedRequest,
        *,
        selected_alias: str,
        reason: str,
        candidates: list[str],
        thinking_requested: bool,
        judge_model_id: Optional[str],
        is_coding: bool,
        task_type: str = "simple",
    ) -> RouteDecision:
        escalated_alias, escalation_source, repetition_streak, similarity = self._find_repetition_escalation_alias(
            cfg,
            req,
            candidates,
            selected_alias,
            is_coding,
        )
        if escalated_alias and escalated_alias != selected_alias:
            decision_source = escalation_source or selected_alias
            logger.info(
                "route_eval_repetition_escalation from=%s to=%s streak=%s similarity=%.3f original_reason=%s",
                selected_alias,
                escalated_alias,
                repetition_streak,
                similarity,
                reason,
            )
            selected_alias = escalated_alias
            reason = f"repetition_escalation_{decision_source}_to_{selected_alias}"
            thinking_requested = self._heuristic_thinking_requested(cfg, req, selected_alias, task_type)

        if req.needs_tooluse or self._is_no_thinking_task(req):
            thinking_requested = False
        if thinking_requested and not cfg.models[selected_alias].supports_thinking:
            thinking_requested = False

        decision = self._make_route_decision(
            req=req,
            selected_alias=selected_alias,
            reason=reason,
            candidates=candidates,
            thinking_requested=thinking_requested,
            judge_model_id=judge_model_id,
            is_coding=is_coding,
            task_type=task_type,
        )
        session_id = self._effective_session_id(cfg, req)
        if session_id:
            self.request_memory.remember(
                session_id,
                {
                    "request_id": decision.request_id,
                    "session_id": session_id,
                    "source_api": decision.source_api,
                    "selected_alias": decision.selected_alias,
                    "reason": decision.reason,
                    "repetition_key": decision.repetition_key,
                    "normalized_text": self._normalized_repetition_text(req),
                    "prompt_excerpt": decision.routing_latest_user_prompt_text,
                    "is_coding": decision.is_coding_request,
                    "needs_tooluse": decision.needs_tooluse,
                    "needs_vision": decision.needs_vision,
                    "thinking_requested": decision.thinking_requested,
                    "candidates": candidates,
                    "judge_model_id": judge_model_id,
                }
            )
        return decision

    def _make_route_decision(
        self,
        req: UnifiedRequest,
        selected_alias: str,
        reason: str,
        candidates: list[str],
        thinking_requested: bool,
        judge_model_id: Optional[str],
        is_coding: bool,
        task_type: str = "simple",
    ) -> RouteDecision:
        prompt_log_max_chars = max(200, int(os.getenv("ROUTER_PROMPT_LOG_MAX_CHARS", "4000")))
        expected_route_class = self._expected_route_class(req, is_coding)
        routing_efficiency_label, routing_efficiency_score = self._routing_efficiency(
            expected_route_class,
            selected_alias,
            initial_alias=selected_alias,
            used_fallback=False,
        )
        return RouteDecision(
            selected_alias=selected_alias,
            reason=reason,
            candidate_aliases=candidates,
            request_id=_request_id_ctx.get(),
            session_id=req.session_id,
            thinking_requested=thinking_requested,
            is_commit_message_task=req.is_commit_message_task,
            judge_model_id=judge_model_id,
            is_coding_request=is_coding,
            task_type=task_type,
            source_api=req.source_api,
            requested_model=req.requested_model,
            stream=req.stream,
            required_capabilities=sorted(req.required_capabilities),
            estimated_input_tokens=req.estimated_input_tokens,
            estimated_total_tokens=req.estimated_total_tokens,
            full_input_tokens=req.full_input_tokens,
            full_estimated_total_tokens=req.full_estimated_total_tokens,
            routing_input_tokens=req.routing_input_tokens,
            routing_estimated_total_tokens=req.routing_estimated_total_tokens,
            max_tokens=req.max_tokens,
            routing_max_tokens_budget=req.routing_max_tokens_budget,
            needs_vision=req.needs_vision,
            needs_tooluse=req.needs_tooluse,
            has_wrapper_noise=req.has_wrapper_noise,
            tool_loop_context=req.tool_loop_context,
            complexity=self._complexity_bucket(req, is_coding),
            context_signature=self._context_signature(req, is_coding),
            repetition_key=self._repetition_key(req),
            prompt_text=(req.prompt_text or "")[:prompt_log_max_chars],
            user_prompt_text=(req.user_prompt_text or "")[:prompt_log_max_chars],
            latest_user_prompt_text=(req.latest_user_prompt_text or "")[:prompt_log_max_chars],
            routing_prompt_text=(req.routing_prompt_text or "")[:prompt_log_max_chars],
            routing_user_prompt_text=(req.routing_user_prompt_text or "")[:prompt_log_max_chars],
            routing_latest_user_prompt_text=(req.routing_latest_user_prompt_text or "")[:prompt_log_max_chars],
            expected_route_class=expected_route_class,
            routing_efficiency_label=routing_efficiency_label,
            routing_efficiency_score=routing_efficiency_score,
        )

    @staticmethod
    def _upstream_for_alias(cfg: RouterConfig, alias: str) -> LMStudioSettings:
        return cfg.upstream_for_alias(alias)

    @staticmethod
    def _normalize_openai_chat_token_param(
        settings: LMStudioSettings,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        # Newer OpenAI chat models reject `max_tokens` and require `max_completion_tokens`.
        if settings.provider != "openai" or path != "/v1/chat/completions":
            return payload
        if "max_completion_tokens" in payload:
            return payload
        if "max_tokens" not in payload:
            return payload
        normalized = dict(payload)
        normalized["max_completion_tokens"] = normalized.get("max_tokens")
        normalized.pop("max_tokens", None)
        return normalized

    @staticmethod
    def _apply_default_request_temperature(
        cfg: RouterConfig,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if "temperature" in payload:
            return payload
        default_temperature = cfg.routing.default_temperature
        if default_temperature is None:
            return payload
        normalized = dict(payload)
        normalized["temperature"] = default_temperature
        return normalized

    @staticmethod
    def _apply_alias_token_budget(
        alias: str,
        payload: dict[str, Any],
        decision: RouteDecision,
    ) -> dict[str, Any]:
        if alias != "small":
            return payload
        budget = decision.routing_max_tokens_budget
        if budget is None:
            return payload
        normalized = dict(payload)
        for key in ("max_tokens", "max_completion_tokens"):
            if key in normalized:
                try:
                    normalized[key] = min(int(normalized[key]), int(budget))
                except Exception:  # noqa: BLE001
                    normalized[key] = int(budget)
        return normalized

    @staticmethod
    def _normalize_thinking_param(
        settings: LMStudioSettings,
        path: str,
        payload: dict[str, Any],
        thinking_enabled: bool,
    ) -> dict[str, Any]:
        normalized = dict(payload)

        def _clear_lmstudio_thinking_flags() -> None:
            chat_kwargs = normalized.get("chat_template_kwargs")
            if isinstance(chat_kwargs, dict):
                chat_kwargs = dict(chat_kwargs)
                chat_kwargs.pop("enable_thinking", None)
                if chat_kwargs:
                    normalized["chat_template_kwargs"] = chat_kwargs
                else:
                    normalized.pop("chat_template_kwargs", None)

            extra_body = normalized.get("extra_body")
            if isinstance(extra_body, dict):
                extra_body = dict(extra_body)
                extra_body.pop("thinking", None)
                extra_body.pop("reasoning", None)
                if extra_body:
                    normalized["extra_body"] = extra_body
                else:
                    normalized.pop("extra_body", None)

            options = normalized.get("options")
            if isinstance(options, dict):
                options = dict(options)
                options.pop("thinking", None)
                if options:
                    normalized["options"] = options
                else:
                    normalized.pop("options", None)

        def _set_lmstudio_thinking_flags(value: bool) -> None:
            chat_kwargs = normalized.get("chat_template_kwargs")
            if not isinstance(chat_kwargs, dict):
                chat_kwargs = {}
            else:
                chat_kwargs = dict(chat_kwargs)
            chat_kwargs["enable_thinking"] = value
            normalized["chat_template_kwargs"] = chat_kwargs

            extra_body = normalized.get("extra_body")
            if not isinstance(extra_body, dict):
                extra_body = {}
            else:
                extra_body = dict(extra_body)
            extra_body["thinking"] = value
            extra_body["reasoning"] = value
            normalized["extra_body"] = extra_body

            options = normalized.get("options")
            if not isinstance(options, dict):
                options = {}
            else:
                options = dict(options)
            options["thinking"] = value
            normalized["options"] = options

            normalized["thinking"] = value

        if path != "/v1/chat/completions":
            if not thinking_enabled:
                normalized.pop("reasoning", None)
                normalized.pop("thinking", None)
                if settings.provider == "lm_studio":
                    _clear_lmstudio_thinking_flags()
            return normalized

        if not thinking_enabled:
            normalized.pop("reasoning", None)
            if settings.provider == "lm_studio":
                # LM Studio / some qwen templates may default to thinking unless explicitly disabled.
                _set_lmstudio_thinking_flags(False)
            else:
                normalized.pop("thinking", None)
            return normalized

        if settings.provider == "openai":
            reasoning = normalized.get("reasoning")
            if isinstance(reasoning, dict):
                effort = str(reasoning.get("effort") or "").strip()
                if not effort:
                    reasoning["effort"] = "medium"
            else:
                normalized["reasoning"] = {"effort": "medium"}
        elif settings.provider == "lm_studio":
            _set_lmstudio_thinking_flags(True)
        return normalized

    @staticmethod
    def _normalize_commit_message_payload(
        path: str,
        payload: dict[str, Any],
        decision: RouteDecision,
    ) -> dict[str, Any]:
        if not decision.is_commit_message_task or path != "/v1/chat/completions":
            return payload

        normalized = dict(payload)
        token_cap = 160
        if "max_completion_tokens" in normalized:
            try:
                if int(normalized["max_completion_tokens"]) > token_cap:
                    normalized["max_completion_tokens"] = token_cap
            except Exception:  # noqa: BLE001
                normalized["max_completion_tokens"] = token_cap
        elif "max_tokens" in normalized:
            try:
                if int(normalized["max_tokens"]) > token_cap:
                    normalized["max_tokens"] = token_cap
            except Exception:  # noqa: BLE001
                normalized["max_tokens"] = token_cap
        else:
            normalized["max_completion_tokens"] = token_cap

        hint = ""
        if COMMIT_MESSAGE_HINT_PATH.exists():
            loaded_hint = yaml.safe_load(COMMIT_MESSAGE_HINT_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded_hint, dict):
                hint = str(loaded_hint.get("commit_message_hint", "")).strip()
            elif loaded_hint is not None:
                hint = str(loaded_hint).strip()
        messages = normalized.get("messages")
        if not isinstance(messages, list):
            messages = []
            normalized["messages"] = messages

        no_thinking_instruction = "Do not think. Do not output any thinking process or internal reasoning. Just output the final result."
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            current = str(messages[0].get("content") or "")
            new_content = current
            if hint not in current:
                sep = "\n\n" if new_content else ""
                new_content = f"{new_content}{sep}{hint}"
            if no_thinking_instruction not in new_content:
                sep = "\n\n" if new_content else ""
                new_content = f"{new_content}{sep}{no_thinking_instruction}"
            messages[0]["content"] = new_content
        else:
            combined_hint = f"{hint}\n\n{no_thinking_instruction}".strip()
            messages.insert(0, {"role": "system", "content": combined_hint})
        return normalized

    def _eligible_aliases(self, cfg: RouterConfig, req: UnifiedRequest) -> list[str]:
        required = req.required_capabilities
        total_tokens = req.routing_estimated_total_tokens
        is_coding = self._is_coding_request(req)
        is_agentic = self._is_agentic_request(req)
        primary_aliases: list[str] = []
        backup_aliases: list[str] = []
        is_lightweight = (
            self._is_lightweight_anthropic_request(cfg, req, is_coding)
            or self._is_lightweight_tool_scaffold_request(req, is_coding)
        )
        for alias, profile in cfg.models.items():
            if not profile.enabled:
                continue
            if is_agentic and alias not in ("large", "deep") and not is_lightweight:
                continue
            if alias == "deep" and not self._is_deep_enabled(cfg):
                continue
            if profile.has_capabilities(required) and profile.context_window >= total_tokens:
                if alias == "backup":
                    backup_aliases.append(alias)
                else:
                    primary_aliases.append(alias)
        if primary_aliases:
            return primary_aliases
        if backup_aliases and not self._has_available_primary_alias(cfg):
            return backup_aliases
        return []

    @staticmethod
    def _has_available_primary_alias(cfg: RouterConfig) -> bool:
        for alias in ("small", "large", "deep"):
            if not cfg.is_alias_enabled(alias):
                continue
            if alias == "deep" and not RouterService._is_deep_enabled(cfg):
                continue
            return True
        return False

    def _find_alias_by_model_id(self, cfg: RouterConfig, model_id: Optional[str]) -> Optional[str]:
        if not model_id:
            return None
        for alias, profile in cfg.models.items():
            if profile.enabled and profile.model_id == model_id:
                return alias
        return None

    @staticmethod
    def _is_router_public_model_name(cfg: RouterConfig, requested_model: Optional[str]) -> bool:
        if not requested_model:
            return False
        return requested_model.strip() == cfg.router_identity.exposed_model_name.strip()

    @staticmethod
    def _is_coding_request(req: UnifiedRequest) -> bool:
        text = (
            req.routing_latest_user_prompt_text
            or req.routing_user_prompt_text
            or req.routing_prompt_text
            or req.latest_user_prompt_text
            or req.user_prompt_text
            or req.prompt_text
            or ""
        ).strip()
        if not text.strip():
            return False
        if CODING_SYNTAX_RE.search(text):
            return True
        return bool(CODING_TOPIC_RE.search(text))

    @staticmethod
    def _classify_task_type(req: UnifiedRequest, is_coding: bool) -> str:
        text = (
            req.routing_latest_user_prompt_text
            or req.routing_user_prompt_text
            or req.routing_prompt_text
            or req.latest_user_prompt_text
            or req.user_prompt_text
            or req.prompt_text
            or ""
        ).strip()
        
        if not text:
            return "simple"
            
        if DEBUG_TASK_RE.search(text):
            return "debug"
        if ARCHITECTURE_TASK_RE.search(text):
            return "architecture"
        if AGENT_TASK_RE.search(text):
            return "agent"
        if COMPLEX_CODE_TASK_RE.search(text):
            return "complex_code"
        if is_coding:
            return "code"
        if BOILERPLATE_TASK_RE.search(text):
            return "simple"  # Boilerplate/Formatting gilt als simple laut Anforderung
        
        return "simple"

    @staticmethod
    def _is_file_search_request(req: UnifiedRequest) -> bool:
        text = (
            req.routing_latest_user_prompt_text
            or req.routing_user_prompt_text
            or req.routing_prompt_text
            or req.latest_user_prompt_text
            or req.user_prompt_text
            or req.prompt_text
            or ""
        ).strip()
        if not text:
            return False
        return bool(FILE_SEARCH_RE.search(text))

    @staticmethod
    def _is_commit_message_task(req: UnifiedRequest) -> bool:
        text = (
            req.routing_latest_user_prompt_text
            or req.routing_user_prompt_text
            or req.routing_prompt_text
            or req.latest_user_prompt_text
            or req.user_prompt_text
            or req.prompt_text
            or ""
        ).strip()
        if not text:
            return False
        return bool(COMMIT_MESSAGE_TASK_RE.search(text))

    @staticmethod
    def _is_no_thinking_task(req: UnifiedRequest) -> bool:
        text = (
            req.routing_latest_user_prompt_text
            or req.routing_user_prompt_text
            or req.routing_prompt_text
            or req.latest_user_prompt_text
            or req.user_prompt_text
            or req.prompt_text
            or ""
        ).strip()
        if not text:
            return False
        return bool(NO_THINKING_TASK_RE.search(text))

    def _heuristic_thinking_requested(
        self,
        cfg: RouterConfig,
        req: UnifiedRequest,
        selected_alias: str,
        task_type: str,
    ) -> bool:
        profile = cfg.models.get(selected_alias)
        if not profile or not profile.supports_thinking:
            return False
            
        # Thinking standardmäßig AUS.
        # Thinking nur aktivieren bei: Debugging, Architekturentscheidungen, komplexem Refactoring,
        # Root-Cause-Analyse, agentischen Planungsaufgaben, sehr großen Kontexten.
        
        if task_type in ("debug", "architecture", "complex_code", "agent"):
            return True
            
        if req.routing_estimated_total_tokens > 60000 and task_type != "simple":
            return True
            
        return False

    async def _judge_alias(
        self,
        cfg: RouterConfig,
        req: UnifiedRequest,
        candidates: Iterable[str],
        *,
        is_deep_reasoning: bool = False,
        is_websearch: bool = False,
        is_commit_task: bool = False,
        is_file_search: bool = False,
    ) -> tuple[Optional[str], Optional[bool]]:
        judge_alias = self._judge_model_alias(cfg)
        if not judge_alias:
            logger.warning("judge_unavailable reason=no_enabled_judge_model")
            return None, None
        judge_model = cfg.models[judge_alias].model_id
        candidate_list = list(candidates)
        if len(candidate_list) <= 1:
            return candidate_list[0] if candidate_list else None, None
        logger.info(
            "judge_start candidates=%s requested_model=%r est_input_tokens=%s est_total_tokens=%s is_deep=%s is_websearch=%s",
            candidate_list,
            req.requested_model,
            req.routing_input_tokens,
            req.routing_estimated_total_tokens,
            is_deep_reasoning,
            is_websearch,
        )
        context_chars = max(500, cfg.routing.heuristics.judge_prompt_context_chars)
        latest_user_text = (
            req.routing_latest_user_prompt_text
            or req.routing_user_prompt_text
            or req.routing_prompt_text
            or req.latest_user_prompt_text
            or req.user_prompt_text
            or req.prompt_text
            or ""
        )
        latest_user_excerpt = latest_user_text[:context_chars]
        recent_user_context = (req.routing_user_prompt_text or req.routing_prompt_text or latest_user_text)[-context_chars:]
        recent_request_memory = self._recent_request_memory_for_judge(cfg, req, self._is_coding_request(req))
        judge_prompt = {
            "instruction": (
                "Return only JSON: "
                "{\"route\":\"small|large|deep\",\"thinking\":\"on|off\",\"reason_code\":\"short_code\"}."
            ),
            "features": {
                "source_api": req.source_api,
                "routing_input_tokens": req.routing_input_tokens,
                "routing_estimated_total_tokens": req.routing_estimated_total_tokens,
                "full_input_tokens": req.full_input_tokens,
                "full_estimated_total_tokens": req.full_estimated_total_tokens,
                "max_tokens": req.max_tokens,
                "routing_max_tokens_budget": req.routing_max_tokens_budget,
                "needs_vision": req.needs_vision,
                "needs_tooluse": req.needs_tooluse,
                "has_wrapper_noise": req.has_wrapper_noise,
                "tool_loop_context": req.tool_loop_context,
                "lightweight_greeting": bool(LIGHTWEIGHT_TASK_RE.match(latest_user_excerpt.strip())),
                "requested_model": req.requested_model,
                "session_id": self._effective_session_id(cfg, req) or None,
                "heuristic_signals": {
                    "is_deep_reasoning": is_deep_reasoning,
                    "is_websearch": is_websearch,
                    "is_commit_task": is_commit_task,
                    "is_file_search": is_file_search,
                },
                "recent_request_memory": recent_request_memory,
            },
            "latest_user_prompt_excerpt": latest_user_excerpt,
            "recent_user_context_excerpt": recent_user_context,
            "candidates": candidate_list,
            "candidate_summary": {
                alias: {
                    "supports_thinking": cfg.models[alias].supports_thinking,
                    "context_window": cfg.models[alias].context_window,
                    "relative_speed": cfg.models[alias].relative_speed,
                    "suitable_for": cfg.models[alias].suitable_for,
                    "capabilities": cfg.models[alias].capabilities,
                }
                for alias in candidate_list
            },
            "edge_arguments": [
                "Client wrappers, system reminders, tool schemas, and local command echoes are not a reason for large.",
                "A high max_tokens value can be a generic client default and is not sufficient evidence for large.",
                "Short acknowledgements or greetings (e.g. 'hallo') should route to small.",
                "Do not choose deep solely because prompt/max_tokens are large.",
                "Use the latest actionable user ask, not wrapper noise, as the main routing signal.",
                "Choose deep only for clear multi-step reasoning, strict rule compliance, high-risk decisions or web-search.",
                "Choose deep if the user explicitly asks for a web search or mentions stm32cube/cubecli related troubleshooting.",
                "Choose large only when the latest user ask clearly requires stronger coding/programming depth.",
                "Set thinking=on only if the selected route supports thinking and the task clearly benefits from it.",
                "Never set thinking=on when tools/tool-use are required.",
                "Set thinking=off for lightweight text tasks like commit messages, PR titles/descriptions, changelogs, or summaries.",
                "When tools, agent flows, or tool-loop context are involved, only choose large or deep.",
                "Prefer small for file search / file lookup tasks when available.",
                "Only compare with recent requests from the same session_id; ignore all other sessions.",
                "If no session_id is available, do not assume prior context from other clients.",
                "If the immediately previous request in the same session is highly similar and compatible, treat it as loop risk.",
                "When loop risk is high, prefer the next stronger candidate to break repetition.",
            ],
            "policy": "Prefer small by default. Use large for regular coding tasks. Use deep only for explicitly complex/high-stakes reasoning or web-search. Agent/tool-use flows must stay on large or deep.",
            "preference": "Prefer small by default.",
        }
        judge_system_prompt = ""
        if JUDGE_PROMPT_SYSTEM_PATH.exists():
            loaded_prompt = yaml.safe_load(JUDGE_PROMPT_SYSTEM_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded_prompt, dict):
                judge_system_prompt = str(loaded_prompt.get("judge_prompt_system", "")).strip()
            elif loaded_prompt is not None:
                judge_system_prompt = str(loaded_prompt).strip()

        payload = {
            "model": judge_model,
            "temperature": cfg.routing.heuristics.judge_temperature,
            "max_tokens": cfg.routing.heuristics.judge_max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": judge_system_prompt,
                },
                {
                    "role": "user",
                    "content": json.dumps(judge_prompt, ensure_ascii=False),
                },
            ],
        }
        judge_settings = cfg.upstream_for_alias(judge_alias).model_copy(
            update={"timeout_seconds": cfg.routing.judge_timeout_seconds}
        )
        judge_request_key = self._judge_request_key(
            cfg,
            req,
            candidate_list,
            judge_model=judge_model,
            is_deep_reasoning=is_deep_reasoning,
            is_websearch=is_websearch,
            is_commit_task=is_commit_task,
            is_file_search=is_file_search,
            recent_request_memory=recent_request_memory,
        )
        shared_future: Optional[asyncio.Future[tuple[Optional[str], Optional[bool]]]] = None
        created_future = False
        async with self._judge_inflight_lock:
            shared_future = self._judge_inflight.get(judge_request_key)
            if shared_future is None:
                shared_future = asyncio.get_running_loop().create_future()
                self._judge_inflight[judge_request_key] = shared_future
                created_future = True
            else:
                logger.info("judge_join_inflight candidates=%s key=%s", candidate_list, judge_request_key[:12])
        if not created_future:
            return await asyncio.shield(shared_future)
        try:
            try:
                response = await self.lm_client.post_json(judge_settings, "/v1/chat/completions", payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("judge_failed error=%s", exc)
                result = (None, None)
            else:
                if not response:
                    logger.warning("judge_empty_json_response")
                    result = (None, None)
                else:
                    text = _extract_assistant_text(response).strip()
                    if not text:
                        logger.warning("judge_empty_response")
                        result = (None, None)
                    else:
                        route = None
                        thinking_requested: Optional[bool] = None
                        try:
                            parsed = json.loads(text)
                            route = parsed.get("route")
                            thinking_raw = str(parsed.get("thinking") or "").strip().lower()
                            if thinking_raw in {"on", "true", "1"}:
                                thinking_requested = True
                            elif thinking_raw in {"off", "false", "0"}:
                                thinking_requested = False
                        except json.JSONDecodeError:
                            match = re.search(r"\b(small|large|deep)\b", text.lower())
                            if match:
                                route = match.group(1)
                            think_match = re.search(r"\b(on|off)\b", text.lower())
                            if think_match:
                                thinking_requested = think_match.group(1) == "on"

                        if route in candidate_list:
                            logger.info("judge_result route=%s thinking=%s", route, thinking_requested)
                            result = (route, thinking_requested)
                        else:
                            logger.warning("judge_unusable_response text=%s", text[:200])
                            result = (None, None)
            if shared_future is not None and not shared_future.done():
                shared_future.set_result(result)
            return result
        except BaseException as exc:
            if shared_future is not None and not shared_future.done():
                shared_future.set_exception(exc)
            raise
        finally:
            async with self._judge_inflight_lock:
                current_future = self._judge_inflight.get(judge_request_key)
                if current_future is shared_future:
                    self._judge_inflight.pop(judge_request_key, None)

    def _heuristic_alias(self, cfg: RouterConfig, req: UnifiedRequest, candidates: list[str]) -> str:
        if len(candidates) == 1:
            return candidates[0]
        if "deep" in candidates and self._is_deep_reasoning_request(req):
            return "deep"
        if "large" in candidates:
            h = cfg.routing.heuristics
            if req.routing_input_tokens >= h.large_prompt_token_threshold:
                return "large"
            if req.effective_routing_max_tokens_budget >= h.large_max_tokens_threshold:
                return "large"
            if req.needs_tooluse and "small" not in candidates:
                return "large"
        return "small" if "small" in candidates else candidates[0]

    @staticmethod
    def _preferred_alias_for_request(
        cfg: RouterConfig,
        req: UnifiedRequest,
        candidates: list[str],
        preferred_alias: Optional[str],
        is_coding: bool,
    ) -> Optional[str]:
        if not cfg.routing.hybrid_client_model_override or not preferred_alias:
            return None
        if preferred_alias not in candidates:
            return None
        if preferred_alias == "large" and not is_coding:
            logger.info("route_eval_skip_client_large_non_coding requested_model=%r", req.requested_model)
            return None
        return preferred_alias

    async def choose_route(self, cfg: RouterConfig, req: UnifiedRequest) -> RouteDecision:
        is_coding = self._is_coding_request(req)
        task_type = self._classify_task_type(req, is_coding)
        self._apply_routing_budget(cfg, req, is_coding)
        candidates = self._eligible_aliases(cfg, req)
        is_commit_task = req.is_commit_message_task
        is_no_thinking_task = self._is_no_thinking_task(req)
        is_file_search = self._is_file_search_request(req)
        session_id = self._effective_session_id(cfg, req)

        # 1. Check for exact repetition cache
        repetition_key = self._repetition_key(req)
        recent_entries = self.request_memory.recent_entries(session_id, limit=3) if session_id else []
        for entry in recent_entries:
            if entry.get("repetition_key") == repetition_key:
                cached_alias = entry.get("selected_alias")
                cached_candidates = entry.get("candidates") or []
                if cached_alias in candidates and set(cached_candidates) == set(candidates):
                    logger.info(
                        "route_eval_cache_hit alias=%s reason=%s request_id=%s",
                        cached_alias,
                        entry.get("reason"),
                        entry.get("request_id"),
                    )
                    return self._make_route_decision(
                        req=req,
                        selected_alias=cached_alias,
                        reason=f"cache_hit_{entry.get('request_id')}",
                        candidates=candidates,
                        thinking_requested=bool(entry.get("thinking_requested")),
                        judge_model_id=entry.get("judge_model_id"),
                        is_coding=is_coding,
                        task_type=task_type,
                    )

        is_first_request = not recent_entries
        total_tokens = req.routing_estimated_total_tokens

        if not candidates:
            if not self._has_available_primary_alias(cfg) and cfg.is_alias_enabled("backup"):
                selected = "backup"
                reason = "no_primary_available_fallback_to_backup"
                thinking_requested = self._heuristic_thinking_requested(cfg, req, selected, task_type)
                decision = self._build_decision(
                    cfg,
                    req,
                    selected_alias=selected,
                    reason=reason,
                    candidates=[selected],
                    thinking_requested=thinking_requested,
                    judge_model_id=cfg.models["small"].model_id if "small" in cfg.models else None,
                    is_coding=is_coding,
                    task_type=task_type,
                )
                logger.info("routing_decision selected_model=%s task_type=%s total_estimated_tokens=%s thinking_enabled=%s routing_reason=%s",
                            decision.selected_alias, decision.task_type, decision.routing_estimated_total_tokens, decision.thinking_requested, decision.reason)
                return decision
            raise HTTPException(status_code=503, detail="No eligible primary model available for this request")

        # 2. Model selection logic
        selected = "small"
        reason = "default_gemma"
        shortcut_reason = self._prefer_small_shortcut(cfg, req, is_coding)
        if shortcut_reason and "small" in candidates:
            selected = "small"
            reason = shortcut_reason
        
        # Policy rules
        if reason == "default_gemma" and total_tokens > cfg.routing.heuristics.qwen_safe_limit:
            if "deep" in candidates:
                selected = "deep"
                reason = "context_exceeds_qwen_safe_limit"
            else:
                logger.warning("route_eval_limit_exceeded total_tokens=%s limit=%s", total_tokens, cfg.routing.heuristics.qwen_safe_limit)
                selected = "large" 
                reason = "context_exceeds_qwen_safe_limit_trying_large"
        elif reason == "default_gemma" and total_tokens > cfg.routing.heuristics.gemma_safe_limit:
            selected = "large"
            reason = "context_exceeds_gemma_safe_limit"
        elif reason == "default_gemma" and task_type in ("debug", "architecture", "complex_code", "agent"):
            selected = "large"
            reason = f"complex_task_{task_type}"
        
        # 3. Validation and fallback within candidates
        if selected not in candidates:
            if "large" in candidates:
                selected = "large"
                reason = f"fallback_to_large_from_{selected}"
            elif "small" in candidates:
                selected = "small"
                reason = f"fallback_to_small_from_{selected}"
            elif "deep" in candidates:
                selected = "deep"
                reason = f"fallback_to_deep_from_{selected}"
            else:
                selected = candidates[0]
                reason = "fallback_to_first_available"

        # 4. Thinking decision
        thinking_requested = self._heuristic_thinking_requested(cfg, req, selected, task_type)
        
        # 5. Build decision
        decision = self._build_decision(
            cfg,
            req,
            selected_alias=selected,
            reason=reason,
            candidates=candidates,
            thinking_requested=thinking_requested,
            judge_model_id=cfg.models["small"].model_id if "small" in cfg.models else None,
            is_coding=is_coding,
            task_type=task_type,
        )
        
        # 6. Logging
        logger.info(
            "routing_decision selected_model=%s task_type=%s total_estimated_tokens=%s thinking_enabled=%s routing_reason=%s",
            decision.selected_alias,
            decision.task_type,
            decision.routing_estimated_total_tokens,
            decision.thinking_requested,
            decision.reason
        )
        
        return decision


    @staticmethod
    def _attempt_order(cfg: RouterConfig, decision: RouteDecision) -> list[str]:
        # Enforce policy: non-coding requests must not spill over to large.
        if not decision.is_coding_request and decision.selected_alias == "small":
            if decision.is_commit_message_task and cfg.routing.fallback_enabled:
                order = [decision.selected_alias]
                if "large" in decision.candidate_aliases:
                    order.append("large")
                return order
            return [decision.selected_alias]
        if not cfg.routing.fallback_enabled:
            return [decision.selected_alias]
        order = [decision.selected_alias]
        for alias in decision.candidate_aliases:
            if alias != decision.selected_alias:
                order.append(alias)
        return order

    async def _attempt_json_with_fallback(
        self,
        cfg: RouterConfig,
        path: str,
        base_payload: dict[str, Any],
        decision: RouteDecision,
    ) -> tuple[str, dict[str, Any], bool]:
        last_error: Optional[UpstreamError] = None
        order = self._attempt_order(cfg, decision)
        logger.info("upstream_json_attempt_order path=%s order=%s", path, order)
        for idx, alias in enumerate(order):
            settings = self._upstream_for_alias(cfg, alias)
            payload_raw = dict(base_payload)
            payload_raw["model"] = cfg.models[alias].model_id
            payload_after_commit = self._normalize_commit_message_payload(path, payload_raw, decision)
            payload_after_temperature = self._apply_default_request_temperature(cfg, payload_after_commit)
            payload_after_budget = self._apply_alias_token_budget(alias, payload_after_temperature, decision)
            thinking_enabled = (
                decision.thinking_requested
                and not decision.needs_tooluse
                and cfg.models[alias].supports_thinking
            )
            payload_after_thinking = self._normalize_thinking_param(
                settings, path, payload_after_budget, thinking_enabled
            )
            payload = payload_after_thinking
            payload = self._normalize_openai_chat_token_param(settings, path, payload)
            if _thinking_debug_enabled():
                logger.info(
                    "thinking_debug_upstream_json path=%s alias=%s provider=%s decision_thinking=%s applied_thinking=%s raw=%s after_commit=%s after_temperature=%s after_budget=%s after_thinking=%s final=%s",
                    path,
                    alias,
                    settings.provider,
                    int(decision.thinking_requested),
                    int(thinking_enabled),
                    _thinking_payload_probe(payload_raw),
                    _thinking_payload_probe(payload_after_commit),
                    _thinking_payload_probe(payload_after_temperature),
                    _thinking_payload_probe(payload_after_budget),
                    _thinking_payload_probe(payload_after_thinking),
                    _thinking_payload_probe(payload),
                )
            logger.info(
                "upstream_json_attempt path=%s alias=%s model=%s thinking=%s attempt=%s/%s",
                path,
                alias,
                cfg.models[alias].model_id,
                int(thinking_enabled),
                idx + 1,
                len(order),
            )
            try:
                result = await self.lm_client.post_json(settings, path, payload)
                if path == "/v1/chat/completions":
                    assistant_text = _extract_assistant_text(result).strip()
                    tool_calls = _extract_openai_tool_call_count(result)
                    if not assistant_text and tool_calls == 0:
                        choices = result.get("choices") if isinstance(result, dict) else None
                        message = {}
                        finish_reason = ""
                        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                            finish_reason = str(choices[0].get("finish_reason") or "")
                            raw_msg = choices[0].get("message")
                            if isinstance(raw_msg, dict):
                                message = raw_msg
                        reasoning_only = bool(str(message.get("reasoning_content") or "").strip())
                        logger.warning(
                            "upstream_json_empty_output alias=%s finish_reason=%s reasoning_only=%s fallback_candidate=%s",
                            alias,
                            finish_reason or "none",
                            int(reasoning_only),
                            idx + 1 < len(order),
                        )
                        if idx + 1 < len(order):
                            continue
                logger.info(
                    "upstream_json_selected path=%s alias=%s fallback=%s",
                    path,
                    alias,
                    idx > 0,
                )
                return alias, result, idx > 0
            except UpstreamError as exc:
                last_error = exc
                logger.warning(
                    "upstream_json_failed alias=%s status=%s body=%s",
                    alias,
                    exc.status_code,
                    exc.body[:300],
                )

        if last_error is not None:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream model call failed after fallback attempts: {last_error.body}",
            )
        raise HTTPException(status_code=500, detail="Unexpected routing failure")

    async def _attempt_stream_with_fallback(
        self,
        cfg: RouterConfig,
        path: str,
        base_payload: dict[str, Any],
        decision: RouteDecision,
    ) -> tuple[str, AsyncIterator[bytes], bool]:
        def _openai_stream_chunk_has_visible_output(chunk: bytes) -> bool:
            text = chunk.decode("utf-8", errors="replace")
            for raw_event in text.split("\n\n"):
                if not raw_event:
                    continue
                for line in raw_event.splitlines():
                    if not line.startswith("data:"):
                        continue
                    data_line = line[5:].strip()
                    if not data_line or data_line == "[DONE]":
                        continue
                    try:
                        parsed = json.loads(data_line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    choices = parsed.get("choices")
                    if not isinstance(choices, list):
                        continue
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            content = delta.get("content")
                            if isinstance(content, str) and content.strip():
                                return True
                            if isinstance(content, list):
                                for part in content:
                                    if (
                                        isinstance(part, dict)
                                        and str(part.get("type") or "").strip().lower() in {"text", "output_text"}
                                        and str(part.get("text") or "").strip()
                                    ):
                                        return True
                            tool_calls = delta.get("tool_calls")
                            if isinstance(tool_calls, list) and tool_calls:
                                return True
                        message = choice.get("message")
                        if isinstance(message, dict):
                            content = message.get("content")
                            if isinstance(content, str) and content.strip():
                                return True
                            if isinstance(content, list):
                                for part in content:
                                    if (
                                        isinstance(part, dict)
                                        and str(part.get("type") or "").strip().lower() in {"text", "output_text"}
                                        and str(part.get("text") or "").strip()
                                    ):
                                        return True
                            tool_calls = message.get("tool_calls")
                            if isinstance(tool_calls, list) and tool_calls:
                                return True
            return False

        def _openai_stream_chunk_is_done(chunk: bytes) -> bool:
            text = chunk.decode("utf-8", errors="replace")
            for raw_event in text.split("\n\n"):
                if not raw_event:
                    continue
                for line in raw_event.splitlines():
                    if line.startswith("data:") and line[5:].strip() == "[DONE]":
                        return True
            return False

        last_error: Optional[UpstreamError] = None
        order = self._attempt_order(cfg, decision)
        logger.info("upstream_stream_attempt_order path=%s order=%s", path, order)
        semantic_check = decision.is_commit_message_task and path == "/v1/chat/completions"
        last_buffered: list[bytes] = []
        last_alias = order[0] if order else decision.selected_alias
        for idx, alias in enumerate(order):
            settings = self._upstream_for_alias(cfg, alias)
            payload_raw = dict(base_payload)
            payload_raw["model"] = cfg.models[alias].model_id
            payload_after_commit = self._normalize_commit_message_payload(path, payload_raw, decision)
            payload_after_temperature = self._apply_default_request_temperature(cfg, payload_after_commit)
            payload_after_budget = self._apply_alias_token_budget(alias, payload_after_temperature, decision)
            thinking_enabled = (
                decision.thinking_requested
                and not decision.needs_tooluse
                and cfg.models[alias].supports_thinking
            )
            payload_after_thinking = self._normalize_thinking_param(
                settings, path, payload_after_budget, thinking_enabled
            )
            payload = payload_after_thinking
            payload = self._normalize_openai_chat_token_param(settings, path, payload)
            if _thinking_debug_enabled():
                logger.info(
                    "thinking_debug_upstream_stream path=%s alias=%s provider=%s decision_thinking=%s applied_thinking=%s raw=%s after_commit=%s after_temperature=%s after_budget=%s after_thinking=%s final=%s",
                    path,
                    alias,
                    settings.provider,
                    int(decision.thinking_requested),
                    int(thinking_enabled),
                    _thinking_payload_probe(payload_raw),
                    _thinking_payload_probe(payload_after_commit),
                    _thinking_payload_probe(payload_after_temperature),
                    _thinking_payload_probe(payload_after_budget),
                    _thinking_payload_probe(payload_after_thinking),
                    _thinking_payload_probe(payload),
                )
            logger.info(
                "upstream_stream_attempt path=%s alias=%s model=%s thinking=%s attempt=%s/%s",
                path,
                alias,
                cfg.models[alias].model_id,
                int(thinking_enabled),
                idx + 1,
                len(order),
            )
            stream_gen = self.lm_client.stream_openai(settings, path, payload)
            try:
                first_chunk = await stream_gen.__anext__()
                if _thinking_debug_enabled():
                    logger.info(
                        "thinking_debug_stream_first_chunk path=%s alias=%s hint=%s chunk_bytes=%s",
                        path,
                        alias,
                        _stream_chunk_thinking_hint(first_chunk),
                        len(first_chunk),
                    )
            except StopAsyncIteration:
                logger.warning("upstream_stream_empty_on_first_chunk path=%s alias=%s", path, alias)
                continue
            except UpstreamError as exc:
                last_error = exc
                logger.warning(
                    "upstream_stream_failed alias=%s status=%s body=%s",
                    alias,
                    exc.status_code,
                    exc.body[:300],
                )
                continue

            async def chained() -> AsyncIterator[bytes]:
                yield first_chunk
                async for chunk in stream_gen:
                    yield chunk

            if semantic_check:
                buffered: list[bytes] = [first_chunk]
                meaningful = _openai_stream_chunk_has_visible_output(first_chunk)
                done = _openai_stream_chunk_is_done(first_chunk)

                if not meaningful and not done:
                    try:
                        async for chunk in stream_gen:
                            buffered.append(chunk)
                            if _openai_stream_chunk_has_visible_output(chunk):
                                meaningful = True
                                break
                            if _openai_stream_chunk_is_done(chunk):
                                done = True
                                break
                    except UpstreamError as exc:
                        last_error = exc
                        logger.warning(
                            "upstream_stream_failed alias=%s status=%s body=%s",
                            alias,
                            exc.status_code,
                            exc.body[:300],
                        )
                        continue

                last_buffered = buffered
                last_alias = alias

                if not meaningful:
                    logger.warning(
                        "upstream_stream_semantic_empty alias=%s commit_task=1 fallback_candidate=%s",
                        alias,
                        idx + 1 < len(order),
                    )
                    with contextlib.suppress(Exception):
                        await stream_gen.aclose()
                    continue

                async def semantic_chained() -> AsyncIterator[bytes]:
                    for chunk in buffered:
                        yield chunk
                    async for chunk in stream_gen:
                        yield chunk

                logger.info("upstream_stream_selected path=%s alias=%s fallback=%s", path, alias, idx > 0)
                return alias, semantic_chained(), idx > 0

            logger.info("upstream_stream_selected path=%s alias=%s fallback=%s", path, alias, idx > 0)
            return alias, chained(), idx > 0

        if last_error is not None:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream streaming call failed after fallback attempts: {last_error.body}",
            )
        if semantic_check:
            async def replay_last() -> AsyncIterator[bytes]:
                for chunk in last_buffered:
                    yield chunk

            logger.warning(
                "upstream_stream_semantic_no_meaningful_output path=%s alias=%s replaying_last=%s",
                path,
                last_alias,
                bool(last_buffered),
            )
            return last_alias, replay_last(), len(order) > 1 and last_alias != order[0]
        raise HTTPException(status_code=500, detail="Unexpected streaming routing failure")

    async def _attempt_anthropic_stream_with_semantic_fallback(
        self,
        cfg: RouterConfig,
        path: str,
        base_payload: dict[str, Any],
        decision: RouteDecision,
    ) -> tuple[str, AsyncIterator[bytes], bool]:
        last_error: Optional[UpstreamError] = None
        order = self._attempt_order(cfg, decision)

        # For Anthropic streaming, allow one semantic-empty retry on large.
        if (
            len(order) == 1
            and decision.selected_alias == "small"
            and "large" in decision.candidate_aliases
            and "large" not in order
        ):
            order.append("large")
            logger.info("anthropic_stream_semantic_retry_extend_order path=%s order=%s", path, order)

        logger.info("anthropic_stream_semantic_attempt_order path=%s order=%s", path, order)
        last_buffered: list[bytes] = []
        last_alias = order[0] if order else decision.selected_alias

        for idx, alias in enumerate(order):
            settings = self._upstream_for_alias(cfg, alias)
            payload_raw = dict(base_payload)
            payload_raw["model"] = cfg.models[alias].model_id
            payload_after_commit = self._normalize_commit_message_payload(path, payload_raw, decision)
            payload_after_temperature = self._apply_default_request_temperature(cfg, payload_after_commit)
            payload_after_budget = self._apply_alias_token_budget(alias, payload_after_temperature, decision)
            thinking_enabled = (
                decision.thinking_requested
                and not decision.needs_tooluse
                and cfg.models[alias].supports_thinking
            )
            payload_after_thinking = self._normalize_thinking_param(
                settings, path, payload_after_budget, thinking_enabled
            )
            payload = payload_after_thinking
            payload = self._normalize_openai_chat_token_param(settings, path, payload)
            if _thinking_debug_enabled():
                logger.info(
                    "thinking_debug_upstream_anthropic_stream path=%s alias=%s provider=%s decision_thinking=%s applied_thinking=%s raw=%s after_commit=%s after_temperature=%s after_budget=%s after_thinking=%s final=%s",
                    path,
                    alias,
                    settings.provider,
                    int(decision.thinking_requested),
                    int(thinking_enabled),
                    _thinking_payload_probe(payload_raw),
                    _thinking_payload_probe(payload_after_commit),
                    _thinking_payload_probe(payload_after_temperature),
                    _thinking_payload_probe(payload_after_budget),
                    _thinking_payload_probe(payload_after_thinking),
                    _thinking_payload_probe(payload),
                )
            logger.info(
                "anthropic_stream_semantic_attempt path=%s alias=%s model=%s thinking=%s attempt=%s/%s",
                path,
                alias,
                cfg.models[alias].model_id,
                int(thinking_enabled),
                idx + 1,
                len(order),
            )

            upstream_stream = self.lm_client.stream_openai(settings, path, payload)
            translated = translate_openai_stream_to_anthropic(
                upstream_stream,
                cfg.router_identity.exposed_model_name,
                source_api=decision.source_api,
                decision=decision,
                final_alias=alias,
                final_model_id=cfg.models[alias].model_id,
                used_fallback=idx > 0,
            )

            buffered: list[bytes] = []
            meaningful = False
            try:
                async for event_chunk in translated:
                    buffered.append(event_chunk)
                    event_name, event_payload = _parse_sse_event(event_chunk)
                    if _is_meaningful_anthropic_event(event_name, event_payload):
                        meaningful = True
                        break
            except UpstreamError as exc:
                last_error = exc
                logger.warning(
                    "anthropic_stream_semantic_failed alias=%s status=%s body=%s",
                    alias,
                    exc.status_code,
                    exc.body[:300],
                )
                continue

            last_buffered = buffered
            last_alias = alias

            if not meaningful:
                logger.warning(
                    "anthropic_stream_semantic_empty alias=%s buffered_events=%s",
                    alias,
                    len(buffered),
                )
                continue

            async def chained() -> AsyncIterator[bytes]:
                for chunk in buffered:
                    yield chunk
                async for chunk in translated:
                    yield chunk

            logger.info("anthropic_stream_semantic_selected path=%s alias=%s fallback=%s", path, alias, idx > 0)
            return alias, chained(), idx > 0

        if last_error is not None:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream streaming call failed after fallback attempts: {last_error.body}",
            )

        async def replay_last() -> AsyncIterator[bytes]:
            for chunk in last_buffered:
                yield chunk

        logger.warning(
            "anthropic_stream_semantic_no_meaningful_output path=%s alias=%s replaying_last=%s",
            path,
            last_alias,
            bool(last_buffered),
        )
        return last_alias, replay_last(), len(order) > 1 and last_alias != order[0]

    async def handle_openai_chat(self, payload: dict[str, Any], *, session_id: str = "") -> tuple[RouteDecision, str, bool, Any]:
        cfg = self.config_store.get_config()
        req = normalize_openai_chat(payload, session_id=session_id)
        decision = await self.choose_route(cfg, req)
        
        # Override stream in payload if required by task normalization
        effective_payload = dict(payload)
        effective_payload["stream"] = req.stream

        if req.stream:
            alias, stream_gen, used_fallback = await self._attempt_stream_with_fallback(
                cfg, "/v1/chat/completions", effective_payload, decision
            )
            public_stream = rewrite_openai_stream_model_name(
                stream_gen,
                cfg.router_identity.exposed_model_name,
                source_api=req.source_api,
                decision=decision,
                final_alias=alias,
                final_model_id=cfg.models[alias].model_id,
                used_fallback=used_fallback,
            )
            return decision, alias, used_fallback, public_stream
        alias, body, used_fallback = await self._attempt_json_with_fallback(
            cfg, "/v1/chat/completions", effective_payload, decision
        )
        public_body = _apply_public_model_name_to_openai_response(
            body,
            cfg.router_identity.exposed_model_name,
        )
        usage = public_body.get("usage") or {}
        _log_output_analytics(
            source_api=req.source_api,
            decision=decision,
            final_alias=alias,
            final_model_id=cfg.models[alias].model_id,
            used_fallback=used_fallback,
            stream=False,
            output_text=_extract_assistant_text(public_body),
            stop_reason=((public_body.get("choices") or [{}])[0].get("finish_reason")),
            output_tokens=usage.get("completion_tokens"),
            input_tokens=usage.get("prompt_tokens"),
            tool_calls=_extract_openai_tool_call_count(public_body),
        )
        return decision, alias, used_fallback, public_body

    async def handle_openai_completions(self, payload: dict[str, Any], *, session_id: str = "") -> tuple[RouteDecision, str, bool, Any]:
        cfg = self.config_store.get_config()
        req = normalize_openai_completion(payload, session_id=session_id)
        decision = await self.choose_route(cfg, req)
        
        # Override stream in payload if required by task normalization
        effective_payload = dict(payload)
        effective_payload["stream"] = req.stream

        if req.stream:
            alias, stream_gen, used_fallback = await self._attempt_stream_with_fallback(
                cfg, "/v1/completions", effective_payload, decision
            )
            public_stream = rewrite_openai_stream_model_name(
                stream_gen,
                cfg.router_identity.exposed_model_name,
                source_api=req.source_api,
                decision=decision,
                final_alias=alias,
                final_model_id=cfg.models[alias].model_id,
                used_fallback=used_fallback,
            )
            return decision, alias, used_fallback, public_stream
        alias, body, used_fallback = await self._attempt_json_with_fallback(
            cfg, "/v1/completions", effective_payload, decision
        )
        public_body = _apply_public_model_name_to_openai_response(
            body,
            cfg.router_identity.exposed_model_name,
        )
        usage = public_body.get("usage") or {}
        _log_output_analytics(
            source_api=req.source_api,
            decision=decision,
            final_alias=alias,
            final_model_id=cfg.models[alias].model_id,
            used_fallback=used_fallback,
            stream=False,
            output_text=_extract_assistant_text(public_body),
            stop_reason=((public_body.get("choices") or [{}])[0].get("finish_reason")),
            output_tokens=usage.get("completion_tokens"),
            input_tokens=usage.get("prompt_tokens"),
            tool_calls=_extract_openai_tool_call_count(public_body),
        )
        return decision, alias, used_fallback, public_body

    async def handle_anthropic_messages(
        self, payload: dict[str, Any], *, session_id: str = ""
    ) -> tuple[RouteDecision, str, bool, bool, Any]:
        cfg = self.config_store.get_config()
        req = normalize_anthropic_messages(payload, session_id=session_id)
        decision = await self.choose_route(cfg, req)

        openai_payload = anthropic_to_openai_payload(payload)
        openai_payload["stream"] = req.stream
        openai_payload["model"] = cfg.models[decision.selected_alias].model_id

        if req.stream:
            alias, translated, used_fallback = await self._attempt_anthropic_stream_with_semantic_fallback(
                cfg, "/v1/chat/completions", openai_payload, decision
            )
            return decision, alias, used_fallback, True, translated

        alias, response_json, used_fallback = await self._attempt_json_with_fallback(
            cfg, "/v1/chat/completions", openai_payload, decision
        )
        anthropic_response = openai_to_anthropic_response(
            response_json,
            cfg.router_identity.exposed_model_name,
        )
        usage = anthropic_response.get("usage") or {}
        text_blocks = anthropic_response.get("content") or []
        text_parts: list[str] = []
        tool_calls = 0
        if isinstance(text_blocks, list):
            for block in text_blocks:
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get("type") or "")
                if block_type == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif block_type == "tool_use":
                    tool_calls += 1
        _log_output_analytics(
            source_api=req.source_api,
            decision=decision,
            final_alias=alias,
            final_model_id=cfg.models[alias].model_id,
            used_fallback=used_fallback,
            stream=False,
            output_text="".join(text_parts),
            stop_reason=anthropic_response.get("stop_reason"),
            output_tokens=usage.get("output_tokens"),
            input_tokens=usage.get("input_tokens"),
            tool_calls=tool_calls,
        )
        return decision, alias, used_fallback, False, anthropic_response
