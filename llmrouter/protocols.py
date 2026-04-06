from __future__ import annotations

from .shared import *
from .shared import (
    _clip_for_log,
    _current_request_latency_ms,
    _estimate_tokens_from_text,
    _extract_assistant_text,
    _extract_openai_tool_call_count,
    _extract_text_and_vision,
    _hash_text,
    _log_text_max_chars,
    _routing_efficiency,
)
from .settings import *
from .requests import *

_analytics_store: Any = None


def set_analytics_store(store: Any) -> None:
    global _analytics_store
    _analytics_store = store


def anthropic_to_openai_payload(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages") or []
    openai_messages: list[dict[str, Any]] = []

    system = payload.get("system")
    if isinstance(system, str) and system.strip():
        openai_messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        parts: list[str] = []
        for item in system:
            text, _ = _extract_text_and_vision(item)
            if text:
                parts.append(text)
        if parts:
            openai_messages.append({"role": "system", "content": "\n".join(parts)})

    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content")
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue
        if isinstance(content, list):
            translated_parts: list[dict[str, Any]] = []
            assistant_tool_calls: list[dict[str, Any]] = []
            tool_result_messages: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    translated_parts.append({"type": "text", "text": str(part)})
                    continue
                p_type = part.get("type")
                if p_type == "text":
                    translated_parts.append({"type": "text", "text": part.get("text", "")})
                elif p_type in {"image", "input_image"}:
                    source = part.get("source") or {}
                    media_type = source.get("media_type", "image/png")
                    if source.get("type") == "base64":
                        data = source.get("data", "")
                        translated_parts.append(
                            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
                        )
                    elif source.get("url"):
                        translated_parts.append({"type": "image_url", "image_url": {"url": source["url"]}})
                elif p_type == "tool_result":
                    tool_use_id = str(part.get("tool_use_id") or f"toolu_{uuid.uuid4().hex[:24]}")
                    result_content = part.get("content")
                    result_text, _ = _extract_text_and_vision(result_content)
                    if not result_text and result_content is not None:
                        result_text = json.dumps(result_content, ensure_ascii=False)
                    if part.get("is_error"):
                        result_text = f"[tool_error] {result_text}"
                    tool_result_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": result_text or "",
                        }
                    )
                elif p_type == "tool_use":
                    tool_id = str(part.get("id") or f"toolu_{uuid.uuid4().hex[:24]}")
                    tool_name = str(part.get("name") or "tool")
                    tool_input = part.get("input", {})
                    if not isinstance(tool_input, (dict, list)):
                        tool_input = {"value": tool_input}
                    assistant_tool_calls.append(
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_input, ensure_ascii=False),
                            },
                        }
                    )

            if assistant_tool_calls:
                text_content = translated_parts if translated_parts else ""
                openai_messages.append(
                    {
                        "role": "assistant",
                        "content": text_content,
                        "tool_calls": assistant_tool_calls,
                    }
                )
            elif translated_parts:
                openai_messages.append({"role": role, "content": translated_parts})

            if tool_result_messages:
                openai_messages.extend(tool_result_messages)
            continue
        openai_messages.append({"role": role, "content": str(content)})

    out: dict[str, Any] = {
        "messages": openai_messages,
        "stream": bool(payload.get("stream")),
    }

    passthrough_fields = [
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
    ]
    for key in passthrough_fields:
        if key in payload:
            if key == "stop_sequences":
                out["stop"] = payload[key]
            else:
                out[key] = payload[key]

    tools = payload.get("tools") or []
    if tools:
        mapped_tools = []
        for tool in tools:
            mapped_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
            )
        out["tools"] = mapped_tools

        # Helps models avoid silent/empty turns in tool loops.
        tool_hint = os.getenv("ROUTER_TOOLUSE_SYSTEM_HINT", DEFAULT_TOOLUSE_SYSTEM_HINT).strip()
        if tool_hint:
            if openai_messages and openai_messages[0].get("role") == "system":
                current = str(openai_messages[0].get("content") or "")
                if tool_hint not in current:
                    sep = "\n\n" if current else ""
                    openai_messages[0]["content"] = f"{current}{sep}{tool_hint}"
            else:
                openai_messages.insert(0, {"role": "system", "content": tool_hint})
    return out


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, list):
        return {"items": raw_arguments}
    if not isinstance(raw_arguments, str):
        return {"value": raw_arguments}
    trimmed = raw_arguments.strip()
    if not trimmed:
        return {}
    try:
        parsed = json.loads(trimmed)
    except json.JSONDecodeError:
        return {"_raw": raw_arguments}
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"items": parsed}
    return {"value": parsed}


def openai_to_anthropic_response(openai_response: dict[str, Any], model_id: str) -> dict[str, Any]:
    content_text = _extract_assistant_text(openai_response)
    openai_usage = openai_response.get("usage") or {}
    stop_reason = "end_turn"
    choices = openai_response.get("choices") or []
    content_blocks: list[dict[str, Any]] = []

    if content_text:
        content_blocks.append({"type": "text", "text": content_text})

    if choices:
        finish_reason = choices[0].get("finish_reason")
        if finish_reason in {"length", "max_tokens"}:
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"

        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            tool_name = str(function.get("name") or "tool")
            tool_id = str(tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}")
            tool_input = _parse_tool_arguments(function.get("arguments"))
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                }
            )
        if tool_calls:
            stop_reason = "tool_use"

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_usage.get("prompt_tokens", 0),
            "output_tokens": openai_usage.get("completion_tokens", 0),
        },
    }


def _apply_public_model_name_to_openai_response(
    openai_response: dict[str, Any], public_model_name: str
) -> dict[str, Any]:
    patched = dict(openai_response)
    patched["model"] = public_model_name
    choices = patched.get("choices")
    if isinstance(choices, list):
        sanitized_choices: list[Any] = []
        for choice in choices:
            if not isinstance(choice, dict):
                sanitized_choices.append(choice)
                continue
            c = dict(choice)
            message = c.get("message")
            if isinstance(message, dict):
                m = dict(message)
                m.pop("reasoning_content", None)
                c["message"] = m
            delta = c.get("delta")
            if isinstance(delta, dict):
                d = dict(delta)
                d.pop("reasoning_content", None)
                c["delta"] = d
            sanitized_choices.append(c)
        patched["choices"] = sanitized_choices
    return patched


async def rewrite_openai_stream_model_name(
    upstream_stream: AsyncIterator[bytes],
    public_model_name: str,
    source_api: str = "openai_chat",
    decision: Optional[RouteDecision] = None,
    final_alias: str = "unknown",
    final_model_id: str = "unknown",
    used_fallback: bool = False,
) -> AsyncIterator[bytes]:
    buffer = ""
    output_chunks: list[str] = []
    output_text_chars = 0
    stop_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    tool_calls = 0
    async for chunk in upstream_stream:
        text = chunk.decode("utf-8", errors="replace")
        buffer += text
        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            out_lines: list[str] = []
            for line in raw_event.splitlines():
                if line.startswith("data:"):
                    data_line = line[5:].strip()
                    if data_line and data_line != "[DONE]":
                        try:
                            parsed = json.loads(data_line)
                            if isinstance(parsed, dict):
                                parsed["model"] = public_model_name
                                choices = parsed.get("choices")
                                if isinstance(choices, list):
                                    sanitized_choices: list[Any] = []
                                    for choice in choices:
                                        if not isinstance(choice, dict):
                                            sanitized_choices.append(choice)
                                            continue
                                        c = dict(choice)
                                        delta = c.get("delta")
                                        if isinstance(delta, dict):
                                            d = dict(delta)
                                            d.pop("reasoning_content", None)
                                            c["delta"] = d
                                        message = c.get("message")
                                        if isinstance(message, dict):
                                            m = dict(message)
                                            m.pop("reasoning_content", None)
                                            c["message"] = m
                                        sanitized_choices.append(c)
                                    parsed["choices"] = sanitized_choices
                                choices = parsed.get("choices") or []
                                if choices:
                                    choice0 = choices[0]
                                    delta_text = _extract_delta_text(choice0 if isinstance(choice0, dict) else {})
                                    if delta_text:
                                        remaining = _log_text_max_chars() - output_text_chars
                                        if remaining > 0:
                                            clipped = delta_text[:remaining]
                                            output_chunks.append(clipped)
                                            output_text_chars += len(clipped)
                                    if isinstance(choice0, dict):
                                        finish_reason = choice0.get("finish_reason")
                                        if finish_reason:
                                            stop_reason = str(finish_reason)
                                        delta = choice0.get("delta") or {}
                                        raw_tool_calls = delta.get("tool_calls") or []
                                        if isinstance(raw_tool_calls, list):
                                            tool_calls += len(raw_tool_calls)
                                usage = parsed.get("usage") or {}
                                if isinstance(usage, dict):
                                    in_tok = usage.get("prompt_tokens")
                                    out_tok = usage.get("completion_tokens")
                                    if isinstance(in_tok, int):
                                        input_tokens = max(input_tokens or 0, in_tok)
                                    if isinstance(out_tok, int):
                                        output_tokens = max(output_tokens or 0, out_tok)
                                line = f"data: {json.dumps(parsed, ensure_ascii=False)}"
                        except json.JSONDecodeError:
                            pass
                out_lines.append(line)
            yield ("\n".join(out_lines) + "\n\n").encode("utf-8")
    if buffer:
        yield buffer.encode("utf-8")
    if decision is not None:
        _log_output_analytics(
            source_api=source_api,
            decision=decision,
            final_alias=final_alias,
            final_model_id=final_model_id,
            used_fallback=used_fallback,
            stream=True,
            output_text="".join(output_chunks),
            stop_reason=stop_reason,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            tool_calls=tool_calls,
        )


def _sse_encode(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _parse_sse_event(chunk: bytes) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    event_name: Optional[str] = None
    data_line: Optional[str] = None
    for line in chunk.decode("utf-8", errors="replace").splitlines():
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_line = line[5:].strip()
    if not data_line:
        return event_name, None
    try:
        payload = json.loads(data_line)
    except json.JSONDecodeError:
        return event_name, None
    if not isinstance(payload, dict):
        return event_name, None
    return event_name, payload


def _is_meaningful_anthropic_event(event_name: Optional[str], payload: Optional[dict[str, Any]]) -> bool:
    if not event_name or not payload:
        return False
    if event_name == "content_block_delta":
        delta = payload.get("delta") or {}
        if not isinstance(delta, dict):
            return False
        if delta.get("type") == "text_delta":
            return bool(str(delta.get("text") or ""))
        if delta.get("type") == "input_json_delta":
            return bool(str(delta.get("partial_json") or ""))
        return False
    if event_name == "content_block_start":
        content_block = payload.get("content_block") or {}
        if not isinstance(content_block, dict):
            return False
        return str(content_block.get("type") or "") == "tool_use"
    return False


def _extract_delta_text(choice: dict[str, Any]) -> str:
    delta = choice.get("delta", {})
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "output_text"}:
                    parts.append(str(item.get("text", "")))
        return "".join(parts)
    if "text" in choice:
        return str(choice.get("text") or "")
    return ""


async def translate_openai_stream_to_anthropic(
    upstream_stream: AsyncIterator[bytes],
    model_id: str,
    source_api: str = "anthropic_messages",
    decision: Optional[RouteDecision] = None,
    final_alias: str = "unknown",
    final_model_id: str = "unknown",
    used_fallback: bool = False,
) -> AsyncIterator[bytes]:
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    usage_input = 0
    usage_output = 0
    stop_reason = "end_turn"
    text_block_index = -1
    next_block_index = 0
    text_block_open = False
    text_delta_events = 0
    tool_blocks_emitted = 0
    tool_call_state: dict[int, dict[str, Any]] = {}
    output_chunks: list[str] = []
    output_text_chars = 0

    logger.info("anthropic_stream_translate_start model=%s", model_id)

    yield _sse_encode(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model_id,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    buffer = ""
    async for chunk in upstream_stream:
        text = chunk.decode("utf-8", errors="replace")
        buffer += text
        while "\n\n" in buffer:
            raw_event, buffer = buffer.split("\n\n", 1)
            data_lines = [line[5:].strip() for line in raw_event.splitlines() if line.startswith("data:")]
            for data_line in data_lines:
                if not data_line or data_line == "[DONE]":
                    continue
                try:
                    parsed = json.loads(data_line)
                except json.JSONDecodeError:
                    continue
                choices = parsed.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta_text = _extract_delta_text(choice)
                if delta_text:
                    remaining = _log_text_max_chars() - output_text_chars
                    if remaining > 0:
                        clipped = delta_text[:remaining]
                        output_chunks.append(clipped)
                        output_text_chars += len(clipped)
                    if not text_block_open:
                        text_block_open = True
                        text_block_index = next_block_index
                        next_block_index += 1
                        yield _sse_encode(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": text_block_index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                    text_delta_events += 1
                    usage_output += _estimate_tokens_from_text(delta_text)
                    yield _sse_encode(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": text_block_index,
                            "delta": {"type": "text_delta", "text": delta_text},
                        },
                    )

                delta = choice.get("delta") or {}
                raw_tool_calls = delta.get("tool_calls") or []
                for tool_delta in raw_tool_calls:
                    if not isinstance(tool_delta, dict):
                        continue
                    index_raw = tool_delta.get("index", 0)
                    try:
                        tool_index = int(index_raw)
                    except (TypeError, ValueError):
                        tool_index = len(tool_call_state)
                    state = tool_call_state.setdefault(
                        tool_index,
                        {
                            "id": "",
                            "name": "",
                            "arguments_parts": [],
                        },
                    )
                    if tool_delta.get("id"):
                        state["id"] = str(tool_delta.get("id"))
                    function_delta = tool_delta.get("function") or {}
                    if isinstance(function_delta, dict):
                        if function_delta.get("name"):
                            state["name"] = str(function_delta.get("name"))
                        arguments_part = function_delta.get("arguments")
                        if isinstance(arguments_part, str) and arguments_part:
                            state["arguments_parts"].append(arguments_part)

                finish_reason = choice.get("finish_reason")
                if finish_reason in {"length", "max_tokens"}:
                    stop_reason = "max_tokens"
                elif finish_reason == "tool_calls":
                    stop_reason = "tool_use"
                usage = parsed.get("usage") or {}
                usage_input = max(usage_input, int(usage.get("prompt_tokens", 0) or 0))
                usage_output = max(usage_output, int(usage.get("completion_tokens", usage_output) or usage_output))

    if text_block_open:
        yield _sse_encode(
            "content_block_stop",
            {
                "type": "content_block_stop",
                "index": text_block_index,
            },
        )

    if tool_call_state:
        for _, state in sorted(tool_call_state.items(), key=lambda item: item[0]):
            tool_name = state.get("name") or "tool"
            tool_id = state.get("id") or f"toolu_{uuid.uuid4().hex[:24]}"
            arguments_str = "".join(state.get("arguments_parts") or []).strip() or "{}"
            try:
                json.loads(arguments_str)
            except json.JSONDecodeError:
                logger.warning(
                    "anthropic_stream_tool_args_invalid tool_id=%s name=%s args=%s",
                    tool_id,
                    tool_name,
                    arguments_str[:200],
                )
                arguments_str = json.dumps({"_raw": arguments_str}, ensure_ascii=False)

            block_index = next_block_index
            next_block_index += 1
            tool_blocks_emitted += 1

            yield _sse_encode(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": {},
                    },
                },
            )
            yield _sse_encode(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "input_json_delta", "partial_json": arguments_str},
                },
            )
            yield _sse_encode(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": block_index,
                },
            )

        if stop_reason == "end_turn":
            stop_reason = "tool_use"

    yield _sse_encode(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"input_tokens": usage_input, "output_tokens": usage_output},
        },
    )
    yield _sse_encode("message_stop", {"type": "message_stop"})
    if stop_reason == "end_turn" and text_delta_events == 0 and tool_blocks_emitted == 0:
        logger.warning("anthropic_stream_empty_assistant_response")
    logger.info(
        "anthropic_stream_translate_done stop_reason=%s text_delta_events=%s tool_blocks=%s usage_input=%s usage_output=%s",
        stop_reason,
        text_delta_events,
        tool_blocks_emitted,
        usage_input,
        usage_output,
    )
    if decision is not None:
        _log_output_analytics(
            source_api=source_api,
            decision=decision,
            final_alias=final_alias,
            final_model_id=final_model_id,
            used_fallback=used_fallback,
            stream=True,
            output_text="".join(output_chunks),
            stop_reason=stop_reason,
            output_tokens=usage_output,
            input_tokens=usage_input,
            tool_calls=tool_blocks_emitted,
        )


def _route_headers(cfg: RouterConfig, decision: RouteDecision, final_alias: str, used_fallback: bool) -> dict[str, str]:
    thinking_applied = (
        decision.thinking_requested
        and not decision.needs_tooluse
        and cfg.models[final_alias].supports_thinking
    )
    return {
        "x-router-selected-model": cfg.models[final_alias].model_id,
        "x-router-judge-model": decision.judge_model_id or "none",
        "x-router-reason": decision.reason,
        "x-router-fallback": "1" if used_fallback else "0",
        "x-router-thinking-requested": "1" if decision.thinking_requested else "0",
        "x-router-thinking-applied": "1" if thinking_applied else "0",
    }


def _log_route_analytics(
    cfg: RouterConfig,
    decision: RouteDecision,
    final_alias: str,
    used_fallback: bool,
) -> None:
    selected_model_id = cfg.models[final_alias].model_id
    routing_efficiency_label, routing_efficiency_score = _routing_efficiency(
        decision.expected_route_class,
        final_alias,
        initial_alias=decision.selected_alias,
        used_fallback=used_fallback,
    )
    payload = {
        "event": "route_analytics",
        "v": 1,
        "request_id": decision.request_id,
        "source": decision.source_api,
        "requested_model": decision.requested_model,
        "initial_alias": decision.selected_alias,
        "selected_alias": final_alias,
        "selected_model": selected_model_id,
        "reason": decision.reason,
        "effective_reason": decision.reason if not used_fallback else f"fallback_from_{decision.reason}",
        "fallback_used": used_fallback,
        "stream": decision.stream,
        "candidate_aliases": decision.candidate_aliases,
        "required_capabilities": decision.required_capabilities,
        "context_signature": decision.context_signature,
        "complexity": decision.complexity,
        "estimated_input_tokens": decision.estimated_input_tokens,
        "estimated_total_tokens": decision.estimated_total_tokens,
        "full_input_tokens": decision.full_input_tokens,
        "full_estimated_total_tokens": decision.full_estimated_total_tokens,
        "routing_input_tokens": decision.routing_input_tokens,
        "routing_estimated_total_tokens": decision.routing_estimated_total_tokens,
        "max_tokens": decision.max_tokens,
        "routing_max_tokens_budget": decision.routing_max_tokens_budget,
        "needs_vision": decision.needs_vision,
        "needs_tooluse": decision.needs_tooluse,
        "is_coding": decision.is_coding_request,
        "has_wrapper_noise": decision.has_wrapper_noise,
        "tool_loop_context": decision.tool_loop_context,
        "repetition_key": decision.repetition_key,
        "prompt_text": decision.prompt_text,
        "user_prompt_text": decision.user_prompt_text,
        "latest_user_prompt_text": decision.latest_user_prompt_text,
        "routing_prompt_text": decision.routing_prompt_text,
        "routing_user_prompt_text": decision.routing_user_prompt_text,
        "routing_latest_user_prompt_text": decision.routing_latest_user_prompt_text,
        "thinking_requested": decision.thinking_requested,
        "thinking_applied": (
            decision.thinking_requested
            and not decision.needs_tooluse
            and cfg.models[final_alias].supports_thinking
        ),
        "expected_route_class": decision.expected_route_class,
        "routing_efficiency_label": routing_efficiency_label,
        "routing_efficiency_score": routing_efficiency_score,
        "latency_ms": _current_request_latency_ms(),
    }
    logger.info("route_analytics %s", json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    if _analytics_store is not None:
        _analytics_store.write_route(
            {
                **payload,
                "prompt_text_hash": _hash_text(decision.prompt_text),
                "user_prompt_text_hash": _hash_text(decision.user_prompt_text),
                "latest_user_text_hash": _hash_text(decision.latest_user_prompt_text),
                "routing_prompt_text_hash": _hash_text(decision.routing_prompt_text),
                "routing_user_text_hash": _hash_text(decision.routing_user_prompt_text),
                "routing_latest_user_text": decision.routing_latest_user_prompt_text,
            }
        )


def _log_output_analytics(
    source_api: str,
    decision: RouteDecision,
    final_alias: str,
    final_model_id: str,
    used_fallback: bool,
    stream: bool,
    output_text: str,
    stop_reason: Optional[str] = None,
    output_tokens: Optional[int] = None,
    input_tokens: Optional[int] = None,
    tool_calls: Optional[int] = None,
) -> None:
    routing_efficiency_label, routing_efficiency_score = _routing_efficiency(
        decision.expected_route_class,
        final_alias,
        initial_alias=decision.selected_alias,
        used_fallback=used_fallback,
        stop_reason=stop_reason,
    )
    payload: dict[str, Any] = {
        "event": "output_analytics",
        "v": 1,
        "request_id": decision.request_id,
        "source": source_api,
        "initial_alias": decision.selected_alias,
        "selected_alias": final_alias,
        "selected_model": final_model_id,
        "reason": decision.reason,
        "fallback_used": used_fallback,
        "stream": stream,
        "stop_reason": stop_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tool_calls": tool_calls,
        "output_text_chars": len(output_text or ""),
        "output_text_excerpt": _clip_for_log(output_text),
        "routing_efficiency_label": routing_efficiency_label,
        "routing_efficiency_score": routing_efficiency_score,
        "latency_ms": _current_request_latency_ms(),
    }
    logger.info("output_analytics %s", json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    if _analytics_store is not None:
        _analytics_store.write_output(payload)


def _build_models_response(cfg: RouterConfig) -> dict[str, Any]:
    models = [
        {
            "id": cfg.router_identity.exposed_model_name,
            "object": "model",
            "created": 0,
            "owned_by": "llm-router",
        }
    ]
    if cfg.router_identity.publish_underlying_models:
        for alias, profile in cfg.models.items():
            models.append(
                {
                    "id": profile.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": f"llm-router-{alias}",
                }
            )
    return {"object": "list", "data": models}
