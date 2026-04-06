from __future__ import annotations

from .shared import *
from .shared import (
    _estimate_tokens_from_text,
    _extract_text_and_vision,
    _extract_text_fragments,
    _sanitize_routing_text,
)

class UnifiedRequest(BaseModel):
    source_api: str
    requested_model: Optional[str] = None
    stream: bool = False
    max_tokens: Optional[int] = None
    prompt_text: str = ""
    user_prompt_text: str = ""
    latest_user_prompt_text: str = ""
    estimated_input_tokens: int = 1
    routing_prompt_text: str = ""
    routing_user_prompt_text: str = ""
    routing_latest_user_prompt_text: str = ""
    full_input_tokens: int = 1
    routing_input_tokens: int = 1
    routing_max_tokens_budget: Optional[int] = None
    needs_vision: bool = False
    needs_tooluse: bool = False
    has_wrapper_noise: bool = False
    tool_loop_context: bool = False
    required_base_capability: str

    @model_validator(mode="after")
    def _normalize_routing_fields(self) -> "UnifiedRequest":
        if not self.routing_prompt_text:
            self.routing_prompt_text = self.prompt_text
        if not self.routing_user_prompt_text:
            self.routing_user_prompt_text = self.user_prompt_text
        if not self.routing_latest_user_prompt_text:
            self.routing_latest_user_prompt_text = (
                self.latest_user_prompt_text or self.routing_user_prompt_text or self.routing_prompt_text
            )
        if self.full_input_tokens <= 0 or (self.full_input_tokens == 1 and self.estimated_input_tokens != 1):
            self.full_input_tokens = max(1, self.estimated_input_tokens)
        if self.routing_input_tokens <= 0 or (self.routing_input_tokens == 1 and self.full_input_tokens != 1):
            self.routing_input_tokens = max(1, self.full_input_tokens)
        if self.estimated_input_tokens <= 0:
            self.estimated_input_tokens = self.full_input_tokens
        return self

    @property
    def required_capabilities(self) -> set[str]:
        base = {self.required_base_capability}
        if self.needs_vision:
            base.add("vision")
        if self.needs_tooluse:
            base.add("tooluse")
        return base

    @property
    def estimated_total_tokens(self) -> int:
        target = self.max_tokens if self.max_tokens is not None else 1024
        return self.estimated_input_tokens + max(1, target)

    @property
    def full_estimated_total_tokens(self) -> int:
        target = self.max_tokens if self.max_tokens is not None else 1024
        return self.full_input_tokens + max(1, target)

    @property
    def effective_routing_max_tokens_budget(self) -> int:
        target = self.max_tokens if self.max_tokens is not None else 1024
        if self.routing_max_tokens_budget is None:
            return max(1, target)
        return max(1, min(target, self.routing_max_tokens_budget))

    @property
    def routing_estimated_total_tokens(self) -> int:
        return self.routing_input_tokens + self.effective_routing_max_tokens_budget


def normalize_openai_chat(payload: dict[str, Any]) -> UnifiedRequest:
    messages = payload.get("messages") or []
    prompt_chunks: list[str] = []
    user_chunks: list[str] = []
    needs_vision = False
    needs_tooluse = bool(payload.get("tools"))
    for msg in messages:
        content = msg.get("content")
        text, msg_vision = _extract_text_and_vision(content)
        if text:
            prompt_chunks.append(text)
            if str(msg.get("role", "")).lower() == "user":
                user_chunks.append(text)
        needs_vision = needs_vision or msg_vision
        if msg.get("tool_calls"):
            needs_tooluse = True
    prompt_text = "\n".join(prompt_chunks)
    user_prompt_text = "\n".join(user_chunks)
    latest_user_prompt_text = user_chunks[-1] if user_chunks else ""
    max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")
    return UnifiedRequest(
        source_api="openai_chat",
        requested_model=payload.get("model"),
        stream=bool(payload.get("stream")),
        max_tokens=max_tokens,
        prompt_text=prompt_text,
        user_prompt_text=user_prompt_text,
        latest_user_prompt_text=latest_user_prompt_text,
        estimated_input_tokens=_estimate_tokens_from_text(prompt_text),
        routing_prompt_text=prompt_text,
        routing_user_prompt_text=user_prompt_text,
        routing_latest_user_prompt_text=latest_user_prompt_text,
        full_input_tokens=_estimate_tokens_from_text(prompt_text),
        routing_input_tokens=_estimate_tokens_from_text(prompt_text),
        needs_vision=needs_vision,
        needs_tooluse=needs_tooluse,
        required_base_capability="chat",
    )


def normalize_openai_completion(payload: dict[str, Any]) -> UnifiedRequest:
    prompt = payload.get("prompt", "")
    if isinstance(prompt, list):
        prompt_text = "\n".join(str(x) for x in prompt)
    else:
        prompt_text = str(prompt)
    return UnifiedRequest(
        source_api="openai_completions",
        requested_model=payload.get("model"),
        stream=bool(payload.get("stream")),
        max_tokens=payload.get("max_tokens"),
        prompt_text=prompt_text,
        user_prompt_text=prompt_text,
        latest_user_prompt_text=prompt_text,
        estimated_input_tokens=_estimate_tokens_from_text(prompt_text),
        routing_prompt_text=prompt_text,
        routing_user_prompt_text=prompt_text,
        routing_latest_user_prompt_text=prompt_text,
        full_input_tokens=_estimate_tokens_from_text(prompt_text),
        routing_input_tokens=_estimate_tokens_from_text(prompt_text),
        needs_vision=False,
        needs_tooluse=False,
        required_base_capability="completions",
    )


def normalize_anthropic_messages(payload: dict[str, Any]) -> UnifiedRequest:
    messages = payload.get("messages") or []
    system = payload.get("system")
    prompt_chunks: list[str] = []
    user_chunks: list[str] = []
    routing_prompt_chunks: list[str] = []
    routing_user_chunks: list[str] = []
    needs_vision = False
    needs_tooluse = bool(payload.get("tools"))
    has_wrapper_noise = False
    tool_loop_context = False
    if system:
        if isinstance(system, str):
            prompt_chunks.append(system)
            _, wrapper_noise = _sanitize_routing_text(system)
            has_wrapper_noise = has_wrapper_noise or wrapper_noise
        elif isinstance(system, list):
            for item in system:
                fragments, is_vision, has_tool_loop_content = _extract_text_fragments(item)
                if fragments:
                    full_text = " ".join(fragments)
                    prompt_chunks.append(full_text)
                    _, wrapper_noise = _sanitize_routing_text(full_text)
                    has_wrapper_noise = has_wrapper_noise or wrapper_noise
                needs_vision = needs_vision or is_vision
                tool_loop_context = tool_loop_context or has_tool_loop_content
    for msg in messages:
        fragments, msg_vision, has_tool_loop_content = _extract_text_fragments(msg.get("content"))
        text = " ".join(fragments)
        role = str(msg.get("role", "")).lower()
        if text:
            prompt_chunks.append(text)
            sanitized, wrapper_noise = _sanitize_routing_text(text)
            has_wrapper_noise = has_wrapper_noise or wrapper_noise
            if role == "user":
                user_chunks.append(text)
                if sanitized:
                    routing_prompt_chunks.append(sanitized)
                    routing_user_chunks.append(sanitized)
        needs_vision = needs_vision or msg_vision
        tool_loop_context = tool_loop_context or has_tool_loop_content
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"tool_use", "tool_result"}:
                    needs_tooluse = True
                    tool_loop_context = True
    prompt_text = "\n".join(prompt_chunks)
    user_prompt_text = "\n".join(user_chunks)
    latest_user_prompt_text = user_chunks[-1] if user_chunks else ""
    routing_user_prompt_text = "\n".join(routing_user_chunks) or user_prompt_text
    routing_prompt_text = "\n".join(routing_prompt_chunks) or routing_user_prompt_text or latest_user_prompt_text or prompt_text
    routing_latest_user_prompt_text = routing_user_chunks[-1] if routing_user_chunks else (
        routing_user_prompt_text or latest_user_prompt_text
    )
    full_input_tokens = _estimate_tokens_from_text(prompt_text)
    routing_input_tokens = _estimate_tokens_from_text(routing_prompt_text)
    return UnifiedRequest(
        source_api="anthropic_messages",
        requested_model=payload.get("model"),
        stream=bool(payload.get("stream")),
        max_tokens=payload.get("max_tokens"),
        prompt_text=prompt_text,
        user_prompt_text=user_prompt_text,
        latest_user_prompt_text=latest_user_prompt_text,
        estimated_input_tokens=full_input_tokens,
        routing_prompt_text=routing_prompt_text,
        routing_user_prompt_text=routing_user_prompt_text,
        routing_latest_user_prompt_text=routing_latest_user_prompt_text,
        full_input_tokens=full_input_tokens,
        routing_input_tokens=routing_input_tokens,
        needs_vision=needs_vision,
        needs_tooluse=needs_tooluse,
        has_wrapper_noise=has_wrapper_noise,
        tool_loop_context=tool_loop_context,
        required_base_capability="chat",
    )
@dataclass
class RouteDecision:
    selected_alias: str
    reason: str
    candidate_aliases: list[str]
    request_id: str = "-"
    thinking_requested: bool = False
    is_commit_message_task: bool = False
    judge_model_id: Optional[str] = None
    is_coding_request: bool = False
    source_api: str = "unknown"
    requested_model: Optional[str] = None
    stream: bool = False
    required_capabilities: Optional[list[str]] = None
    estimated_input_tokens: int = 1
    estimated_total_tokens: int = 1
    full_input_tokens: int = 1
    full_estimated_total_tokens: int = 1
    routing_input_tokens: int = 1
    routing_estimated_total_tokens: int = 1
    max_tokens: Optional[int] = None
    routing_max_tokens_budget: Optional[int] = None
    needs_vision: bool = False
    needs_tooluse: bool = False
    has_wrapper_noise: bool = False
    tool_loop_context: bool = False
    complexity: str = "low"
    context_signature: str = "none"
    repetition_key: str = "none"
    prompt_text: str = ""
    user_prompt_text: str = ""
    latest_user_prompt_text: str = ""
    routing_prompt_text: str = ""
    routing_user_prompt_text: str = ""
    routing_latest_user_prompt_text: str = ""
    expected_route_class: str = "small"
    routing_efficiency_label: str = "good_fit"
    routing_efficiency_score: int = 100

    def __post_init__(self) -> None:
        if self.required_capabilities is None:
            self.required_capabilities = []
