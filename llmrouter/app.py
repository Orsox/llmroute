import asyncio
import argparse
import contextvars
import json
import hashlib
import logging
import math
import os
import re
import threading
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, Literal, Optional

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator


_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get()
        return True


def _configure_logging() -> logging.Logger:
    app_logger = logging.getLogger("llm-router")
    if app_logger.handlers:
        return app_logger

    level_name = os.getenv("ROUTER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file_path = Path(os.getenv("ROUTER_LOG_FILE", "logs/router.log"))
    max_bytes = int(os.getenv("ROUTER_LOG_MAX_BYTES", str(5 * 1024 * 1024)))
    backup_count = int(os.getenv("ROUTER_LOG_BACKUP_COUNT", "3"))
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s [req=%(request_id)s] %(message)s")
    req_filter = _RequestIdFilter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(req_filter)

    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(req_filter)

    app_logger.setLevel(level)
    app_logger.addHandler(stream_handler)
    app_logger.addHandler(file_handler)
    app_logger.propagate = False
    return app_logger


logger = _configure_logging()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


DEFAULT_CONFIG_PATH = Path(os.getenv("ROUTER_CONFIG_PATH", "config/router_config.yaml"))
WINDOWS_STARTUP_REG_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
WINDOWS_STARTUP_VALUE_NAME = "LMRouterTray"
START_SCRIPT_RELATIVE_PATH = Path("scripts/start_llm_router.ps1")
CODING_SYNTAX_RE = re.compile(
    r"```|^\s*(def|class|function|import|from|select|insert|update|delete)\b|traceback|stack\s*trace",
    re.IGNORECASE | re.MULTILINE,
)
CODING_TOPIC_RE = re.compile(
    r"\b("
    r"code|coding|programmieren|programming|python|javascript|typescript|java|c\+\+|c#|sql|regex|"
    r"debug|bug|refactor|unit\s*test|pytest|compile|compiler|script|funktion|klasse|"
    r"docker|git|bash|powershell|html|css|react|vue|angular|node|npm|pip|fastapi|flask|"
    r"django|rust|golang|kotlin|swift|php|ruby"
    r")\b",
    re.IGNORECASE,
)
FILE_SEARCH_RE = re.compile(
    r"\b("
    r"datei|dateien|file|files|file search|find file|find files|"
    r"suche nach dateien|dateisuche|datei suchen|"
    r"pfad|paths?|verzeichnis|ordner|directory|directories|"
    r"grep|ripgrep|rg|glob|filename|filenames"
    r")\b",
    re.IGNORECASE,
)
COMMIT_MESSAGE_TASK_RE = re.compile(
    r"\b("
    r"commit message|git commit|commit-msg|conventional commit|"
    r"commit titel|commit title|commit text|"
    r"schreibe.*commit|generiere.*commit"
    r")\b",
    re.IGNORECASE,
)
NO_THINKING_TASK_RE = re.compile(
    r"\b("
    r"commit message|git commit|commit-msg|"
    r"changelog|release notes?|"
    r"pull request title|pr title|pr description|merge request|"
    r"zusammenfassen|summary|summarize|"
    r"kurzfassung|kurz zusammenfassen"
    r")\b",
    re.IGNORECASE,
)
DEEP_REASONING_RE = re.compile(
    r"\b("
    r"architecture|architektur|trade[- ]?off|abw[aä]gung|compliance|policy|regelwerk|"
    r"risk|risiko|threat model|bedrohungsmodell|entscheidungsmatrix|grundsatz|"
    r"designentscheidung|governance|multi[- ]step|mehrstufig|reasoning|"
    r"fehleranalyse|root cause|ursachenanalyse|sicherheitskonzept|"
    r"migrationsstrategie|rollout|validierungsstrategie"
    r")\b",
    re.IGNORECASE,
)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


class LMStudioSettings(BaseModel):
    provider: Literal["lm_studio", "openai"] = "lm_studio"
    base_url: str = "http://127.0.0.1:1234"
    timeout_seconds: float = 90.0
    api_key: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    organization: Optional[str] = None
    project: Optional[str] = None

    def resolve_api_key(self) -> Optional[str]:
        direct = (self.api_key or "").strip()
        if direct:
            return direct
        env_name = (self.api_key_env or "").strip()
        if not env_name:
            return None
        env_value = os.getenv(env_name, "").strip()
        return env_value or None


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 12345


class SecuritySettings(BaseModel):
    shared_bearer_token: Optional[str] = None


class HeuristicSettings(BaseModel):
    large_prompt_token_threshold: int = 2200
    large_max_tokens_threshold: int = 1800
    judge_temperature: float = 0.0
    judge_max_tokens: int = 96
    judge_prompt_context_chars: int = 6000


class RoutingSettings(BaseModel):
    judge_timeout_seconds: float = 15.0
    fallback_enabled: bool = True
    hybrid_client_model_override: bool = True
    heuristics: HeuristicSettings = Field(default_factory=HeuristicSettings)


class RouterIdentitySettings(BaseModel):
    exposed_model_name: str = "borg-cpu"
    publish_underlying_models: bool = False


class ModelProfile(BaseModel):
    model_id: str
    context_window: int
    capabilities: list[str]
    upstream_ref: str = "local"
    supports_thinking: bool = False
    relative_speed: float = 1.0
    suitable_for: str = ""

    def has_capabilities(self, required: set[str]) -> bool:
        return required.issubset(set(self.capabilities))


class RouterConfig(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    # Backward compatibility: legacy single-upstream key.
    lm_studio: Optional[LMStudioSettings] = None
    upstreams: Dict[str, LMStudioSettings] = Field(default_factory=dict)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    router_identity: RouterIdentitySettings = Field(default_factory=RouterIdentitySettings)
    models: Dict[str, ModelProfile]

    @model_validator(mode="after")
    def _normalize_legacy_upstreams(self) -> "RouterConfig":
        if not self.upstreams:
            self.upstreams = {"local": self.lm_studio or LMStudioSettings()}
        elif self.lm_studio is None and "local" in self.upstreams:
            self.lm_studio = self.upstreams["local"]
        return self

    def default_upstream(self) -> LMStudioSettings:
        if "local" in self.upstreams:
            return self.upstreams["local"]
        return next(iter(self.upstreams.values()))

    def upstream_for_alias(self, alias: str) -> LMStudioSettings:
        profile = self.models.get(alias)
        if profile is None:
            raise ValueError(f"Unknown model alias: {alias}")
        upstream_ref = (profile.upstream_ref or "").strip() or "local"
        settings = self.upstreams.get(upstream_ref)
        if settings is None:
            raise ValueError(f"Unknown upstream_ref '{upstream_ref}' for alias '{alias}'")
        return settings


class WindowsStartupToggleRequest(BaseModel):
    enabled: bool


class ConfigStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._config = self._load_from_disk()

    def _load_from_disk(self) -> RouterConfig:
        if not self.path.exists():
            default = _default_config()
            self._write_yaml_atomic(yaml.safe_dump(default, sort_keys=False, allow_unicode=False))
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        cfg = RouterConfig.model_validate(raw)
        self._validate_required_model_aliases(cfg)
        return cfg

    @staticmethod
    def _validate_required_model_aliases(cfg: RouterConfig) -> None:
        required = {"small", "large", "deep"}
        missing = required.difference(cfg.models.keys())
        if missing:
            missing_s = ", ".join(sorted(missing))
            raise ValueError(f"Missing required model aliases in config: {missing_s}")
        upstreams = set(cfg.upstreams.keys())
        for alias, profile in cfg.models.items():
            upstream_ref = (profile.upstream_ref or "").strip() or "local"
            if upstream_ref not in upstreams:
                raise ValueError(f"Model alias '{alias}' references unknown upstream '{upstream_ref}'")

    def get_config(self) -> RouterConfig:
        return self._config

    def get_yaml(self) -> str:
        data = self._config.model_dump(mode="python")
        data.pop("lm_studio", None)
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=False)

    async def update_from_yaml(self, yaml_text: str) -> RouterConfig:
        async with self._lock:
            parsed = yaml.safe_load(yaml_text) or {}
            cfg = RouterConfig.model_validate(parsed)
            self._validate_required_model_aliases(cfg)
            self._write_yaml_atomic(yaml.safe_dump(parsed, sort_keys=False, allow_unicode=False))
            self._config = cfg
            return cfg

    def _write_yaml_atomic(self, data: str) -> None:
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(data, encoding="utf-8")
        os.replace(temp_path, self.path)


def _default_config() -> dict[str, Any]:
    deep_base_url = os.getenv("DEEP_BASE_URL", "https://api.openai.com").strip() or "https://api.openai.com"
    deep_model_id = os.getenv("DEEP_MODEL_ID", "gpt-5.4-mini").strip() or "gpt-5.4-mini"
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 12345,
        },
        "upstreams": {
            "local": {
                "provider": "lm_studio",
                "base_url": "http://192.168.178.2:1234",
                "timeout_seconds": 120,
                "api_key": None,
                "api_key_env": "OPENAI_API_KEY",
                "organization": None,
                "project": None,
            },
            "deep": {
                "provider": "openai",
                "base_url": deep_base_url,
                "timeout_seconds": 180,
                "api_key": None,
                "api_key_env": "DEEP_API_KEY",
                "organization": None,
                "project": None,
            },
        },
        "security": {
            "shared_bearer_token": None,
        },
        "routing": {
            "judge_timeout_seconds": 15,
            "fallback_enabled": True,
            "hybrid_client_model_override": True,
            "heuristics": {
                "large_prompt_token_threshold": 2200,
                "large_max_tokens_threshold": 1800,
                "judge_temperature": 0.0,
                "judge_max_tokens": 96,
                "judge_prompt_context_chars": 6000,
            },
        },
        "router_identity": {
            "exposed_model_name": "borg-cpu",
            "publish_underlying_models": False,
        },
        "models": {
            "small": {
                "model_id": "qwen/qwen3-vl-8b",
                "context_window": 32996,
                "capabilities": ["chat", "completions", "vision", "tooluse"],
                "upstream_ref": "local",
                "supports_thinking": True,
                "relative_speed": 3.0,
                "suitable_for": "Fast routing judge, low latency chat, multimodal light tasks.",
            },
            "large": {
                "model_id": "qwen/qwen3.5-35b-a3b",
                "context_window": 262144,
                "capabilities": ["chat", "completions", "tooluse"],
                "upstream_ref": "local",
                "supports_thinking": False,
                "relative_speed": 1.0,
                "suitable_for": "Higher complexity reasoning and long-context workloads.",
            },
            "deep": {
                "model_id": deep_model_id,
                "context_window": 400000,
                "capabilities": ["chat", "completions", "tooluse"],
                "upstream_ref": "deep",
                "supports_thinking": True,
                "relative_speed": 0.5,
                "suitable_for": "High-stakes reasoning and strict rule/compliance tasks.",
            },
        },
    }


def _payload_summary(payload: dict[str, Any]) -> str:
    model = payload.get("model")
    stream = bool(payload.get("stream"))
    max_tokens = payload.get("max_tokens", payload.get("max_completion_tokens"))
    messages = payload.get("messages")
    msg_count = len(messages) if isinstance(messages, list) else 0
    has_tools = bool(payload.get("tools"))
    has_prompt = "prompt" in payload
    has_thinking = "thinking" in payload
    has_reasoning = "reasoning" in payload
    reasoning = payload.get("reasoning")
    reasoning_effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    return (
        f"model={model!r} stream={stream} max_tokens={max_tokens} "
        f"messages={msg_count} tools={int(has_tools)} prompt={int(has_prompt)} "
        f"thinking={int(has_thinking)} reasoning={int(has_reasoning)} reasoning_effort={reasoning_effort!r}"
    )


DEFAULT_TOOLUSE_SYSTEM_HINT = (
    "When tools are available, never return an empty assistant response. "
    "If you need external data or an action, call the appropriate tool. "
    "If no tool is needed, return a concise direct answer."
)
THINKING_DEBUG_ENV = "ROUTER_DEBUG_THINKING"


def _thinking_debug_enabled() -> bool:
    return _env_flag(THINKING_DEBUG_ENV, default=False)


def _thinking_payload_probe(payload: dict[str, Any]) -> str:
    def _safe_get(container: Any, key: str) -> Any:
        if isinstance(container, dict):
            return container.get(key)
        return None

    reasoning = payload.get("reasoning")
    chat_kwargs = payload.get("chat_template_kwargs")
    extra_body = payload.get("extra_body")
    options = payload.get("options")
    probe = {
        "keys": sorted([k for k in payload.keys() if k in {"thinking", "reasoning", "chat_template_kwargs", "extra_body", "options"}]),
        "thinking": payload.get("thinking"),
        "reasoning_effort": reasoning.get("effort") if isinstance(reasoning, dict) else None,
        "chat_template_enable_thinking": _safe_get(chat_kwargs, "enable_thinking"),
        "extra_body_thinking": _safe_get(extra_body, "thinking"),
        "extra_body_reasoning": _safe_get(extra_body, "reasoning"),
        "options_thinking": _safe_get(options, "thinking"),
    }
    return json.dumps(probe, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _stream_chunk_thinking_hint(chunk: bytes) -> str:
    text = chunk.decode("utf-8", errors="ignore").lower()[:600]
    markers: list[str] = []
    if "<think" in text:
        markers.append("xml_think_tag")
    if "reasoning_content" in text:
        markers.append("reasoning_content_field")
    if '"reasoning"' in text:
        markers.append("reasoning_field")
    if '"thinking"' in text:
        markers.append("thinking_field")
    return ",".join(markers) if markers else "none"


def _extract_text_and_vision(content: Any) -> tuple[str, bool]:
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, False
    if isinstance(content, list):
        chunks: list[str] = []
        has_vision = False
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if not isinstance(item, dict):
                chunks.append(str(item))
                continue
            item_type = item.get("type")
            if item_type in {"image", "image_url", "input_image"}:
                has_vision = True
            elif item_type in {"text", "input_text"}:
                chunks.append(str(item.get("text", "")))
            elif item_type == "tool_result":
                nested = item.get("content")
                if isinstance(nested, str):
                    chunks.append(nested)
            else:
                text = item.get("text")
                if text:
                    chunks.append(str(text))
        return " ".join(x for x in chunks if x), has_vision
    return str(content), False


def _estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 1
    return max(1, math.ceil(len(text) / 4))


def _extract_assistant_text(openai_response: dict[str, Any]) -> str:
    choices = openai_response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content")
    text, _ = _extract_text_and_vision(content)
    if text:
        return text
    completion_text = choices[0].get("text")
    return completion_text or ""


class UnifiedRequest(BaseModel):
    source_api: str
    requested_model: Optional[str] = None
    stream: bool = False
    max_tokens: Optional[int] = None
    prompt_text: str = ""
    user_prompt_text: str = ""
    latest_user_prompt_text: str = ""
    estimated_input_tokens: int = 1
    needs_vision: bool = False
    needs_tooluse: bool = False
    required_base_capability: str

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
        needs_vision=False,
        needs_tooluse=False,
        required_base_capability="completions",
    )


def normalize_anthropic_messages(payload: dict[str, Any]) -> UnifiedRequest:
    messages = payload.get("messages") or []
    system = payload.get("system")
    prompt_chunks: list[str] = []
    user_chunks: list[str] = []
    needs_vision = False
    needs_tooluse = bool(payload.get("tools"))
    if system:
        if isinstance(system, str):
            prompt_chunks.append(system)
        elif isinstance(system, list):
            for item in system:
                text, is_vision = _extract_text_and_vision(item)
                if text:
                    prompt_chunks.append(text)
                needs_vision = needs_vision or is_vision
    for msg in messages:
        text, msg_vision = _extract_text_and_vision(msg.get("content"))
        if text:
            prompt_chunks.append(text)
            if str(msg.get("role", "")).lower() == "user":
                user_chunks.append(text)
        needs_vision = needs_vision or msg_vision
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"tool_use", "tool_result"}:
                    needs_tooluse = True
    prompt_text = "\n".join(prompt_chunks)
    user_prompt_text = "\n".join(user_chunks)
    latest_user_prompt_text = user_chunks[-1] if user_chunks else ""
    return UnifiedRequest(
        source_api="anthropic_messages",
        requested_model=payload.get("model"),
        stream=bool(payload.get("stream")),
        max_tokens=payload.get("max_tokens"),
        prompt_text=prompt_text,
        user_prompt_text=user_prompt_text,
        latest_user_prompt_text=latest_user_prompt_text,
        estimated_input_tokens=_estimate_tokens_from_text(prompt_text),
        needs_vision=needs_vision,
        needs_tooluse=needs_tooluse,
        required_base_capability="chat",
    )

@dataclass
class RouteDecision:
    selected_alias: str
    reason: str
    candidate_aliases: list[str]
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
    max_tokens: Optional[int] = None
    needs_vision: bool = False
    needs_tooluse: bool = False
    complexity: str = "low"
    context_signature: str = "none"
    repetition_key: str = "none"

    def __post_init__(self) -> None:
        if self.required_capabilities is None:
            self.required_capabilities = []


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

    async def post_json(
        self,
        settings: LMStudioSettings,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        url = settings.base_url.rstrip("/") + path
        headers = self._upstream_headers(settings)
        timeout = httpx.Timeout(settings.timeout_seconds)
        start = time.perf_counter()
        logger.info("upstream_post_start provider=%s path=%s %s", settings.provider, path, _payload_summary(payload))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload, headers=headers or None)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                if response.status_code >= 400:
                    logger.warning(
                        "upstream_post_failed path=%s status=%s duration_ms=%s body=%s",
                        path,
                        response.status_code,
                        elapsed_ms,
                        response.text[:300],
                    )
                    raise UpstreamError(response.status_code, response.text)
                logger.info(
                    "upstream_post_ok path=%s status=%s duration_ms=%s",
                    path,
                    response.status_code,
                    elapsed_ms,
                )
                try:
                    return response.json()
                except ValueError as exc:
                    body = response.text[:300]
                    logger.warning("upstream_post_invalid_json path=%s body=%s", path, body)
                    raise UpstreamError(502, f"Invalid JSON from upstream: {body}") from exc
        except httpx.TimeoutException as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "upstream_post_timeout path=%s duration_ms=%s timeout_s=%s error=%s",
                path,
                elapsed_ms,
                settings.timeout_seconds,
                exc,
            )
            raise UpstreamError(504, f"Upstream timeout after {settings.timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("upstream_post_http_error path=%s duration_ms=%s error=%s", path, elapsed_ms, exc)
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
            candidate_paths = ["/api/v0/models", "/v1/models"]
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

    async def stream_openai(
        self,
        settings: LMStudioSettings,
        path: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        url = settings.base_url.rstrip("/") + path
        headers = self._upstream_headers(settings)
        timeout = httpx.Timeout(settings.timeout_seconds)
        start = time.perf_counter()
        logger.info(
            "upstream_stream_start provider=%s path=%s %s",
            settings.provider,
            path,
            _payload_summary(payload),
        )
        chunk_count = 0
        byte_count = 0
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload, headers=headers or None) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", errors="replace")
                        elapsed_ms = int((time.perf_counter() - start) * 1000)
                        logger.warning(
                            "upstream_stream_failed path=%s status=%s duration_ms=%s body=%s",
                            path,
                            response.status_code,
                            elapsed_ms,
                            body[:300],
                        )
                        raise UpstreamError(response.status_code, body)

                    async for chunk in response.aiter_bytes():
                        if chunk:
                            chunk_count += 1
                            byte_count += len(chunk)
                            if chunk_count == 1:
                                logger.info("upstream_stream_first_chunk path=%s first_chunk_bytes=%s", path, len(chunk))
                            yield chunk

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "upstream_stream_done path=%s chunks=%s bytes=%s duration_ms=%s",
                path,
                chunk_count,
                byte_count,
                elapsed_ms,
            )
        except httpx.TimeoutException as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning(
                "upstream_stream_timeout path=%s duration_ms=%s timeout_s=%s error=%s",
                path,
                elapsed_ms,
                settings.timeout_seconds,
                exc,
            )
            raise UpstreamError(504, f"Upstream stream timeout after {settings.timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.warning("upstream_stream_http_error path=%s duration_ms=%s error=%s", path, elapsed_ms, exc)
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
        for key in ("id", "model_id", "model", "name"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _normalize_model_id(value: str) -> str:
        return value.strip().lower()

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
                if normalized in {"loaded", "ready", "running", "active", "available"}:
                    return True
                if normalized in {"unloaded", "not_loaded", "stopped", "inactive", "error", "failed"}:
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
                        try:
                            path, items = await list_models_fn(upstream_settings)
                            upstream_catalogs[upstream_ref] = {"path": path, "items": items, "error": None}
                        except Exception as upstream_exc:  # noqa: BLE001
                            upstream_catalogs[upstream_ref] = {
                                "path": None,
                                "items": [],
                                "error": str(upstream_exc),
                            }
                        upstream_status.append(
                            {
                                "upstream_ref": upstream_ref,
                                "provider": upstream_settings.provider,
                                "base_url": upstream_settings.base_url,
                                "catalog_path": upstream_catalogs[upstream_ref]["path"],
                                "error": upstream_catalogs[upstream_ref]["error"],
                            }
                        )

                    expected_models = [(alias, profile.model_id) for alias, profile in cfg.models.items()]
                    for alias, expected_model_id in expected_models:
                        profile = cfg.models[alias]
                        upstream_ref = (profile.upstream_ref or "").strip() or "local"
                        upstream_data = upstream_catalogs.get(upstream_ref, {"items": [], "error": "unknown_upstream_ref"})
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
                                "model_id": expected_model_id,
                                "upstream_ref": upstream_ref,
                                "matched_upstream_id": matched_id or None,
                                "available": available,
                                "loaded": loaded,
                                "loaded_inferred": loaded_inferred,
                                "upstream_error": upstream_data.get("error"),
                            }
                        )

                    all_available = bool(models_status) and all(item["available"] for item in models_status)
                    all_loaded = bool(models_status) and all(item["loaded"] for item in models_status)
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
                    logger.warning("model_availability_check_failed error=%s", exc)

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


class RouterService:
    def __init__(self, config_store: ConfigStore, lm_client: Optional[LMStudioClient] = None):
        self.config_store = config_store
        self.lm_client = lm_client or LMStudioClient()

    @staticmethod
    def _is_deep_reasoning_request(req: UnifiedRequest) -> bool:
        text = (req.latest_user_prompt_text or req.user_prompt_text or req.prompt_text or "").strip()
        if not text:
            return False
        return bool(DEEP_REASONING_RE.search(text))

    @staticmethod
    def _is_deep_enabled(cfg: RouterConfig) -> bool:
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
    def _complexity_bucket(req: UnifiedRequest, is_coding: bool) -> str:
        total_tokens = req.estimated_total_tokens
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
        base = (req.latest_user_prompt_text or req.user_prompt_text or req.prompt_text or "").strip().lower()
        normalized = re.sub(r"\s+", " ", base)[:2000]
        material = f"{req.source_api}|{req.required_base_capability}|{normalized}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def _make_route_decision(
        self,
        req: UnifiedRequest,
        selected_alias: str,
        reason: str,
        candidates: list[str],
        thinking_requested: bool,
        judge_model_id: Optional[str],
        is_coding: bool,
    ) -> RouteDecision:
        return RouteDecision(
            selected_alias=selected_alias,
            reason=reason,
            candidate_aliases=candidates,
            thinking_requested=thinking_requested,
            is_commit_message_task=self._is_commit_message_task(req),
            judge_model_id=judge_model_id,
            is_coding_request=is_coding,
            source_api=req.source_api,
            requested_model=req.requested_model,
            stream=req.stream,
            required_capabilities=sorted(req.required_capabilities),
            estimated_input_tokens=req.estimated_input_tokens,
            estimated_total_tokens=req.estimated_total_tokens,
            max_tokens=req.max_tokens,
            needs_vision=req.needs_vision,
            needs_tooluse=req.needs_tooluse,
            complexity=self._complexity_bucket(req, is_coding),
            context_signature=self._context_signature(req, is_coding),
            repetition_key=self._repetition_key(req),
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
    def _normalize_thinking_param(
        settings: LMStudioSettings,
        path: str,
        payload: dict[str, Any],
        thinking_enabled: bool,
    ) -> dict[str, Any]:
        normalized = dict(payload)
        if path != "/v1/chat/completions":
            if not thinking_enabled:
                normalized.pop("reasoning", None)
                normalized.pop("thinking", None)
            return normalized

        if not thinking_enabled:
            normalized.pop("reasoning", None)
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

        hint = (
            "You are writing a git commit message. "
            "Return only the final commit message text, concise, no explanation."
        )
        messages = normalized.get("messages")
        if isinstance(messages, list):
            if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
                current = str(messages[0].get("content") or "")
                if hint not in current:
                    sep = "\n\n" if current else ""
                    messages[0]["content"] = f"{current}{sep}{hint}"
            else:
                messages.insert(0, {"role": "system", "content": hint})
        return normalized

    def _eligible_aliases(self, cfg: RouterConfig, req: UnifiedRequest) -> list[str]:
        required = req.required_capabilities
        total_tokens = req.estimated_total_tokens
        aliases: list[str] = []
        for alias, profile in cfg.models.items():
            if alias == "deep" and not self._is_deep_enabled(cfg):
                continue
            if profile.has_capabilities(required) and profile.context_window >= total_tokens:
                aliases.append(alias)
        return aliases

    def _find_alias_by_model_id(self, cfg: RouterConfig, model_id: Optional[str]) -> Optional[str]:
        if not model_id:
            return None
        for alias, profile in cfg.models.items():
            if profile.model_id == model_id:
                return alias
        return None

    @staticmethod
    def _is_router_public_model_name(cfg: RouterConfig, requested_model: Optional[str]) -> bool:
        if not requested_model:
            return False
        return requested_model.strip() == cfg.router_identity.exposed_model_name.strip()

    @staticmethod
    def _is_coding_request(req: UnifiedRequest) -> bool:
        text = (req.latest_user_prompt_text or req.user_prompt_text or req.prompt_text or "").strip()
        if not text.strip():
            return False
        if CODING_SYNTAX_RE.search(text):
            return True
        return bool(CODING_TOPIC_RE.search(text))

    @staticmethod
    def _is_file_search_request(req: UnifiedRequest) -> bool:
        text = (req.latest_user_prompt_text or req.user_prompt_text or req.prompt_text or "").strip()
        if not text:
            return False
        return bool(FILE_SEARCH_RE.search(text))

    @staticmethod
    def _is_commit_message_task(req: UnifiedRequest) -> bool:
        text = (req.latest_user_prompt_text or req.user_prompt_text or req.prompt_text or "").strip()
        if not text:
            return False
        return bool(COMMIT_MESSAGE_TASK_RE.search(text))

    @staticmethod
    def _is_no_thinking_task(req: UnifiedRequest) -> bool:
        text = (req.latest_user_prompt_text or req.user_prompt_text or req.prompt_text or "").strip()
        if not text:
            return False
        return bool(NO_THINKING_TASK_RE.search(text))

    def _heuristic_thinking_requested(
        self,
        cfg: RouterConfig,
        req: UnifiedRequest,
        selected_alias: str,
    ) -> bool:
        if req.needs_tooluse or self._is_no_thinking_task(req):
            return False
        profile = cfg.models[selected_alias]
        if not profile.supports_thinking:
            return False
        return self._is_deep_reasoning_request(req)

    async def _judge_alias(
        self, cfg: RouterConfig, req: UnifiedRequest, candidates: Iterable[str]
    ) -> tuple[Optional[str], Optional[bool]]:
        judge_alias = "small"
        judge_model = cfg.models[judge_alias].model_id
        candidate_list = list(candidates)
        if len(candidate_list) <= 1:
            return candidate_list[0] if candidate_list else None, None
        logger.info(
            "judge_start candidates=%s requested_model=%r est_input_tokens=%s est_total_tokens=%s",
            candidate_list,
            req.requested_model,
            req.estimated_input_tokens,
            req.estimated_total_tokens,
        )
        context_chars = max(500, cfg.routing.heuristics.judge_prompt_context_chars)
        latest_user_text = req.latest_user_prompt_text or req.user_prompt_text or req.prompt_text or ""
        latest_user_excerpt = latest_user_text[:context_chars]
        recent_user_context = (req.user_prompt_text or req.prompt_text or "")[-context_chars:]
        judge_prompt = {
            "instruction": (
                "Return only JSON: "
                "{\"route\":\"small|large|deep\",\"thinking\":\"on|off\",\"reason_code\":\"short_code\"}."
            ),
            "features": {
                "source_api": req.source_api,
                "estimated_input_tokens": req.estimated_input_tokens,
                "estimated_total_tokens": req.estimated_total_tokens,
                "max_tokens": req.max_tokens,
                "needs_vision": req.needs_vision,
                "needs_tooluse": req.needs_tooluse,
                "requested_model": req.requested_model,
            },
            "latest_user_prompt_excerpt": latest_user_excerpt,
            "recent_user_context_excerpt": recent_user_context,
            "candidates": candidate_list,
            "model_capabilities": {
                alias: cfg.models[alias].model_dump(mode="python") for alias in candidate_list
            },
            "edge_arguments": [
                "Client wrappers may add large system prompts and tool schemas; this alone is not a reason for large.",
                "A high max_tokens value can be a generic client default and is not sufficient evidence for large.",
                "Short acknowledgements or greetings (e.g. 'hallo') should route to small.",
                "Do not choose deep solely because prompt/max_tokens are large.",
                "Choose deep only for clear multi-step reasoning, strict rule compliance or high-risk decisions.",
                "Choose large only when the latest user ask clearly requires stronger coding/programming depth.",
                "Set thinking=on only if the selected route supports thinking and the task clearly benefits from it.",
                "Never set thinking=on when tools/tool-use are required.",
                "Set thinking=off for lightweight text tasks like commit messages, PR titles/descriptions, changelogs, or summaries.",
                "Prefer small for tool-use flows when available.",
                "Prefer small for file search / file lookup tasks when available.",
            ],
            "policy": "Prefer small by default. Use large for regular coding tasks. Use deep only for explicitly complex/high-stakes reasoning.",
            "preference": "Prefer small by default.",
        }
        payload = {
            "model": judge_model,
            "temperature": cfg.routing.heuristics.judge_temperature,
            "max_tokens": cfg.routing.heuristics.judge_max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an LLM router judge. Output strict JSON only.",
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
        try:
            response = await self.lm_client.post_json(judge_settings, "/v1/chat/completions", payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge_failed error=%s", exc)
            return None, None

        text = _extract_assistant_text(response).strip()
        if not text:
            logger.warning("judge_empty_response")
            return None, None

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
            return route, thinking_requested
        logger.warning("judge_unusable_response text=%s", text[:200])
        return None, None

    def _heuristic_alias(self, cfg: RouterConfig, req: UnifiedRequest, candidates: list[str]) -> str:
        if len(candidates) == 1:
            return candidates[0]
        if "deep" in candidates and self._is_deep_reasoning_request(req):
            return "deep"
        if "large" in candidates:
            h = cfg.routing.heuristics
            if req.estimated_input_tokens >= h.large_prompt_token_threshold:
                return "large"
            if (req.max_tokens or 0) >= h.large_max_tokens_threshold:
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
        candidates = self._eligible_aliases(cfg, req)
        is_coding = self._is_coding_request(req)
        is_commit_task = self._is_commit_message_task(req)
        is_no_thinking_task = self._is_no_thinking_task(req)
        is_file_search = self._is_file_search_request(req)
        logger.info(
            "route_eval_start source=%s requested_model=%r stream=%s required_caps=%s candidates=%s est_total_tokens=%s is_coding=%s",
            req.source_api,
            req.requested_model,
            req.stream,
            sorted(req.required_capabilities),
            candidates,
            req.estimated_total_tokens,
            is_coding,
        )
        if _thinking_debug_enabled():
            logger.info(
                "thinking_debug_route_flags source=%s commit_task=%s no_thinking_task=%s file_search=%s needs_tooluse=%s",
                req.source_api,
                int(is_commit_task),
                int(is_no_thinking_task),
                int(is_file_search),
                int(req.needs_tooluse),
            )
        if not candidates:
            logger.warning(
                "route_eval_no_candidates required_caps=%s est_total_tokens=%s",
                sorted(req.required_capabilities),
                req.estimated_total_tokens,
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    "No configured model satisfies required capabilities/context. "
                    f"required={sorted(req.required_capabilities)} total_tokens={req.estimated_total_tokens}"
                ),
            )

        preferred_alias = None
        if not self._is_router_public_model_name(cfg, req.requested_model):
            preferred_alias = self._find_alias_by_model_id(cfg, req.requested_model)

        small_policy_reason: Optional[str] = None
        if is_commit_task and "small" in candidates:
            small_policy_reason = "policy_commit_message_small"
        elif req.needs_tooluse and "small" in candidates:
            small_policy_reason = "policy_tooluse_small"
        elif is_file_search and "small" in candidates:
            small_policy_reason = "policy_file_search_small"

        if small_policy_reason:
            decision = self._make_route_decision(
                req=req,
                selected_alias="small",
                reason=small_policy_reason,
                candidates=candidates,
                thinking_requested=self._heuristic_thinking_requested(cfg, req, "small"),
                judge_model_id=cfg.models["small"].model_id,
                is_coding=is_coding,
            )
            logger.info("route_eval_decision selected=%s reason=%s", decision.selected_alias, decision.reason)
            return decision

        if len(candidates) == 1:
            selected = candidates[0]
            decision = self._make_route_decision(
                req=req,
                selected_alias=selected,
                reason="constraint_single_candidate",
                candidates=candidates,
                thinking_requested=self._heuristic_thinking_requested(cfg, req, selected),
                judge_model_id=cfg.models["small"].model_id,
                is_coding=is_coding,
            )
            logger.info("route_eval_decision selected=%s reason=%s", decision.selected_alias, decision.reason)
            return decision

        judge_alias, judge_thinking_requested = await self._judge_alias(cfg, req, candidates)
        if judge_alias and judge_alias in candidates:
            selected = judge_alias
            reason = f"judge_{judge_alias}"
            if not is_coding and "small" in candidates and selected == "large":
                selected = "small"
                reason = "judge_policy_non_coding_small"

            preferred = self._preferred_alias_for_request(cfg, req, candidates, preferred_alias, is_coding)
            if preferred:
                selected = preferred
                reason = "client_model_preference"

            thinking_requested = (
                judge_thinking_requested
                if judge_thinking_requested is not None
                else self._heuristic_thinking_requested(cfg, req, selected)
            )
            if req.needs_tooluse or self._is_no_thinking_task(req):
                thinking_requested = False
            if thinking_requested and not cfg.models[selected].supports_thinking:
                thinking_requested = False

            decision = self._make_route_decision(
                req=req,
                selected_alias=selected,
                reason=reason,
                candidates=candidates,
                thinking_requested=thinking_requested,
                judge_model_id=cfg.models["small"].model_id,
                is_coding=is_coding,
            )
            logger.info("route_eval_decision selected=%s reason=%s", decision.selected_alias, decision.reason)
            return decision

        preferred_when_judge_unavailable = self._preferred_alias_for_request(
            cfg, req, candidates, preferred_alias, is_coding
        )
        if preferred_when_judge_unavailable:
            decision = self._make_route_decision(
                req=req,
                selected_alias=preferred_when_judge_unavailable,
                reason="client_model_preference_judge_unavailable",
                candidates=candidates,
                thinking_requested=self._heuristic_thinking_requested(cfg, req, preferred_when_judge_unavailable),
                judge_model_id=cfg.models["small"].model_id,
                is_coding=is_coding,
            )
            logger.info("route_eval_decision selected=%s reason=%s", decision.selected_alias, decision.reason)
            return decision

        if "small" in candidates:
            decision = self._make_route_decision(
                req=req,
                selected_alias="small",
                reason="judge_unavailable_default_small",
                candidates=candidates,
                thinking_requested=self._heuristic_thinking_requested(cfg, req, "small"),
                judge_model_id=cfg.models["small"].model_id,
                is_coding=is_coding,
            )
            logger.info("route_eval_decision selected=%s reason=%s", decision.selected_alias, decision.reason)
            return decision

        heur_alias = self._heuristic_alias(cfg, req, candidates)
        decision = self._make_route_decision(
            req=req,
            selected_alias=heur_alias,
            reason="heuristic_fallback",
            candidates=candidates,
            thinking_requested=self._heuristic_thinking_requested(cfg, req, heur_alias),
            judge_model_id=cfg.models["small"].model_id,
            is_coding=is_coding,
        )
        logger.info("route_eval_decision selected=%s reason=%s", decision.selected_alias, decision.reason)
        return decision

    @staticmethod
    def _attempt_order(cfg: RouterConfig, decision: RouteDecision) -> list[str]:
        # Enforce policy: non-coding requests must not spill over to large.
        if not decision.is_coding_request and decision.selected_alias == "small":
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
            thinking_enabled = (
                decision.thinking_requested
                and not decision.needs_tooluse
                and cfg.models[alias].supports_thinking
            )
            payload_after_thinking = self._normalize_thinking_param(
                settings, path, payload_after_commit, thinking_enabled
            )
            payload = payload_after_thinking
            payload = self._normalize_openai_chat_token_param(settings, path, payload)
            if _thinking_debug_enabled():
                logger.info(
                    "thinking_debug_upstream_json path=%s alias=%s provider=%s decision_thinking=%s applied_thinking=%s raw=%s after_commit=%s after_thinking=%s final=%s",
                    path,
                    alias,
                    settings.provider,
                    int(decision.thinking_requested),
                    int(thinking_enabled),
                    _thinking_payload_probe(payload_raw),
                    _thinking_payload_probe(payload_after_commit),
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
        last_error: Optional[UpstreamError] = None
        order = self._attempt_order(cfg, decision)
        logger.info("upstream_stream_attempt_order path=%s order=%s", path, order)
        for idx, alias in enumerate(order):
            settings = self._upstream_for_alias(cfg, alias)
            payload_raw = dict(base_payload)
            payload_raw["model"] = cfg.models[alias].model_id
            payload_after_commit = self._normalize_commit_message_payload(path, payload_raw, decision)
            thinking_enabled = (
                decision.thinking_requested
                and not decision.needs_tooluse
                and cfg.models[alias].supports_thinking
            )
            payload_after_thinking = self._normalize_thinking_param(
                settings, path, payload_after_commit, thinking_enabled
            )
            payload = payload_after_thinking
            payload = self._normalize_openai_chat_token_param(settings, path, payload)
            if _thinking_debug_enabled():
                logger.info(
                    "thinking_debug_upstream_stream path=%s alias=%s provider=%s decision_thinking=%s applied_thinking=%s raw=%s after_commit=%s after_thinking=%s final=%s",
                    path,
                    alias,
                    settings.provider,
                    int(decision.thinking_requested),
                    int(thinking_enabled),
                    _thinking_payload_probe(payload_raw),
                    _thinking_payload_probe(payload_after_commit),
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
                async def empty_gen() -> AsyncIterator[bytes]:
                    if False:
                        yield b""
                logger.info("upstream_stream_empty path=%s alias=%s fallback=%s", path, alias, idx > 0)
                return alias, empty_gen(), idx > 0
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

            logger.info("upstream_stream_selected path=%s alias=%s fallback=%s", path, alias, idx > 0)
            return alias, chained(), idx > 0

        if last_error is not None:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream streaming call failed after fallback attempts: {last_error.body}",
            )
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
            thinking_enabled = (
                decision.thinking_requested
                and not decision.needs_tooluse
                and cfg.models[alias].supports_thinking
            )
            payload_after_thinking = self._normalize_thinking_param(
                settings, path, payload_after_commit, thinking_enabled
            )
            payload = payload_after_thinking
            payload = self._normalize_openai_chat_token_param(settings, path, payload)
            if _thinking_debug_enabled():
                logger.info(
                    "thinking_debug_upstream_anthropic_stream path=%s alias=%s provider=%s decision_thinking=%s applied_thinking=%s raw=%s after_commit=%s after_thinking=%s final=%s",
                    path,
                    alias,
                    settings.provider,
                    int(decision.thinking_requested),
                    int(thinking_enabled),
                    _thinking_payload_probe(payload_raw),
                    _thinking_payload_probe(payload_after_commit),
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

    async def handle_openai_chat(self, payload: dict[str, Any]) -> tuple[RouteDecision, str, bool, Any]:
        cfg = self.config_store.get_config()
        req = normalize_openai_chat(payload)
        decision = await self.choose_route(cfg, req)
        if req.stream:
            alias, stream_gen, used_fallback = await self._attempt_stream_with_fallback(
                cfg, "/v1/chat/completions", payload, decision
            )
            public_stream = rewrite_openai_stream_model_name(
                stream_gen,
                cfg.router_identity.exposed_model_name,
            )
            return decision, alias, used_fallback, public_stream
        alias, body, used_fallback = await self._attempt_json_with_fallback(
            cfg, "/v1/chat/completions", payload, decision
        )
        public_body = _apply_public_model_name_to_openai_response(
            body,
            cfg.router_identity.exposed_model_name,
        )
        return decision, alias, used_fallback, public_body

    async def handle_openai_completions(self, payload: dict[str, Any]) -> tuple[RouteDecision, str, bool, Any]:
        cfg = self.config_store.get_config()
        req = normalize_openai_completion(payload)
        decision = await self.choose_route(cfg, req)
        if req.stream:
            alias, stream_gen, used_fallback = await self._attempt_stream_with_fallback(
                cfg, "/v1/completions", payload, decision
            )
            public_stream = rewrite_openai_stream_model_name(
                stream_gen,
                cfg.router_identity.exposed_model_name,
            )
            return decision, alias, used_fallback, public_stream
        alias, body, used_fallback = await self._attempt_json_with_fallback(
            cfg, "/v1/completions", payload, decision
        )
        public_body = _apply_public_model_name_to_openai_response(
            body,
            cfg.router_identity.exposed_model_name,
        )
        return decision, alias, used_fallback, public_body

    async def handle_anthropic_messages(
        self, payload: dict[str, Any]
    ) -> tuple[RouteDecision, str, bool, bool, Any]:
        cfg = self.config_store.get_config()
        req = normalize_anthropic_messages(payload)
        decision = await self.choose_route(cfg, req)

        openai_payload = anthropic_to_openai_payload(payload)
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
        return decision, alias, used_fallback, False, anthropic_response


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
    return patched


async def rewrite_openai_stream_model_name(
    upstream_stream: AsyncIterator[bytes], public_model_name: str
) -> AsyncIterator[bytes]:
    buffer = ""
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
                                line = f"data: {json.dumps(parsed, ensure_ascii=False)}"
                        except json.JSONDecodeError:
                            pass
                out_lines.append(line)
            yield ("\n".join(out_lines) + "\n\n").encode("utf-8")
    if buffer:
        yield buffer.encode("utf-8")


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
    upstream_stream: AsyncIterator[bytes], model_id: str
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
    payload = {
        "event": "route_analytics",
        "v": 1,
        "source": decision.source_api,
        "requested_model": decision.requested_model,
        "selected_alias": final_alias,
        "selected_model": selected_model_id,
        "reason": decision.reason,
        "fallback_used": used_fallback,
        "stream": decision.stream,
        "candidate_aliases": decision.candidate_aliases,
        "required_capabilities": decision.required_capabilities,
        "context_signature": decision.context_signature,
        "complexity": decision.complexity,
        "estimated_input_tokens": decision.estimated_input_tokens,
        "estimated_total_tokens": decision.estimated_total_tokens,
        "max_tokens": decision.max_tokens,
        "needs_vision": decision.needs_vision,
        "needs_tooluse": decision.needs_tooluse,
        "is_coding": decision.is_coding_request,
        "repetition_key": decision.repetition_key,
        "thinking_requested": decision.thinking_requested,
        "thinking_applied": (
            decision.thinking_requested
            and not decision.needs_tooluse
            and cfg.models[final_alias].supports_thinking
        ),
    }
    logger.info("route_analytics %s", json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True))


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




def _project_root() -> Path:
    return PROJECT_ROOT


def _start_script_path() -> Path:
    return (_project_root() / START_SCRIPT_RELATIVE_PATH).resolve()


def _windows_startup_command(script_path: Path) -> str:
    return f'powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{script_path}"'


def _get_windows_startup_status() -> dict[str, Any]:
    script_path = _start_script_path()
    command = _windows_startup_command(script_path)

    if os.name != "nt":
        return {
            "supported": False,
            "enabled": False,
            "reason": "windows_only",
            "script_path": str(script_path),
            "script_exists": script_path.exists(),
            "command": command,
        }

    import winreg

    current_command: Optional[str] = None
    read_error: Optional[str] = None

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, WINDOWS_STARTUP_REG_PATH, 0, winreg.KEY_READ) as key:
            current_command = winreg.QueryValueEx(key, WINDOWS_STARTUP_VALUE_NAME)[0]
    except FileNotFoundError:
        current_command = None
    except OSError as exc:
        read_error = str(exc)

    enabled = current_command == command
    return {
        "supported": True,
        "enabled": enabled,
        "value_name": WINDOWS_STARTUP_VALUE_NAME,
        "script_path": str(script_path),
        "script_exists": script_path.exists(),
        "command": command,
        "current_command": current_command,
        "read_error": read_error,
    }


def _set_windows_startup_enabled(enabled: bool) -> dict[str, Any]:
    if os.name != "nt":
        raise HTTPException(status_code=400, detail="Windows startup is only supported on Windows.")

    import winreg

    status = _get_windows_startup_status()
    script_path = Path(status["script_path"])
    if not script_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Start script not found: {script_path}. Please create scripts/start_llm_router.ps1 first.",
        )

    command = _windows_startup_command(script_path)
    access = winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, WINDOWS_STARTUP_REG_PATH, 0, access) as key:
        if enabled:
            winreg.SetValueEx(key, WINDOWS_STARTUP_VALUE_NAME, 0, winreg.REG_SZ, command)
        else:
            try:
                winreg.DeleteValue(key, WINDOWS_STARTUP_VALUE_NAME)
            except FileNotFoundError:
                pass

    return _get_windows_startup_status()


def _admin_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Router Admin</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --card: #ffffff;
      --ink: #1a2230;
      --accent: #0068d6;
      --line: #d6deea;
    }
    body {
      font-family: "Segoe UI", "Source Sans Pro", sans-serif;
      background: radial-gradient(circle at top right, #e8f1ff, var(--bg));
      color: var(--ink);
      margin: 0;
      padding: 24px;
    }
    .card {
      max-width: 1100px;
      margin: 0 auto;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 20px;
      box-shadow: 0 10px 24px rgba(21, 36, 67, 0.08);
    }
    h1 { margin-top: 0; }
    .row {
      display: flex;
      gap: 10px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    input, button, textarea {
      font: inherit;
    }
    input {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      min-width: 280px;
      flex: 1;
    }
    button {
      border: 0;
      background: var(--accent);
      color: #fff;
      border-radius: 8px;
      padding: 9px 14px;
      cursor: pointer;
    }
    textarea {
      width: 100%;
      min-height: 560px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      resize: vertical;
      font-family: Consolas, "Courier New", monospace;
      font-size: 13px;
      line-height: 1.4;
    }
    .status {
      margin-top: 10px;
      min-height: 20px;
      font-weight: 600;
    }
    .substatus {
      min-height: 20px;
      color: #2f4b6d;
      margin-bottom: 10px;
    }
    label.inline {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 12px;
      background: #f7faff;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>LLM Router Admin</h1>
    <div class="row">
      <input id="tokenInput" placeholder="Optional Bearer token for protected API/UI">
      <button onclick="loadConfig()">Load config</button>
      <button onclick="saveConfig()">Save config</button>
    </div>
    <div class="row">
      <label class="inline">
        <input type="checkbox" id="startupToggle">
        Mit Windows starten (Tray)
      </label>
      <button onclick="saveWindowsStartup()">Windows-Start speichern</button>
      <button onclick="createStartupScriptInfo()">PS1-Startpfad anzeigen</button>
    </div>
    <div class="substatus" id="startupStatus"></div>
    <textarea id="configText"></textarea>
    <div class="status" id="status"></div>
  </div>
  <script>
    function headers(contentType = "text/plain") {
      const h = {};
      if (contentType) h["Content-Type"] = contentType;
      const token = document.getElementById("tokenInput").value.trim();
      if (token) h["Authorization"] = "Bearer " + token;
      return h;
    }
    function setStartupMessage(msg) {
      document.getElementById("startupStatus").textContent = msg;
    }
    async function loadConfig() {
      const res = await fetch("/admin/config", { headers: headers() });
      const txt = await res.text();
      document.getElementById("configText").value = txt;
      document.getElementById("status").textContent = res.ok ? "Config loaded." : "Load failed: " + txt;
    }
    async function saveConfig() {
      const payload = document.getElementById("configText").value;
      const res = await fetch("/admin/config", { method: "PUT", headers: headers(), body: payload });
      const txt = await res.text();
      document.getElementById("status").textContent = res.ok ? "Config saved." : "Save failed: " + txt;
    }
    async function loadWindowsStartup() {
      const res = await fetch("/admin/windows-startup", { headers: headers(null) });
      const data = await res.json();
      const toggle = document.getElementById("startupToggle");
      if (!res.ok) {
        toggle.disabled = true;
        setStartupMessage("Startup load failed.");
        return;
      }
      if (!data.supported) {
        toggle.checked = false;
        toggle.disabled = true;
        setStartupMessage("Windows startup is only available on Windows.");
        return;
      }
      toggle.disabled = false;
      toggle.checked = !!data.enabled;
      const state = data.enabled ? "enabled" : "disabled";
      const scriptInfo = data.script_exists ? "PS1 script found." : "PS1 script missing.";
      setStartupMessage("Windows startup is " + state + ". " + scriptInfo);
    }
    async function saveWindowsStartup() {
      const toggle = document.getElementById("startupToggle");
      const payload = JSON.stringify({ enabled: !!toggle.checked });
      const res = await fetch("/admin/windows-startup", {
        method: "PUT",
        headers: headers("application/json"),
        body: payload
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data.detail || "Save failed.";
        setStartupMessage(detail);
        return;
      }
      const state = data.enabled ? "enabled" : "disabled";
      setStartupMessage("Windows startup " + state + ".");
      toggle.checked = !!data.enabled;
    }
    async function createStartupScriptInfo() {
      const res = await fetch("/admin/windows-startup", { headers: headers(null) });
      const data = await res.json();
      if (!res.ok) {
        setStartupMessage("Could not load PS1 path.");
        return;
      }
      setStartupMessage("PS1 start script: " + data.script_path);
    }
    loadConfig();
    loadWindowsStartup();
  </script>
</body>
</html>
"""


def _admin_status_html() -> str:
    return """<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Router Status</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --card: #ffffff;
      --ink: #142033;
      --line: #d4deec;
      --ok-bg: #e8f8ed;
      --ok-ink: #146c2e;
      --warn-bg: #fff7e6;
      --warn-ink: #8a5a00;
      --bad-bg: #fdecec;
      --bad-ink: #9f1f1f;
      --muted: #5b6d84;
      --btn: #0068d6;
    }
    body {
      margin: 0;
      padding: 24px;
      background: radial-gradient(circle at top right, #e8f1ff, var(--bg));
      color: var(--ink);
      font-family: "Segoe UI", "Source Sans Pro", sans-serif;
    }
    .card {
      max-width: 920px;
      margin: 0 auto;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 20px;
      box-shadow: 0 10px 24px rgba(21, 36, 67, 0.08);
    }
    h1 {
      margin: 0 0 12px 0;
    }
    .row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 12px;
      align-items: center;
    }
    input, button {
      font: inherit;
    }
    input {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      min-width: 260px;
      flex: 1;
    }
    button {
      border: 0;
      background: var(--btn);
      color: #fff;
      border-radius: 8px;
      padding: 8px 14px;
      cursor: pointer;
    }
    .pill {
      display: inline-block;
      border-radius: 999px;
      padding: 4px 10px;
      font-weight: 700;
      font-size: 13px;
    }
    .ok { background: var(--ok-bg); color: var(--ok-ink); }
    .warn { background: var(--warn-bg); color: var(--warn-ink); }
    .bad { background: var(--bad-bg); color: var(--bad-ink); }
    .muted { color: var(--muted); }
    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }
    .box {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #f9fbff;
    }
    .box .label {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.4px;
      margin-bottom: 4px;
    }
    .box .value {
      font-weight: 700;
      word-break: break-word;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 14px;
    }
    th, td {
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 700;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Router Status</h1>
    <div class="row">
      <input id="tokenInput" placeholder="Optional Bearer token for protected API/UI">
      <button onclick="refreshStatus()">Aktualisieren</button>
    </div>
    <div class="row">
      <span id="overallBadge" class="pill warn">Pruefung laeuft...</span>
      <span class="muted" id="overallText"></span>
    </div>
    <div class="summary">
      <div class="box">
        <div class="label">Letzter Check</div>
        <div class="value" id="lastChecked">-</div>
      </div>
      <div class="box">
        <div class="label">Provider</div>
        <div class="value" id="provider">-</div>
      </div>
      <div class="box">
        <div class="label">Upstream URL</div>
        <div class="value" id="baseUrl">-</div>
      </div>
      <div class="box">
        <div class="label">Model Catalog</div>
        <div class="value" id="catalogPath">-</div>
      </div>
    </div>
    <div class="row muted" id="errorText"></div>
    <table>
      <thead>
        <tr>
          <th>Alias</th>
          <th>Model ID</th>
          <th>Upstream</th>
          <th>Verfuegbar</th>
          <th>Geladen</th>
          <th>Match</th>
        </tr>
      </thead>
      <tbody id="modelRows"></tbody>
    </table>
  </div>
  <script>
    function headers(contentType = null) {
      const h = {};
      if (contentType) h["Content-Type"] = contentType;
      const token = document.getElementById("tokenInput").value.trim();
      if (token) h["Authorization"] = "Bearer " + token;
      return h;
    }
    function badgeClass(ok, warn) {
      if (ok) return "pill ok";
      if (warn) return "pill warn";
      return "pill bad";
    }
    function formatTime(value) {
      if (!value) return "-";
      const d = new Date(value);
      if (isNaN(d.getTime())) return value;
      return d.toLocaleString();
    }
    function boolPill(value) {
      if (value === true) return '<span class="pill ok">Ja</span>';
      if (value === false) return '<span class="pill bad">Nein</span>';
      return '<span class="pill warn">Unklar</span>';
    }
    async function refreshStatus() {
      const overallBadge = document.getElementById("overallBadge");
      const overallText = document.getElementById("overallText");
      const errorText = document.getElementById("errorText");
      const modelRows = document.getElementById("modelRows");
      try {
        const [healthRes, modelRes] = await Promise.all([
          fetch("/healthz", { headers: headers() }),
          fetch("/admin/model-availability", { headers: headers() }),
        ]);

        const healthOk = healthRes.ok;
        const healthBody = healthOk ? await healthRes.json() : {};
        const modelBody = modelRes.ok ? await modelRes.json() : {};
        const allAvailable = !!modelBody.all_available;
        const allLoaded = !!modelBody.all_loaded;
        const ok = healthOk && healthBody.status === "ok" && allAvailable && allLoaded;

        overallBadge.className = badgeClass(ok, !ok && healthOk);
        overallBadge.textContent = ok ? "Gesund" : (healthOk ? "Warnung" : "Fehler");
        overallText.textContent = ok
          ? "Router erreichbar, alle konfigurierten Modelle verfuegbar und geladen."
          : "Bitte Details unten pruefen.";

        document.getElementById("lastChecked").textContent = formatTime(modelBody.last_checked_at);
        document.getElementById("provider").textContent = modelBody.provider || "-";
        document.getElementById("baseUrl").textContent = modelBody.base_url || "-";
        document.getElementById("catalogPath").textContent = modelBody.catalog_path || "-";

        if (modelBody.error) {
          errorText.textContent = "Letzter Fehler: " + modelBody.error;
        } else if (!modelRes.ok) {
          errorText.textContent = "Model-Status konnte nicht geladen werden.";
        } else {
          errorText.textContent = "";
        }

        const rows = Array.isArray(modelBody.models) ? modelBody.models : [];
        if (!rows.length) {
          modelRows.innerHTML = '<tr><td colspan="6" class="muted">Keine Modellinformationen vorhanden.</td></tr>';
        } else {
          modelRows.innerHTML = rows.map((m) => {
            const match = m.matched_upstream_id || "-";
            return "<tr>"
              + "<td>" + (m.alias || "-") + "</td>"
              + "<td>" + (m.model_id || "-") + "</td>"
              + "<td>" + (m.upstream_ref || "-") + "</td>"
              + "<td>" + boolPill(m.available) + "</td>"
              + "<td>" + boolPill(m.loaded) + "</td>"
              + "<td>" + match + "</td>"
              + "</tr>";
          }).join("");
        }
      } catch (err) {
        overallBadge.className = "pill bad";
        overallBadge.textContent = "Fehler";
        overallText.textContent = "Status konnte nicht geladen werden.";
        errorText.textContent = String(err);
        modelRows.innerHTML = '<tr><td colspan="6" class="muted">Keine Daten.</td></tr>';
      }
    }
    refreshStatus();
    setInterval(refreshStatus, 15000);
  </script>
</body>
</html>
"""


def create_app(
    config_path: Optional[Path] = None,
    lm_client: Optional[LMStudioClient] = None,
    model_check_interval_seconds: float = 60.0,
) -> FastAPI:
    cfg_path = config_path or DEFAULT_CONFIG_PATH
    store = ConfigStore(cfg_path)
    service = RouterService(store, lm_client=lm_client)
    monitor = ModelAvailabilityMonitor(store, service.lm_client, check_interval_seconds=model_check_interval_seconds)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await monitor.start()
        try:
            yield
        finally:
            await monitor.stop()

    app = FastAPI(title="LM Studio Router", version="0.1.0", lifespan=lifespan)
    app.state.config_store = store
    app.state.router_service = service
    app.state.model_availability_monitor = monitor

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        incoming_request_id = request.headers.get("x-request-id", "").strip()
        request_id = incoming_request_id or uuid.uuid4().hex[:12]
        token = _request_id_ctx.set(request_id)
        start = time.perf_counter()
        request.state.request_id = request_id

        client_host = request.client.host if request.client else "-"
        logger.info(
            "request_start method=%s path=%s client=%s query=%s",
            request.method,
            request.url.path,
            client_host,
            request.url.query,
        )
        try:
            response = await call_next(request)
            duration_ms = int((time.perf_counter() - start) * 1000)
            response.headers["x-request-id"] = request_id
            logger.info(
                "request_end method=%s path=%s status=%s duration_ms=%s",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
            return response
        except Exception:  # noqa: BLE001
            duration_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                "request_exception method=%s path=%s duration_ms=%s",
                request.method,
                request.url.path,
                duration_ms,
            )
            raise
        finally:
            _request_id_ctx.reset(token)

    async def require_auth(request: Request) -> None:
        cfg = store.get_config()
        token = (cfg.security.shared_bearer_token or "").strip()
        if not token:
            return
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {token}"
        if auth != expected:
            logger.warning("auth_failed path=%s", request.url.path)
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/admin/model-availability")
    async def admin_get_model_availability(request: Request) -> JSONResponse:
        await require_auth(request)
        status = await monitor.get_status()
        if status.get("last_checked_at") is None:
            await monitor.run_check_once()
            status = await monitor.get_status()
        return JSONResponse(status)

    @app.get("/v1/models")
    async def get_models(request: Request) -> JSONResponse:
        await require_auth(request)
        cfg = store.get_config()
        return JSONResponse(_build_models_response(cfg))

    @app.post("/v1/chat/completions")
    async def post_chat_completions(request: Request):
        await require_auth(request)
        payload = await request.json()
        logger.info("request_payload source=openai_chat %s", _payload_summary(payload))
        if _thinking_debug_enabled():
            logger.info("thinking_debug_request source=openai_chat probe=%s", _thinking_payload_probe(payload))
        decision, alias, used_fallback, result = await service.handle_openai_chat(payload)
        cfg = store.get_config()
        headers = _route_headers(cfg, decision, alias, used_fallback)
        _log_route_analytics(cfg, decision, alias, used_fallback)
        if isinstance(result, dict):
            return JSONResponse(result, headers=headers)
        return StreamingResponse(result, media_type="text/event-stream", headers=headers)

    @app.post("/v1/completions")
    async def post_completions(request: Request):
        await require_auth(request)
        payload = await request.json()
        logger.info("request_payload source=openai_completions %s", _payload_summary(payload))
        decision, alias, used_fallback, result = await service.handle_openai_completions(payload)
        cfg = store.get_config()
        headers = _route_headers(cfg, decision, alias, used_fallback)
        _log_route_analytics(cfg, decision, alias, used_fallback)
        if isinstance(result, dict):
            return JSONResponse(result, headers=headers)
        return StreamingResponse(result, media_type="text/event-stream", headers=headers)

    @app.post("/v1/messages")
    async def post_anthropic_messages(request: Request):
        await require_auth(request)
        payload = await request.json()
        logger.info("request_payload source=anthropic_messages %s", _payload_summary(payload))
        if _thinking_debug_enabled():
            logger.info("thinking_debug_request source=anthropic_messages probe=%s", _thinking_payload_probe(payload))
        decision, alias, used_fallback, is_stream, result = await service.handle_anthropic_messages(payload)
        cfg = store.get_config()
        headers = _route_headers(cfg, decision, alias, used_fallback)
        _log_route_analytics(cfg, decision, alias, used_fallback)
        if is_stream:
            return StreamingResponse(result, media_type="text/event-stream", headers=headers)
        return JSONResponse(result, headers=headers)

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_page(request: Request) -> str:
        await require_auth(request)
        return _admin_html()

    @app.get("/admin/status", response_class=HTMLResponse)
    async def admin_status_page(request: Request) -> str:
        await require_auth(request)
        return _admin_status_html()

    @app.get("/admin/config")
    async def admin_get_config(request: Request) -> PlainTextResponse:
        await require_auth(request)
        return PlainTextResponse(store.get_yaml(), media_type="application/yaml")

    @app.put("/admin/config")
    async def admin_put_config(request: Request) -> JSONResponse:
        await require_auth(request)
        yaml_payload = (await request.body()).decode("utf-8", errors="replace")
        try:
            cfg = await store.update_from_yaml(yaml_payload)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid YAML config: {exc}") from exc
        return JSONResponse(
            {
                "status": "ok",
                "server": {"host": cfg.server.host, "port": cfg.server.port},
                "models": {k: v.model_id for k, v in cfg.models.items()},
                "upstreams": {
                    name: {"provider": upstream.provider, "base_url": upstream.base_url}
                    for name, upstream in cfg.upstreams.items()
                },
                "exposed_model_name": cfg.router_identity.exposed_model_name,
            }
        )

    @app.get("/admin/windows-startup")
    async def admin_get_windows_startup(request: Request) -> JSONResponse:
        await require_auth(request)
        return JSONResponse(_get_windows_startup_status())

    @app.put("/admin/windows-startup")
    async def admin_put_windows_startup(request: Request) -> JSONResponse:
        await require_auth(request)
        payload = WindowsStartupToggleRequest.model_validate(await request.json())
        status = _set_windows_startup_enabled(payload.enabled)
        return JSONResponse(status)

    return app


def _admin_base_url(host: str, port: int) -> str:
    browser_host = host.strip() or "127.0.0.1"
    if browser_host in {"0.0.0.0", "::"}:
        browser_host = "127.0.0.1"
    return f"http://{browser_host}:{port}"


class RouterServerController:
    def __init__(self, app_instance: FastAPI):
        self._app = app_instance
        self._lock = threading.Lock()
        self._start_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Any = None
        self.last_error: Optional[str] = None

    def start(self) -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return True
            self.last_error = None
            self._start_event = threading.Event()
            self._thread = threading.Thread(target=self._serve, name="llm-router-server", daemon=True)
            self._thread.start()
        self._start_event.wait(timeout=5.0)
        return self.last_error is None

    def _serve(self) -> None:
        try:
            import uvicorn

            runtime_cfg = self._app.state.config_store.get_config()
            server_config = uvicorn.Config(
                self._app,
                host=runtime_cfg.server.host,
                port=runtime_cfg.server.port,
                reload=False,
            )
            server = uvicorn.Server(server_config)
            loop = asyncio.new_event_loop()
            with self._lock:
                self._loop = loop
                self._server = server
            self._start_event.set()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server.serve())
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            logger.exception("tray_server_failed error=%s", exc)
            self._start_event.set()
        finally:
            with self._lock:
                loop = self._loop
                self._loop = None
                self._server = None
                self._thread = None
            if loop and not loop.is_closed():
                loop.close()

    def stop(self) -> None:
        with self._lock:
            server = self._server
            loop = self._loop
            thread = self._thread

        if server is not None and loop is not None and loop.is_running():
            loop.call_soon_threadsafe(lambda: setattr(server, "should_exit", True))

        if thread and thread.is_alive():
            thread.join(timeout=10.0)

    def is_running(self) -> bool:
        with self._lock:
            thread = self._thread
            server = self._server

        if not thread or not thread.is_alive() or server is None:
            return False
        if getattr(server, "should_exit", False):
            return False
        return bool(getattr(server, "started", False))

    def is_starting(self) -> bool:
        with self._lock:
            thread = self._thread
            server = self._server

        if not thread or not thread.is_alive() or server is None:
            return False
        return not bool(getattr(server, "started", False)) and not bool(getattr(server, "should_exit", False))


def _build_tray_icon(is_running: bool):
    from PIL import Image, ImageDraw

    size = 64
    asset_path = _project_root() / "assets" / "llmrouter_route_icon.png"
    indicator = "#22c55e" if is_running else "#ef4444"

    if asset_path.exists():
        image = Image.open(asset_path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    else:
        image = Image.new("RGBA", (size, size), "#0b1220")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((2, 2, size - 3, size - 3), radius=14, fill="#0b1220", outline="#1f2937", width=2)
        path_points = [(14, 46), (25, 34), (37, 34), (50, 18)]
        draw.line(path_points, fill="#0ea5e9", width=10, joint="curve")
        draw.line(path_points, fill="#38bdf8", width=6, joint="curve")
        for px, py in [(14, 46), (37, 34), (50, 18)]:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill="#f8fafc", outline="#0ea5e9", width=1)

    draw = ImageDraw.Draw(image)
    draw.ellipse((45, 45, 61, 61), fill=indicator, outline="#ffffff", width=2)
    return image


def run_with_tray(app_instance: FastAPI) -> None:
    try:
        import pystray
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Tray mode requires pystray and pillow. Install with: pip install -r requirements.txt") from exc

    config_store = app_instance.state.config_store

    def current_urls() -> tuple[str, str, str]:
        runtime_cfg = config_store.get_config()
        base_url = _admin_base_url(runtime_cfg.server.host, runtime_cfg.server.port)
        return base_url, f"{base_url}/admin", f"{base_url}/admin/status"

    controller = RouterServerController(app_instance)
    controller.start()

    icon = pystray.Icon("llm-router", icon=_build_tray_icon(controller.is_running()), title="LM Router")
    shutdown_event = threading.Event()

    def status_text() -> str:
        base_url, _, _ = current_urls()
        if controller.is_running():
            return f"Status: Running ({base_url})"
        if controller.is_starting():
            return "Status: Starting..."
        if controller.last_error:
            return f"Status: Error ({controller.last_error})"
        return "Status: Stopped"

    def refresh_visuals() -> None:
        running = controller.is_running()
        icon.icon = _build_tray_icon(running)
        icon.title = "LM Router (Running)" if running else "LM Router (Stopped)"
        icon.update_menu()

    def on_open_admin(_icon, _item) -> None:
        _, admin_url, _ = current_urls()
        webbrowser.open(admin_url, new=2)

    def on_open_health(_icon, _item) -> None:
        _, _, status_url = current_urls()
        webbrowser.open(status_url, new=2)

    def on_restart(_icon, _item) -> None:
        old_cfg = config_store.get_config()
        controller.stop()
        try:
            config_store._config = config_store._load_from_disk()
            logger.info("tray_restart_config_reloaded")
        except Exception as exc:  # noqa: BLE001
            config_store._config = old_cfg
            controller.last_error = f"Config reload failed: {exc}"
            logger.exception("tray_restart_config_reload_failed error=%s", exc)
        controller.start()
        refresh_visuals()

    def on_quit(_icon, _item) -> None:
        shutdown_event.set()
        controller.stop()
        _icon.stop()

    icon.menu = pystray.Menu(
        pystray.MenuItem(lambda _item: status_text(), None, enabled=False),
        pystray.MenuItem("Settings oeffnen", on_open_admin),
        pystray.MenuItem("Status oeffnen", on_open_health),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Router neu starten", on_restart),
        pystray.MenuItem("Beenden", on_quit),
    )

    def monitor() -> None:
        last_state: Optional[bool] = None
        while not shutdown_event.wait(1.5):
            state = controller.is_running()
            if state != last_state:
                refresh_visuals()
                last_state = state

    monitor_thread = threading.Thread(target=monitor, name="llm-router-tray-monitor", daemon=True)
    monitor_thread.start()

    refresh_visuals()
    icon.run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LM Router")
    parser.add_argument("--tray", action="store_true", help="Run with system tray icon and controls")
    return parser.parse_args()


app = create_app()


def main() -> None:
    import uvicorn

    args = _parse_args()
    runtime_cfg = app.state.config_store.get_config()

    if args.tray:
        run_with_tray(app)
    else:
        uvicorn.run(
            app,
            host=runtime_cfg.server.host,
            port=runtime_cfg.server.port,
            reload=False,
        )


if __name__ == "__main__":
    main()
