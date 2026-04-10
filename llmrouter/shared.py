from __future__ import annotations

import asyncio
import argparse
import contextvars
import json
import hashlib
import logging
import math
import os
import re
import sqlite3
import threading
import time
import uuid
import webbrowser
from typing import Any, AsyncIterator, Dict, Iterable, Literal, Optional

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field, model_validator

try:
    from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

from contextlib import asynccontextmanager
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path


_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_request_start_ctx: contextvars.ContextVar[float] = contextvars.ContextVar("request_start", default=0.0)
_session_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")

# Optionaler Import des Konfigurationsfensters
# config_gui.py removed - HTML is now default UI


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get()
        record.session_id = _session_id_ctx.get()
        return True


def _configure_logging() -> logging.Logger:
    app_logger = logging.getLogger("llm-router")
    if app_logger.handlers:
        return app_logger

    level_name = os.getenv("ROUTER_LOG_LEVEL", "DEBUG").upper()
    level = getattr(logging, level_name, logging.INFO)
    app_logger.setLevel(level)
    app_logger.debug("Logger initialized with level=%s from env ROUTER_LOG_LEVEL", level_name)
    log_file_path = Path(os.getenv("ROUTER_LOG_FILE", "logs/router.log"))
    max_bytes = int(os.getenv("ROUTER_LOG_MAX_BYTES", str(5 * 1024 * 1024)))
    backup_count = int(os.getenv("ROUTER_LOG_BACKUP_COUNT", "3"))
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s [req=%(request_id)s sess=%(session_id)s] %(message)s")
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
_analytics_store: Optional[Any] = None


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
    r"migrationsstrategie|rollout|validierungsstrategie|"
    r"stm32cubecli|stm32cube|cubecli|websearch|websuche|web search"
    r")\b|https?://",
    re.IGNORECASE,
)
WEBSEARCH_RE = re.compile(
    r"\b(websearch|websuche|web search|performing.*web search|web search results)\b|https?://",
    re.IGNORECASE,
)
ROUTING_WRAPPER_TAG_RE = re.compile(
    r"<(?P<tag>system-reminder|local-command-caveat|command-name|command-message|command-args|"
    r"local-command-stdout|local-command-stderr|user-prompt-submit-hook|local-command-cwd|"
    r"command-exit-code)[^>]*>.*?</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
ROUTING_INLINE_TAG_RE = re.compile(r"</?[^>\n]+>")
ROUTING_NOISE_LINE_RE = re.compile(
    r"^(?:"
    r"the file .+ (?:has been )?updated(?: successfully)?\.?|"
    r"file created successfully at:.+|"
    r"no matches found|"
    r"found \d+ files?.*|"
    r"done|"
    r"set model to .+"
    r")$",
    re.IGNORECASE,
)
SESSION_ID_RE = re.compile(r"[^a-zA-Z0-9._:-]+")
LIGHTWEIGHT_TASK_RE = re.compile(
    r"^(?:"
    r"hi|hello|hey|hallo|moin|servus|yo|ok|okay|thanks|danke|thx|"
    r"ping|test|kurze frage|kurzfrage|guten morgen|guten tag|guten abend"
    r")(?:[!.?, ]*)$",
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


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _current_request_latency_ms() -> Optional[int]:
    start = _request_start_ctx.get()
    if start <= 0:
        return None
    return int((time.perf_counter() - start) * 1000)

def _payload_summary(payload: dict[str, Any]) -> str:
    def _safe_get(container: Any, key: str) -> Any:
        if isinstance(container, dict):
            return container.get(key)
        return None

    model = payload.get("model")
    stream = bool(payload.get("stream"))
    max_tokens = payload.get("max_tokens", payload.get("max_completion_tokens"))
    messages = payload.get("messages")
    msg_count = len(messages) if isinstance(messages, list) else 0
    has_tools = bool(payload.get("tools"))
    has_prompt = "prompt" in payload
    chat_kwargs = payload.get("chat_template_kwargs")
    extra_body = payload.get("extra_body")
    options = payload.get("options")
    has_thinking = any(
        [
            payload.get("thinking") is True,
            _safe_get(chat_kwargs, "enable_thinking") is True,
            _safe_get(extra_body, "thinking") is True,
            _safe_get(options, "thinking") is True,
        ]
    )
    reasoning = payload.get("reasoning")
    has_reasoning = isinstance(reasoning, dict) and bool(reasoning)
    reasoning_effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    return (
        f"model={model!r} stream={stream} max_tokens={max_tokens} "
        f"messages={msg_count} tools={int(has_tools)} prompt={int(has_prompt)} "
        f"thinking={int(has_thinking)} reasoning={int(has_reasoning)} reasoning_effort={reasoning_effort!r}"
    )


DEFAULT_TOOLUSE_SYSTEM_HINT = ""
TOOLUSE_SYSTEM_HINT_PATH = PROJECT_ROOT / "config" / "tooluse_system_hint.yaml"
COMMIT_MESSAGE_HINT_PATH = PROJECT_ROOT / "config" / "commit_message_hint.yaml"
JUDGE_PROMPT_SYSTEM_PATH = PROJECT_ROOT / "config" / "judge_prompt_system.yaml"
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


def _extract_text_fragments(content: Any) -> tuple[list[str], bool, bool]:
    if content is None:
        return [], False, False
    if isinstance(content, str):
        return [content], False, False
    if not isinstance(content, list):
        return [str(content)], False, False

    chunks: list[str] = []
    has_vision = False
    has_tool_loop_content = False
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
            has_tool_loop_content = True
            nested_text, nested_vision = _extract_text_and_vision(item.get("content"))
            has_vision = has_vision or nested_vision
            if nested_text:
                chunks.append(nested_text)
        elif item_type == "tool_use":
            has_tool_loop_content = True
        else:
            text = item.get("text")
            if text:
                chunks.append(str(text))
    return [chunk for chunk in chunks if chunk], has_vision, has_tool_loop_content


def _sanitize_routing_text(text: str) -> tuple[str, bool]:
    if not text:
        return "", False

    original = text
    cleaned = ROUTING_WRAPPER_TAG_RE.sub(" ", text)
    cleaned = ROUTING_INLINE_TAG_RE.sub(" ", cleaned)
    lines = [line.strip() for line in cleaned.splitlines()]
    kept_lines = [line for line in lines if line and not ROUTING_NOISE_LINE_RE.match(line)]
    cleaned = re.sub(r"\s+", " ", " ".join(kept_lines)).strip()
    return cleaned, cleaned != re.sub(r"\s+", " ", original).strip()


def _hash_text(text: str) -> str:
    normalized = (text or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _sanitize_session_id(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    cleaned = SESSION_ID_RE.sub("-", raw).strip(".:-_")
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned[:128]


def _estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 1
    return max(1, math.ceil(len(text) / 4))


def _log_text_max_chars() -> int:
    return max(200, int(os.getenv("ROUTER_LOG_TEXT_MAX_CHARS", "4000")))


def _clip_for_log(text: str, max_chars: Optional[int] = None) -> str:
    limit = _log_text_max_chars() if max_chars is None else max(50, max_chars)
    return (text or "")[:limit]


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


def _extract_openai_tool_call_count(openai_response: dict[str, Any]) -> int:
    choices = openai_response.get("choices") or []
    if not choices:
        return 0
    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls") or []
    if isinstance(tool_calls, list):
        return len(tool_calls)
    return 0


def _routing_efficiency(
    expected_route_class: str,
    final_alias: str,
    *,
    initial_alias: str,
    used_fallback: bool,
    stop_reason: Optional[str] = None,
) -> tuple[str, int]:
    label = "good_fit"
    score = 100
    if initial_alias == "small" and (used_fallback or stop_reason == "length"):
        label = "undersized_route"
        score -= 45
    elif expected_route_class == "small" and final_alias in {"large", "deep"}:
        label = "oversized_route"
        score -= 30 if final_alias == "large" else 40

    if used_fallback:
        score -= 15
    if stop_reason in {"length", "content_filter"}:
        score -= 10
    return label, max(0, score)
