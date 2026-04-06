from __future__ import annotations

from .shared import *
from .shared import (
    _current_request_latency_ms,
    _env_flag,
    _extract_assistant_text,
    _extract_openai_tool_call_count,
    _hash_text,
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

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS routing_runs (
                request_id TEXT PRIMARY KEY,
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
        conn.commit()

    def _connect(self) -> sqlite3.Connection:
        db_path = self._db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        if self._initialized_path != db_path:
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
                        request_id, created_at, updated_at, route_logged_at, source, requested_model,
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
                        :request_id, :created_at, :updated_at, :route_logged_at, :source, :requested_model,
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
                        request_id, created_at, updated_at, output_logged_at, source,
                        selected_alias, selected_model, reason, fallback_used, stream,
                        output_text_chars, output_text_excerpt, output_tokens, input_tokens,
                        tool_calls, stop_reason, routing_efficiency_label,
                        routing_efficiency_score, latency_ms
                    ) VALUES (
                        :request_id, :created_at, :updated_at, :output_logged_at, :source,
                        :selected_alias, :selected_model, :reason, :fallback_used, :stream,
                        :output_text_chars, :output_text_excerpt, :output_tokens, :input_tokens,
                        :tool_calls, :stop_reason, :routing_efficiency_label,
                        :routing_efficiency_score, :latency_ms
                    )
                    ON CONFLICT(request_id) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        output_logged_at=excluded.output_logged_at,
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


class RouterService:
    def __init__(self, config_store: ConfigStore, lm_client: Optional[LMStudioClient] = None):
        self.config_store = config_store
        self.lm_client = lm_client or LMStudioClient()

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
        total_tokens = req.routing_estimated_total_tokens
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
            req.routing_input_tokens,
            req.routing_estimated_total_tokens,
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
                }
                for alias in candidate_list
            },
            "edge_arguments": [
                "Client wrappers, system reminders, tool schemas, and local command echoes are not a reason for large.",
                "A high max_tokens value can be a generic client default and is not sufficient evidence for large.",
                "Short acknowledgements or greetings (e.g. 'hallo') should route to small.",
                "Do not choose deep solely because prompt/max_tokens are large.",
                "Use the latest actionable user ask, not wrapper noise, as the main routing signal.",
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
        self._apply_routing_budget(cfg, req, is_coding)
        candidates = self._eligible_aliases(cfg, req)
        is_commit_task = self._is_commit_message_task(req)
        is_no_thinking_task = self._is_no_thinking_task(req)
        is_file_search = self._is_file_search_request(req)
        small_coding_context_limit = self._small_coding_context_limit_tokens()
        small_coding_task_limit = self._small_coding_task_limit_tokens()
        if is_coding and "small" in candidates and req.routing_estimated_total_tokens > small_coding_context_limit:
            candidates = [alias for alias in candidates if alias != "small"]
            logger.info(
                "route_eval_filter_small_coding_context est_total_tokens=%s limit=%s",
                req.routing_estimated_total_tokens,
                small_coding_context_limit,
            )
        logger.info(
            "route_eval_start source=%s requested_model=%r stream=%s required_caps=%s candidates=%s routing_total_tokens=%s full_total_tokens=%s is_coding=%s",
            req.source_api,
            req.requested_model,
            req.stream,
            sorted(req.required_capabilities),
            candidates,
            req.routing_estimated_total_tokens,
            req.full_estimated_total_tokens,
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
                "route_eval_no_candidates_using_backup required_caps=%s est_total_tokens=%s",
                sorted(req.required_capabilities),
                req.routing_estimated_total_tokens,
            )
            selected = "backup"
            decision = self._make_route_decision(
                req=req,
                selected_alias=selected,
                reason="no_candidates_fallback_to_backup",
                candidates=[selected],
                thinking_requested=self._heuristic_thinking_requested(cfg, req, selected),
                judge_model_id=cfg.models["small"].model_id,
                is_coding=is_coding,
            )
            logger.info("route_eval_decision selected=%s reason=%s", decision.selected_alias, decision.reason)
            return decision

        preferred_alias = None
        if not self._is_router_public_model_name(cfg, req.requested_model):
            preferred_alias = self._find_alias_by_model_id(cfg, req.requested_model)

        if req.tool_loop_context and "large" in candidates:
            decision = self._make_route_decision(
                req=req,
                selected_alias="large",
                reason="policy_tool_loop_large",
                candidates=candidates,
                thinking_requested=self._heuristic_thinking_requested(cfg, req, "large"),
                judge_model_id=cfg.models["small"].model_id,
                is_coding=is_coding,
            )
            logger.info("route_eval_decision selected=%s reason=%s", decision.selected_alias, decision.reason)
            return decision

        small_policy_reason: Optional[str] = None
        if is_commit_task and "small" in candidates:
            small_policy_reason = "policy_commit_message_small"
        elif (
            req.needs_tooluse
            and "small" in candidates
            and (not is_coding or req.routing_estimated_total_tokens <= small_coding_task_limit)
        ):
            small_policy_reason = "policy_tooluse_small"
        elif (
            is_file_search
            and "small" in candidates
            and (not is_coding or req.routing_estimated_total_tokens <= small_coding_task_limit)
        ):
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
                source_api=req.source_api,
                decision=decision,
                final_alias=alias,
                final_model_id=cfg.models[alias].model_id,
                used_fallback=used_fallback,
            )
            return decision, alias, used_fallback, public_stream
        alias, body, used_fallback = await self._attempt_json_with_fallback(
            cfg, "/v1/chat/completions", payload, decision
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
                source_api=req.source_api,
                decision=decision,
                final_alias=alias,
                final_model_id=cfg.models[alias].model_id,
                used_fallback=used_fallback,
            )
            return decision, alias, used_fallback, public_stream
        alias, body, used_fallback = await self._attempt_json_with_fallback(
            cfg, "/v1/completions", payload, decision
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
