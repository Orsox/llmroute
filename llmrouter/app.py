from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from .desktop import (
    _admin_html,
    _admin_settings_html,
    _admin_status_html,
    _get_windows_startup_status,
    _parse_args,
    _set_windows_startup_enabled,
    run_with_tray,
)
from .protocols import (
    _build_models_response,
    _log_route_analytics,
    _route_headers,
    anthropic_to_openai_payload,
    set_analytics_store,
)
from .requests import RouteDecision, UnifiedRequest, normalize_anthropic_messages, normalize_openai_chat
from .services import AnalyticsStore, LMStudioClient, ModelAvailabilityMonitor, RouterService, UpstreamError
from .settings import (
    DEFAULT_CONFIG_PATH,
    ConfigStore,
    LMStudioSettings,
    RouterConfig,
    WindowsStartupToggleRequest,
)
from .shared import (
    DEFAULT_TOOLUSE_SYSTEM_HINT,
    _payload_summary,
    _request_id_ctx,
    _request_start_ctx,
    _thinking_debug_enabled,
    _thinking_payload_probe,
    logger,
)

__all__ = [
    'DEFAULT_TOOLUSE_SYSTEM_HINT',
    'LMStudioClient',
    'LMStudioSettings',
    'RouteDecision',
    'RouterConfig',
    'RouterService',
    'UnifiedRequest',
    'UpstreamError',
    'anthropic_to_openai_payload',
    'create_app',
    'main',
    'normalize_anthropic_messages',
    'normalize_openai_chat',
    'app',
]


def create_app(
    config_path: Optional[Path] = None,
    lm_client: Optional[LMStudioClient] = None,
    model_check_interval_seconds: float = 60.0,
) -> FastAPI:
    cfg_path = config_path or DEFAULT_CONFIG_PATH
    logger.debug('Creating app with config path=%s', cfg_path)
    store = ConfigStore(cfg_path)
    service = RouterService(store, lm_client=lm_client)
    analytics_store = AnalyticsStore(store)
    set_analytics_store(analytics_store)
    monitor = ModelAvailabilityMonitor(store, service.lm_client, check_interval_seconds=model_check_interval_seconds)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await monitor.start()
        try:
            yield
        finally:
            await monitor.stop()

    app_instance = FastAPI(title='LM Studio Router', version='0.1.0', lifespan=lifespan)
    app_instance.state.config_store = store
    app_instance.state.router_service = service
    app_instance.state.analytics_store = analytics_store
    app_instance.state.model_availability_monitor = monitor

    @app_instance.middleware('http')
    async def request_logging_middleware(request: Request, call_next):
        incoming_request_id = request.headers.get('x-request-id', '').strip()
        request_id = incoming_request_id or uuid.uuid4().hex[:12]
        req_token = _request_id_ctx.set(request_id)
        start_token = _request_start_ctx.set(time.perf_counter())
        start = time.perf_counter()
        request.state.request_id = request_id

        client_host = request.client.host if request.client else '-'
        logger.info(
            'request_start method=%s path=%s client=%s query=%s',
            request.method,
            request.url.path,
            client_host,
            request.url.query,
        )
        try:
            response = await call_next(request)
            duration_ms = int((time.perf_counter() - start) * 1000)
            response.headers['x-request-id'] = request_id
            logger.info(
                'request_end method=%s path=%s status=%s duration_ms=%s',
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
            return response
        except Exception:
            duration_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(
                'request_exception method=%s path=%s duration_ms=%s',
                request.method,
                request.url.path,
                duration_ms,
            )
            raise
        finally:
            _request_start_ctx.reset(start_token)
            _request_id_ctx.reset(req_token)

    async def require_auth(request: Request) -> None:
        cfg = store.get_config()
        token = (cfg.security.shared_bearer_token or '').strip()
        if not token:
            return
        auth = request.headers.get('authorization', '')
        expected = f'Bearer {token}'
        if auth != expected:
            logger.warning('auth_failed path=%s', request.url.path)
            raise HTTPException(status_code=401, detail='Unauthorized')

    @app_instance.get('/healthz')
    async def healthz() -> dict[str, str]:
        return {'status': 'ok'}

    @app_instance.get('/admin/model-availability')
    async def admin_get_model_availability(request: Request) -> JSONResponse:
        await require_auth(request)
        status = await monitor.get_status()
        if status.get('last_checked_at') is None:
            await monitor.run_check_once()
            status = await monitor.get_status()
        return JSONResponse(status)

    @app_instance.get('/v1/models')
    async def get_models(request: Request) -> JSONResponse:
        await require_auth(request)
        cfg = store.get_config()
        return JSONResponse(_build_models_response(cfg))

    @app_instance.post('/v1/chat/completions')
    async def post_chat_completions(request: Request):
        await require_auth(request)
        payload = await request.json()
        logger.info('request_payload source=openai_chat %s', _payload_summary(payload))
        if _thinking_debug_enabled():
            logger.info('thinking_debug_request source=openai_chat probe=%s', _thinking_payload_probe(payload))
        decision, alias, used_fallback, result = await service.handle_openai_chat(payload)
        cfg = store.get_config()
        headers = _route_headers(cfg, decision, alias, used_fallback)
        _log_route_analytics(cfg, decision, alias, used_fallback)
        if isinstance(result, dict):
            return JSONResponse(result, headers=headers)
        return StreamingResponse(result, media_type='text/event-stream', headers=headers)

    @app_instance.post('/v1/completions')
    async def post_completions(request: Request):
        await require_auth(request)
        payload = await request.json()
        logger.info('request_payload source=openai_completions %s', _payload_summary(payload))
        decision, alias, used_fallback, result = await service.handle_openai_completions(payload)
        cfg = store.get_config()
        headers = _route_headers(cfg, decision, alias, used_fallback)
        _log_route_analytics(cfg, decision, alias, used_fallback)
        if isinstance(result, dict):
            return JSONResponse(result, headers=headers)
        return StreamingResponse(result, media_type='text/event-stream', headers=headers)

    @app_instance.post('/v1/messages')
    async def post_anthropic_messages(request: Request):
        await require_auth(request)
        payload = await request.json()
        logger.info('request_payload source=anthropic_messages %s', _payload_summary(payload))
        if _thinking_debug_enabled():
            logger.info('thinking_debug_request source=anthropic_messages probe=%s', _thinking_payload_probe(payload))
        decision, alias, used_fallback, is_stream, result = await service.handle_anthropic_messages(payload)
        cfg = store.get_config()
        headers = _route_headers(cfg, decision, alias, used_fallback)
        _log_route_analytics(cfg, decision, alias, used_fallback)
        if is_stream:
            return StreamingResponse(result, media_type='text/event-stream', headers=headers)
        return JSONResponse(result, headers=headers)

    @app_instance.get('/admin', response_class=HTMLResponse)
    async def admin_page(request: Request) -> str:
        await require_auth(request)
        return _admin_html()

    @app_instance.get('/admin/status', response_class=HTMLResponse)
    async def admin_status_page(request: Request) -> str:
        await require_auth(request)
        return _admin_status_html()

    @app_instance.get('/admin/config')
    async def admin_get_config(request: Request) -> PlainTextResponse:
        await require_auth(request)
        return PlainTextResponse(store.get_yaml(), media_type='application/yaml')

    @app_instance.put('/admin/config')
    async def admin_put_config(request: Request) -> JSONResponse:
        await require_auth(request)
        yaml_payload = (await request.body()).decode('utf-8', errors='replace')
        try:
            cfg = await store.update_from_yaml(yaml_payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f'Invalid YAML config: {exc}') from exc
        return JSONResponse(
            {
                'status': 'ok',
                'server': {'host': cfg.server.host, 'port': cfg.server.port},
                'models': {key: profile.model_id for key, profile in cfg.models.items()},
                'upstreams': {
                    name: {'provider': upstream.provider, 'base_url': upstream.base_url}
                    for name, upstream in cfg.upstreams.items()
                },
                'exposed_model_name': cfg.router_identity.exposed_model_name,
            }
        )

    @app_instance.get('/admin/windows-startup')
    async def admin_get_windows_startup(request: Request) -> JSONResponse:
        await require_auth(request)
        return JSONResponse(_get_windows_startup_status())

    @app_instance.put('/admin/windows-startup')
    async def admin_put_windows_startup(request: Request) -> JSONResponse:
        await require_auth(request)
        payload = WindowsStartupToggleRequest.model_validate(await request.json())
        return JSONResponse(_set_windows_startup_enabled(payload.enabled))

    @app_instance.get('/settings', response_class=HTMLResponse)
    async def settings_page(request: Request) -> str:
        await require_auth(request)
        return _admin_settings_html(store.get_config())

    return app_instance


app = create_app()


def main() -> None:
    import uvicorn

    args = _parse_args()
    log_level = args.log_level.upper() if hasattr(args, 'log_level') else logging.getLevelName(logger.level)
    logger.debug('Starting app with log_level=%s', log_level)

    runtime_cfg = app.state.config_store.get_config()
    if args.tray:
        logger.debug('Running in tray mode')
        run_with_tray(app)
        return

    logger.debug('Running as standalone HTTP server')
    uvicorn.run(
        app,
        host=runtime_cfg.server.host,
        port=runtime_cfg.server.port,
        reload=False,
    )


if __name__ == '__main__':
    main()
