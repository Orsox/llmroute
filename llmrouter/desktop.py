from __future__ import annotations

from .shared import *
from .settings import *

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
        logger.debug("ServerController.start() called, running=%s", self._thread is not None and self._thread.is_alive())
        with self._lock:
            if self._thread and self._thread.is_alive():
                logger.debug("Server already running")
                return True
            self.last_error = None
            self._start_event = threading.Event()
            self._thread = threading.Thread(target=self._serve, name="llm-router-server", daemon=True)
            self._thread.start()
            logger.debug("Server thread started")
        self._start_event.wait(timeout=5.0)
        result = self.last_error is None
        logger.debug("Server start completed, running=%s, error=%s", result, self.last_error if not result else None)
        return result

    def _serve(self) -> None:
        logger.debug("Server _serve() thread started")
        try:
            import uvicorn

            runtime_cfg = self._app.state.config_store.get_config()
            logger.debug("Creating uvicorn server: host=%s port=%s", runtime_cfg.server.host, runtime_cfg.server.port)
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
            logger.debug("Starting uvicorn server")
            loop.run_until_complete(server.serve())
        except Exception as exc:  # noqa: BLE001
            logger.exception("tray_server_failed error=%s", exc)
            self.last_error = str(exc)
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
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        logger.error("PIL not available in _build_tray_icon")
        return None

    size = 64
    asset_path = _project_root() / "assets" / "llmrouter_route_icon.png"
    indicator = "#22c55e" if is_running else "#ef4444"

    try:
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
    except Exception as exc:
        logger.exception("Failed to build tray icon: %s", exc)
        # Fallback: einfaches farbiges Quadrat
        return Image.new("RGBA", (size, size), indicator)


def run_with_tray(app_instance: FastAPI) -> None:
    logger.info("Starting LM Router in tray mode")
    try:
        import pystray
        from PIL import Image  # noqa: F401 - verify pillow is available
    except ImportError as exc:
        logger.error("Tray mode unavailable: pystray/PIL not installed")
        raise RuntimeError(
            "Tray mode requires pystray and pillow. "
            "Install with: pip install -r requirements.txt"
        ) from exc

    config_store = app_instance.state.config_store
    runtime_cfg = config_store.get_config()
    logger.debug("Tray mode config loaded: host=%s port=%s",
                 runtime_cfg.server.host, runtime_cfg.server.port)

    def current_urls() -> tuple[str, str, str]:
        runtime_cfg = config_store.get_config()
        base_url = _admin_base_url(runtime_cfg.server.host, runtime_cfg.server.port)
        return base_url, f"{base_url}/admin", f"{base_url}/admin/status"

    controller = RouterServerController(app_instance)
    logger.debug("Starting router server controller")
    controller.start()
    logger.debug("Router server controller started, running=%s", controller.is_running())

    # Initiales Icon-Bild erstellen
    initial_icon = _build_tray_icon(controller.is_running())
    icon = pystray.Icon("llm-router", icon=initial_icon, title="LM Router")
    shutdown_event = threading.Event()

    def status_text(item=None) -> str:
        base_url, _, _ = current_urls()
        if controller.is_running():
            return f"Status: Running ({base_url})"
        if controller.is_starting():
            return "Status: Starting..."
        if controller.last_error:
            return f"Status: Error ({controller.last_error})"
        return "Status: Stopped"

    def refresh_visuals() -> None:
        icon.title = "LM Router (Running)" if controller.is_running() else "LM Router (Stopped)"
        _refresh_icon()

    admin_url, _admin_url, status_url = current_urls()

    def on_open_admin(_icon, _item) -> None:
        """Öffne Admin Settings Seite im Browser."""
        logger.debug("Opening admin settings page in browser")
        webbrowser.open(admin_url + "/settings", new=2)

    def on_open_settings(_icon, _item) -> None:
        """Öffne Settings Seite."""
        settings_url = f"{admin_url}/settings"
        logger.debug("Opening settings page at %s", settings_url)
        webbrowser.open(settings_url, new=2)

    def on_open_health(_icon, _item) -> None:
        _, _, status_url = current_urls()
        logger.debug("Opening health status in browser")
        webbrowser.open(status_url, new=2)

    def on_restart(_icon, _item) -> None:
        logger.debug("Reloading config and restarting server")
        old_cfg = config_store.get_config()
        controller.stop()
        try:
            config_store._config = config_store._load_from_disk()
            logger.info("tray_restart_config_reloaded")
        except Exception as exc:  # noqa: BLE001
            logger.exception("tray_restart_config_reload_failed error=%s", exc)
            config_store._config = old_cfg
            controller.last_error = f"Config reload failed: {exc}"
        controller.start()
        refresh_visuals()

    def on_quit(_icon, _item) -> None:
        logger.debug("Stopping tray icon and controller")
        shutdown_event.set()
        controller.stop()
        _icon.stop()

    def _refresh_icon() -> None:
        """Tray-Icon neu erstellen (benötigt wird, da pystray keine dynamischen Updates unterstützt)."""
        is_running = controller.is_running()
        new_icon = _build_tray_icon(is_running)
        icon.icon = new_icon

    icon.menu = pystray.Menu(
        pystray.MenuItem(status_text, lambda: None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Settings", on_open_settings),  # Direkter Zugriff auf die Web-Oberfläche
        pystray.MenuItem("Status", on_open_health),
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
    logger.debug("Monitor thread started")

    refresh_visuals()
    icon.run()


def _admin_settings_html(cfg: RouterConfig) -> str:
    """
    Moderne Web-Oberfläche für alle YAML-Einstellungen.
    HTML is now default UI - keine Qt GUI mehr.
    """
    # Config in Python-Objekt konvertieren
    server = cfg.server
    upstreams = cfg.upstreams
    routing = cfg.routing
    identity = cfg.router_identity
    heuristics = routing.heuristics if hasattr(routing, "heuristics") else None

    # Models für Router/Judge
    model_ids = list(cfg.models.keys())

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LM Router Settings</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@sweetalert2/theme-bootstrap-4@5/bootstrap-4.css">
  <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --card-hover: #334155;
      --ink: #f8fafc;
      --muted: #94a3b8;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --success: #10b981;
      --warning: #f59e0b;
      --error: #ef4444;
      --line: #334155;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      color: var(--ink);
      margin: 0;
      padding: 24px;
      min-height: 100vh;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    header {{
      text-align: center;
      margin-bottom: 32px;
      padding: 24px;
      background: linear-gradient(135deg, var(--accent), var(--accent-hover));
      border-radius: 16px;
      box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
    }}
    h1 {{ margin: 0; font-size: 28px; font-weight: 700; }}
    p.subtitle {{ margin: 12px 0 0; color: rgba(255,255,255,0.9); font-size: 14px; }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 24px;
      transition: all 0.2s ease;
    }}
    .card:hover {{ border-color: var(--accent); }}
    .card h2 {{
      margin: 0 0 20px;
      font-size: 18px;
      color: var(--accent);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .card h2::before {{
      content: "⚙️";
    }}

    .form-group {{ margin-bottom: 16px; }}
    .form-group:last-child {{ margin-bottom: 0; }}
    label {{
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    input[type="text"], input[type="number"], select {{
      width: 100%;
      padding: 12px 16px;
      background: var(--bg);
      border: 2px solid var(--line);
      border-radius: 10px;
      color: var(--ink);
      font-size: 14px;
      transition: all 0.2s;
    }}
    input:focus, select:focus {{
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }}
    select {{ cursor: pointer; }}

    .toggle {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 14px 16px;
      background: var(--bg);
      border: 2px solid var(--line);
      border-radius: 10px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .toggle:hover {{ border-color: var(--accent); }}
    .toggle input {{ margin: 0; cursor: pointer; }}
    .toggle-label {{ font-size: 14px; color: var(--ink); }}

    .status {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px;
      background: var(--card);
      border-radius: 12px;
      margin-bottom: 24px;
    }}
    .status-dot {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 16px var(--success);
    }}
    .status-dot.error {{ background: var(--error); box-shadow: 0 0 16px var(--error); }}
    .status-info {{ color: var(--muted); font-size: 13px; }}

    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 14px 24px;
      border: none;
      border-radius: 10px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      text-decoration: none;
    }}
    .btn-primary {{
      background: var(--accent);
      color: white;
    }}
    .btn-primary:hover {{
      background: var(--accent-hover);
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(99, 102, 241, 0.4);
    }}
    .btn-secondary {{
      background: var(--card);
      border: 2px solid var(--line);
      color: var(--ink);
    }}
    .btn-secondary:hover {{
      border-color: var(--accent);
      background: var(--card-hover);
    }}

    .config-actions {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .section-title {{
      font-size: 14px;
      font-weight: 700;
      color: var(--muted);
      margin: 24px 0 16px;
      text-transform: uppercase;
      letter-spacing: 1px;
    }}

    .upstream-item {{
      background: var(--bg);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 12px;
    }}
    .upstream-item:last-child {{ margin-bottom: 0; }}

    @media (max-width: 768px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>⚡ LLM Router Settings</h1>
      <p class="subtitle">Kontrolliere alle Einstellungen mit einfachen Dropdowns und Toggles</p>
    </header>

    <div class="status">
      <div class="status-dot" id="statusDot"></div>
      <div class="status-info" id="statusInfo">Lade Konfiguration...</div>
    </div>

    <div class="config-actions">
      <button class="btn btn-primary" onclick="loadConfig()">📥 Config laden</button>
      <button class="btn btn-secondary" onclick="saveConfig()">💾 Speichern</button>
      <a href="/admin/config" class="btn btn-secondary">📄 Raw YAML</a>
    </div>

    <div class="section-title">🌐 Server Einstellungen</div>
    <div class="grid">
      <div class="card">
        <h2>Host & Port</h2>
        <div class="form-group">
          <label>API Host</label>
          <input type="text" id="server_host" placeholder="0.0.0.0">
        </div>
        <div class="form-group">
          <label>Port</label>
          <input type="number" id="server_port" placeholder="12345">
        </div>
      </div>
    </div>

    <div class="section-title">🤖 Router Modelle</div>
    <div class="grid">
      <div class="card">
        <h2>Router Model</h2>
        <div class="form-group">
          <label>Wähle ein Modell</label>
          <select id="router_model"></select>
        </div>
        <div class="form-group">
          <label>Context Window</label>
          <input type="number" id="router_context" placeholder="100000">
        </div>
      </div>

      <div class="card">
        <h2>Judge Model</h2>
        <div class="form-group">
          <label>Wähle ein Modell</label>
          <select id="judge_model"></select>
        </div>
        <div class="form-group">
          <label>Context Window</label>
          <input type="number" id="judge_context" placeholder="262144">
        </div>
      </div>
    </div>

    <div class="section-title">🔗 Upstreams</div>
    <div class="grid">
      <div class="card">
        <h2>Local Upstream</h2>
        <div id="local_upstream" class="upstream-item">
          <div class="form-group">
            <label>Provider</label>
            <input type="text" id="local_provider" placeholder="lm_studio">
          </div>
          <div class="form-group">
            <label>Base URL</label>
            <input type="text" id="local_base_url" placeholder="http://localhost:1234">
          </div>
          <div class="form-group">
            <label>Timeout (Sekunden)</label>
            <input type="number" id="local_timeout" placeholder="120">
          </div>
        </div>
      </div>

      <div class="card">
        <h2>Deep Upstream</h2>
        <div id="deep_upstream" class="upstream-item">
          <div class="form-group">
            <label>Provider</label>
            <input type="text" id="deep_provider" placeholder="openai">
          </div>
          <div class="form-group">
            <label>Base URL</label>
            <input type="text" id="deep_base_url" placeholder="https://api.openai.com">
          </div>
          <div class="form-group">
            <label>Timeout (Sekunden)</label>
            <input type="number" id="deep_timeout" placeholder="180">
          </div>
        </div>
      </div>
    </div>

    <div class="section-title">🛤️ Routing</div>
    <div class="grid">
      <div class="card">
        <h2>Fallback & Hybrid</h2>
        <label class="toggle">
          <input type="checkbox" id="routing_fallback_enabled">
          <span class="toggle-label">Fallback enabled</span>
        </label>
        <label class="toggle" style="margin-top: 8px;">
          <input type="checkbox" id="routing_hybrid_client_model_override">
          <span class="toggle-label">Hybrid client model override</span>
        </label>
        <div class="form-group" style="margin-top: 12px;">
          <label>Judge Timeout (Sekunden)</label>
          <input type="number" id="judge_timeout" placeholder="15">
        </div>
      </div>

      <div class="card">
        <h2>Router Identity</h2>
        <div class="form-group">
          <label>Exposed Model Name</label>
          <input type="text" id="exposed_model_name" placeholder="borg-cpu">
        </div>
        <label class="toggle" style="margin-top: 8px;">
          <input type="checkbox" id="publish_underlying_models" {str(cfg.router_identity.publish_underlying_models).lower()}>
          <span class="toggle-label">Publish underlying models</span>
        </label>
      </div>
    </div>

    <div class="section-title">🔧 Heuristics</div>
    <div class="grid">
      <div class="card">
        <h2>Token Thresholds</h2>
        <div class="form-group">
          <label>Large Prompt Token Threshold</label>
          <input type="number" id="large_prompt_threshold" placeholder="2200">
        </div>
        <div class="form-group">
          <label>Large Max Tokens Threshold</label>
          <input type="number" id="large_max_tokens_threshold" placeholder="1800">
        </div>
      </div>

      <div class="card">
        <h2>Judge Settings</h2>
        <div class="form-group">
          <label>Judge Temperature</label>
          <input type="number" id="judge_temperature" placeholder="0.0" step="0.1">
        </div>
        <div class="form-group">
          <label>Judge Max Tokens</label>
          <input type="number" id="judge_max_tokens" placeholder="512">
        </div>
        <div class="form-group">
          <label>Judge Prompt Context Chars</label>
          <input type="number" id="judge_prompt_context_chars" placeholder="6000">
        </div>
      </div>
    </div>

    <div class="section-title">🌐 API Endpoints</div>
    <div class="grid">
      <div class="card">
        <h2>Client Model Override</h2>
        <div class="form-group">
          <label>Small Models (JSON Array)</label>
          <input type="text" id="small_models" placeholder='["small"]'>
        </div>
      </div>
    </div>

    <footer style="text-align: center; padding: 24px; color: var(--muted); font-size: 13px;">
      <p>💡 Tipp: Ändere die Einstellungen und klicke auf "Speichern" um den Router neu zu starten.</p>
    </footer>
  </div>

  <script>
    const API_BASE = "{_admin_base_url(server.host, server.port)}";
    let token = "";

    function headers(contentType = "text/plain") {{
      const h = {{}};
      if (contentType) h["Content-Type"] = contentType;
      if (token) h["Authorization"] = "Bearer " + token;
      return h;
    }}

    async function loadConfig() {{
      try {{
        const res = await fetch(`${{API_BASE}}/admin/config`, {{ headers: headers() }});
        const txt = await res.text();
        document.getElementById("statusDot").className = "status-dot " + (res.ok ? "" : "error");
        document.getElementById("statusInfo").textContent = res.ok ? "✅ Config loaded successfully!" : "❌ " + txt;

        // Parse YAML and populate fields
        try {{
          const config = await parseYaml(txt);
          if (!config) {{
            setField("server_host", server.host);
            setField("server_port", server.port);
            setModel("router_model", "{{models}}");
            setModel("judge_model", "{{models}}");
            setJsonField("small_models", config.models);
            setField("judge_timeout", routing.judge_timeout_seconds);
            setField("judge_temperature", routing.heuristics.judge_temperature);
            setField("judge_max_tokens", routing.heuristics.judge_max_tokens);
            setField("judge_prompt_context_chars", routing.heuristics.judge_prompt_context_chars);
            setField("exposed_model_name", identity.exposed_model_name);
            setField("large_prompt_threshold", routing.heuristics.large_prompt_token_threshold);
            setField("large_max_tokens_threshold", routing.heuristics.large_max_tokens_threshold);
            setField("routing_fallback_enabled", routing.fallback_enabled);
            setField("routing_hybrid_client_model_override", routing.hybrid_client_model_override);
            document.getElementById("publish_underlying_models").checked = identity.publish_underlying_models;
            document.getElementById("local_provider").value = config.upstreams.local.provider;
            document.getElementById("local_base_url").value = config.upstreams.local.base_url;
            document.getElementById("local_timeout").value = config.upstreams.local.timeout_seconds;
            document.getElementById("deep_provider").value = config.upstreams.deep.provider;
            document.getElementById("deep_base_url").value = config.upstreams.deep.base_url;
            document.getElementById("deep_timeout").value = config.upstreams.deep.timeout_seconds;
          }}
        }} catch (e) {{
          console.error("YAML parse failed:", e);
          document.getElementById("statusInfo").textContent = "⚠️ Failed to parse YAML";
        }}
      }} catch (err) {{
        document.getElementById("statusInfo").textContent = "❌ " + err.message;
      }}
    }}

    async function saveConfig() {{
      // Collect all values
      const data = collectFormData();
      if (Object.keys(data).length === 0) {{
        return Swal.fire("⚠️", "Keine Änderungen gefunden", "warning");
      }}

      try {{
        const yaml = yamlify(data);
        const res = await fetch(`${{API_BASE}}/admin/config`, {{
          method: "PUT",
          headers: headers("application/yaml"),
          body: yaml
        }});

        if (res.ok) {{
          document.getElementById("statusDot").className = "status-dot";
          document.getElementById("statusInfo").textContent = "✅ Config saved! Router wird neu gestartet...";
          setTimeout(() => window.location.reload(), 2000);
          Swal.fire({{
            icon: "success",
            title: "Saved!",
            text: "Der Router wurde erfolgreich gespeichert und wird neu gestartet.",
            confirmButtonText: "OK"
          }});
        }} else {{
          document.getElementById("statusInfo").textContent = "❌ Save failed";
          Swal.fire("❌", "Speichern fehlgeschlagen", "error");
        }}
      }} catch (err) {{
        document.getElementById("statusInfo").textContent = "❌ " + err.message;
      }}
    }}

    function setField(name, value) {{
      const el = document.getElementById(name);
      if (el) {{
        el.value = value;
      }}
    }}

    function setModel(name, value) {{
      const el = document.getElementById(name);
      if (el && value) {{
        const parts = value.split(",");
        el.value = parts.join(", ");
      }}
    }}

    function setJsonField(name, value) {{
      try {{
        const json = JSON.stringify(value);
        const el = document.getElementById(name);
        if (el) el.value = json;
      }} catch (e) {{
        console.error("JSON stringify failed:", e);
      }}
    }}

    function collectFormData() {{
      const data = {{}};
      // Simple field collection
      const fields = document.querySelectorAll("input[type='text'], input[type='number'], select");
      fields.forEach(f => {{
        if (f.id) {{
          const value = f.value;
          const name = f.id;
          if (value && value.trim()) {{
            try {{
              const parsed = JSON.parse(value);
              data[name] = parsed;
            }} catch (e) {{
              data[name] = value.trim();
            }}
          }}
        }}
      }});

      // Toggles
      document.querySelectorAll(".toggle input[type='checkbox']").forEach(t => {{
        if (t.id) {{
          data[t.id] = t.checked;
        }}
      }});

      return data;
    }}

    function yamlify(data) {{
      // Simple YAML generation
      const lines = [];
      const indent = 2;

      function indentLines(arr, level) {{
        return arr.map(line => "  ".repeat(level) + line).join("\\n");
      }}

      function toYamlValue(v) {{
        if (typeof v === "string" && (v.includes("://") || v.includes(".") || /^[\\d\\.,\\-+]+\\s*$/.test(v))) {{
          return \`'${{v}}'\`;
        }}
        return String(v);
      }}

      function renderSection(prefix, obj) {{
        if (!obj) return "";
        const lines = [];
        for (const key in obj) {{
          if (obj.hasOwnProperty(key)) {{
            const value = obj[key];
            const keyPath = prefix ? `${{prefix}}.${{key}}` : key;
            let val = toYamlValue(value);
            lines.push(`${{keyPath}}: ${{val}}`);
          }}
        }}
        return lines.join("\\n");
      }}

      lines.push("server:");
      lines.push(`  host: ${{toYamlValue(data.server_host)}}`);
      lines.push(`  port: ${{data.server_port}}`);
      lines.push("models:");
      const models = data.models || {{}};
      const modelKeys = Object.keys(models);
      modelKeys.forEach(k => {{
        const m = models[k];
        lines.push(`  ${{k}}:`);
        lines.push(`    model_id: ${{m.model_id}}`);
        lines.push(`    context_window: ${{m.context_window || 100000}}`);
        lines.push(`    upstream_ref: local`);
        lines.push(`    supports_thinking: ${{m.supports_thinking || "true"}}`);
      }});
      lines.push("upstreams:");
      lines.push("  local:");
      lines.push(`    provider: lm_studio`);
      lines.push(`    base_url: ${{data.local_base_url}}`);
      lines.push(`    timeout_seconds: ${{data.local_timeout}}`);
      lines.push(`    api_key_env: OPENAI_API_KEY`);
      lines.push("  deep:");
      lines.push(`    provider: openai`);
      lines.push(`    base_url: ${{data.deep_base_url}}`);
      lines.push(`    timeout_seconds: ${{data.deep_timeout}}`);
      lines.push("routing:");
      lines.push(`  judge_timeout_seconds: ${{data.judge_timeout}}`);
      lines.push(`  fallback_enabled: ${{data.routing_fallback_enabled}}`);
      lines.push(`  hybrid_client_model_override: ${{data.routing_hybrid_client_model_override}}`);
      lines.push("router_identity:");
      lines.push(`  exposed_model_name: ${{toYamlValue(data.exposed_model_name)}}`);
      lines.push(`  publish_underlying_models: ${{data.publish_underlying_models}}`);
      lines.push("models:");
      if (data.small_models) {{
        try {{
          const sm = JSON.parse(data.small_models);
          lines.push(`  ${{sm[0] || "small"}}:`);
          lines.push(`    model_id: qwen/qwen3.5-9b`);
          lines.push(`    context_window: 100000`);
          lines.push(`    upstream_ref: local`);
          lines.push(`    supports_thinking: true`);
        }} catch (e) {{
          // skip
        }}
      }}
      lines.push("heuristics:");
      lines.push(`  large_prompt_token_threshold: ${{data.large_prompt_threshold}}`);
      lines.push(`  large_max_tokens_threshold: ${{data.large_max_tokens_threshold}}`);
      lines.push(`  judge_temperature: ${{data.judge_temperature}}`);
      lines.push(`  judge_max_tokens: ${{data.judge_max_tokens}}`);
      lines.push(`  judge_prompt_context_chars: ${{data.judge_prompt_context_chars}}`);

      return lines.join("\\n");
    }}

    async function parseYaml(txt) {{
      try {{
        return await (await fetch(API_BASE + "/admin/config")).text();
      }} catch (e) {{ return null; }}
    }}

    // Initialize on load
    window.addEventListener("load", () => {{ loadConfig(); }});
  </script>
</body>
</html>"""

    return html

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LM Router")
    parser.add_argument("--tray", action="store_true", help="Run with system tray icon and controls")
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Log level: DEBUG, INFO, WARNING, ERROR (default: DEBUG)")
    return parser.parse_args()
