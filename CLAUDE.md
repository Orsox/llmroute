# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Überblick

LM Router ist ein Python/FastAPI-Proxy, der OpenAI-kompatible und Anthropic-kompatible Requests annimmt und anhand von Heuristiken und einem Judge-Modell zwischen `small`, `large` und `deep` routet. Die gesamte Laufzeitlogik sitzt überwiegend in `app.py`; das `llmrouter/`-Paket dient vor allem als Import-/CLI-Hülle.

## Wichtige Architektur

- `app.py` enthält Konfiguration, Routing-Logik, Upstream-Client, Request-Normalisierung, Anthropic/OpenAI-Translation, Admin-Endpoints, Tray-Modus und Server-Start.
- `ConfigStore` lädt und schreibt `config/router_config.yaml` atomar und stellt sicher, dass die Aliase `small`, `large` und `deep` vorhanden sind.
- `RouterService` trifft die Routing-Entscheidung. Es gibt einen Judge-Pfad plus Heuristiken für Coding-/Reasoning-/Kontext-Fälle.
- `LMStudioClient` kapselt die Upstream-Aufrufe gegen LM Studio bzw. OpenAI-kompatible Backends.
- `create_app()` baut die FastAPI-App und registriert API-, Admin- und Health-Endpunkte.
- Tests in `tests/test_router.py` decken Routing, Auth, OpenAI/Anthropic-Translation, Streaming und Model-Availability ab.

## Häufige Befehle

### Installation

```powershell
pip install -r requirements.txt
```

### Server starten

```powershell
python -m llmrouter
```

Tray-Modus:

```powershell
python -m llmrouter --tray
```

PowerShell-Launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_llm_router.ps1
```

### Tests

Alle Tests:

```powershell
pytest
```

Einzelne Testdatei:

```powershell
pytest tests/test_router.py
```

Einzelner Test:

```powershell
pytest tests/test_router.py -k test_name
```

### Manuelle Prüfskripte

Deep-Verbindung prüfen:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test_deep_connection.ps1
```

Demo-Requests ausführen:

```powershell
python demo_requests.py --router-url http://127.0.0.1:12345
```

## Konfiguration

- Laufzeitkonfiguration: `config/router_config.yaml`
- Vorlage: `config/router_config.example.yaml`
- Umgebungsvariablen werden aus `.env` geladen, falls vorhanden.
- Relevante Flags für Deep-Routing stehen in `.env.example`.

## API- und Verhaltenshinweise

- Unterstützte Endpunkte: `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/models`, `/healthz`, `/admin`, `/admin/status`, `/admin/config`, `/admin/model-availability`.
- Das Router-Modell, das nach außen erscheint, heißt standardmäßig `borg-cpu`.
- Request-IDs werden über `x-request-id` korreliert und in `logs/router.log` protokolliert.
- Anthropic-Payloads werden nach OpenAI-Form übersetzt; Tool-Use und Streaming werden in beiden Richtungen unterstützt.

## Wichtige Dateien

- `app.py` — zentrale Anwendung und Routing-Logik
- `llmrouter/__main__.py` — CLI-Einstiegspunkt
- `llmrouter/app.py` — Paket-Wrapper für `app.py`
- `tests/test_router.py` — Haupttests
- `README.md` — Projektübersicht und Laufzeitdoku
