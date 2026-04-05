# CLAUDE.md

This file guides Claude Code (claude.ai/code) for this repository.

## Core Rule: Single App File

- The only application module is `llmrouter/app.py`.
- Do not create or modify a root-level `app.py`.
- If a second `app.py` appears, remove it and keep `llmrouter/app.py` as source of truth.
- Imports must target `llmrouter.app` (never `from app import ...`).

## Architecture

- `llmrouter/app.py`: FastAPI app, routing logic, config, upstream client, tray mode.
- `llmrouter/__main__.py`: module entrypoint.
- `run.py`: local launcher (tray by default).
- `config/router_config.yaml`: runtime config.
- `tests/test_router.py`: main tests.

## Start Commands

```powershell
python -m llmrouter
python -m llmrouter --tray
python run.py
```

## Safety Checklist (before finishing changes)

Run these checks:

```powershell
Get-ChildItem -Recurse -Filter app.py | Select-Object -ExpandProperty FullName
```

Expected result:

- Exactly one hit: `<repo>/llmrouter/app.py`

And:

```powershell
Select-String -Path *.py,llmrouter\*.py,tests\*.py,scripts\*.py -Pattern "from app import|import app" -CaseSensitive:$false
```

Expected result:

- No imports from root `app`.
