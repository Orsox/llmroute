from pathlib import Path


def test_warp_trace_records_live_local_model_integration_expectations() -> None:
    document = Path("warp-trace.md").read_text(encoding="utf-8")

    expected_markers = [
        "QA-004: Integration tests shall be provided that actually call the locally configured and running models, not only mocked adapters, and verify that requests are routed to the intended backend models.",
        "QA-005: If local-model integration coverage exists in the repository, it shall include real end-to-end pytest cases that contact running local model endpoints with short-context requests, long-context requests, and several prompt variations so that routing behavior is verified against actual LLM execution rather than schema-only or fixture-only checks.",
        "QA-006: Integration tests against running local model endpoints shall validate successful routing, fallback behavior, incorrect-route detection, and preservation of the selected backend across the defined short-context, long-context, and prompt-variation scenarios.",
        "QA-007: Test execution shall be suitable for local development and CI, with clear separation between fast routing tests and environment-dependent local-model integration tests.",
    ]

    for marker in expected_markers:
        assert marker in document


def test_warp_trace_adds_live_model_context_matrix_step() -> None:
    document = Path("warp-trace.md").read_text(encoding="utf-8")

    expected_markers = [
        "### [x] WT-069 - Add live-model integration matrix for context length and prompt variation",
        "- Goal: Verify that real running local LLMs are contacted across materially different request shapes, not only fallback and offline scenarios.",
        "- Result: `tests/test_live_local_model_integration.py` runs only when `LLMROUTER_LOCAL_MODEL_MATRIX_JSON` or `LLMROUTER_LOCAL_MODEL_MATRIX_PATH` is provided, then sends short-context, long-context, and prompt-variation requests through the router to real externally running OpenAI-compatible local model endpoints such as LM Studio and asserts the expected routed backend plus structured routing-log evidence for each scenario.",
        "- Dependencies: WT-059, WT-062, WT-065, WT-068",
        "- QA/release agents: WT-059 through WT-069",
    ]

    for marker in expected_markers:
        assert marker in document


def test_warp_trace_records_local_dotenv_bootstrap_step() -> None:
    document = Path("warp-trace.md").read_text(encoding="utf-8")

    expected_markers = [
        "### [x] WT-070 - Add local `.env` bootstrap support",
        "- Goal: Allow local router configuration to be stored in a repository or explicitly referenced `.env` file without weakening explicit environment-variable control.",
        "- Result: Router bootstrap loads `LLMROUTER_*` values from `.env` in the current working directory or from `LLMROUTER_ENV_FILE`, keeps process environment values authoritative, and documents the precedence plus local usage in the configuration and operations docs.",
        "- Dependencies: WT-006, WT-008, WT-014, WT-064",
    ]

    for marker in expected_markers:
        assert marker in document
