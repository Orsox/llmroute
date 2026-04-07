from __future__ import annotations

from .shared import *

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
    lightweight_max_tokens_cap: int = 768
    suspect_default_max_tokens_threshold: int = 8192


class RoutingSettings(BaseModel):
    judge_timeout_seconds: float = 15.0
    fallback_enabled: bool = True
    hybrid_client_model_override: bool = True
    default_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    analytics_enabled: bool = True
    analytics_sqlite_path: str = "logs/router_analytics.sqlite"
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
        required = {"small", "large", "deep", "backup"}
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
            "default_temperature": None,
            "analytics_enabled": True,
            "analytics_sqlite_path": "logs/router_analytics.sqlite",
            "heuristics": {
                "large_prompt_token_threshold": 2200,
                "large_max_tokens_threshold": 1800,
                "judge_temperature": 0.0,
                "judge_max_tokens": 96,
                "judge_prompt_context_chars": 6000,
                "lightweight_max_tokens_cap": 768,
                "suspect_default_max_tokens_threshold": 8192,
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
            "backup": {
                "model_id": "gpt-4o-mini",
                "context_window": 128000,
                "capabilities": ["chat", "completions", "tooluse"],
                "upstream_ref": "deep",
                "supports_thinking": False,
                "relative_speed": 2.0,
                "suitable_for": "Ultimate fallback when no other model fits constraints.",
            },
        },
    }
