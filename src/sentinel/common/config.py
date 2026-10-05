"""Typed access to params.yaml (pipeline parameters) and .env (secrets, service URLs)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class DataParams(BaseModel):
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    reference_dir: Path
    reports_dir: Path = Path("reports")
    attempted_policy: Literal["benign", "drop", "keep"] = "benign"
    dev_sample_frac: float = Field(0.1, gt=0, le=1)


class SplitParams(BaseModel):
    train_frac: float = Field(0.6, gt=0, lt=1)
    val_frac: float = Field(0.2, gt=0, lt=1)
    gap_rows: int = Field(50, ge=0)
    holdout_families: list[str] = Field(
        default_factory=lambda: ["Infiltration", "Bot", "WebAttack"]
    )


class FeatureParams(BaseModel):
    log_transform: bool = True
    drop_constant: bool = True


class LightGBMParams(BaseModel):
    n_estimators: int = 2000
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 20
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    early_stopping_rounds: int = 100
    # Overrides ids.class_weight_power for this model when set.
    class_weight_power: float | None = Field(None, ge=0, le=1)


class XGBoostParams(BaseModel):
    n_estimators: int = 2000
    learning_rate: float = 0.05
    max_depth: int = 8
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 100
    class_weight_power: float | None = Field(None, ge=0, le=1)


class MLPParams(BaseModel):
    hidden: list[int] = Field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2048
    epochs: int = 30
    patience: int = 4
    focal_gamma: float = 2.0


class IDSParams(BaseModel):
    benign_fpr_target: float = Field(0.01, gt=0, lt=1)
    class_weight_power: float = Field(0.5, ge=0, le=1)
    max_class_weight: float = Field(1000, ge=1)
    severity_bands: list[float] = Field(default_factory=lambda: [0.6, 0.8, 0.95])
    tune_trials: int = 30
    tune_timeout_minutes: float = 60
    lightgbm: LightGBMParams = LightGBMParams()
    xgboost: XGBoostParams = XGBoostParams()
    mlp: MLPParams = MLPParams()


class AutoencoderParams(BaseModel):
    hidden: list[int] = Field(default_factory=lambda: [64, 32])
    bottleneck_sweep: list[int] = Field(default_factory=lambda: [8, 12, 16])
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 4096
    epochs: int = 30
    patience: int = 3


class IForestParams(BaseModel):
    n_estimators: int = 200
    max_samples: int = 256
    train_rows: int = 200_000


class AnomalyParams(BaseModel):
    fpr_target: float = Field(0.01, gt=0, lt=1)
    fpr_sweep: list[float] = Field(default_factory=lambda: [0.001, 0.005, 0.01, 0.02])
    autoencoder: AutoencoderParams = AutoencoderParams()
    iforest: IForestParams = IForestParams()


class PhishingParams(BaseModel):
    raw_dir: Path = Path("data/raw/phishing_emails")
    processed_dir: Path = Path("data/processed/phishing")
    kaggle_dir: Path = Path("kaggle/phishing-emails")
    max_chars: int = Field(3000, gt=100)
    n_folds: int = Field(5, ge=3)
    kaggle_output_dir: Path = Path("models/phishing/phishing_lora_output")
    max_len: int = 256
    fpr_budget: float = Field(0.02, gt=0, lt=1)
    quantization: Literal["embeddings", "full", "none"] = "embeddings"
    benchmark_emails: int = Field(1000, ge=50)
    url_raw_dir: Path = Path("data/raw/phishing_urls")
    url_fpr_budget: float = Field(0.01, gt=0, lt=1)
    url_max_features: int = Field(50_000, ge=1000)
    url_email_extra_fpr: float = Field(0.005, ge=0, lt=1)


class AdversarialParams(BaseModel):
    per_family: int = 250
    eps: list[float] = Field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0, 2.0])
    pgd_iter: int = 20
    hsj_samples: int = 80
    hsj_max_iter: int = 10
    hsj_max_eval: int = 400
    pads: list[float] = Field(default_factory=lambda: [0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
    delays: list[float] = Field(default_factory=lambda: [1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0])
    budgets: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {"low": (0.1, 1.5), "medium": (0.5, 3.0), "high": (1.0, 10.0)}
    )
    adv_train_eps: float = 0.5
    adv_train_steps: int = 5
    adv_train_epochs: int = 3
    adv_train_lr: float = 3e-4
    adv_train_per_class: int = 40_000
    adv_train_benign: int = 150_000
    review_budget: float = Field(0.01, gt=0, lt=1)


class CopilotLLMParams(BaseModel):
    providers: list[str] = Field(default_factory=lambda: ["gemini", "ollama"])
    gemini_model: str = "gemini-3.5-flash-lite"
    ollama_model: str = "llama3.2:3b"
    # Free-tier budget; set to your limits from https://aistudio.google.com/rate-limit.
    gemini_rpm: int = Field(8, ge=1)
    gemini_rpd: int = Field(200, ge=1)
    # Evaluation judge: a different model from the generator (report section 8.3).
    judge_model: str = "gemini-3.1-flash-lite"
    judge_rpm: int = Field(5, ge=1)
    judge_rpd: int = Field(100, ge=1)
    cache_path: Path = Path("data/cache/llm.sqlite")


class GuardParams(BaseModel):
    deepset_dir: Path = Path("data/raw/prompt_injections")
    n_train: int = 4000  # synthetic documents per split (half carry an injection)
    n_val: int = 1000
    n_test: int = 2000
    max_fpr: float = Field(0.01, gt=0, lt=1)  # false alarms on clean validation documents
    train_deberta: bool = True
    epochs: int = 2
    lr: float = 3e-5
    batch_size: int = 16
    max_len: int = 128


class AssetParams(BaseModel):
    cidr: str  # a single host ("192.168.10.50") or a network ("192.168.10.0/24")
    name: str
    criticality: Literal["low", "medium", "high", "critical"] = "medium"


Level = Literal["Low", "Medium", "High", "Critical"]


class SeverityParams(BaseModel):
    family_impact: dict[str, Level] = Field(
        default_factory=lambda: {  # type: ignore[arg-type]
            "PortScan": "Medium",
            "BruteForce": "Medium",
            "DoS": "High",
            "DDoS": "High",
            "WebAttack": "High",
            "Bot": "High",
            "Infiltration": "Critical",
            "Unknown anomaly": "Medium",
            "Phishing/Spam": "Medium",
        }
    )
    family_cap: dict[str, Level] = Field(
        default_factory=lambda: {"PortScan": "High", "BruteForce": "High"}  # type: ignore[arg-type]
    )
    volume_high: int = Field(100, ge=1)  # related alerts in the window that make a campaign
    url_malicious: float = Field(0.9, gt=0, le=1)


class BehaviourParams(BaseModel):
    """Thresholds for behaviour descriptions; tuned on the development alert set only."""

    enabled: bool = True
    # Candidates added from a behaviour-only query (of copilot.n_techniques in total).
    extra_candidates: int = Field(0, ge=0)
    scan_ports: int = Field(20, ge=2)  # distinct ports from one source in the window
    sweep_hosts: int = Field(10, ge=2)  # distinct hosts from one source
    repeat_min: int = Field(20, ge=2)  # pair connections for "repeated" (login services)
    web_repeat_min: int = Field(10, ge=2)  # same, for web requests
    alike_cv: float = Field(0.5, gt=0)  # coefficient of variation of request sizes
    flood_alerts: int = Field(500, ge=10)  # alerts to one target in the window
    flood_sources: int = Field(5, ge=2)  # sources for "distributed"
    slow_seconds: float = Field(30.0, gt=0)  # mean connection duration
    slow_pps: float = Field(5.0, gt=0)  # packets per second
    slow_bwd_bytes: float = Field(1_000, ge=0)  # the server barely answers
    beacon_min: int = Field(5, ge=3)
    beacon_min_interval_s: float = Field(5.0, gt=0)
    beacon_cv: float = Field(0.5, gt=0)  # regularity of the intervals
    leak_bytes: float = Field(10_000, gt=0)
    leak_ratio: float = Field(10.0, gt=1)
    few_requests: int = Field(10, ge=1)
    large_request: float = Field(1_000, gt=0)


class CopilotParams(BaseModel):
    behaviour: BehaviourParams = BehaviourParams()
    llm: CopilotLLMParams = CopilotLLMParams()
    guard: GuardParams = GuardParams()
    severity: SeverityParams = SeverityParams()
    assets: list[AssetParams] = Field(default_factory=list)
    related_window_minutes: int = Field(30, ge=1)
    # Development alert set (validation split) for tuning the copilot without touching test.
    dev_database: str = "sentinel_dev"
    dev_email_sample: dict[str, int] = Field(default_factory=lambda: {"phishing": 60, "fraud": 60})
    n_techniques: int = Field(8, ge=1)  # retrieved candidates the LLM may choose from
    n_cves: int = Field(3, ge=0)
    # Cross-encoder score a CVE needs to be offered at all. Flow records carry no product
    # or version, so irrelevant CVEs score far below 0; a real match scores well above 1.
    cve_min_score: float = 1.0
    # Emails scored per kind when filling the alerts table (flagged ones become alerts).
    email_sample: dict[str, int] = Field(
        default_factory=lambda: {"phishing": 309, "fraud": 300, "spam": 300, "legitimate": 300}
    )


class Params(BaseModel):
    seed: int = 42
    data: DataParams
    split: SplitParams = SplitParams()
    features: FeatureParams = FeatureParams()
    ids: IDSParams = IDSParams()
    anomaly: AnomalyParams = AnomalyParams()
    phishing: PhishingParams = PhishingParams()
    adversarial: AdversarialParams = AdversarialParams()
    copilot: CopilotParams = CopilotParams()

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path


class Settings(BaseSettings):
    """Secrets and service endpoints, read from environment / .env."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    postgres_url: str = "postgresql+psycopg://sentinel:sentinel@localhost:5432/sentinel"
    # The copilot's tools connect with this SELECT-only role (created by `sentinel alerts load`).
    postgres_ro_password: str = "sentinel_ro"
    postgres_ro_url: str = "postgresql+psycopg://sentinel_ro:sentinel_ro@localhost:5432/sentinel"
    mlflow_tracking_uri: str = "http://localhost:5000"
    kafka_bootstrap: str = "localhost:19092"
    gemini_api_key: str | None = None
    ollama_url: str = "http://localhost:11434"
    # Comma-separated override of copilot.llm.providers, e.g. LLM_PROVIDERS=ollama (offline).
    llm_providers: str | None = None
    # The API loads this registry alias; if MLflow is unreachable it falls back to the
    # bundle directory written by the last local training run.
    ids_model_uri: str = "models:/ids-classifier@production"
    ids_model_path: Path = PROJECT_ROOT / "models" / "ids"
    anomaly_model_uri: str = "models:/anomaly-detector@production"
    anomaly_model_path: Path = PROJECT_ROOT / "models" / "anomaly"
    phishing_model_uri: str = "models:/phishing-classifier@production"
    phishing_model_path: Path = PROJECT_ROOT / "models" / "phishing" / "bundle"
    url_model_uri: str = "models:/url-classifier@production"
    url_model_path: Path = PROJECT_ROOT / "models" / "phishing" / "url_bundle"
    log_level: str = "INFO"


def load_params(path: Path | None = None) -> Params:
    path = path or PROJECT_ROOT / "params.yaml"
    with path.open(encoding="utf-8") as f:
        return Params.model_validate(yaml.safe_load(f))


@lru_cache
def get_settings() -> Settings:
    return Settings()
