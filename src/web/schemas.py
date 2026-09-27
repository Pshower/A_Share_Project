"""Bounded requests. The browser cannot choose commands, interpreters or paths."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Training(Request):
    training_date_policy: Literal["intersection", "legacy_masked"] = "intersection"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dataset_id: str
    codes: list[str] = Field(min_length=1, max_length=500)
    total_timesteps: int = Field(default=8192, ge=8, le=1000000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    lookback: int = Field(default=1, ge=1, le=60)
    hidden_sizes: list[int] = Field(default=[128, 64], min_length=2, max_length=2)
    n_steps: int = Field(default=256, ge=8, le=4096)
    batch_size: int = Field(default=64, ge=2, le=4096)
    n_epochs: int = Field(default=10, ge=1, le=30)
    learning_rate: float = Field(default=0.0003, gt=0, le=0.01)
    learning_rate_schedule: Literal["constant", "linear"] = "constant"
    learning_rate_end_factor: float = Field(default=0.1, ge=0, le=1)
    gamma: float = Field(default=0.99, ge=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    clip_range: float = Field(default=0.2, gt=0, le=0.5)
    target_kl: float = Field(default=0.03, gt=0, le=0.2)
    ent_coef: float = Field(default=0, ge=0, le=0.1)
    vf_coef: float = Field(default=0.5, ge=0, le=2)
    max_grad_norm: float = Field(default=0.5, gt=0, le=10)
    max_validation_drawdown: float | None = Field(default=None, gt=0, le=1)
    max_validation_turnover: float | None = Field(default=None, gt=0, le=10)
    min_validation_steps: int = Field(default=20, ge=2, le=2000)
    validation_rollouts: int = Field(default=10, ge=1, le=100)
    name: str = Field(default="PPO experiment", max_length=80)

    @model_validator(mode="after")
    def valid_shape(self):
        if len(set(self.codes)) != len(self.codes) or self.n_steps % self.batch_size:
            raise ValueError("Unique stocks required; batch_size must divide n_steps")
        if any(v < 8 or v > 512 for v in self.hidden_sizes):
            raise ValueError("Hidden layers must be between 8 and 512 units")
        if self.n_steps * len(self.codes) * self.lookback * 20 * 4 * 3 > 512 * 1024 * 1024:
            raise ValueError("Estimated rollout allocation exceeds the local 512 MB limit")
        return self


class Coverage(Request):
    codes: list[str] = Field(min_length=1, max_length=500)
    lookback: int = Field(default=1, ge=1, le=60)


class FeatureDefinition(Request):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,39}$")
    kind: Literal["momentum", "ma_gap", "log_volatility", "drawdown", "atr_ratio", "relative_volume", "relative_amount", "bb_width"]
    window: int = Field(ge=2, le=250, strict=True)


class FeaturePreview(Request):
    code: str = Field(pattern=r"^[0-9]{6}$")
    extra_features: list[FeatureDefinition] = Field(min_length=1, max_length=12)


class Build(Request):
    extra_features: list[FeatureDefinition] | None = Field(default=None, max_length=12)
    dataset_id: str
    version: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{2,47}$")
    codes: list[str] = Field(min_length=1, max_length=500)
    start_date: date
    train_end: date
    val_end: date
    end_date: date

    @model_validator(mode="after")
    def valid_dates(self):
        if self.extra_features is not None:
            from src.data.feature_definitions import validate_definitions
            from src.data.preprocess import FEATURES
            validate_definitions([d.model_dump() for d in self.extra_features], FEATURES)
        if not self.start_date < self.train_end < self.val_end < self.end_date:
            raise ValueError("Expected start < train_end < val_end < end")
        if len(set(self.codes)) != len(self.codes):
            raise ValueError("Duplicate stock codes")
        return self


class ExperimentRecipe(Request):
    name: str = Field(min_length=1, max_length=80)
    training: Training
    build: Build | None = None
    history_ids: list[str] = Field(default_factory=list, max_length=20)
    source_mode: Literal["dataset", "history"] = "dataset"
    seeds: list[int] = Field(default=[42], min_length=1, max_length=5)
    parent_id: str | None = None

    @model_validator(mode="after")
    def coherent(self):
        if len(set(self.seeds)) != len(self.seeds) or any(s < 0 or s > 2147483647 for s in self.seeds):
            raise ValueError("Choose up to five unique nonnegative seeds")
        if len(set(self.history_ids)) != len(self.history_ids):
            raise ValueError("Repeated source snapshots")
        if self.history_ids and self.build is None:
            raise ValueError("Raw captures require a new dataset build")
        if self.source_mode == "history" and self.build is None:
            raise ValueError("History mode requires a dataset build")
        if self.source_mode == "dataset" and self.history_ids:
            raise ValueError("Choose history mode for snapshot sources")
        if self.build and (self.build.dataset_id != self.training.dataset_id or self.build.codes != self.training.codes):
            raise ValueError("Build and training must use the same template and ordered stock pool")
        from src.data.online import valid_codes
        valid_codes(self.training.codes)
        return self


class ExperimentApproval(Request):
    allow_network: bool = False
    allow_training: bool = False


class ModelRequest(Request):
    model_id: str


class Evaluation(ModelRequest):
    split: Literal["val", "test"] = "val"
    frozen_job_id: str | None = None
    acknowledge_test: bool = False

    @model_validator(mode="after")
    def test_guard(self):
        if self.split == "test" and (not self.frozen_job_id or not self.acknowledge_test):
            raise ValueError("Testing requires a freeze and explicit exposure acknowledgement")
        if self.split == "val" and self.frozen_job_id:
            raise ValueError("Validation does not use a frozen test selection")
        return self


class Transfer(ModelRequest):
    codes: list[str] = Field(min_length=1, max_length=500)


class Position(Request):
    code: str = Field(pattern=r"^\d{6}$")
    shares: int = Field(ge=0, le=1000000000)
    available_shares: int = Field(ge=0, le=1000000000)


class Receivable(Request):
    date: date
    amount: float = Field(ge=0, le=1000000000000)


class Account(Request):
    cash: float = Field(ge=0, le=1000000000000)
    positions: list[Position] = Field(default=[], max_length=500)
    receivables: list[Receivable] = Field(default=[], max_length=1000)


class Prediction(ModelRequest):
    as_of: date
    display_codes: list[str] = Field(min_length=1, max_length=500)
    account: Account | None = None


class Label(Request):
    label: str = Field(min_length=1, max_length=80)


class OnlineHistory(Request):
    source: Literal["eastmoney", "sina"] = "sina"
    codes: list[str] = Field(min_length=1, max_length=500)
    start_date: date
    end_date: date
    bases: list[Literal["hfq", "unadjusted"]] = Field(default=["hfq", "unadjusted"], min_length=1, max_length=2)
    allow_network: Literal[True]
    resume_id: str | None = None

    @model_validator(mode="after")
    def validate_request(self):
        from src.data.online import valid_codes, CHINA
        from datetime import datetime
        valid_codes(self.codes)
        if self.start_date > self.end_date or self.end_date > datetime.now(CHINA).date() or len(set(self.bases)) != len(self.bases):
            raise ValueError("Invalid download dates or repeated price basis")
        return self


class StockList(Request):
    text: str = Field(min_length=1, max_length=200000)
    format: Literal["txt", "csv"] = "txt"


class Supplement(Request):
    data_type: Literal["profile", "fund_flow", "minute", "index_daily", "dividend"]
    codes: list[str] = Field(min_length=1, max_length=20)
    market: Literal["sh", "sz", "bj"] = "sz"
    start_date: date
    end_date: date
    period: Literal["1", "5", "15", "30", "60"] = "5"
    allow_network: Literal[True]
    resume_id: str | None = None

    @model_validator(mode="after")
    def valid_request(self):
        from src.data.supplement import validate_request
        validate_request(self.model_dump(mode="json", exclude={"allow_network", "resume_id"}))
        return self


class OnlineQuotes(Request):
    codes: list[str] = Field(min_length=1, max_length=50)
    allow_network: Literal[True]
    plan_id: str | None = None

    @model_validator(mode="after")
    def validate_codes(self):
        from src.data.online import valid_codes
        valid_codes(self.codes)
        return self


class DailyPlan(ModelRequest):
    history_id: str
    close_as_of: date | None = None
    accept_revisions: bool = False
    acknowledge_research: Literal[True]
    research_weights: dict[str, float] = Field(default={})

    @model_validator(mode="after")
    def weights(self):
        if any(v < 0 for v in self.research_weights.values()) or sum(self.research_weights.values()) > 1:
            raise ValueError("Research weights must be nonnegative and sum to at most one")
        return self


class OnlineBuild(Build):
    history_id: str


class ModelComparison(Request):
    model_ids: list[str] = Field(min_length=1, max_length=3)
    history_id: str
    close_as_of: date
    display_code: str = Field(pattern=r"^\d{6}$")
    account_mode: Literal["cash", "weights"]
    research_weights: dict[str, float] = Field(default={})
    accept_revisions: bool = False
    acknowledge_research: Literal[True]

    @model_validator(mode="after")
    def validate_comparison(self):
        from src.data.online import valid_codes
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("Select unique models")
        if self.research_weights:
            valid_codes(list(self.research_weights))
        if any(v < 0 for v in self.research_weights.values()) or sum(self.research_weights.values()) > 1:
            raise ValueError("Research weights must be nonnegative and sum to at most one")
        if self.account_mode == "cash" and self.research_weights:
            raise ValueError("Cash scenario cannot include positions")
        if self.account_mode == "weights" and not self.research_weights:
            raise ValueError("Weights scenario requires explicit positions")
        return self


class MonitorRequest(OnlineQuotes):
    interval_seconds: int = Field(default=60, ge=60, le=600)
    duration_seconds: int = Field(default=300, ge=60, le=1800)
