"""Bounded requests. The browser cannot choose commands, interpreters or paths."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Training(Request):
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


class Build(Request):
    dataset_id: str
    version: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{2,47}$")
    codes: list[str] = Field(min_length=1, max_length=500)
    start_date: date
    train_end: date
    val_end: date
    end_date: date

    @model_validator(mode="after")
    def valid_dates(self):
        if not self.start_date < self.train_end < self.val_end < self.end_date:
            raise ValueError("Expected start < train_end < val_end < end")
        if len(set(self.codes)) != len(self.codes):
            raise ValueError("Duplicate stock codes")
        return self


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
    codes: list[str] = Field(min_length=1, max_length=100)
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


class MonitorRequest(OnlineQuotes):
    interval_seconds: int = Field(default=60, ge=60, le=600)
    duration_seconds: int = Field(default=300, ge=60, le=1800)
