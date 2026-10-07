"""Offline protection of the matched-initial-function summary study."""

from copy import deepcopy
from types import SimpleNamespace
import zipfile

import pytest
import torch

from src.data.preprocess import write_json
from src.training import research_market_summary as study


@pytest.fixture
def reference(tmp_path):
    folder = tmp_path / "reference"
    folder.mkdir()
    write_json(folder / "protocol.json", dict(folds={fold: dict(directory=str(tmp_path / "folds" / fold))
                                                   for fold in study.FOLDS}))
    return folder


@pytest.mark.parametrize("fold", study.FOLDS)
@pytest.mark.parametrize("arm", study.ARMS)
def test_config_changes_only_pooling_and_fold(reference, tmp_path, fold, arm):
    base = dict(ppo={})
    original = deepcopy(base)
    config = study.config_for(base, reference, tmp_path / "output", fold, arm, 42)
    assert base == original
    assert config["research_config"].endswith(f"{fold}\\research.json") or config["research_config"].endswith(f"{fold}/research.json")
    assert config["market_summary"] == dict(pooling=arm, fold=fold, critic_features="shared",
        zero_query_initialization=True, zero_variance_projection_initialization=True)
    assert config["exploration"] == dict(architecture="split", sampling="recent", common_lookback=5,
        reward_basis="simple_net", size_annealing=False)
    assert config["lookback"] == 1 and config["actor_output_gain"] == .1
    assert config["initial_log_std"] == -.7 and config["ppo"]["gamma"] == .99
    assert config["prior_feature_index"] is None
    assert config["initial_portfolio"] == "mixed" and config["top_n_choices"] == [10, 20, 40]
    assert config["training_episode_steps"] == 256 and config["recent_episode_every"] == 4


@pytest.mark.parametrize("fold,arm,seed", [("2025", "mean", 42), ("2023", "max", 42), ("2023", "mean", 7)])
def test_unregistered_conditions_fail_before_reading_reference(tmp_path, fold, arm, seed):
    with pytest.raises(ValueError, match="Unregistered"):
        study.config_for(dict(ppo={}), tmp_path / "missing", tmp_path, fold, arm, seed)


def fake_records():
    result = []
    for fold in study.FOLDS:
        for arm in study.ARMS:
            for seed in study.SEEDS:
                for budget, steps in study.BUDGETS.items():
                    initial = .01 + (seed == 123) * .001
                    gain = .02 + {"mean": 0., "attention": .01, "dispersion": .02}[arm]
                    roi = initial + gain + (fold == "2024") * .002
                    metric = dict(cumulative_return=roi, max_drawdown=-.03, total_fees=100., total_slippage_cost=50.)
                    result.append(dict(fold=fold, arm=arm, seed=seed, budget=budget, steps=steps,
                        metrics={mode: dict(ppo=deepcopy(metric), initialized=dict(metric, cumulative_return=initial))
                                 for mode in ("cash", "equal", "concentrated", "random")}))
    return result


def fake_rules():
    return {fold: {mode: dict(momentum_20day=dict(cumulative_return=.005))
                   for mode in ("cash", "equal", "concentrated", "random")} for fold in study.FOLDS}


def test_budget_counts_do_not_double_count_short_prefixes():
    result = study.summarize(fake_records(), fake_rules())
    assert result["actual_training_steps"] == 147456
    assert result["checkpoint_steps_sum"] == 221184
    assert result["trained_runs"] == 18 and result["snapshots"] == 36
    assert result["arms"]["attention_long"]["metrics"]["cumulative_return"]["count"] == 3
    assert result["qualified_candidate"] == dict(arm="dispersion", budget="short", representative_seed=42,
                                                representative_fold="2024")
    assert not result["actionable"] and not result["test_evaluated"]


def test_paired_statistics_use_seed_means_and_both_folds():
    result = study.summarize(fake_records(), fake_rules())
    for budget in study.BUDGETS:
        assert result["arms"][f"attention_{budget}"]["paired_roi_vs_mean"]["mean"] == pytest.approx(.01)
        assert result["arms"][f"dispersion_{budget}"]["paired_roi_vs_mean"]["mean"] == pytest.approx(.02)
        assert result["arms"][f"mean_{budget}"]["paired_roi_vs_mean"]["mean"] == 0
        assert not result["arms"][f"mean_{budget}"]["improves_mean"]
        assert result["arms"][f"dispersion_{budget}"]["improves_mean"]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "wrong_steps", "wrong_fold", "wrong_seed", "wrong_arm"])
def test_partial_or_changed_results_never_claim_complete(failure):
    records = fake_records()
    if failure == "missing":
        records.pop()
    elif failure == "duplicate":
        records[-1] = deepcopy(records[-2])
    else:
        field, value = {"wrong_steps": ("steps", 4096), "wrong_fold": ("fold", "2025"),
                        "wrong_seed": ("seed", 7), "wrong_arm": ("arm", "max")}[failure]
        records[-1][field] = value
    with pytest.raises(ValueError):
        study.validate_records(records)


def test_risk_is_worst_account_not_average_and_each_fold_must_pass():
    records = fake_records()
    for row in records:
        if row["fold"] == "2023":
            row["metrics"]["cash"]["ppo"]["max_drawdown"] = -.101
    result = study.summarize(records, fake_rules())
    assert result["qualified_candidate"] is None
    assert all(not item["qualified"] for item in result["arms"].values())


def test_two_positive_learning_seeds_required_despite_positive_mean():
    records = fake_records()
    for row in records:
        for mode in row["metrics"].values():
            mode["ppo"]["cumulative_return"] = .50 if row["seed"] == 42 else -.01
    result = study.summarize(records, fake_rules())
    assert result["qualified_candidate"] is None
    assert all(item["own_initialization_gain"]["mean"] > 0 for item in result["arms"].values())


def test_new_arm_improvement_requires_each_fold_not_only_pooled_average():
    records = fake_records()
    for row in records:
        if row["fold"] == "2023" and row["arm"] != "mean":
            for mode in row["metrics"].values():
                mode["ppo"]["cumulative_return"] -= .03
    result = study.summarize(records, fake_rules())
    assert not result["arms"]["attention_long"]["improves_mean"]
    assert not result["arms"]["dispersion_long"]["improves_mean"]
    assert result["qualified_candidate"]["arm"] == "mean"


def test_new_arm_improvement_requires_two_positive_paired_seeds():
    records = fake_records()
    for row in records:
        if row["arm"] != "mean":
            for mode in row["metrics"].values():
                mode["ppo"]["cumulative_return"] = .30 if row["seed"] == 42 else .02
    result = study.summarize(records, fake_rules())
    assert result["arms"]["attention_long"]["paired_roi_vs_mean"]["mean"] > 0
    assert not result["arms"]["attention_long"]["improves_mean"]


def test_unequal_initialization_cannot_be_silently_called_learning_gain():
    records = fake_records()
    records[-1]["metrics"]["cash"]["initialized"]["cumulative_return"] += .01
    with pytest.raises(ValueError, match="identical initialized"):
        study.summarize(records, fake_rules())


def test_input_order_does_not_change_seed_pairing():
    records = fake_records()
    assert study.summarize(records, fake_rules()) == study.summarize(list(reversed(records)), fake_rules())


def test_canonical_contract_only_removes_pooling():
    contract = dict(market_summary=dict(pooling="attention", fold="2023", critic_features="shared"),
                    features=["return"], cost=.001)
    original = deepcopy(contract)
    canonical = study._canonical_contract(contract)
    assert contract == original
    assert canonical == dict(market_summary=dict(fold="2023", critic_features="shared"), features=["return"], cost=.001)


def tiny_policy(pooling="mean"):
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.summary_query = torch.nn.Parameter(torch.zeros(2))
            self.variance_projection = torch.nn.Linear(2, 2, bias=False)
            torch.nn.init.zeros_(self.variance_projection.weight)
            self.body = torch.nn.Linear(2, 1)
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = Network()
            self.pooling = pooling
            self.optimizer = torch.optim.Adam(self.parameters())
    return Policy()


@pytest.mark.parametrize("field", ["summary_query", "variance_projection.weight"])
def test_initial_equivalence_requires_both_exact_zero_parameters(field):
    policy = tiny_policy()
    assert study.neutral_initialization(policy)["query_exact_zero"]
    parameter = dict(policy.network.named_parameters())[field]
    with torch.no_grad():
        parameter.reshape(-1)[0] = 1e-10
    with pytest.raises(ValueError, match="exactly zero"):
        study.neutral_initialization(policy)


@pytest.mark.parametrize("pooling,active", [("mean", 3), ("attention", 5), ("dispersion", 7)])
def test_parameter_participation_does_not_claim_inactive_adam_states(pooling, active):
    policy = tiny_policy(pooling)
    loss = policy.network.body(torch.ones(1, 2)).sum()
    if pooling == "attention":
        loss += policy.network.summary_query.sum()
    elif pooling == "dispersion":
        loss += policy.network.variance_projection.weight.sum()
    loss.backward()
    policy.optimizer.step()
    result = study.parameter_participation(SimpleNamespace(policy=policy))
    assert result["total_parameters"] == 9
    assert result["expected_functionally_active_parameters"] == active
    assert result["adam_state_parameters"] == active
    assert not result["inactive_adam_state_names"] and not result["inactive_nonzero_names"]
    assert (result["summary_query_l2"] > 0) == (pooling == "attention")
    assert (result["variance_projection_l2"] > 0) == (pooling == "dispersion")


def test_initial_byte_pair_detects_different_theta(tmp_path):
    directory = tmp_path / "runs" / "2023_mean_42"
    other = tmp_path / "runs" / "2023_attention_42"
    for folder in (directory, other):
        (folder / "initial_model").mkdir(parents=True)
        with zipfile.ZipFile(folder / "initial_model" / "model.zip", "w") as archive:
            archive.writestr("policy.pth", b"theta")
    result = study.verify_initial_pair(tmp_path, "2023", "mean", 42, directory, tiny_policy())
    assert result["matched_existing_arms"] == ["attention"]
    with zipfile.ZipFile(other / "initial_model" / "model.zip", "w") as archive:
        archive.writestr("policy.pth", b"changed")
    with pytest.raises(AssertionError, match="byte-identical"):
        study.verify_initial_pair(tmp_path, "2023", "mean", 42, directory, tiny_policy())


@pytest.mark.parametrize("wrong", ["class", "architecture", "policy_pooling", "network_pooling"])
def test_loader_rejects_policy_flags_not_matching_contract(monkeypatch, wrong):
    class Policy:
        architecture = "split"
        pooling = "attention"
        network = SimpleNamespace(pooling="attention")
    monkeypatch.setattr(study, "MarketSummaryPolicy", Policy)
    policy = Policy()
    if wrong == "class":
        policy = SimpleNamespace(architecture="split", pooling="attention", network=SimpleNamespace(pooling="attention"))
    elif wrong == "architecture":
        policy.architecture = "baseline"
    elif wrong == "policy_pooling":
        policy.pooling = "mean"
    else:
        policy.network = SimpleNamespace(pooling="dispersion")
    monkeypatch.setattr(study.PPOAgent, "load", lambda *args: SimpleNamespace(model=SimpleNamespace(policy=policy)))
    with pytest.raises(ValueError, match="flags"):
        study.load_agent("unused", SimpleNamespace(contract=dict(market_summary=dict(pooling="attention"))))


def test_preparation_rejects_untrusted_audit_before_trusting_protocol(monkeypatch, tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "final_audit.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="sealed reference audit"):
        study.prepare(tmp_path / "output", reference)


@pytest.mark.parametrize("relation", ["same", "output_inside_reference", "reference_inside_output"])
def test_preparation_never_reuses_or_writes_inside_sealed_reference(tmp_path, relation):
    reference, output = tmp_path / "reference", tmp_path / "output"
    if relation == "same":
        output = reference
    elif relation == "output_inside_reference":
        output = reference / "nested_output"
    else:
        reference = output / "nested_reference"
    with pytest.raises(ValueError, match="must not overlap"):
        study.prepare(output, reference)


def test_frozen_source_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr(study, "source_hashes", lambda: {"new.py": "changed"})
    with pytest.raises(ValueError, match="Frozen"):
        study.check_frozen(dict(source_hashes={"new.py": "old"}, document_sha256="old"))


def test_cache_rejects_identity_change_before_file_or_model_reads(tmp_path):
    write_json(tmp_path / "completed.json", dict(config_sha256="changed", protocol_sha256="changed"))
    with pytest.raises(ValueError, match="configuration"):
        study.cached_trial(tmp_path, {}, {})


def test_cache_rejects_partial_snapshots_even_if_marker_hashes_match(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "_verify_files", lambda *args: None)
    write_json(tmp_path / "completed.json", dict(config_sha256=study.json_digest({}), protocol_sha256=study.json_digest({}),
        files={}, records=[dict(budget="short", steps=4096)]))
    with pytest.raises(ValueError, match="two registered"):
        study.cached_trial(tmp_path, {}, {})
