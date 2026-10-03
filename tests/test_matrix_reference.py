"""CLI preparation checks use a temporary reference matrix and no training/data IO."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.data.preprocess import file_hash, read_config, write_json
from src.training import matrix_compare, research_matrix


@pytest.fixture
def reference_matrix(tmp_path, monkeypatch):
    codes = [f"{index:06d}" for index in range(300)]
    core = dict(data_source="sina", stock_codes=codes,
                feature_cols=[f"signal_{index}" for index in range(14)] + ["momentum_20"], lookback=1,
                price_basis="hfq_research", broker=dict(max_weight=0.05), initial_capital=1000000,
                mapping_version="masked_topk_softmax_cap_cash_v2", action_top_k=20,
                manifest_sha256="synthetic-manifest", artifacts={"prices.parquet": "synthetic-prices"})
    prior = dict(feature_index=14, strength=4.0, residual_scale=0.25)
    base = dict(stock_codes=codes, lookback=1, seed=42, hidden_sizes=[256, 128],
                device="cuda", torch_threads=1, logit_bound=8.0, prior_feature_index=14,
                prior_strength=4.0, residual_scale=0.25, initial_log_std=-3.0, action_top_k=20,
                training_episode_steps=256, recent_episode_every=4, output_root="registered-prior-runs",
                ppo=dict(n_steps=512, batch_size=128, n_epochs=5, learning_rate=1e-5,
                         ent_coef=0.0, vf_coef=0.5, gamma=0.99, gae_lambda=0.95,
                         clip_range=0.2, target_kl=0.03, max_grad_norm=0.5))
    design = dict(small_hidden=[256, 128], large_hidden=[2048, 1024], long_steps=65536,
                  memory_selection="synthetic registered design")
    reference = tmp_path / "prior_matrix"
    reference.mkdir()
    reference_protocol = dict(design=design, seeds=[42, 123, 2026], short_steps=8192, long_steps=65536,
                              core_contract=dict(core, prior=prior), base_config=deepcopy(base))
    write_json(reference / "protocol.json", reference_protocol)
    write_json(reference / "status.json", dict(status="complete"))
    config = tmp_path / "base.json"
    write_json(config, base)
    source_state = {"src/training/research_matrix.py": "synthetic-source-a"}
    prepared_configs = []
    audit_calls = []
    audit_state = dict(error=None)
    evidence = dict(verified_file_count=624, training_runs=6, checkpoints=12,
                    total_trained_steps=393216, test_evaluated=False)

    class FakeResearchData:
        def __init__(self, cfg):
            prepared_configs.append(deepcopy(cfg))
            self.config = deepcopy(cfg)
            self.codes = list(cfg["stock_codes"])
            self.contract = dict(deepcopy(core), hidden_sizes=list(cfg["hidden_sizes"]))
            if cfg.get("prior_feature_index") is not None:
                self.contract["prior"] = deepcopy(prior)
            self.manifest = dict(train_end="2026-03-31", val_end="2026-06-30")

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference preparation must not probe capacity or start training")

    def fake_load_matrix(root, mode):
        assert root == reference.resolve() and mode == "prior"
        audit_calls.append((root, mode))
        if audit_state["error"] is not None:
            raise audit_state["error"]
        return dict(evidence=deepcopy(evidence))

    monkeypatch.setattr(matrix_compare, "load_matrix", fake_load_matrix)
    monkeypatch.setattr(research_matrix, "ResearchData", FakeResearchData)
    monkeypatch.setattr(research_matrix, "source_hashes", lambda: deepcopy(source_state))
    monkeypatch.setattr(research_matrix, "choose_design", forbidden)
    monkeypatch.setattr(research_matrix, "run_trial", forbidden)
    monkeypatch.setattr(research_matrix.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(research_matrix.torch.cuda, "get_device_name", lambda device: "Synthetic GPU")
    return SimpleNamespace(config=config, base=base, reference=reference, design=design,
                           output=tmp_path / "random_matrix", sources=source_state,
                           prepared_configs=prepared_configs, audit_calls=audit_calls,
                           audit_state=audit_state, evidence=evidence)


def arguments(fixture, mode="random"):
    return ["--config", str(fixture.config), "--output", str(fixture.output),
            "--initialization-mode", mode, "--reference-matrix", str(fixture.reference)]


def test_random_prepare_uses_registered_design_budgets_and_seeds_without_training(reference_matrix):
    fixture = reference_matrix
    inputs = [fixture.config, fixture.reference / "protocol.json", fixture.reference / "status.json"]
    original = {path: path.read_bytes() for path in inputs}

    assert research_matrix.main(arguments(fixture)) == fixture.output.resolve()

    protocol_path = fixture.output / "protocol.json"
    protocol = read_config(protocol_path)
    assert protocol["initialization_mode"] == "random"
    assert protocol["engineering_smoke"] is False
    assert protocol["design"] == fixture.design
    assert protocol["design"]["small_hidden"] == [256, 128]
    assert protocol["design"]["large_hidden"] == [2048, 1024]
    assert protocol["short_steps"] == 8192 and protocol["long_steps"] == 65536
    assert protocol["seeds"] == [42, 123, 2026]
    assert protocol["maximum_training_steps"] == 393216
    assert protocol["base_config"]["prior_feature_index"] is None
    assert "prior" not in protocol["core_contract"]
    assert protocol["base_config"]["initial_log_std"] == -3.0
    assert protocol["base_config"]["ppo"] == fixture.base["ppo"]
    assert protocol["source_hashes"] == fixture.sources
    assert protocol["gpu"] == "Synthetic GPU" and protocol["test_evaluated"] is False
    assert protocol["reference_matrix"]["root"] == str(fixture.reference.resolve())
    assert protocol["reference_matrix"]["protocol_sha256"] == file_hash(fixture.reference / "protocol.json")
    assert protocol["reference_matrix"]["evidence"] == fixture.evidence
    assert fixture.audit_calls == [(fixture.reference.resolve(), "prior")]
    assert fixture.prepared_configs[-1]["hidden_sizes"] == [256, 128]
    assert fixture.prepared_configs[-1]["prior_feature_index"] is None
    frozen = protocol_path.read_bytes()

    assert research_matrix.main(arguments(fixture)) == fixture.output.resolve()

    assert protocol_path.read_bytes() == frozen
    assert fixture.audit_calls == [(fixture.reference.resolve(), "prior")] * 2
    assert {path: path.read_bytes() for path in inputs} == original
    assert not (fixture.output / "runs").exists()
    assert not (fixture.output / "summary.json").exists()
    assert not (fixture.output / "status.json").exists()


@pytest.mark.parametrize("status", ["training", "failed", "prepared"])
def test_prepare_rejects_a_reference_matrix_that_is_not_complete(reference_matrix, status):
    fixture = reference_matrix
    write_json(fixture.reference / "status.json", dict(status=status))
    with pytest.raises(ValueError, match="reference matrix must be complete"):
        research_matrix.main(arguments(fixture))
    assert not fixture.prepared_configs
    assert not fixture.audit_calls
    assert not fixture.output.exists()


@pytest.mark.parametrize("changed", ["broker", "artifacts", "stock_codes"])
def test_prepare_rejects_reference_data_or_execution_contract_changes(reference_matrix, changed):
    fixture = reference_matrix
    reference = read_config(fixture.reference / "protocol.json")
    if changed == "broker":
        reference["core_contract"]["broker"]["max_weight"] = 0.1
    elif changed == "artifacts":
        reference["core_contract"]["artifacts"]["prices.parquet"] = "different-prices"
    else:
        reference["core_contract"]["stock_codes"].reverse()
    write_json(fixture.reference / "protocol.json", reference)
    with pytest.raises(ValueError, match="Reference matrix data or execution contract changed"):
        research_matrix.main(arguments(fixture))
    assert not fixture.output.exists()


def test_prior_mode_rejects_a_missing_policy_prior(reference_matrix):
    fixture = reference_matrix
    base = deepcopy(fixture.base)
    base["prior_feature_index"] = None
    write_json(fixture.config, base)
    with pytest.raises(ValueError, match="initialization mode differs from the policy prior"):
        research_matrix.main(arguments(fixture, mode="prior"))
    assert not fixture.output.exists()


@pytest.mark.parametrize("changed", ["source", "mode"])
def test_prepared_protocol_rejects_source_or_initialization_mode_changes(reference_matrix, changed):
    fixture = reference_matrix
    research_matrix.main(arguments(fixture))
    path = fixture.output / "protocol.json"
    frozen = path.read_bytes()
    mode = "random"
    if changed == "source":
        fixture.sources["src/training/research_matrix.py"] = "synthetic-source-b"
    else:
        mode = "prior"
    with pytest.raises(ValueError, match="Matrix protocol changed"):
        research_matrix.main(arguments(fixture, mode=mode))
    assert path.read_bytes() == frozen
    assert not (fixture.output / "runs").exists()


@pytest.mark.parametrize("override", [
    ["--short-steps", "4096"],
    ["--long-steps", "32768"],
    ["--seeds", "42", "123"],
    ["--seeds", "2026", "123", "42"],
])
def test_formal_prepare_requires_the_registered_budgets_and_seed_order(reference_matrix, override):
    fixture = reference_matrix
    with pytest.raises(ValueError, match="preserve the reference budgets and seeds"):
        research_matrix.main(arguments(fixture) + override)
    assert not fixture.prepared_configs
    assert not fixture.output.exists()


def test_explicit_engineering_smoke_allows_reduced_budgets_without_training(reference_matrix):
    fixture = reference_matrix
    smoke = arguments(fixture) + ["--smoke", "--short-steps", "512", "--long-steps", "1024", "--seeds", "42"]

    assert research_matrix.main(smoke) == fixture.output.resolve()

    path = fixture.output / "protocol.json"
    protocol = read_config(path)
    assert protocol["engineering_smoke"] is True
    assert protocol["design"] == fixture.design
    assert protocol["short_steps"] == 512 and protocol["long_steps"] == 1024
    assert protocol["seeds"] == [42]
    assert protocol["maximum_training_steps"] == 2048
    frozen = path.read_bytes()
    assert research_matrix.main(smoke) == fixture.output.resolve()
    assert path.read_bytes() == frozen
    assert not (fixture.output / "runs").exists()
    assert not (fixture.output / "status.json").exists()


@pytest.mark.parametrize("overlap", ["same", "descendant", "ancestor"])
def test_prepare_rejects_overlapping_reference_and_output_directories(reference_matrix, overlap):
    fixture = reference_matrix
    protected = [fixture.config, fixture.reference / "protocol.json", fixture.reference / "status.json"]
    original = {path: path.read_bytes() for path in protected}
    if overlap == "same":
        fixture.output = fixture.reference
    elif overlap == "descendant":
        fixture.output = fixture.reference / "nested_random_matrix"
    else:
        fixture.output = fixture.reference.parent
    with pytest.raises(ValueError, match="directories must be separate"):
        research_matrix.main(arguments(fixture))
    assert not fixture.prepared_configs
    assert not fixture.audit_calls
    assert {path: path.read_bytes() for path in protected} == original
    if overlap == "descendant":
        assert not fixture.output.exists()
    elif overlap == "ancestor":
        assert not (fixture.output / "protocol.json").exists()


@pytest.mark.parametrize("changed", ["noise", "learning_rate", "entropy"])
def test_prepare_rejects_hyperparameter_or_exploration_changes(reference_matrix, changed):
    fixture = reference_matrix
    base = deepcopy(fixture.base)
    if changed == "noise":
        base["initial_log_std"] = -2.0
    elif changed == "learning_rate":
        base["ppo"]["learning_rate"] = 2e-5
    else:
        base["ppo"]["ent_coef"] = 1e-5
    write_json(fixture.config, base)
    with pytest.raises(ValueError, match="hyperparameters or exploration changed"):
        research_matrix.main(arguments(fixture))
    assert fixture.audit_calls == [(fixture.reference.resolve(), "prior")]
    assert not fixture.prepared_configs
    assert not fixture.output.exists()


def test_prepare_allows_a_new_output_root_while_preserving_registered_hyperparameters(reference_matrix):
    fixture = reference_matrix
    base = deepcopy(fixture.base)
    base["output_root"] = "different-requested-runs"
    write_json(fixture.config, base)

    assert research_matrix.main(arguments(fixture)) == fixture.output.resolve()

    protocol = read_config(fixture.output / "protocol.json")
    assert protocol["base_config"]["output_root"] == "different-requested-runs"
    assert protocol["base_config"]["prior_feature_index"] is None
    assert protocol["base_config"]["initial_log_std"] == -3.0
    assert protocol["reference_matrix"]["evidence"] == fixture.evidence


def test_reference_evidence_corruption_propagates_before_any_output_is_created(reference_matrix):
    fixture = reference_matrix
    protected = [fixture.config, fixture.reference / "protocol.json", fixture.reference / "status.json"]
    original = {path: path.read_bytes() for path in protected}
    corruption = ValueError("Synthetic reference model hash mismatch")
    fixture.audit_state["error"] = corruption

    with pytest.raises(ValueError, match="reference model hash mismatch") as raised:
        research_matrix.main(arguments(fixture))

    assert raised.value is corruption
    assert fixture.audit_calls == [(fixture.reference.resolve(), "prior")]
    assert not fixture.prepared_configs
    assert not fixture.output.exists()
    assert {path: path.read_bytes() for path in protected} == original
