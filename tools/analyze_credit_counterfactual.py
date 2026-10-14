"""Same-path GAE sensitivity from sealed arrays; not new training or model evaluation."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import file_hash, read_config, write_json
from src.training.freeze import json_digest
from tools import research_timing_credit as study
from tools.analyze_timing_credit import read_stage
from tools.audit_timing_credit import estimator_audit, gae_targets


def target_metrics(targets, advantages, old_values):
    target, advantage, old = (np.asarray(item, dtype=float) for item in (targets, advantages, old_values))
    if not target.size or target.shape != advantage.shape or target.shape != old.shape or not all(np.isfinite(item).all() for item in (target, advantage, old)):
        raise ValueError("Same-path targets, advantages and old values must be finite aligned arrays")
    variance = float(np.var(target))
    mse = float(np.mean((target - old) ** 2))
    return dict(return_targets_std=float(np.std(target)), raw_advantages_std=float(np.std(advantage)),
        preupdate_old_value_mse=mse, target_variance=variance,
        mse_to_target_variance=mse / variance if variance > 0. else None,
        ratio_status="defined" if variance > 0. else "undefined_zero_target_variance")


def counterfactual_arrays(buffer, context):
    actual = context["gae_lambda"].item()
    study.cell_label("log_net", actual)
    if float(context["gamma"]) != .995:
        raise ValueError("Counterfactual comparison keeps gamma fixed at .995")
    summaries, recomputed = {}, {}
    for value in study.LAMBDAS:
        advantages, targets = gae_targets(buffer["rewards"], buffer["values"], buffer["episode_starts"],
            context["last_values"], context["last_dones"], .995, value)
        recomputed[value] = (advantages, targets)
        summaries[str(value)] = target_metrics(targets, advantages, buffer["values"])
    actual_adv, actual_returns = recomputed[actual]
    if not np.allclose(actual_adv, buffer["advantages"], rtol=1e-6, atol=1e-6) or not np.allclose(actual_returns, buffer["returns"], rtol=1e-6, atol=1e-6):
        raise ValueError("Actual lambda does not reproduce its original saved GAE targets")
    difference = np.asarray(recomputed[1.][1], dtype=float) - recomputed[.95][1]
    return dict(actual_gae_lambda=actual, gamma=.995, steps=len(buffer["rewards"]), estimators=summaries,
        raw_environment_reward_std=float(np.std(context["raw_rewards"])),
        actual_bootstrap_corrected_reward_std=float(np.std(buffer["rewards"])),
        lambda_one_minus_095_targets=dict(mean=float(difference.mean()), std=float(difference.std()), maximum_absolute=float(np.max(np.abs(difference)))),
        timeout_bootstraps=int(context["truncated"].sum()), natural_terminals=int((context["dones"] & ~context["truncated"]).sum()),
        actual_tail_value=float(context["last_values"][0]), actual_tail_done=bool(context["last_dones"][0]),
        held_fixed="Actual trajectory, raw/corrected rewards, old values, terminal values, tail value, episode/reset masks and gamma")


def output_path(root, requested=None):
    root = Path(root).resolve()
    output = Path(requested or root / "diagnostics" / "credit_counterfactual").resolve()
    if not output.is_relative_to(root / "diagnostics") or output == root / "diagnostics" or output.exists():
        raise ValueError("Output must be a new diagnostic child, never an existing result or signed leaf")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/runs/ppo_timing_credit_20261015")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    protocol = read_config(root / "protocol.json")
    study.check_frozen(protocol)
    if not (root / "selection.json").exists():
        raise ValueError("Same-path diagnostics wait for all eight sealed screens and locked selection")
    selected = read_config(root / "selection.json")
    rows = read_stage(root, protocol, "screen", selected)
    if selected["protocol_sha256"] != json_digest(protocol) or any(selected.get(key) != value for key, value in study.select_screen(rows).items()):
        raise ValueError("Counterfactual scope differs from the frozen eight-trial selection")
    output = output_path(root, args.output)
    results = []
    for row in rows:
        identifier = study.run_id(row["fold"], row["reward_basis"], row["gae_lambda"], "screen", row["seed"])
        directory = root / "runs" / identifier
        config = read_config(directory / "config.json")
        estimator_audit(directory, row["audit"], config)
        for buffer_record, context_record in zip(row["audit"]["rollout_buffers"], row["audit"]["credit_estimator_buffers"]):
            with np.load(directory / buffer_record["path"], allow_pickle=False) as saved:
                buffer = {name: saved[name] for name in saved.files}
            with np.load(directory / context_record["path"], allow_pickle=False) as saved:
                context = {name: saved[name] for name in saved.files}
            results.append(dict(run_id=identifier, fold=row["fold"], reward_basis=row["reward_basis"], seed=row["seed"],
                first_step=buffer_record["first_step"], last_step=buffer_record["last_step"],
                original_buffer_sha256=buffer_record["sha256"], actual_boundary_sha256=context_record["sha256"],
                **counterfactual_arrays(buffer, context)))
    report = dict(schema_version=1, status="complete_for_sealed_screen", scope="All eight registered final-budget screen trials only",
        registered_trials=len(rows), original_training_steps=sum(row["steps"] for row in rows), analyzed_original_buffers=len(results),
        added_rl_experience=0, added_environment_steps=0, added_forward_calls=0, added_gradient_updates=0,
        no_tensor_deserialization=True, no_source_or_signed_leaf_writes=True,
        protocol_sha256=file_hash(root / "protocol.json"), selection_sha256=file_hash(root / "selection.json"), tool_sha256=file_hash(Path(__file__)),
        buffers=results, interpretation=["This is an explanatory posthoc estimator comparison on the same actual path and old Critic values.",
            "Changing lambda here changes targets only; no different policy was trained, no new market path was simulated and no new return was earned.",
            "MSE is against different bootstrapped targets, so a smaller value is not evidence of a better model or investment strategy.",
            "A zero target variance makes the MSE/variance ratio undefined and is recorded as null.",
            "These diagnostics do not participate in selection, admission or promotion."])
    output.mkdir(parents=True)
    write_json(output / "diagnostics.json", report)
    pd.DataFrame([dict(run_id=item["run_id"], fold=item["fold"], reward_basis=item["reward_basis"],
        actual_lambda=item["actual_gae_lambda"], estimated_lambda=value, first_step=item["first_step"], last_step=item["last_step"],
        **metrics) for item in results for value, metrics in item["estimators"].items()]).to_csv(output / "same_path_scales.csv", index=False)
    print(f'CREDIT_COUNTERFACTUAL sealed_trials={len(rows)} original_buffers={len(results)} added_RL=0 output={output}', flush=True)
    return report


if __name__ == "__main__":
    main()
