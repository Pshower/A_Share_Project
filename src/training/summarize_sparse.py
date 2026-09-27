"""Read completed evidence and qualify a research candidate, never trade or train."""

import argparse
from pathlib import Path
from statistics import mean

from src.data.preprocess import ROOT, read_config, write_json, file_hash
from src.envs.initial_account import MODES
from src.web.catalog import identifier


def summarize(root):
    rows = read_config(root / "summary.json")
    protocol = read_config(root / "protocol.json")
    selected = read_config(root / "selection.json")["selected"]
    if selected not in protocol["arms"]:
        raise ValueError("No selected research candidate")
    expected = ({("screen", arm, 42) for arm in protocol["arms"]}
                | {("confirm", selected, seed) for seed in [42, 123, 2026]}
                | {("walk", arm, seed) for arm in {"D0", selected} for seed in [42, 123]})
    if (len(rows) != len(expected) or {(r["phase"], r["arm"], r["seed"]) for r in rows} != expected
            or any(r["steps"] != (4096 if r["phase"] == "screen" else 8192) for r in rows)
            or sum(r["steps"] for r in rows) > protocol["maximum_steps"]):
        raise ValueError("The pre-registered research batch is incomplete")
    confirm = [r for r in rows if r["phase"] == "confirm"]
    walk = [r for r in rows if r["phase"] == "walk" and r["arm"] == selected]
    controls = [r for r in rows if r["phase"] == "walk" and r["arm"] == "D0"]
    if sorted(r["seed"] for r in confirm) != [42, 123, 2026] or len(walk) != 2 or len(controls) != 2:
        raise ValueError("Missing confirmation or matched walk-forward evidence")
    old = read_config(ROOT / "reports/runs/sina_hs300_study_20260927/summary.json")
    gaps = {m: mean(r["metrics"][m]["cumulative_return"] for r in confirm)
                  - mean(r["metrics"]["cumulative_return"] for r in old if r["mode"] == m) for m in MODES}
    worst = max(abs(m["max_drawdown"]) for r in confirm + walk for m in r["metrics"].values())
    walk_gap = mean(r["mean_return"] for r in walk) - mean(r["mean_return"] for r in controls)
    qualified = all(v >= .01 for v in gaps.values()) and worst <= .1 and walk_gap > 0
    audit_rows = []
    for r in rows:
        folder = Path(r["output"])
        audit = read_config(folder / "training_audit.json")
        metadata = read_config(folder / "final_model/metadata.json")
        dataset = read_config(folder / "research_config.json")["dataset"]
        expected_dates = ("2025-06-30", "2025-12-31") if r["phase"] == "walk" else ("2026-03-31", "2026-06-30")
        if (dataset["train_end"], dataset["val_end"]) != expected_dates:
            raise ValueError("Research time boundaries differ from the registered protocol")
        if (audit["actual_timesteps"] != r["steps"] or not audit["all_parameters_finite"]
                or not audit["save_load_deterministic_equal"] or audit["changed_tensors"] < 1
                or metadata["contract"]["data_source"] != "sina"
                or (r["phase"] != "walk" and metadata["contract"]["manifest_sha256"] != protocol["source_manifest"])
                or file_hash(folder / "final_model/model.zip") != metadata["model_sha256"]):
            raise ValueError("Invalid completed audit: " + str(folder))
        audit_rows.append(dict(run=folder.name, steps=audit["actual_timesteps"],
            optimizer_steps=audit["optimizer_steps_min"], changed_tensors=audit["changed_tensors"],
            minimum_valid_observations=min(audit["valid_observation_steps"].values()), git=metadata["git"]))
    candidate = next(r for r in confirm if r["seed"] == 42)
    model = Path(candidate["output"]) / "final_model"
    report = dict(qualified_under_development_protocol=qualified,
        status="research_candidate_only" if qualified else "not_qualified", selected_arm=selected,
        model_path=str(model), model_id=identifier(model, ROOT), example_seed=42,
        confirmation_mean_return=mean(r["mean_return"] for r in confirm),
        confirmation_old_ppo_gaps=gaps, walk_mean_return=mean(r["mean_return"] for r in walk),
        walk_control_mean_return=mean(r["mean_return"] for r in controls), walk_return_gap=walk_gap,
        worst_candidate_drawdown=worst,
        confirmation_learning_increment=mean(r["mean_return"] - mean(v["cumulative_return"] for v in r["initialized"].values()) for r in confirm),
        walk_learning_increment=mean(r["mean_return"] - mean(v["cumulative_return"] for v in r["initialized"].values()) for r in walk),
        total_steps=sum(r["steps"] for r in rows), total_optimizer_steps=sum(r["optimizer_steps"] for r in audit_rows),
        audits=audit_rows, test_evaluated=False, actionable=False,
        limitations=["Development comparisons only; not proof of unseen performance or real execution.",
                     "Initialization comparisons, not parameter updates alone, determine evidence of PPO learning alpha.",
                     "Higher returns come with higher drawdown; not a Pareto improvement.",
                     "HFQ execution, corporate-action gaps, fixed-universe bias and stale current bars remain."])
    write_json(root / "qualification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.directory.resolve())
    print({k: report[k] for k in ["qualified_under_development_protocol", "confirmation_mean_return", "walk_mean_return",
                                 "walk_control_mean_return", "worst_candidate_drawdown", "confirmation_learning_increment",
                                 "walk_learning_increment", "total_steps", "total_optimizer_steps", "model_id"]})
