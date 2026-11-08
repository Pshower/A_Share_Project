"""Describe the sealed single calendar skip; no models or market replay."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import file_hash, read_config, write_json


def main():
    root = Path("reports/runs/ppo_timing_20261014").resolve()
    folder = root / "runs" / "confirm_2026_log_net_42" / "trained" / "cash"
    result = read_config(folder / "result.json")
    manifest = result["trace"]
    if len(manifest["chunks"]) != 1 or len(manifest["benchmark_trace"]["chunks"]) != 1:
        raise ValueError("This sidecar describes exactly the already sealed sixty-step cash path")
    agent_path = folder / "trace" / manifest["chunks"][0]["path"]
    benchmark_path = folder / "trace" / "benchmark" / manifest["benchmark_trace"]["chunks"][0]["path"]
    if (file_hash(agent_path) != manifest["chunks"][0]["sha256"]
            or file_hash(benchmark_path) != manifest["benchmark_trace"]["chunks"][0]["sha256"]):
        raise ValueError("Sealed native evidence changed")
    with np.load(agent_path, allow_pickle=False) as agent, np.load(benchmark_path, allow_pickle=False) as benchmark:
        different = np.flatnonzero(agent["action_macro_action"] != (agent["action_phase"] == 0))
        if len(different) != 1:
            raise ValueError("Expected exactly one divergent macro decision")
        index = int(different[0])
        if not np.array_equal(agent["decision_positions"][:index + 1], benchmark["decision_positions"][:index + 1]) or not np.array_equal(agent["decision_cash"][:index + 1], benchmark["decision_cash"][:index + 1]):
            raise ValueError("The accounts were not identical before this sole branch")
        initial_equity = float(agent["decision_equity"][0])
        output = dict(schema_version=1, scope="sealed 2026 seed42 cash account; four-account summary is separate",
            run_id="confirm_2026_log_net_42", initial_mode="cash", step=int(agent["step"][index]),
            decision_date=str(agent["decision_date"][index]), execution_date=str(agent["execution_date"][index]),
            probabilities=agent["action_macro_probabilities"][index].tolist(), critic_value=float(agent["action_macro_value"][index]),
            identical_accounts_until_branch=True, divergent_macro_decisions=1,
            agent_next_day_net_return=float(agent["action_raw_net_return"][index]),
            benchmark_next_day_net_return=float(agent["action_benchmark_net_return"][index]),
            immediate_return_gap=float(agent["action_raw_net_return"][index] - agent["action_benchmark_net_return"][index]),
            saved_fee=float(benchmark["execution_fee"][index].sum() - agent["execution_fee"][index].sum()),
            fewer_stock_day_fills=int((benchmark["execution_filled_shares"][index] != 0).sum() - (agent["execution_filled_shares"][index] != 0).sum()),
            agent_final_equity=float(agent["after_equity"][-1]), benchmark_final_equity=float(benchmark["after_equity"][-1]),
            terminal_equity_gap=float(agent["after_equity"][-1] - benchmark["after_equity"][-1]),
            terminal_roi_gap=float((agent["after_equity"][-1] - benchmark["after_equity"][-1]) / initial_equity),
            no_forward=True, no_model_load=True, no_extra_environment_steps=True,
            interpretation="The immediate holding-day gain reverses over the remaining fixed historical path. Only this macro differs, but this paired historical counterfactual does not establish a general causal effect or reveal the model's internal reason.")
    output["evidence_sha256"] = {str(path.relative_to(root)): file_hash(path) for path in
        (root / "protocol.json", folder.parent.parent / "completed.json", folder / "result.json", agent_path, benchmark_path)}
    output["tool_sha256"] = file_hash(Path(__file__))
    path = root / "diagnostics" / "timing_all" / "single_branch_sidecar.json"
    if path.exists():
        raise ValueError("Preserve the original independent sidecar; do not overwrite it")
    write_json(path, output)
    print(f'SIDECAR {path} SHA256={file_hash(path)}', flush=True)
    return output


if __name__ == "__main__":
    main()
