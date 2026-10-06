"""Training-period functional structure probes; no trades or learning."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.agents.trade_mapping import TradeActionWrapper
from src.data.preprocess import file_hash, read_config, write_json
from src.envs.broker import BrokerConfig
from .exploration_env import ExplorationData, ExplorationEnv, ExplorationMarket
from .freeze import json_digest, source_hashes
from .research_exploration import config_for, make_exploration_model
from .research_matrix import preserve_training_rng
from .exploration_probes import parameter_digest


CELLS = {"baseline_l1": 126343, "lookback5": 146823, "split_l1": 125063}
INDICES = (-52, -32, -12)
MODES = ("cash", "random")
ACCOUNT_SEED = 20260927
STOCK_FIELDS = ("stock_features", "valid_mask", "buy_mask", "sell_mask",
                "execution_features", "share_state", "trade_lot_bounds", "legal_directions")


def _market_summary(network, observation):
    if hasattr(network, "encode_components"):
        return network.encode_components(observation)["market"]
    weights, valid = observation["portfolio"][:, :-1].float(), observation["valid_mask"].float()
    flags = torch.stack([weights, valid, observation["buy_mask"].float(),
                         observation["sell_mask"].float()], dim=-1)
    execution = observation["execution_features"].float()
    notionals = observation["trade_lot_bounds"].float() * execution[..., :1]
    encoded = network.encoder(torch.cat([observation["stock_features"].float().flatten(start_dim=2),
                                         flags, execution, notionals], dim=-1))
    return (encoded * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)


def _checks(policy, observation):
    tensor, _ = policy.obs_to_tensor(observation)
    output = policy.network(tensor)
    if not all(torch.isfinite(value).all() for value in output):
        raise FloatingPointError("Structure output must be finite")
    n = len(observation["valid_mask"])
    permutation = torch.arange(n - 1, -1, -1, device=output[0].device)
    permuted = {name: value.clone() for name, value in tensor.items()}
    for name in STOCK_FIELDS:
        permuted[name] = tensor[name][:, permutation]
    permuted["portfolio"] = torch.cat([tensor["portfolio"][:, :-1][:, permutation],
                                       tensor["portfolio"][:, -1:]], dim=1)
    other = policy.network(permuted)
    errors = []
    for expected, actual in ((output[0][:, permutation], other[0]),
                             (output[1][:, permutation], other[1]), (output[2], other[2])):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        errors.append(float((actual - expected).abs().max().cpu()))
    changed = {name: value.clone() for name, value in tensor.items()}
    changed["portfolio"][:, :-1] += .001
    changed["portfolio"][:, -1:] = 0.
    changed["buy_mask"] = 1 - changed["buy_mask"]
    changed["sell_mask"] = 1 - changed["sell_mask"]
    changed["execution_features"] += .01
    changed["trade_lot_bounds"] += 1
    # This deliberately nonphysical account perturbation is never mapped or traded.
    before, after = _market_summary(policy.network, tensor), _market_summary(policy.network, changed)
    delta = float((after - before).abs().max().cpu())
    if policy.architecture == "split":
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    arrays = {name: value.copy() for name, value in observation.items()}
    arrays.update(logits=output[0][0].cpu().numpy(), size_mean=output[1][0].cpu().numpy(),
        value=output[2][0].cpu().numpy(), log_std=output[3].cpu().numpy(),
        market_before=before.cpu().numpy(), market_after_account_perturbation=after.cpu().numpy(),
        permutation=permutation.cpu().numpy())
    return dict(finite_outputs=True, permutation_equivariant=True,
        permutation_max_errors=errors, account_perturbation_market_max_delta=delta,
        split_market_account_invariant=delta == 0 if policy.architecture == "split" else None), arrays


def run_structure_diagnostic(root, old_root="reports/runs/ppo_trade_actor_20261005"):
    root, old_root = Path(root).resolve(), Path(old_root).resolve()
    destination = root / "structure"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite structure diagnostic evidence")
    old = read_config(old_root / "protocol.json")
    before_sources, previous_threads = source_hashes(), torch.get_num_threads()
    rows, arrays, identities = [], {}, {}
    # Model construction seeds torch globally, including available CUDA devices.
    rng_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        with preserve_training_rng(rng_device), torch.no_grad():
            for cell, expected_count in CELLS.items():
                config = config_for(old["base_config"], root, cell, 42)
                config.update(device="cpu", torch_threads=1)
                data = ExplorationData(config)
                train = data.market("train")
                if train.n_dates < -min(INDICES):
                    raise ValueError("Registered training dates are unavailable")
                vector = data.vector_env()
                try:
                    model = make_exploration_model(vector, config)
                    model.policy.set_training_mode(False)
                    digest = parameter_digest(model.policy)
                    count = sum(parameter.numel() for parameter in model.policy.parameters())
                    if count != expected_count:
                        raise AssertionError("Registered structure parameter count differs")
                    identities[cell] = dict(parameters=count, parameter_sha256=digest,
                        contract=data.contract, config_sha256=json_digest(config))
                    end = pd.Timestamp(data.manifest["train_end"])
                    for negative in INDICES:
                        date = train.dates[negative]
                        for mode in MODES:
                            market = ExplorationMarket(data.features.loc[:end, pd.IndexSlice[data.codes, :]],
                                data.manifest["feature_cols"], price_path=data.prices.loc[:end, pd.IndexSlice[data.codes, :]],
                                price_basis=data.manifest["price_basis"], start_date=date, end_date=end,
                                lookback=config["lookback"], common_lookback=5)
                            env = TradeActionWrapper(ExplorationEnv(market, data.settings["initial_capital"],
                                BrokerConfig(**data.settings["broker"]), initial_portfolio=mode), top_n=20,
                                max_trade_weight=.05, max_commission_ratio=.001)
                            try:
                                observation, account = env.reset(seed=ACCOUNT_SEED)
                                checked, values = _checks(model.policy, observation)
                                prefix = f"{cell}_index{negative}_{mode}"
                                arrays.update({prefix + "__" + name: value for name, value in values.items()})
                                rows.append(dict(cell=cell, mode=mode, train_index=negative,
                                    date=str(date.date()), account_seed=ACCOUNT_SEED,
                                    stock_count=len(data.codes), account_equity=float(account["equity"]),
                                    environment_steps=0, **checked))
                                if market.current_index != 0:
                                    raise AssertionError("A structure probe advanced the account")
                            finally:
                                env.close()
                    if (parameter_digest(model.policy) != digest or model.num_timesteps
                            or model.policy.optimizer.state):
                        raise AssertionError("Structure diagnostics changed model or optimizer")
                finally:
                    vector.close()
    finally:
        torch.set_num_threads(previous_threads)
    if source_hashes() != before_sources:
        raise ValueError("Structure source changed while probing")
    destination.mkdir(parents=True)
    path = destination / "observations.npz"
    np.savez_compressed(path, **arrays)
    summary = dict(schema_version=1, status="complete", created_at=datetime.now(timezone.utc).isoformat(),
        stage="structure", device="cpu", case_count=len(rows), cases=rows, models=identities,
        training_steps=0, training_updates=0, environment_steps=0, test_evaluated=False,
        validation_evaluated=False, old_protocol_sha256=file_hash(old_root / "protocol.json"),
        source_hashes=before_sources, artifacts={path.name: file_hash(path)},
        interpretation="Functional forward-only training-date probes, not ROI or unseen-stock generalization.",
        account_perturbation="Nonphysical weights/flags/execution/bounds changed only inside forward; never executed.")
    write_json(destination / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="reports/runs/ppo_exploration_20261006")
    parser.add_argument("--old-root", default="reports/runs/ppo_trade_actor_20261005")
    args = parser.parse_args(argv)
    run_structure_diagnostic(args.output_root, args.old_root)


if __name__ == "__main__":
    main()
