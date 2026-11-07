"""What the R1 momentum prior actually forced, and what training could not undo.

The 2026-09 rounds initialised the stock head to zero and added a hard prior:

    latent = prior_strength * rank(momentum_20) + residual_scale * tanh(head)

with ``prior_strength = 4.0`` and ``residual_scale = 0.25``.  Every later claim
that the policy "should have learned the reversal" assumes the learned part could
have changed the ordering.  This tool measures that assumption instead of
asserting it, using only saved models and a fixed probe of observations:

* the initial model must reproduce ``prior_strength * rank`` exactly, because the
  head weights start at zero -- that is the machinery check;
* the trained model's distance from that starting latent is bounded by
  ``residual_scale``, so the largest rank displacement it can buy is
  ``residual_scale * (count - 1) / prior_strength`` positions, i.e.
  ``residual_scale / prior_strength`` of the rank axis regardless of pool size;
* the sign of the learned correction is read off the correlation between the
  residual and the prior signal.

No return is computed and no reward window is consumed.  Probes are restricted to
``PROBE_END`` so the run never reaches the reserved or protected segments, and the
tool only reads observations, latents and correlations.

The prior studies trained on ``sina_hs300_study_20260927``, whose ``train_end`` is
2026-03-31: those models already saw 2025-2026.  This tool does not add exposure,
but it does not undo that one either.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.agents.ppo_agent import PPOAgent
from src.data.preprocess import write_json
from src.training.env_factory import ResearchData

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "runs" / "prior_attribution"

PROBE_END = "2024-12-31"
PROBE_STRIDE = 20
PROBE_SPLIT = "train"
TRAINED_DIRS = ("final_model", "best_8192", "long_model", "best_5120", "best_7680")
TOP_N = 20


def prior_rank(signal, eligible):
    """Midrank of the eligible signal on [-1, 1]; excluded rows receive zero.

    Mirrors ``SharedStockNetwork.forward``: equal signals share a midrank and every
    ineligible row is pushed behind the eligible ones.
    """
    signal = np.asarray(signal, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if signal.shape != eligible.shape:
        raise ValueError("Signal and eligibility must share a shape")
    count = int(eligible.sum())
    if count == 0:
        return np.zeros_like(signal)
    order = np.where(eligible, signal, np.inf)
    left = np.searchsorted(np.sort(order), order, side="left")
    right = np.searchsorted(np.sort(order), order, side="right")
    midrank = (left + right - 1) / 2.0
    # A lone candidate gets +1 rather than a division by zero.
    rank = np.ones_like(signal) if count == 1 else 2.0 * midrank / (count - 1) - 1.0
    return np.where(eligible, rank, 0.0)


def spearman(left, right):
    """Rank correlation without scipy; ties keep their average rank."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.shape != b.shape or a.size < 3:
        return float("nan")
    ar = pd.Series(a).rank().to_numpy()
    br = pd.Series(b).rank().to_numpy()
    ar -= ar.mean()
    br -= br.mean()
    scale = np.sqrt((ar * ar).sum() * (br * br).sum())
    return float((ar * br).sum() / scale) if scale > 0 else float("nan")


def displacement_bound(count, prior_strength, residual_scale):
    """Largest number of rank positions the learned residual can move a name.

    Adjacent ranks differ by ``2 / (count - 1)`` on the [-1, 1] axis, so after the
    ``prior_strength`` scaling one step is ``2 * prior_strength / (count - 1)``.
    The residual lives in ``[-residual_scale, residual_scale]``, a width of
    ``2 * residual_scale``.
    """
    if count < 2 or prior_strength <= 0 or residual_scale < 0:
        return float("nan")
    step = 2.0 * prior_strength / (count - 1)
    return 2.0 * residual_scale / step


def fixed_observation(features, valid, buy, sell):
    """Observation with a neutral portfolio so both models see the same state.

    Weights are equal across the valid names, cash is zero and there are no
    receivables.  The account is deliberately not the model's own: the question is
    what each policy prefers in an identical state, not where it happened to be.
    """
    n = len(valid)
    weights = np.zeros(n + 1, dtype=np.float32)
    if valid.any():
        weights[:n] = np.where(valid, 1.0 / valid.sum(), 0.0)
    return dict(stock_features=np.asarray(features, dtype=np.float32),
                portfolio=weights,
                receivable_ratio=np.zeros(1, dtype=np.float32),
                valid_mask=valid.astype(np.int8),
                buy_mask=buy.astype(np.int8),
                sell_mask=sell.astype(np.int8))


def stock_latent(agent, observation):
    """Deterministic mean latent for the stock coordinates."""
    agent.model.policy.set_training_mode(False)
    tensor, _ = agent.model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        distribution = agent.model.policy.get_distribution(tensor)
        latent = distribution.get_actions(deterministic=True).cpu().numpy()[0]
    n = len(observation["valid_mask"])
    return np.asarray(latent, dtype=float)[:n]


def probe_dates(market, end=PROBE_END, stride=PROBE_STRIDE):
    """Indices whose date is on or before ``end``, sampled every ``stride`` days.

    The filter is by date, so a calendar that continues past ``end`` can never leak
    a later observation into the probe: later rows are simply not in the list.
    """
    dates = market.dates
    limit = pd.Timestamp(end)
    picked = [i for i, day in enumerate(dates) if day <= limit]
    if not picked:
        raise ValueError("No probe date falls on or before the requested end")
    return np.asarray(picked, dtype=int)[::stride]


def measure_model(agent, data, contract, indices, market):
    """Per-date latents, prior ranks and the correlations between them."""
    index = contract["prior"]["feature_index"]
    strength = float(contract["prior"]["strength"])
    signal_index = market.feature_cols.index(contract["feature_cols"][index])
    rows = []
    for step in indices:
        features, valid, buy, sell = market.observation(step)
        observation = fixed_observation(features, valid, buy, sell)
        signal = features[:, -1, signal_index].astype(float)
        eligible = valid & buy
        rank = prior_rank(signal, eligible)
        latent = stock_latent(agent, observation)
        prior = strength * rank
        residual = latent - prior
        keep = eligible if eligible.any() else valid
        top = np.argsort(-latent)[:TOP_N]
        prior_top = np.argsort(-prior)[:TOP_N]
        rows.append(dict(date=str(market.dates[step].date()), eligible=int(eligible.sum()),
                         spearman_signal=spearman(latent[keep], signal[keep]),
                         spearman_prior=spearman(latent[keep], prior[keep]),
                         residual_signal=spearman(residual[keep], signal[keep]),
                         residual_max=float(np.abs(residual[keep]).max()),
                         top_overlap=int(len(set(top) & set(prior_top)))))
    return rows


def summarize(name, rows, strength, residual_scale, count):
    """Aggregate the per-date probe into one record."""
    frame = pd.DataFrame(rows)
    bound = displacement_bound(count, strength, residual_scale)
    return dict(
        name=name, probes=int(len(frame)),
        spearman_signal=float(frame["spearman_signal"].mean()),
        spearman_prior=float(frame["spearman_prior"].mean()),
        residual_signal=float(frame["residual_signal"].mean()),
        residual_max=float(frame["residual_max"].max()),
        residual_share_of_prior=float(frame["residual_max"].max() / strength),
        top_overlap=float(frame["top_overlap"].mean()),
        displacement_bound=float(bound),
        displacement_share=float(residual_scale / strength) if strength else float("nan"))


def discover(roots):
    """Run directories that hold both an initial model and a trained model."""
    found = []
    for root in roots:
        for config_path in sorted(Path(root).rglob("config.json")):
            run = config_path.parent
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("prior_feature_index") is None:
                continue
            initial = run / "initial_model" / "model.zip"
            trained = next((run / name for name in TRAINED_DIRS if (run / name / "model.zip").exists()), None)
            if initial.exists() and trained is not None:
                found.append((run, config))
    return found


def measure(run, config, *, device="cpu", end=PROBE_END, stride=PROBE_STRIDE):
    """Load both models of one run and probe them on identical observations.

    The observation is a neutral, equally weighted book over the valid names, so
    the two models answer the same question in the same state.  Probes stop at
    ``end`` (2024-12-31 by default) and never reach the reserved segment.
    """
    data = ResearchData(config)
    market = data.market(PROBE_SPLIT)
    indices = probe_dates(market, end=end, stride=stride)
    metadata = json.loads((run / "initial_model" / "metadata.json").read_text(encoding="utf-8"))
    contract = metadata["contract"]
    trained_dir = next(run / name for name in TRAINED_DIRS if (run / name / "model.zip").exists())

    records = {}
    count = 0
    for label, directory in (("initial", run / "initial_model"), ("trained", trained_dir)):
        agent = PPOAgent.load(directory, contract, device=device)
        rows = measure_model(agent, data, contract, indices, market)
        count = max(2, int(round(float(np.mean([row["eligible"] for row in rows])))))
        records[label] = summarize(label, rows, contract["prior"]["strength"],
                                   contract["prior"]["residual_scale"], count)
    return dict(run=str(Path(run).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
                trained_dir=Path(trained_dir).name,
                probe_start=str(market.dates[indices[0]].date()),
                probe_end=str(market.dates[indices[-1]].date()),
                prior=dict(contract["prior"], feature=contract["feature_cols"][contract["prior"]["feature_index"]]),
                pool=int(market.n_stocks), eligible_mean=count, splits=records)


def verdict(payload):
    """The three statements the artefact actually supports."""
    rows = payload["runs"]
    if not rows:
        return dict(status="empty")
    initial = [row["splits"]["initial"] for row in rows]
    trained = [row["splits"]["trained"] for row in rows]
    machinery = max(abs(row["residual_max"]) for row in initial)
    return dict(
        status="ready",
        runs=len(rows),
        machinery_max_deviation=float(machinery),
        machinery_exact=bool(machinery < 1e-4),
        initial_spearman_prior=float(np.mean([row["spearman_prior"] for row in initial])),
        trained_spearman_prior=float(np.mean([row["spearman_prior"] for row in trained])),
        trained_residual_share=float(np.mean([row["residual_share_of_prior"] for row in trained])),
        trained_residual_signal=float(np.mean([row["residual_signal"] for row in trained])),
        trained_residual_positive=int(sum(1 for row in trained if row["residual_signal"] > 0)),
        trained_residual_negative=int(sum(1 for row in trained if row["residual_signal"] <= 0)),
        trained_top_overlap=float(np.mean([row["top_overlap"] for row in trained])),
        displacement_bound=float(np.mean([row["displacement_bound"] for row in trained])),
        displacement_share=float(np.mean([row["displacement_share"] for row in trained])),
        prior_feature=rows[0]["prior"]["feature"],
        prior_strength=float(rows[0]["prior"]["strength"]),
        prior_residual_scale=float(rows[0]["prior"]["residual_scale"]))


def _markdown(payload):
    v = payload["verdict"]
    lines = ["# R1 动量先验的只读归因", "",
             f"日期：{payload['date']}　探测区间 {payload['probe_start']} ~ {payload['probe_end']}，"
             f"每 {payload['stride']} 个交易日一次，股票池 {payload['pool']} 只。", "",
             f"先验特征 `{v['prior_feature']}`，`prior_strength={v['prior_strength']}`，"
             f"`residual_scale={v['prior_residual_scale']}`。", "",
             "## 1. 机件自检", "",
             f"- 初始模型与 `strength × rank` 的最大偏差 **{v['machinery_max_deviation']:.2e}**"
             f"（{'通过，先验确实被精确复现' if v['machinery_exact'] else '**未通过**'}）。", "",
             "## 2. 训练前后", "",
             "| 指标 | 初始 | 训练后 |", "| --- | ---: | ---: |",
             f"| 与先验排序的 Spearman | {v['initial_spearman_prior']:+.4f} | {v['trained_spearman_prior']:+.4f} |",
             f"| 残差占先验振幅 | — | {v['trained_residual_share']:.2%} |",
             f"| 残差与先验信号的 Spearman | — | {v['trained_residual_signal']:+.4f} "
             f"（{v['trained_residual_positive']} 同向 / {v['trained_residual_negative']} 反向） |",
             f"| top-{TOP_N} 与前验 top-{TOP_N} 的交集 | — | {v['trained_top_overlap']:.1f} / {TOP_N} |",
             f"| 残差最多能挪动的名次 | {v['displacement_bound']:.1f} 位 | 名次轴的 {v['displacement_share']:.2%} |",
             ""]
    lines += ["## 3. 逐 run", "",
              "| run | 池 | 训练后 ρ(latent, 先验) | 残差/先验 | ρ(残差, 信号) | top 重叠 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in payload["runs"]:
        trained = row["splits"]["trained"]
        lines.append("| `{}` | {} | {:+.4f} | {:.2%} | {:+.4f} | {:.1f} |".format(
            row["run"], row["pool"], trained["spearman_prior"], trained["residual_share_of_prior"],
            trained["residual_signal"], trained["top_overlap"]))
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Attribute the R1 momentum prior from saved models.")
    parser.add_argument("--root", action="append", default=None,
                        help="Study root to scan; repeatable. Defaults to both R1 studies.")
    parser.add_argument("--end", default=PROBE_END)
    parser.add_argument("--stride", type=int, default=PROBE_STRIDE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    roots = args.root or [PROJECT_ROOT / "reports" / "runs" / "agent_improvement_20260927",
                          PROJECT_ROOT / "reports" / "runs" / "ppo_matrix_20261003"]
    runs = discover(roots)
    if not runs:
        raise SystemExit("No run with both an initial model and a trained model was found")

    payload = dict(date="2026-11-07", stride=args.stride, runs=[], status="ready")
    for run, config in runs:
        record = measure(run, config, device=args.device, end=args.end, stride=args.stride)
        payload["runs"].append(record)
        print("%-58s trained rho=%.4f residual=%.8f top=%d" % (
            record["run"], record["splits"]["trained"]["spearman_prior"],
            record["splits"]["trained"]["residual_max"], record["splits"]["trained"]["top_overlap"]))
    payload["pool"] = payload["runs"][0]["pool"]
    payload["probe_start"] = payload["runs"][0]["probe_start"]
    payload["probe_end"] = payload["runs"][0]["probe_end"]
    payload["verdict"] = verdict(payload)

    target = Path(args.output) if args.output else REPORT_ROOT
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "prior_attribution.json", payload)
    (target / "prior_attribution.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["verdict"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
