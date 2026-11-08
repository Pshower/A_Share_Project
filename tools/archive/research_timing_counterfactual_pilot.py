"""Registered, two-pass train-only rule counterfactual diagnostic. No fitting."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.archive import timing_counterfactual as paired

FOLDS = ("2023", "2024")
PARENT = ROOT / "reports/runs/ppo_timing_phase_state_repair_20261015"
OUTPUT = ROOT / "reports/runs/ppo_counterfactual_pilot_20261016"
PARENT_AUDIT_SHA256 = "af731fb246da9dfb15999a73a1761d1e4c804a7525f3865e7395398d9e04b78d"
PARENT_FILE_COUNT = 6712
SOURCE_COUNT = 96
INVENTORY = (
    "tools/archive/research_timing_counterfactual_pilot.py", "tools/archive/timing_counterfactual.py",
    "tests/test_timing_counterfactual.py", "tests/test_timing_counterfactual_review.py",
    "tests/test_counterfactual_pilot.py", "doc/ppo_counterfactual_pilot_protocol_20261016.md",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def fixed_registration():
    return dict(study="train_only_counterfactual_pilot", training_folds=list(FOLDS), seed=42,
        episodes_per_fold=48, offsets=[0, 60], collection_passes=2, collection_steps_per_episode=60,
        collection_episode_steps=80, horizon=20, gamma=.995, signal_threshold=.0001,
        states_per_fold=96, collection_agent_steps_per_fold=5760, branch_agent_steps_per_fold=3840,
        total_agent_steps=19200, total_benchmark_steps=19200, total_book_steps=38400,
        rl_steps=0, nn_forwards=0, gradient_updates=0, optimizer_updates=0,
        primary_label="discounted_relative_delta", reward_basis="relative_log",
        continuation="original phase calendar M20; never reset phase", test_evaluated=False, actionable=False)


def output_root(root):
    root = Path(root).resolve()
    require(root == OUTPUT.resolve() and root.is_relative_to(ROOT.resolve()), "Use the registered fresh pilot root")
    require(not root.is_relative_to(PARENT.resolve()) and not PARENT.resolve().is_relative_to(root),
            "Pilot output overlaps the immutable parent")
    return root


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in (ROOT / "src").rglob("*.py")}


def parent_reference():
    path = PARENT / "final_audit.json"
    require(file_hash(path) == PARENT_AUDIT_SHA256, "Parent final audit identity differs")
    audit = read_json(path)
    files = audit["signed_files_sha256"]
    require(audit["status"] == "passed" and len(files) == PARENT_FILE_COUNT, "Incomplete parent signature")
    for name, expected in files.items():
        logical = PurePosixPath(name)
        require(not logical.is_absolute() and logical.as_posix() == name and "\\" not in name
                and ".." not in logical.parts, "Parent path is not workspace-relative POSIX")
        target = (ROOT / name).resolve()
        require(target.is_relative_to(ROOT.resolve()) and file_hash(target) == expected, "Parent signed input changed")
    sources = source_hashes()
    require(len(sources) == SOURCE_COUNT and all(files.get(name) == value for name, value in sources.items()),
            "Frozen source inventory changed")
    return audit, sources


def reference_config(fold):
    require(fold in FOLDS, "Only the two registered training folds are allowed")
    return PARENT / "runs" / f"screen_{fold}_linear_42" / "config.json"


def config_for(root, fold):
    config = deepcopy(read_json(reference_config(fold)))
    require(type(config["seed"]) is int and config["seed"] == 42 and config["timing"]["fold"] == fold
            and config["timing"]["reward_basis"] == "relative_log", "Parent collection configuration differs")
    config["training_episode_steps"] = 80
    config["output_root"] = str(output_root(root) / "folds")
    return config


def make_data(config):
    # The original data/environment factory is used without constructing PPO.
    from src.training.research_timing import TimingData
    return TimingData(deepcopy(config))


def data_identity(data, config, fold):
    cutoff = fold + "-03-31"
    require(data.manifest["train_end"] == cutoff and len(data.contract["stock_codes"]) == 300
            and len(data.contract["feature_cols"]) == 20 and data.contract["price_basis"] == "hfq_research"
            and type(data.contract["training_episode_steps"]) is int
            and data.contract["training_episode_steps"] == 80, "Data, price basis or collection horizon differs")
    research = Path(config["research_config"])
    research = research if research.is_absolute() else ROOT / research
    return dict(train_end=cutoff, contract=deepcopy(data.contract), contract_sha256=digest(data.contract),
        research_config_sha256=file_hash(research), manifest_sha256=data.contract["manifest_sha256"],
        artifacts=deepcopy(data.contract["artifacts"]))


def check_frozen(protocol, root):
    output_root(root)
    require(all(canonical(protocol.get(k)) == canonical(v) for k, v in fixed_registration().items()),
            "Registered pilot design or budget changed")
    require(protocol["output_root"] == str(output_root(root)) and set(protocol["inventory_sha256"]) == set(INVENTORY),
            "Output or exact six-file inventory differs")
    for name, value in protocol["inventory_sha256"].items():
        require(file_hash(ROOT / name) == value, "Frozen pilot tool, test or document changed")
    require(file_hash(Path(__file__)) == protocol["inventory_sha256"][INVENTORY[0]], "Driver file differs")
    audit, sources = parent_reference()
    require(protocol["parent_audit_sha256"] == PARENT_AUDIT_SHA256 and protocol["source_hashes"] == sources
            and set(protocol["folds"]) == set(FOLDS), "Parent, source or fold inventory differs")
    for fold in FOLDS:
        path = reference_config(fold)
        name = path.resolve().relative_to(ROOT.resolve()).as_posix()
        identity = protocol["folds"][fold]
        require(audit["signed_files_sha256"].get(name) == file_hash(path) == identity["reference_config_sha256"]
                and canonical(config_for(root, fold)) == canonical(identity["config"]),
                "Signed parent config or explicit 80-step override changed")


def prepare(root=OUTPUT):
    root = output_root(root)
    if (root / "protocol.json").exists():
        protocol = read_json(root / "protocol.json")
        check_frozen(protocol, root)
        return protocol
    require(not root.exists(), "Unregistered or partial output cannot be overwritten")
    audit, sources = parent_reference()
    inventory = {name: file_hash(ROOT / name) for name in INVENTORY}
    folds = {}
    for fold in FOLDS:
        path = reference_config(fold)
        key = path.resolve().relative_to(ROOT.resolve()).as_posix()
        require(audit["signed_files_sha256"].get(key) == file_hash(path), "Reference config is not parent signed")
        config = config_for(root, fold)
        data = make_data(config)
        folds[fold] = dict(reference_config_sha256=file_hash(path), config=config,
                           data_identity=data_identity(data, config, fold))
    from src.backtest.run import git_metadata
    protocol = dict(schema_version=1, **fixed_registration(), output_root=str(root),
        parent_audit_sha256=PARENT_AUDIT_SHA256, source_hashes=sources, inventory_sha256=inventory,
        folds=folds, git=git_metadata())
    check_frozen(protocol, root)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "protocol.json", protocol)
    write_json(root / "status.json", dict(status="prepared", rl_steps=0, actionable=False))
    return protocol


def counters():
    return dict(collection_agent_steps=0, collection_benchmark_steps=0, branch_agent_steps=0,
                branch_benchmark_steps=0, branch_macro_attempts=0, completed_pairs=0,
                rl_steps=0, nn_forwards=0, gradient_updates=0, optimizer_updates=0)


def new_environment(protocol, fold):
    record = protocol["folds"][fold]
    data = make_data(record["config"])
    require(canonical(data_identity(data, record["config"], fold)) == canonical(record["data_identity"]),
            "Runtime data identity changed")
    env = data.env("train")
    try:
        cutoff = pd.Timestamp(record["data_identity"]["train_end"])
        require(env.training is True and env.reward_basis == "relative_log", "Only training relative-reward environments are allowed")
        for base in (env.env.env, env.benchmark.env):
            market = base.market
            require(type(base.max_steps) is int and base.max_steps == 80 and len(market.dates) > 20
                    and market.dates.max() <= cutoff and market.panel.index.max() <= cutoff
                    and market.prices.index.max() <= cutoff, "Provider is not physically cropped to train_end")
    except Exception:
        env.close()
        raise
    return env


def input_hash(observation):
    array = np.asarray(observation)
    require(array.shape == (129,) and array.dtype == np.float32 and np.isfinite(array).all(), "Malformed current 129D input")
    return digest(dict(dtype=array.dtype.str, shape=list(array.shape), sha256=hashlib.sha256(array.tobytes()).hexdigest()))


def state_record(env, fold, episode, offset, schedule):
    observation = np.asarray(env.observe()).copy()
    base = env.env.env
    market = base.market
    require(env._done is False and base._steps == env.benchmark.env._steps == offset
            and base._steps + 20 <= base.max_steps and market.current_index + 20 < len(market.dates),
            "State is not reset, alive or eligible for the full horizon")
    endpoint = str(market.dates[market.current_index + 20].date())
    require(endpoint <= fold + "-03-31", "Label endpoint exceeds training cutoff")
    return dict(state_id=f"{episode:03d}_{offset:02d}", episode=episode, offset=offset,
        decision_date=str(market.current_date.date()), endpoint=endpoint, phase=int(env.phase),
        initial_mode=schedule["initial_mode"], date_stratum=schedule["date_stratum"], phase_bin=int(schedule["phase_bin"]),
        observation=observation.tolist(), input_sha256=input_hash(observation),
        full_state_sha256=paired.snapshot_state_fingerprint(env))


def collection_states(env, fold, counts):
    for episode in range(48):
        _, info = env.reset(seed=42) if episode == 0 else env.reset()
        schedule = deepcopy(info["timing"]["schedule"])
        yield state_record(env, fold, episode, 0, schedule), env
        for _ in range(60):
            before = (env.env.env._steps, env.benchmark.env._steps)
            try:
                _, _, terminated, truncated, _ = env.step(int(env.phase == 0))
            finally:
                counts["collection_agent_steps"] += env.env.env._steps - before[0]
                counts["collection_benchmark_steps"] += env.benchmark.env._steps - before[1]
            require(not terminated and not truncated, "Collection ended before its offset60 state")
        yield state_record(env, fold, episode, 60, schedule), env


def validate_plan(plan, fold):
    require(plan["fold"] == fold and plan["seed"] == 42 and len(plan["states"]) == 96, "Sampling plan is incomplete")
    expected = {(mode, date, phase) for mode in ("cash", "equal", "concentrated", "random")
                for date in ("early", "middle", "latest") for phase in range(4)}
    observed = []
    for index, record in enumerate(plan["states"]):
        episode, offset = divmod(index, 2)[0], (0, 60)[index % 2]
        require(record["episode"] == episode and record["offset"] == offset
                and record["state_id"] == f"{episode:03d}_{offset:02d}"
                and record["input_sha256"] == input_hash(np.asarray(record["observation"], dtype=np.float32))
                and record["decision_date"] <= record["endpoint"] <= fold + "-03-31"
                and type(record["phase"]) is int and 0 <= record["phase"] < 20,
                "Sampling state identity, input or training endpoint differs")
        if offset == 0:
            observed.append((record["initial_mode"], record["date_stratum"], record["phase_bin"]))
        else:
            previous = plan["states"][index - 1]
            require(all(record[k] == previous[k] for k in ("phase", "initial_mode", "date_stratum", "phase_bin")),
                    "Offset60 changed the sampled phase or strata")
    require(set(observed) == expected and len(observed) == len(set(observed)), "Sampling must cover one complete 48-strata cycle")


def validate_label(label, record, fold):
    require(type(label["schema_version"]) is int and label["schema_version"] == 1
            and label["label_version"] == paired.LABEL_VERSION, "Unknown label schema or version")
    inp, target, proof, count = (label[key] for key in ("input", "labels", "provenance", "counters"))
    require(inp["phase"] == record["phase"] and inp["decision_date"] == record["decision_date"]
            and input_hash(np.asarray(inp["observation"], dtype=np.float32)) == record["input_sha256"]
            and tuple(inp["feature_names"]) == tuple(paired.FEATURE_NAMES), "Label input differs from the frozen sampling plan")
    require(target["horizon"] == 20 and type(target["horizon"]) is int and type(target["gamma"]) is float
            and target["gamma"] == .995 and target["train_end"] == fold + "-03-31", "Label estimator or cutoff differs")
    require(proof["source_before_sha256"] == proof["source_after_sha256"] == record["full_state_sha256"]
            and all(proof[key] is True for key in ("source_unchanged", "global_rng_unchanged", "benchmark_paths_equal",
                "clone_initial_state_equal", "clone_storage_independent", "current_close_observation_verified"))
            and proof["price_basis"] == "hfq_research" and proof["research_only"] is True
            and proof["actionable"] is False, "Counterfactual source or branch isolation proof failed")
    require(count["branch_agent_steps"] == count["branch_benchmark_steps"] == count["macro_attempts"] == 40
            and all(type(count[key]) is int for key in ("branch_agent_steps", "branch_benchmark_steps", "macro_attempts",
                "collection_steps", "rl_steps", "nn_forwards", "optimizer_updates"))
            and all(count[key] == 0 for key in ("collection_steps", "rl_steps", "nn_forwards", "optimizer_updates")),
            "Counterfactual label execution counts differ")
    differences = np.asarray(target["per_step_relative_reward_difference"], dtype=float)
    require(differences.shape == (20,) and np.isfinite(differences).all(), "Invalid reward differences")
    hold, rebalance = target["branches"]["hold"], target["branches"]["rebalance"]
    require(hold["initial_agent_nav"] == rebalance["initial_agent_nav"] > 0, "Branch initial NAV differs")
    for first, branch in enumerate((hold, rebalance)):
        require(branch["endpoint"] == record["endpoint"] and len(branch["steps"]) == 20
                and branch["actual_agent_steps"] == branch["actual_benchmark_steps"] == 20, "Branch horizon differs")
        for j, step in enumerate(branch["steps"]):
            phase = (record["phase"] + j) % 20
            require(step["phase"] == phase and step["macro_action"] == (first if j == 0 else int(phase == 0))
                    and (j == 19 or not step["terminated"] and not step["truncated"]), "Branch changed continuation or ended early")
    for j in range(20):
        h, r = hold["steps"][j], rebalance["steps"][j]
        require(canonical(h["benchmark_ledger"]) == canonical(r["benchmark_ledger"])
                and abs(r["relative_log_reward"] - h["relative_log_reward"] - differences[j]) <= 1e-10,
                "Shadow path or per-step reward cancellation differs")
    require(all(np.isfinite(branch["final_agent_nav"]) and branch["final_agent_nav"] > 0 for branch in (hold, rebalance)),
            "Branch terminal NAV must be positive and finite")
    terminal = np.log(rebalance["final_agent_nav"]) - np.log(hold["final_agent_nav"])
    require(all(np.isfinite(target[key]) for key in ("discounted_relative_delta", "undiscounted_relative_delta", "terminal_log_nav_ratio"))
            and abs(float(np.dot(.995 ** np.arange(20), differences)) - target["discounted_relative_delta"]) <= 1e-10
            and abs(float(differences.sum()) - target["undiscounted_relative_delta"]) <= 1e-10
            and abs(terminal - target["terminal_log_nav_ratio"]) <= 1e-10
            and abs(terminal - target["undiscounted_relative_delta"]) <= 1e-10,
            "Primary discounted label or terminal audit identity differs")


def leaf_files(directory):
    return {p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p != directory / "completed.json"}


def index_row(record, label, label_hash):
    delta = label["labels"]["discounted_relative_delta"]
    return {**{key: record[key] for key in ("state_id", "episode", "offset", "decision_date", "endpoint", "phase",
            "initial_mode", "date_stratum", "phase_bin", "input_sha256", "full_state_sha256")},
        "schema_version": 1, "label_sha256": label_hash, "discounted_relative_delta": delta,
        "terminal_log_nav_ratio": label["labels"]["terminal_log_nav_ratio"],
        "oracle_calendar_advantage": max(-delta if record["phase"] == 0 else delta, 0.)}


def write_index(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def expected_counts():
    return {**counters(), "collection_agent_steps": 5760, "collection_benchmark_steps": 5760,
            "branch_agent_steps": 3840, "branch_benchmark_steps": 3840, "branch_macro_attempts": 3840, "completed_pairs": 96}


def cached_fold(root, protocol, fold):
    require(fold in FOLDS, "Unknown diagnostic fold")
    check_frozen(protocol, root)
    directory = Path(root) / "folds" / fold
    if not (directory / "completed.json").exists():
        return None
    completed = read_json(directory / "completed.json")
    require(completed["protocol_sha256"] == digest(protocol) and completed["fold"] == fold
            and completed["counters"] == expected_counts() and completed["files"] == leaf_files(directory),
            "Completed fold identity, budget or exact leaf inventory changed")
    plan = read_json(directory / "sampling_plan.json")
    validate_plan(plan, fold)
    seal = read_json(directory / "sampling_plan_seal.json")
    require(plan["protocol_sha256"] == digest(protocol) and completed["plan_sha256"] == file_hash(directory / "sampling_plan.json")
            and plan["data_identity"] == protocol["folds"][fold]["data_identity"]
            and seal == dict(protocol_sha256=digest(protocol), sampling_plan_sha256=completed["plan_sha256"],
                inputs_sha256=file_hash(directory / "inputs.npz"), states=96, labels_started=False),
            "Frozen sampling plan identity differs")
    with np.load(directory / "inputs.npz", allow_pickle=False) as archive:
        require(set(archive.files) == {"observations"} and archive["observations"].shape == (96, 129)
                and archive["observations"].dtype == np.float32, "Input archive schema differs")
        inputs = archive["observations"]
    rows = []
    for i, record in enumerate(plan["states"]):
        require(input_hash(inputs[i]) == record["input_sha256"], "Stored input changed")
        path = directory / "labels" / (record["state_id"] + ".json.gz")
        label = json.loads(gzip.decompress(path.read_bytes()))
        validate_label(label, record, fold)
        rows.append(index_row(record, label, file_hash(path)))
    with (directory / "index.csv").open(encoding="utf-8", newline="") as stream:
        saved = list(csv.DictReader(stream))
    require(saved == [{k: str(v) for k, v in row.items()} for row in rows], "Scalar index differs from actual labels")
    return rows


def run_fold(root, protocol, fold):
    require(fold in FOLDS, "Unknown diagnostic fold")
    check_frozen(protocol, root)
    cached = cached_fold(root, protocol, fold)
    if cached is not None:
        return cached
    directory = Path(root) / "folds" / fold
    require(not directory.exists(), "Never resume or overwrite a partial diagnostic fold")
    directory.mkdir(parents=True)
    counts, rows, env = counters(), [], None
    try:
        env = new_environment(protocol, fold)
        states = [record for record, _ in collection_states(env, fold, counts)]
        env.close()
        env = None
        plan = dict(schema_version=1, protocol_sha256=digest(protocol), fold=fold, seed=42,
                    data_identity=deepcopy(protocol["folds"][fold]["data_identity"]), states=states)
        validate_plan(plan, fold)
        write_json(directory / "sampling_plan.json", plan)
        plan_sha = file_hash(directory / "sampling_plan.json")
        np.savez_compressed(directory / "inputs.npz", observations=np.asarray([r["observation"] for r in states], dtype=np.float32))
        inputs_sha = file_hash(directory / "inputs.npz")
        write_json(directory / "sampling_plan_seal.json", dict(protocol_sha256=digest(protocol),
            sampling_plan_sha256=plan_sha, inputs_sha256=inputs_sha, states=96, labels_started=False))
        seal_sha = file_hash(directory / "sampling_plan_seal.json")
        (directory / "labels").mkdir()
        env = new_environment(protocol, fold)
        for index, (record, state) in enumerate(collection_states(env, fold, counts)):
            require(file_hash(directory / "sampling_plan.json") == plan_sha
                    and file_hash(directory / "sampling_plan_seal.json") == seal_sha
                    and file_hash(directory / "inputs.npz") == inputs_sha
                    and canonical(record) == canonical(states[index]),
                    "Second-pass full state, input or frozen plan differs")
            try:
                label = paired.paired_rule_label(state, horizon=20, train_end=fold + "-03-31", gamma=.995)
            except paired.CounterfactualLabelError as error:
                for target, source in (("branch_agent_steps", "branch_agent_steps"),
                        ("branch_benchmark_steps", "branch_benchmark_steps"), ("branch_macro_attempts", "macro_attempts")):
                    counts[target] += error.counters[source]
                raise
            for target, source in (("branch_agent_steps", "branch_agent_steps"),
                    ("branch_benchmark_steps", "branch_benchmark_steps"), ("branch_macro_attempts", "macro_attempts")):
                counts[target] += label["counters"][source]
            validate_label(label, record, fold)
            path = directory / "labels" / (record["state_id"] + ".json.gz")
            path.write_bytes(gzip.compress(canonical(label), mtime=0))
            rows.append(index_row(record, label, file_hash(path)))
            counts["completed_pairs"] += 1
        env.close()
        env = None
        require(counts == expected_counts() and len(rows) == 96, "Actual diagnostic budget or label count differs")
        write_index(directory / "index.csv", rows)
        write_json(directory / "counters.json", counts)
        check_frozen(protocol, root)
        write_json(directory / "completed.json", dict(schema_version=1, fold=fold, protocol_sha256=digest(protocol),
                   plan_sha256=plan_sha, counters=counts, files=leaf_files(directory)))
        return rows
    except Exception as error:
        write_json(directory / "failure.json", dict(status="failed_no_resume", protocol_sha256=digest(protocol),
            fold=fold, counters=counts, reason=str(error), traceback=traceback.format_exc(),
            evidence=getattr(error, "evidence", {}), completed_pairs=len(rows), no_replacement_sampling=True))
        raise
    finally:
        if env is not None:
            env.close()


def describe(rows):
    deltas = np.asarray([row["discounted_relative_delta"] for row in rows], dtype=float)
    groups = defaultdict(list)
    for row in rows:
        for key in ("phase", "phase_bin", "initial_mode", "date_stratum", "offset"):
            groups[f"{key}={row[key]}"].append(row["discounted_relative_delta"])
    overlaps = sum(a["decision_date"] <= b["endpoint"] and b["decision_date"] <= a["endpoint"]
                   for i, a in enumerate(rows) for b in rows[i + 1:])
    return dict(states=len(rows), mean=float(deltas.mean()), quantiles={str(q): float(np.quantile(deltas, q)) for q in (0., .25, .5, .75, 1.)},
        positive_signal=int((deltas >= .0001).sum()), negative_signal=int((deltas <= -.0001).sum()),
        neutral=int((np.abs(deltas) < .0001).sum()), mean_oracle_calendar_advantage=statistics.mean(r["oracle_calendar_advantage"] for r in rows),
        strata={k: dict(count=len(v), mean=statistics.mean(v)) for k, v in groups.items()},
        overlapping_interval_pairs=overlaps, duplicated_inputs=sum(n - 1 for n in Counter(r["input_sha256"] for r in rows).values()),
        interpretation="Single-seed overlapping historical rule-continuation outcomes; oracle opportunity is not predictability or an executable NAV curve.")


def write_report(root, protocol):
    check_frozen(protocol, root)
    folds = {}
    for fold in FOLDS:
        rows = cached_fold(root, protocol, fold)
        require(rows is not None, "Report cannot collect or invent a missing fold")
        folds[fold] = describe(rows)
    result = dict(status="diagnostic_complete", protocol_sha256=digest(protocol), folds=folds,
        states=192, agent_steps=19200, benchmark_steps=19200, book_steps=38400,
        rl_steps=0, nn_forwards=0, gradient_updates=0, optimizer_updates=0, model_fits=0,
        primary_label="discounted_relative_delta", actionable=False, qualified_candidate=None,
        limitations=["HFQ/static cohort research proxy", "No outer validation/2025/2026/protected-period evaluation",
            "Outer-train scaler is not an inner causal validation scaler", "No model was fitted or initialized from these labels"])
    write_json(Path(root) / "summary.json", result)
    write_json(Path(root) / "status.json", dict(status="diagnostic_complete", actionable=False, rl_steps=0))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "run", "report"))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--fold", choices=FOLDS)
    args = parser.parse_args(argv)
    if (args.stage == "run") != (args.fold is not None):
        parser.error("Only run requires exactly one registered --fold")
    root = output_root(args.output)
    if args.stage == "prepare":
        return prepare(root)
    require((root / "protocol.json").exists(), "Prepare and freeze before running diagnostics")
    protocol = read_json(root / "protocol.json")
    return run_fold(root, protocol, args.fold) if args.stage == "run" else write_report(root, protocol)


if __name__ == "__main__":
    main()
