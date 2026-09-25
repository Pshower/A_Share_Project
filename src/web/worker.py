"""One task per process, using Graduate's interpreter and existing research code."""

import argparse
from copy import deepcopy
import re
import threading
import traceback

from src.data.preprocess import ROOT, read_config, write_json
from src.runtime import events
from .catalog import Catalog, identifier, inside
from .store import Store


def training_config(catalog, job, directory):
    payload = job["payload"]
    research = catalog.research_config(payload["dataset_id"])
    research_path = directory / "research.json"
    write_json(research_path, research)
    cfg = read_config(ROOT / "configs/ppo.json")
    cfg.update(research_config=str(research_path), stock_codes=payload["codes"], seed=payload["seed"],
               lookback=payload["lookback"], hidden_sizes=payload["hidden_sizes"],
               validation_rollouts=payload["validation_rollouts"], device="cpu", torch_threads=1,
               output_root=str(ROOT / "reports/runs/workbench"))
    for key in ["n_steps", "batch_size", "n_epochs", "learning_rate"]:
        cfg["ppo"][key] = payload[key]
    write_json(directory / "config.json", cfg)
    return cfg


def execute(job, directory):
    catalog = Catalog()
    payload, kind = job["payload"], job["kind"]
    output = ROOT / "reports/runs/workbench" / job["id"]
    events.phase("validating")
    if kind in {"training", "check"}:
        from src.training.train import main
        training_config(catalog, job, directory)
        args = ["--config", str(directory / "config.json"), "--run-id", job["id"]]
        args += ["--train", "--total-timesteps", str(payload["total_timesteps"])] if kind == "training" else ["--check-only"]
        main(args)
        return dict(output=str(output.relative_to(ROOT)), trained=kind == "training")
    if kind == "build":
        from src.data.preprocess import build_from_config
        cfg = catalog.research_config(payload["dataset_id"])
        final = inside(ROOT / "data/clean" / payload["version"], ROOT / "data/clean")
        if final.exists():
            raise FileExistsError("Dataset version already exists")
        staging = ROOT / "data/clean" / (".building_" + job["id"])
        stock_path = directory / "stocks.txt"
        stock_path.write_text("\n".join(payload["codes"]) + "\n", encoding="utf-8")
        cfg["dataset"].update(version=payload["version"], clean_dir=str(staging), stock_list_path=str(stock_path),
                              train_codes=payload["codes"], **{k: payload[k] for k in ["start_date", "train_end", "val_end", "end_date"]})
        events.phase("building")
        build_from_config(cfg)
        events.check_cancel()
        cfg["dataset"]["clean_dir"] = str(final)
        write_json(staging / "research_config.json", cfg)
        staging.rename(final)
        return dict(dataset_id=identifier(final, ROOT), output=str(final.relative_to(ROOT)))
    import torch
    from src.agents.ppo_agent import PPOAgent
    from src.training.env_factory import ResearchData
    from src.training.evaluate import evaluate_agent
    from src.training.freeze import freeze_selection
    from src.inference.predict import predict_action

    cfg, record, model_path = catalog.model_config(payload["model_id"])
    torch.set_num_threads(cfg.get("torch_threads", 1))
    data = ResearchData(cfg)
    if kind != "prediction" and not record["trained"]:
        raise ValueError("Initialized models cannot be used as trained candidates")
    agent = PPOAgent.load(model_path, data.contract, cfg["device"])
    if kind == "evaluation":
        frozen = None
        if payload["split"] == "test":
            freeze_job = Store(ROOT / ".workbench").get(payload["frozen_job_id"])
            if (freeze_job["kind"] != "freeze" or freeze_job["status"] != "succeeded"
                    or freeze_job["payload"]["model_id"] != payload["model_id"]):
                raise ValueError("Select a successful freeze for this model")
            frozen = read_config(inside(ROOT / freeze_job["result"]["frozen_path"], ROOT / "reports/runs"))
        events.phase("evaluating")
        evaluate_agent(agent, data, output, compare_baselines=True, split=payload["split"], frozen=frozen)
        return dict(report_id=identifier(output, ROOT), output=str(output.relative_to(ROOT)))
    if kind == "freeze":
        output.mkdir(parents=True, exist_ok=False)
        freeze_selection(model_path, data, output / "frozen_selection.json")
        return dict(frozen_path=str((output / "frozen_selection.json").relative_to(ROOT)), output=str(output.relative_to(ROOT)))
    if kind == "prediction":
        events.phase("predicting")
        result = predict_action(agent, data, payload["as_of"], payload.get("account"), payload["display_codes"])
        result["model_id"] = payload["model_id"]
        result["model_sha256"] = record["metadata"]["model_sha256"]
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "prediction.json", result)
        if payload["as_of"] > data.manifest["val_end"]:
            Store(ROOT / ".workbench").expose("test_observation", data.manifest["version"], payload["as_of"], payload["as_of"], job["id"])
        return dict(prediction_path=str((output / "prediction.json").relative_to(ROOT)), output=str(output.relative_to(ROOT)))
    if kind == "transfer":
        config = deepcopy(cfg)
        config["stock_codes"] = payload["codes"]
        target = ResearchData(config)
        env = target.vector_env()
        try:
            transferred = agent.transfer_to(env, config, target.contract)
            transferred.save(output / "transfer_model", config)
        finally:
            env.close()
        return dict(model_id=identifier(output / "transfer_model", ROOT), output=str(output.relative_to(ROOT)))
    raise ValueError("Unknown task type")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"j_[0-9a-f]{16}", args.job):
        raise ValueError("Invalid job ID")
    directory = ROOT / ".workbench/jobs" / args.job
    job = read_config(directory / "request.json")
    events.configure(directory)
    stopped = threading.Event()

    def heartbeat():
        while not stopped.is_set():
            (directory / "heartbeat").touch()
            stopped.wait(2)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        events.check_cancel()
        result = execute(job, directory)
        events.emit("finished", result=result)
        record = dict(status="succeeded", result=result)
    except events.Cancelled as error:
        events.emit("cancelled", message=str(error))
        record = dict(status="cancelled", error=str(error))
    except Exception as error:
        traceback.print_exc()
        events.emit("error", message=str(error))
        record = dict(status="failed", error=str(error))
    finally:
        stopped.set()
        thread.join(timeout=3)
    temporary = directory / "result.tmp"
    write_json(temporary, record)
    temporary.replace(directory / "result.json")


if __name__ == "__main__":
    main()
