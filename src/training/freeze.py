"""Bind a validation-selected checkpoint and configuration before final testing."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from src.agents.ppo_agent import PPOAgent
from src.backtest.run import git_metadata
from src.data.preprocess import ROOT, file_hash, read_config, resolve_path, write_json
from .env_factory import ResearchData


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def policy_digest(agent):
    digest = hashlib.sha256()
    for name, tensor in sorted(agent.model.policy.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT / "src").rglob("*.py")}


def freeze_selection(model_path, data, output):
    model_path, output = Path(model_path).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    selection_path = model_path.parent / "selection.json"
    selection = read_config(selection_path)
    if selection["best_path"] != model_path.name or selection["split"] != "val":
        raise ValueError("Only the validation-selected checkpoint can be frozen")
    agent = PPOAgent.load(model_path, data.contract, data.config["device"])
    metadata = read_config(model_path / "metadata.json")
    if agent.model.num_timesteps <= 0 or metadata["config"] != data.config:
        raise ValueError("Freeze requires a trained checkpoint with the original config")
    frozen = dict(created_at=datetime.now(timezone.utc).isoformat(), model_path=str(model_path),
                  model_sha256=file_hash(model_path / "model.zip"), policy_sha256=policy_digest(agent),
                  config_sha256=json_digest(data.config), research_config_sha256=json_digest(data.research),
                  contract=data.contract, selection=selection, selection_sha256=file_hash(selection_path),
                  source_hashes=source_hashes(), git=git_metadata(),
                  protocol="one_pass_test_no_retuning", test_evaluated=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, frozen)
    return frozen


def validate_frozen(agent, data, frozen):
    if (frozen["protocol"] != "one_pass_test_no_retuning" or frozen["contract"] != data.contract
            or frozen["config_sha256"] != json_digest(data.config)
            or frozen["research_config_sha256"] != json_digest(data.research)
            or frozen["policy_sha256"] != policy_digest(agent)
            or frozen["model_sha256"] != agent.provenance.get("loaded_model_sha256")
            or frozen["source_hashes"] != source_hashes()):
        raise ValueError("Frozen model, config, dataset or source mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data = ResearchData(read_config(args.config))
    freeze_selection(resolve_path(args.model), data, resolve_path(args.output))
    print(f"Frozen selection: {args.output}")


if __name__ == "__main__":
    main()
