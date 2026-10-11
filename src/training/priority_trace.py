"""Native priority-action traces layered on the unchanged economic trade audit."""

from copy import deepcopy

import numpy as np

from .research_trade import TradeTrace, verify_trace_prefix


class PriorityTrace(TradeTrace):
    """Preserve original 3N samples, including priorities rejected by the mapper.

    The parent still records all exact account snapshots, economic fills and
    integer-lot invariants. Passing original samples as arrays prevents its
    legacy TradeOrder branch from reconstructing only 2N coordinates.
    """

    DIAG_FIELDS = (*TradeTrace.DIAG_FIELDS, "priority_latent", "raw_quantity_latent",
                   "raw_priority_latent", "priority_active")

    def append(self, date, observation, action, info, **kwargs):
        diagnostics = action.diagnostics if hasattr(action, "diagnostics") else info.get("action_diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError("Priority trace requires original action diagnostics")
        n = len(observation["portfolio"]) - 1
        packed = np.asarray(diagnostics.get("raw_packed_action", diagnostics.get("latent"))) if hasattr(action, "diagnostics") else np.asarray(action)
        if packed.shape != (3 * n,) or not np.isfinite(packed).all():
            raise ValueError("Priority trace requires a finite original 3N action")
        variant = self.identity.get("actor_variant")
        if variant not in {"coupled", "decoupled"}:
            raise ValueError("Priority trace requires the registered actor variant")
        directions, quantities, priorities = np.split(packed, 3)
        if not np.isin(directions, range(4)).all():
            raise ValueError("Priority trace direction codes must be integers 0..3")
        buy = directions == 1
        if np.any(priorities[~buy] != 0) or (variant == "coupled" and np.any(priorities[buy] != quantities[buy])):
            raise ValueError("Priority action does not match its conditional atom support")
        enriched = dict(diagnostics, raw_quantity_latent=quantities.copy(),
                        raw_priority_latent=priorities.copy(), priority_active=buy.copy())
        copied_info = dict(info, action_diagnostics=enriched)
        super().append(date, observation, packed, copied_info, **kwargs)

    def manifest(self):
        result = super().manifest()
        result.update(schema_version=2, action_layout="direction_N_quantity_N_priority_N",
                      raw_action_semantics="Unclipped on-policy sample; mapped intent and actual fills stored separately.")
        return result


def verify_priority_trace(directory, manifest, *, actor_variant=None, steps=None):
    """Check stored 3N support without executing a policy or financial environment."""
    if manifest.get("schema_version") != 2 or manifest.get("action_layout") != "direction_N_quantity_N_priority_N":
        raise ValueError("Unsupported priority trace schema")
    if actor_variant is not None and manifest["identity"].get("actor_variant") != actor_variant:
        raise ValueError("Priority trace belongs to another actor variant")
    if steps is not None and manifest.get("total_steps") != steps:
        raise ValueError("Priority trace has a different number of actual steps")
    verify_trace_prefix(directory, manifest)
    from pathlib import Path
    n = len(manifest["identity"]["codes"])
    variant = manifest["identity"]["actor_variant"]
    if variant not in {"coupled", "decoupled"}:
        raise ValueError("Unregistered stored priority actor variant")
    for chunk in manifest["chunks"]:
        with np.load(Path(directory) / chunk["path"], allow_pickle=False) as values:
            raw = values["raw_packed_action"]
            if raw.shape != (chunk["steps"], 3 * n) or not np.isfinite(raw).all():
                raise ValueError("Priority trace lost original finite 3N actions")
            directions, quantities, priorities = np.split(raw, 3, axis=1)
            buy = directions == 1
            if (not np.isin(directions, range(4)).all() or np.any(priorities[~buy] != 0)
                    or variant == "coupled" and np.any(priorities[buy] != quantities[buy])):
                raise ValueError("Stored priority samples violate conditional support")
            legal = values["obs_legal_directions"]
            bounds = values["obs_trade_lot_bounds"]
            if (legal.shape != (chunk["steps"], n, 4) or not np.isin(legal, (0, 1)).all()
                    or not np.array_equal(directions, values["action_directions"])
                    or not np.take_along_axis(legal, directions.astype(np.int64)[..., None], axis=2).all()):
                raise ValueError("Original directions differ from their stored state-known legal support")
            active = buy | ((directions == 2) & (bounds[..., 3] > bounds[..., 2]))
            if np.any(quantities[~active] != 0):
                raise ValueError("Stored inactive quantity coordinates are not zero atoms")
            if (not np.array_equal(priorities, values["action_raw_priority_latent"])
                    or not np.array_equal(quantities, values["action_raw_quantity_latent"])
                    or not np.array_equal(buy, values["action_priority_active"])):
                raise ValueError("Stored priority coordinates disagree")
            if any(not np.isfinite(values[name]).all() for name in ("log_prob", "value", "reward") if name in values):
                raise ValueError("Priority trace has nonfinite rollout targets")
    return deepcopy(manifest)
