from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PROFILE_NAME = "hcoxfix_trial010_forceB_softC"
G_CHECKPOINT_ROOT = Path(
    os.environ.get(
        "MTGP_G_CHECKPOINT_ROOT",
        str(
            ROOT
            / "single_endpoint_phaseA_G_checkpoint_hcoxfix_trial010"
            / "outputs"
            / "profiles"
            / PROFILE_NAME
            / "single_endpoint_knowledge_outputs"
        ),
    )
)
G_KNOWLEDGE_ROOT = Path(
    os.environ.get(
        "MTGP_G_KNOWLEDGE_ROOT",
        str(ROOT / "knowledge_hcoxfix_trial010_forceB_softC"),
    )
)
AUTO_2G_KNOWLEDGE_ROOT = Path(
    os.environ.get(
        "MTGP_AUTO_2G_KNOWLEDGE_ROOT",
        str(ROOT / "knowledge_hcoxfix_trial010_auto_2G"),
    )
)

_DEFAULT_ENDPOINT_NAMES = {
    "death_hospice": "event_1_death_hospice",
    "institutional_nonhome": "event_2_institutional_nonhome",
    "icu_los_gt7d": "event_3_icu_los_gt7d",
    "home": "event_5_home",
}


def _runtime_endpoint_names() -> dict[str, str]:
    raw = os.environ.get("MTGP_ENDPOINT_CONFIG_JSON", "").strip()
    if not raw:
        return dict(_DEFAULT_ENDPOINT_NAMES)
    payload = json.loads(raw)
    rows = payload.get("endpoints") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) < 2:
        raise RuntimeError("MTGP_ENDPOINT_CONFIG_JSON has no valid endpoint list")
    names = {str(row["name"]): str(row["id"]) for row in rows}
    if len(names) != len(rows) or len(set(names.values())) != len(rows):
        raise RuntimeError("Runtime endpoint names and IDs must be unique")
    return names


ENDPOINT_NAMES = _runtime_endpoint_names()


def original_endpoint_seeds() -> dict[str, int]:
    supplied = os.environ.get("MTGP_ENDPOINT_SEEDS_JSON", "").strip()
    if supplied:
        values = json.loads(supplied)
        expected = set(ENDPOINT_NAMES.values())
        if set(values) != expected:
            raise RuntimeError(
                "MTGP_ENDPOINT_SEEDS_JSON endpoint mismatch: "
                f"observed={sorted(values)}, expected={sorted(expected)}"
            )
        return {str(key): int(value) for key, value in values.items()}
    source = (
        ROOT
        / "single_endpoint_historical_cox_fixed_gp_trials_route1"
        / "profiles"
        / PROFILE_NAME
        / "single_endpoint_knowledge_outputs"
    )
    return {
        path.parent.name: int(json.loads(path.read_text(encoding="utf-8-sig"))["seed"])
        for path in source.glob("event_*/run_status.json")
    }


def validate_g_checkpoints() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for target_id in ENDPOINT_NAMES.values():
        paths = sorted((G_CHECKPOINT_ROOT / target_id / "phaseA_checkpoints").glob("*.pkl"))
        if len(paths) != 5:
            raise RuntimeError(f"Expected five G checkpoints for {target_id}, got {len(paths)}")
        generations = []
        for expected_call, path in enumerate(paths, 1):
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if int(payload["phaseA_call_index"]) != expected_call:
                raise RuntimeError(f"Checkpoint call mismatch: {path}")
            if not payload.get("python_random_state") or payload.get("numpy_random_state") is None:
                raise RuntimeError(f"Checkpoint RNG state missing: {path}")
            generations.append(int(payload["generation"]))
        supplied_expected = os.environ.get(
            "MTGP_G_EXPECTED_GENERATIONS_JSON", ""
        ).strip()
        if supplied_expected:
            expected_map = json.loads(supplied_expected)
            expected = [int(value) for value in expected_map[target_id]]
        else:
            expected = (
                [10, 10, 10, 10, 30]
                if target_id.startswith("event_3_")
                else [8, 8, 8, 8, 30]
            )
        if generations != expected:
            raise RuntimeError(
                f"Wrong effective G budgets for {target_id}: "
                f"{generations} != {expected}"
            )
        rows.append({"target_id": target_id, "generations": generations})
    return {
        "passed": True,
        "checkpoint_n": len(ENDPOINT_NAMES) * 5,
        "endpoints": rows,
    }


def checkpoint_phasea_source_pool(mt: Any, max_per_endpoint: int = 100) -> dict[str, list[dict[str, Any]]]:
    validate_g_checkpoints()
    pool: dict[str, list[dict[str, Any]]] = {}
    for endpoint, target_id in mt.ENDPOINT_IDS.items():
        rows_by_call: dict[int, list[dict[str, Any]]] = {}
        for checkpoint_path in sorted((G_CHECKPOINT_ROOT / target_id / "phaseA_checkpoints").glob("*.pkl")):
            with checkpoint_path.open("rb") as handle:
                payload = pickle.load(handle)
            call_index = int(payload["phaseA_call_index"])
            source_names = [str(value) for value in payload["gp_input_vars"]]
            groups = (
                (
                    "checkpoint_hof",
                    payload["hall_of_fame_expressions"],
                    payload["hall_of_fame_fitness"],
                    True,
                ),
                (
                    "checkpoint_population",
                    payload["population_expressions"],
                    payload["population_fitness"],
                    False,
                ),
            )
            call_rows: list[dict[str, Any]] = []
            seen_in_call: set[str] = set()
            for source_kind, individuals, fitness_values, is_hof in groups:
                fitness_scalars = [
                    float(value[0] if isinstance(value, (list, tuple)) else value)
                    for value in fitness_values
                ]
                ranked_indices = sorted(
                    range(len(fitness_scalars)),
                    key=lambda index: fitness_scalars[index],
                    reverse=True,
                )
                rank_by_index = {index: rank + 1 for rank, index in enumerate(ranked_indices)}
                group_n = max(1, len(fitness_scalars))
                for individual_index, expressions in enumerate(individuals):
                    for tree_index, expression_value in enumerate(expressions):
                        expression = str(expression_value).strip()
                        normalized = re.sub(r"\s+", "", expression)
                        if not expression or normalized in seen_in_call:
                            continue
                        used_indices = sorted({int(value) for value in re.findall(r"\bx(\d+)\b", expression)})
                        if any(index >= len(source_names) for index in used_indices):
                            continue
                        seen_in_call.add(normalized)
                        source_rank = int(rank_by_index[individual_index])
                        fitness_percentile = (
                            1.0
                            if group_n == 1
                            else 1.0 - float(source_rank - 1) / float(group_n - 1)
                        )
                        call_rows.append({
                            "source_target_id": target_id,
                            "source_target_name": endpoint,
                            "source_endpoint": endpoint,
                            "expression": expression,
                            "normalized_expression": normalized,
                            "source_feature_names": source_names,
                            "feature_names": source_names,
                            "used_features": [source_names[index] for index in used_indices],
                            "selected_in_final_model": bool(is_hof and payload["phaseA_call_index"] == 5),
                            "selected_by_fold": bool(is_hof and payload["phaseA_call_index"] < 5),
                            "rank": len(call_rows) + 1,
                            "source": source_kind,
                            "candidate_id": f"{target_id}_call{payload['phaseA_call_index']:02d}_{source_kind}_ind{individual_index:03d}_tree{tree_index:02d}",
                            "checkpoint_path": str(checkpoint_path.resolve()),
                            "checkpoint_call_index": call_index,
                            "checkpoint_generation": int(payload["generation"]),
                            "source_individual_index": int(individual_index),
                            "source_tree_slot": int(tree_index),
                            "source_individual_fitness": float(fitness_scalars[individual_index]),
                            "source_fitness_rank": source_rank,
                            "source_fitness_group_n": group_n,
                            "source_fitness_percentile": float(fitness_percentile),
                        })
            rows_by_call[call_index] = call_rows

        # Keep every fold/full-train boundary represented before filling the
        # remaining capacity with the strongest within-source individuals.
        per_call_quota = max(1, int(max_per_endpoint) // max(1, len(rows_by_call)))
        selected_ids: set[str] = set()
        rows: list[dict[str, Any]] = []
        for call_index in sorted(rows_by_call):
            for row in rows_by_call[call_index][:per_call_quota]:
                rows.append(row)
                selected_ids.add(str(row["candidate_id"]))
        remaining = [
            row
            for call_rows in rows_by_call.values()
            for row in call_rows
            if str(row["candidate_id"]) not in selected_ids
        ]
        remaining.sort(
            key=lambda row: (
                bool(row["selected_in_final_model"]),
                bool(row["selected_by_fold"]),
                float(row["source_fitness_percentile"]),
                str(row["source"]) == "checkpoint_hof",
            ),
            reverse=True,
        )
        rows.extend(remaining[:max(0, int(max_per_endpoint) - len(rows))])
        rows = rows[:int(max_per_endpoint)]
        if not rows:
            raise RuntimeError(f"Empty checkpoint source pool for {endpoint}")
        pool[endpoint] = rows
    return pool


def call_aware_phasea_transfer_candidates(
    mt: Any,
    source_pool: dict[str, list[dict[str, Any]]],
    target_gp_input_vars: dict[str, list[str]],
    transfer_matrix: np.ndarray,
    min_transfer_weight: float = 0.0,
    relation_weight_power: float = 1.0,
    max_trees_per_target: int = 100,
    max_trees_per_target_by_endpoint: dict[str, int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Map checkpoint trees while retaining one provenance copy per call."""
    endpoint_index = {name: index for index, name in enumerate(mt.ENDPOINTS)}
    by_target: dict[str, list[dict[str, Any]]] = {}
    transfer_log: list[dict[str, Any]] = []
    for target in mt.ENDPOINTS:
        candidates: list[dict[str, Any]] = []
        target_features = target_gp_input_vars[target]
        for source in mt.ENDPOINTS:
            if source == target:
                continue
            raw_weight = float(transfer_matrix[endpoint_index[target], endpoint_index[source]])
            weight = raw_weight ** float(relation_weight_power)
            if raw_weight < float(min_transfer_weight) or weight <= 0:
                continue
            for row in source_pool.get(source, []):
                mapped, transferable, reason = mt.remap_expression_by_feature_names(
                    row["expression"], row["source_feature_names"], target_features
                )
                log_row = {
                    **row,
                    "target_endpoint": target,
                    "transfer_weight": weight,
                    "whether_transferred": transferable,
                    "mapped_expression": mapped,
                    "transfer_reason": reason,
                }
                transfer_log.append(log_row)
                if not transferable:
                    continue
                candidate = dict(log_row)
                candidate.update({
                    "target_gp_input_vars": target_features,
                    "relation_score": weight,
                    "reestimated_in_target": True,
                    "source_coefficient_transferred": False,
                })
                candidates.append(candidate)

        candidates.sort(
            key=lambda row: (
                -float(row.get("transfer_weight", 0.0)),
                int(row.get("checkpoint_call_index", 10**9)),
                not bool(row.get("selected_in_final_model")),
                not bool(row.get("selected_by_fold")),
                -float(row.get("source_fitness_percentile", 0.0) or 0.0),
                int(row.get("rank") or 10**9),
            )
        )
        target_limit = int(
            (max_trees_per_target_by_endpoint or {}).get(target, max_trees_per_target)
        )
        per_call: dict[int, list[dict[str, Any]]] = {}
        for row in candidates:
            per_call.setdefault(int(row.get("checkpoint_call_index", -1)), []).append(row)
        quota = max(1, max(0, target_limit) // max(1, len(per_call)))
        selected: list[dict[str, Any]] = []
        seen: set[tuple[int, str]] = set()
        for call_index in sorted(per_call):
            for row in per_call[call_index]:
                key = (call_index, re.sub(r"\s+", "", str(row.get("mapped_expression", ""))))
                if key in seen:
                    continue
                seen.add(key)
                selected.append(row)
                if sum(int(value.get("checkpoint_call_index", -1)) == call_index for value in selected) >= quota:
                    break
        if len(selected) < target_limit:
            for row in candidates:
                call_index = int(row.get("checkpoint_call_index", -1))
                key = (call_index, re.sub(r"\s+", "", str(row.get("mapped_expression", ""))))
                if key in seen:
                    continue
                seen.add(key)
                selected.append(row)
                if len(selected) >= target_limit:
                    break
        by_target[target] = selected[:max(0, target_limit)]
    return by_target, transfer_log
