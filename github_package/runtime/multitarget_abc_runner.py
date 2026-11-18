from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from locked_trial31_safe_repair_20260802.safe_repair_primitives import (
    soft_div as _audited_soft_div,
    soft_log as _audited_soft_log,
    soft_sqrt as _audited_soft_sqrt,
)

import run_multitarget_profile_optimization as legacy
from endpoint_relation_config import (
    ENDPOINTS,
    ENDPOINT_GROUPS,
    ENDPOINT_IDS,
    EXCLUSIVE_GROUPS,
    build_exclusive_conflict_matrix,
    build_legacy_phaseC_transfer_matrix,
    build_phaseA_transfer_matrix,
    build_phaseC_transfer_matrix,
)
from phaseC_relation_transfer import (
    aggregate_phaseC_rule_pool,
    collect_phaseC_rule_pool,
    evaluate_transferred_phaseC_rule_in_target,
    get_transferable_phaseC_rules,
)
from relation_aware_mtgps import remap_expression_by_feature_names
from single_endpoint_abc_pipeline import SingleEndpointABCPipeline, load_result_json


WORKDIR = Path(
    os.environ.get("MTGP_DATA_DIR", str(Path(__file__).resolve().parent))
)
OUTPUT_ROOT = WORKDIR / "relation_aware_multitarget_outputs"
SUMMARY_DIR = OUTPUT_ROOT / "summaries"
SELECTED_DIR = OUTPUT_ROOT / "selected_profile"
LOCKED_STRUCTURE_DIRNAME = "locked_structures"
LOCKED_STRUCTURE_MANIFEST_NAME = "locked_structure_manifest.json"
LOCKED_EXTERNAL_VALIDATION_DIRNAME = "locked_structure_external_validation"
DEFAULT_SINGLE_ENDPOINT_REFERENCE_ROOT = (
    WORKDIR
    / "multitarget_profile_optimization_outputs_b_sensitive_v2"
    / "selected_profile_final_external"
)
SINGLE_ENDPOINT_KNOWLEDGE_ROOT = WORKDIR / "single_endpoint_knowledge_outputs"
SINGLE_ENDPOINT_KNOWLEDGE_CACHE = (
    SINGLE_ENDPOINT_KNOWLEDGE_ROOT / "single_endpoint_knowledge_cache.json"
)
USE_SINGLE_ENDPOINT_KNOWLEDGE_CACHE = True
SKIP_MULTITARGET_SELF_PRE_DISCOVERY = True
MAX_PHASEA_SOURCE_TREES_PER_ENDPOINT = 100
MAX_PHASEC_SOURCE_RULES_PER_ENDPOINT = 300

TARGET_NAME_BY_ID = {target_id: name for name, target_id in ENDPOINT_IDS.items()}
EXPECTED_TARGET_IDS = set(TARGET_NAME_BY_ID)

SCORE_WEIGHTS = {
    "mean_delta_vs_cox": 1.0,
    "min_delta_vs_cox": 0.8,
    "n_positive_vs_cox": 0.05,
    "negative_delta_vs_cox_penalty": -3.0,
    "mean_delta_vs_single": 0.4,
    "min_delta_vs_single": 0.3,
    "n_positive_vs_single": 0.02,
    "negative_delta_vs_single_penalty": -1.0,
    "exclusive_conflict_penalty": -0.2,
    "complexity_penalty": -0.01,
    "failure_penalty": -1000.0,
}

# The terminal-disposition constraint is an internal model-selection rule.  It
# is intentionally fixed: making this coefficient a tunable metadata field
# lets an optimiser improve its score by setting the coefficient to zero while
# leaving the fitted models unchanged.
DEFAULT_INTERNAL_EXCLUSIVE_WEIGHT = 0.20
DEFAULT_INTERNAL_EXCLUSIVE_HIGH_QUANTILE = 0.75

ABLATIONS = {
    "M0": {"relation_aware_phaseA": False, "relation_aware_phaseC": False},
    "M1": {"relation_aware_phaseA": True, "relation_aware_phaseC": False},
    "M2": {"relation_aware_phaseA": False, "relation_aware_phaseC": True},
    "M3": {"relation_aware_phaseA": True, "relation_aware_phaseC": True},
}

PHASEC_MIN_TRANSFER_WEIGHT = 0.30
PHASEC_MAX_RULES_PER_TARGET = 40
PHASEC_POOL_MODE = "raw_plus_final_fold_feasible_multisource_v1"
PHASEC_SUPPORT_GATE_MODE = "adaptive_uniform"
PHASEA_CANDIDATE_KEYS = [
    "phaseA_top_tree_candidates",
    "candidate_gp_expressions",
    "phaseA_fold_selected_gp_expressions",
    "final_gp_expressions",
]

PHASEC_CANDIDATE_KEYS = [
    "phasec_stgp_and_final_selected_candidates",
    "phasec_final_selected_candidates",
    "phaseC_selected_candidates",
    "phasec_stgp_and_selected_candidates_by_fold",
    "phasec_selected_candidates_by_fold",
    "phasec_stgp_and_rule_universe_feasible",
    "phasec_stgp_and_candidate_pool",
    "phaseC_candidate_pool",
    "phasec_all_candidates",
    "phasec_evaluated_candidates",
]

PHASEC_FINAL_KEYS = {
    "phasec_stgp_and_final_selected_candidates",
    "phasec_final_selected_candidates",
    "phaseC_selected_candidates",
}

PHASEC_FOLD_KEYS = {
    "phasec_stgp_and_selected_candidates_by_fold",
    "phasec_selected_candidates_by_fold",
}


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return data


def load_single_endpoint_knowledge_cache(cache_path: Path) -> dict:
    """Load and validate the four-endpoint knowledge cache manifest."""
    path = Path(cache_path)
    cache = _read_json_object(path, "single-endpoint knowledge cache")
    if not cache.get("schema_version"):
        raise ValueError("single_endpoint_knowledge_cache.json is missing schema_version")
    allowed_profile_names = {
        "profile_007_high_mutation",
        "hcoxfix_trial010_forceB_softC",
    }
    if cache.get("profile_name") not in allowed_profile_names:
        raise ValueError(
            "single-endpoint knowledge profile_name must be one of "
            f"{sorted(allowed_profile_names)}, got {cache.get('profile_name')!r}"
        )
    if cache.get("external_used_for_selection") is not False:
        raise ValueError("single-endpoint knowledge cache must set external_used_for_selection=false")
    endpoints = cache.get("endpoints")
    if not isinstance(endpoints, dict) or len(endpoints) != len(EXPECTED_TARGET_IDS):
        raise ValueError(
            "single-endpoint knowledge cache endpoint count mismatch: "
            f"observed={len(endpoints) if isinstance(endpoints, dict) else None}, "
            f"expected={len(EXPECTED_TARGET_IDS)}"
        )
    found_ids = {
        str(row.get("target_id"))
        for row in endpoints.values()
        if isinstance(row, dict) and row.get("target_id")
    }
    if found_ids != EXPECTED_TARGET_IDS:
        raise ValueError(
            "single-endpoint knowledge cache target_id mismatch: "
            f"missing={sorted(EXPECTED_TARGET_IDS - found_ids)} "
            f"unexpected={sorted(found_ids - EXPECTED_TARGET_IDS)}"
        )
    cache["_cache_path"] = str(path.resolve())
    return cache


def load_single_endpoint_knowledge_packages(cache: dict) -> dict:
    """Load the Phase A, Phase C, combined knowledge, and result JSON per endpoint."""
    packages: dict[str, dict[str, Any]] = {}
    cache_path = Path(str(cache.get("_cache_path") or SINGLE_ENDPOINT_KNOWLEDGE_CACHE))
    for cache_name, entry in cache["endpoints"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid cache endpoint entry: {cache_name}")
        target_id = str(entry.get("target_id") or "")
        target_name = TARGET_NAME_BY_ID.get(target_id)
        if not target_name:
            raise ValueError(f"Unknown target_id in knowledge cache: {target_id}")
        if str(cache_name) != target_name:
            raise ValueError(
                f"Knowledge cache endpoint name mismatch for {target_id}: "
                f"expected {target_name}, got {cache_name}"
            )
        combined_path = Path(str(entry.get("single_endpoint_knowledge_json") or ""))
        if not combined_path.is_absolute():
            combined_path = cache_path.parent / combined_path
        endpoint_dir = combined_path.parent
        paths = {
            "single_endpoint_knowledge": combined_path,
            "phaseA_knowledge": endpoint_dir / "phaseA_knowledge.json",
            "phaseC_knowledge": endpoint_dir / "phaseC_knowledge.json",
            "result": Path(str(entry.get("result_json") or endpoint_dir / "result.json")),
        }
        packages[target_id] = {
            "target_id": target_id,
            "target_name": target_name,
            "cache_entry": dict(entry),
            "paths": {key: str(path.resolve()) for key, path in paths.items()},
            **{key: _read_json_object(path, key) for key, path in paths.items()},
            "warnings": [],
            "errors": [],
        }
    return packages


def _phasea_input_vars(bundle: dict[str, Any]) -> list[str]:
    for source in (
        bundle.get("phaseA_knowledge", {}),
        bundle.get("single_endpoint_knowledge", {}),
        bundle.get("cache_entry", {}),
    ):
        values = source.get("phaseA_gp_input_vars") or source.get("gp_input_vars")
        if isinstance(values, list) and values:
            return [str(value) for value in values]
    return []


def _phasea_candidate_source(bundle: dict[str, Any]) -> tuple[str, list[Any]]:
    sources = (
        bundle.get("phaseA_knowledge", {}),
        bundle.get("single_endpoint_knowledge", {}),
        bundle.get("result", {}),
    )
    for key in PHASEA_CANDIDATE_KEYS:
        for source in sources:
            rows = source.get(key)
            if isinstance(rows, list) and rows:
                return key, rows
    return "", []


def _phasec_fields_present(bundle: dict[str, Any]) -> list[str]:
    found = []
    for key in PHASEC_CANDIDATE_KEYS:
        if any(key in source for source in (
            bundle.get("phaseC_knowledge", {}),
            bundle.get("single_endpoint_knowledge", {}),
            bundle.get("result", {}),
        )):
            found.append(key)
    return found


def validate_single_endpoint_knowledge_for_multitarget(packages: dict) -> dict:
    """Validate fields needed by internal relation-aware transfer."""
    if set(packages) != EXPECTED_TARGET_IDS:
        raise ValueError("Knowledge packages do not contain the configured target_id values")
    rows = []
    errors = []
    for target_id in sorted(EXPECTED_TARGET_IDS):
        bundle = packages[target_id]
        combined = bundle["single_endpoint_knowledge"]
        endpoint_errors: list[str] = []
        endpoint_warnings: list[str] = []
        gp_vars = _phasea_input_vars(bundle)
        phasea_key, phasea_rows = _phasea_candidate_source(bundle)
        phasec_fields = _phasec_fields_present(bundle)
        if combined.get("external_used_for_selection") is not False:
            endpoint_errors.append("external_used_for_selection_must_be_false")
        if combined.get("target_id") not in (None, target_id):
            endpoint_errors.append("single_endpoint_knowledge_target_id_mismatch")
        if combined.get("target_name") not in (None, bundle["target_name"]):
            endpoint_errors.append("single_endpoint_knowledge_target_name_mismatch")
        if not gp_vars:
            endpoint_errors.append("phaseA_gp_input_vars_missing_or_empty")
        if not phasea_rows:
            endpoint_errors.append("phaseA_candidates_missing")
        if not phasec_fields:
            endpoint_errors.append("phaseC_fields_missing")
        bundle["warnings"].extend(endpoint_warnings)
        bundle["errors"].extend(endpoint_errors)
        rows.append({
            "target_id": target_id,
            "target_name": bundle["target_name"],
            "phaseA_gp_input_vars_n": len(gp_vars),
            "phaseA_candidate_source": phasea_key,
            "phaseA_candidate_raw_n": len(phasea_rows),
            "phaseC_fields_n": len(phasec_fields),
            "knowledge_validation_passed": not endpoint_errors,
            "warnings": endpoint_warnings,
            "errors": endpoint_errors,
        })
        errors.extend(f"{target_id}:{error}" for error in endpoint_errors)
    if errors:
        raise ValueError("Invalid single-endpoint knowledge packages: " + "; ".join(errors))
    return {"passed": True, "endpoint_n": len(rows), "rows": rows, "errors": []}


def load_baseline_summary_from_single_endpoint_knowledge(
    packages: dict,
) -> dict[str, dict[str, Any]]:
    """Load ordinary Cox references from the four knowledge packages."""
    baseline_by_endpoint: dict[str, dict[str, Any]] = {}
    for target_id in sorted(EXPECTED_TARGET_IDS):
        bundle = packages.get(target_id)
        if not isinstance(bundle, dict):
            raise ValueError(f"Missing single-endpoint knowledge package for {target_id}")
        sources = (
            bundle.get("single_endpoint_knowledge", {}),
            bundle.get("result", {}),
            bundle.get("cache_entry", {}),
        )

        def first_finite(fields: tuple[str, ...]) -> float:
            for source in sources:
                for field in fields:
                    value = legacy.to_float(source.get(field))
                    if math.isfinite(value):
                        return value
            return math.nan

        internal = first_finite(
            ("baseline_internal_for_report", "reference_top10_internal_cindex")
        )
        external = first_finite(
            ("baseline_external_for_report", "reference_top10_external_cindex")
        )
        if not math.isfinite(internal):
            raise ValueError(
                f"Missing internal ordinary Cox baseline in single-endpoint knowledge for {target_id}"
            )
        target_name = bundle["target_name"]
        baseline_by_endpoint[target_name] = {
            "baseline_internal_for_report": internal,
            "baseline_external_for_report": external,
            "baseline_source_for_report": "single_endpoint_knowledge_cache_reference_top10_cox",
            "baseline_file_path": bundle["paths"]["single_endpoint_knowledge"],
            "reference_top10_internal_cindex": internal,
            "reference_top10_external_cindex": external,
        }
    return baseline_by_endpoint


def _tree_used_features(expression: str, source_feature_names: list[str]) -> tuple[list[str], list[int]]:
    indices = sorted({int(value) for value in re.findall(r"\bx(\d+)\b", expression)})
    invalid = [index for index in indices if index >= len(source_feature_names)]
    return [source_feature_names[index] for index in indices if index < len(source_feature_names)], invalid


def build_phaseA_source_pool_from_single_endpoint_knowledge(
    packages: dict,
    max_per_endpoint: int = MAX_PHASEA_SOURCE_TREES_PER_ENDPOINT,
) -> dict:
    """Build a normalized Phase A tree pool without external-metric ranking."""
    pool: dict[str, list[dict[str, Any]]] = {}
    for target_id, bundle in packages.items():
        target_name = bundle["target_name"]
        gp_vars = _phasea_input_vars(bundle)
        source_key, candidates = _phasea_candidate_source(bundle)
        seen = set()
        rows = []
        for rank, raw in enumerate(candidates, 1):
            candidate = dict(raw) if isinstance(raw, dict) else {"expression": raw}
            expression = str(candidate.get("expression") or "").strip()
            if not expression:
                bundle["warnings"].append(f"{source_key}:tree_without_expression")
                continue
            source_features = [str(value) for value in candidate.get("source_feature_names") or gp_vars]
            if source_features != gp_vars:
                bundle["warnings"].append(
                    f"{source_key}:source_feature_names_mismatch:{candidate.get('candidate_id', rank)}"
                )
                continue
            used_features, invalid_indices = _tree_used_features(expression, source_features)
            if invalid_indices:
                bundle["warnings"].append(
                    f"{source_key}:x_index_out_of_range:{candidate.get('candidate_id', rank)}:"
                    + ",".join(f"x{index}" for index in invalid_indices)
                )
                continue
            normalized = re.sub(r"\s+", "", str(candidate.get("normalized_expression") or expression))
            dedupe_key = (target_name, normalized)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            rows.append({
                "source_target_id": target_id,
                "source_target_name": target_name,
                "source_endpoint": target_name,
                "expression": expression,
                "normalized_expression": normalized,
                "source_feature_names": source_features,
                "feature_names": source_features,
                "used_features": [str(value) for value in candidate.get("used_features") or used_features],
                "selected_in_final_model": bool(candidate.get("selected_in_final_model", False)),
                "selected_by_fold": bool(candidate.get("selected_by_fold", False)),
                "valid_ab_cindex": candidate.get("valid_ab_cindex"),
                "valid_main_gp_cindex": candidate.get("valid_main_gp_cindex"),
                "cox_cv_mean_cindex": candidate.get("cox_cv_mean_cindex"),
                "tree_height": candidate.get("tree_height"),
                "tree_length": candidate.get("tree_length"),
                "tree_complexity": candidate.get("tree_complexity"),
                "rank": candidate.get("rank", rank),
                "source": source_key,
                "candidate_id": candidate.get("candidate_id"),
            })
            if len(rows) >= int(max_per_endpoint):
                break
        if not rows:
            raise ValueError(f"No valid Phase A source trees for {target_id}")
        pool[target_name] = rows
    return pool


def _phasec_pair(candidate: dict[str, Any]) -> list[str] | None:
    pair = candidate.get("interaction_pair")
    if isinstance(pair, list) and len(pair) == 2:
        out = [str(pair[0]).strip(), str(pair[1]).strip()]
    else:
        parsed = _pair_key(
            candidate.get("candidate_name") or candidate.get("name") or candidate.get("expr")
        )
        out = list(parsed) if parsed else []
    if len(out) != 2 or not all(out) or out[0] == out[1]:
        return None
    return sorted(out)


def build_phaseC_source_pool_from_single_endpoint_knowledge(
    packages: dict,
    max_per_endpoint: int = MAX_PHASEC_SOURCE_RULES_PER_ENDPOINT,
) -> dict:
    """Build a normalized Phase C AND-rule pool without external-metric ranking."""
    pool: dict[str, list[dict[str, Any]]] = {}
    for target_id, bundle in packages.items():
        target_name = bundle["target_name"]
        sources = (
            bundle.get("phaseC_knowledge", {}),
            bundle.get("single_endpoint_knowledge", {}),
            bundle.get("result", {}),
        )
        if not _phasec_fields_present(bundle):
            raise ValueError(f"No Phase C fields for {target_id}")
        seen = set()
        rows = []
        for key in PHASEC_CANDIDATE_KEYS:
            candidates = next(
                (source.get(key) for source in sources if isinstance(source.get(key), list)),
                [],
            )
            for raw in candidates:
                candidate = dict(raw) if isinstance(raw, dict) else {"expr": raw}
                pair = _phasec_pair(candidate)
                operator = str(candidate.get("operator") or "AND").upper()
                if pair is None or operator != "AND":
                    bundle["warnings"].append(f"{key}:invalid_and_rule")
                    continue
                dedupe_key = (target_name, tuple(pair))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                final_selected = key in PHASEC_FINAL_KEYS or bool(candidate.get("final_selected"))
                fold_selected = key in PHASEC_FOLD_KEYS or bool(candidate.get("selected_by_fold"))
                rows.append({
                    "source_target_id": target_id,
                    "source_target_name": target_name,
                    "source_endpoint": target_name,
                    "operator": "AND",
                    "interaction_pair": pair,
                    "expr": f"AND::{pair[0]} AND {pair[1]}",
                    "candidate_name": f"{pair[0]} AND {pair[1]}",
                    "support_inter_n": candidate.get("support_inter_n"),
                    "support_inter_events": candidate.get("support_inter_events"),
                    "train_loglik_improvement": candidate.get("train_loglik_improvement"),
                    "cv_cindex_delta": candidate.get("cv_cindex_delta"),
                    "fitness": candidate.get("fitness"),
                    "coefficient": candidate.get("coefficient"),
                    "hr": candidate.get("hr"),
                    "semantic_valid": candidate.get("semantic_valid", True),
                    "selected": bool(candidate.get("selected", False)),
                    "final_selected": final_selected,
                    "selected_by_fold": fold_selected,
                    "appeared_as_final_selected": final_selected,
                    "appeared_as_fold_selected": fold_selected,
                    "appeared_in_feasible_universe": key == "phasec_stgp_and_rule_universe_feasible",
                    "source_result_key": key,
                    "source": key,
                    "source_support_n": candidate.get("support_inter_n"),
                    "source_support_events": candidate.get("support_inter_events"),
                    "source_fitness": candidate.get("fitness"),
                    "source_loglik_improvement": candidate.get("train_loglik_improvement"),
                    "source_cv_cindex_delta": candidate.get("cv_cindex_delta"),
                })
                if len(rows) >= int(max_per_endpoint):
                    break
            if len(rows) >= int(max_per_endpoint):
                break
        if not rows:
            bundle["warnings"].append("phaseC_feasible_rules_empty")
        pool[target_name] = rows
    return pool


def _phasec_knowledge_counts(bundle: dict[str, Any]) -> dict[str, int]:
    phasec = bundle.get("phaseC_knowledge", {})

    def first_count(keys: tuple[str, ...]) -> int:
        for key in keys:
            rows = phasec.get(key)
            if isinstance(rows, list):
                return len(rows)
        return 0

    return {
        "phaseC_final_selected_n": first_count((
            "phasec_stgp_and_final_selected_candidates",
            "phasec_final_selected_candidates",
            "phaseC_selected_candidates",
        )),
        "phaseC_fold_selected_n": first_count((
            "phasec_stgp_and_selected_candidates_by_fold",
            "phasec_selected_candidates_by_fold",
        )),
        "phaseC_feasible_n": first_count((
            "phasec_stgp_and_rule_universe_feasible",
        )),
        "phaseC_candidate_pool_n": first_count((
            "phasec_stgp_and_candidate_pool",
            "phaseC_candidate_pool",
        )),
        "phaseC_all_or_evaluated_n": max(
            first_count(("phasec_all_candidates",)),
            first_count(("phasec_evaluated_candidates",)),
        ),
    }


def build_phaseA_transfer_candidates_from_source_pool(
    source_pool: dict[str, list[dict[str, Any]]],
    target_gp_input_vars: dict[str, list[str]],
    transfer_matrix: np.ndarray,
    min_transfer_weight: float = 0.0,
    relation_weight_power: float = 1.0,
    max_trees_per_target: int = 100,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Relation-filter and remap cached source trees for target-side evaluation."""
    idx = {name: i for i, name in enumerate(ENDPOINTS)}
    by_target: dict[str, list[dict[str, Any]]] = {}
    transfer_log = []
    for target in ENDPOINTS:
        candidates = []
        target_features = target_gp_input_vars[target]
        for source in ENDPOINTS:
            if source == target:
                continue
            raw_weight = float(transfer_matrix[idx[target], idx[source]])
            weight = raw_weight ** float(relation_weight_power)
            if raw_weight < float(min_transfer_weight) or weight <= 0:
                continue
            for row in source_pool.get(source, []):
                mapped, transferable, reason = remap_expression_by_feature_names(
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
                candidate = dict(row)
                candidate.update({
                    "target_endpoint": target,
                    "mapped_expression": mapped,
                    "target_gp_input_vars": target_features,
                    "transfer_weight": weight,
                    "relation_score": weight,
                    "whether_transferred": True,
                    "reestimated_in_target": True,
                    "source_coefficient_transferred": False,
                })
                candidates.append(candidate)
        candidates.sort(
            key=lambda row: (
                -float(row.get("transfer_weight", 0.0)),
                not bool(row.get("selected_in_final_model")),
                not bool(row.get("selected_by_fold")),
                int(row.get("rank") or 10**9),
                str(row.get("normalized_expression", "")),
            )
        )
        seen = set()
        by_target[target] = []
        for row in candidates:
            key = _norm_expr(row.get("expression"))
            if key in seen:
                continue
            seen.add(key)
            by_target[target].append(row)
            if len(by_target[target]) >= max_trees_per_target:
                break
    return by_target, transfer_log


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return str(obj)


def safe_to_excel(df: pd.DataFrame, path: Path) -> Path:
    """Write an Excel file, falling back to a timestamped name if locked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_excel(path, index=False)
        return path
    except OSError:
        fallback = path.with_name(path.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + path.suffix)
        df.to_excel(fallback, index=False)
        return fallback


LOCKED_FITTED_VALUE_KEYS = {
    "coef",
    "coefficient",
    "coefficients",
    "hazard_ratio",
    "hr",
    "p",
    "p_value",
    "se",
    "standard_error",
    "z",
}


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        value = value.to_dict(orient="records")
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, dict)]


def _first_records(data: dict[str, Any], keys: list[str]) -> list[dict[str, Any]]:
    for key in keys:
        rows = _records(data.get(key))
        if rows:
            return rows
    return []


def _list_value(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _first_list(data: dict[str, Any], keys: list[str]) -> list[str]:
    for key in keys:
        values = _list_value(data.get(key))
        if values:
            return values
    for summary in _records(data.get("summary")):
        for key in keys:
            values = _list_value(summary.get(key))
            if values:
                return values
    return []


def _structure_only_record(record: dict[str, Any]) -> dict[str, Any]:
    """Remove fitted statistics; Cell 4 locks structure, never coefficients."""
    return {
        str(key): value
        for key, value in record.items()
        if str(key).lower() not in LOCKED_FITTED_VALUE_KEYS
        and not str(key).lower().endswith("_coefficient")
    }


def _dedupe_structure_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for raw in rows:
        row = _structure_only_record(raw)
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _compact_locked_records(
    rows: list[dict[str, Any]], allowed_keys: tuple[str, ...], identity_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for raw in rows:
        row = _structure_only_record(raw)
        compact = {key: row[key] for key in allowed_keys if key in row}
        identity = tuple(
            json.dumps(compact.get(key), sort_keys=True, default=_json_default)
            for key in identity_keys
        )
        if identity in seen:
            continue
        seen.add(identity)
        out.append(compact)
    return out


def extract_locked_endpoint_structure(
    result_json_path: Path,
    endpoint: str,
    target_id: str,
    selected_profile: dict[str, Any],
) -> dict[str, Any]:
    """Convert one Cell 4 result into an immutable, coefficient-free structure."""
    data = load_result_json(Path(result_json_path))
    profile_params = data.get("profile_params") or data.get("final_params") or {}
    if not isinstance(profile_params, dict):
        profile_params = {}
    ab_params = profile_params.get("ab_params", profile_params)
    if not isinstance(ab_params, dict):
        ab_params = {}
    c_params = profile_params.get("c_params", {})
    if not isinstance(c_params, dict):
        c_params = {}

    main_vars = _first_list(data, ["main_effect_vars", "selected_main_effect_vars"])
    gp_vars = _first_list(data, ["gp_input_vars", "phaseA_gp_input_vars"])
    gp_rows = _compact_locked_records(
        _first_records(data, [
            "final_gp_expressions",
            "phaseA_final_selected_gp_expressions",
            "phaseA_fold_selected_gp_expressions",
        ]),
        (
            "expression",
            "mapped_expression",
            "notebook_mapped_expression",
            "normalized_expression",
            "source_feature_names",
            "target_gp_input_vars",
            "origin",
            "source_endpoint",
        ),
        (
            "mapped_expression",
            "notebook_mapped_expression",
            "expression",
            "normalized_expression",
        ),
    )
    for row in gp_rows:
        original = str(row.get("expression") or row.get("normalized_expression") or "")
        mapped = str(
            row.get("mapped_expression")
            or row.get("notebook_mapped_expression")
            or original
        )
        row["expression_original"] = original
        row["expression_mapped"] = mapped
        source_names = row.get("source_feature_names") or gp_vars
        target_names = row.get("target_gp_input_vars") or gp_vars
        row["source_feature_names"] = (
            [str(value) for value in source_names]
            if isinstance(source_names, list)
            else list(gp_vars)
        )
        row["target_gp_input_vars"] = (
            [str(value) for value in target_names]
            if isinstance(target_names, list)
            else list(gp_vars)
        )
    phaseb_rows = _compact_locked_records(
        _first_records(data, [
            "phaseb_timebasis_final_selected_candidates",
            "phaseb_timebasis_final_selected",
            "phaseb_timebasis_selected_by_fold",
        ]),
        ("name", "feature", "variable", "basis", "interaction_col", "type", "expr"),
        ("feature", "variable", "basis"),
    )
    phasec_rows = []
    for key in [
        "phasec_stgp_and_final_selected_candidates",
        "phasec_final_selected_candidates",
        "phaseC_selected_candidates",
        "phasec_stgp_and_selected_candidates_by_fold",
    ]:
        candidates = [
            row
            for row in _records(data.get(key))
            if str(row.get("operator") or "").upper() == "AND"
            and all(_locked_rule_pair(row))
        ]
        if candidates:
            phasec_rows = _compact_locked_records(
                candidates,
                (
                    "name",
                    "candidate_name",
                    "type",
                    "operator",
                    "main_effects",
                    "interaction",
                    "interaction_pair",
                    "cols_to_add",
                    "expr",
                    "complexity",
                    "leaf_terms",
                ),
                ("interaction_pair", "name", "candidate_name"),
            )
            break
    selected_k = int(data.get("selected_k_cat", data.get("selected_k", len(phasec_rows))) or 0)
    if selected_k >= 0 and phasec_rows:
        phasec_rows = phasec_rows[:selected_k]

    cox_penalizer = ab_params.get("COX_PENALIZER", data.get("COX_PENALIZER", 0.01))
    cox_l1_ratio = ab_params.get("COX_L1_RATIO", data.get("COX_L1_RATIO", 0.0))
    raw_profile = str(selected_profile.get("profile_name") or "")
    ablation = str(selected_profile.get("ablation") or "")
    if ablation and raw_profile.startswith(f"{ablation}_"):
        raw_profile = raw_profile.replace(f"{ablation}_", "", 1)
    train_file = WORKDIR / f"{target_id}_eicu.xlsx"
    external_file = WORKDIR / f"{target_id}_mimic.xlsx"
    structure = {
        "schema_version": "locked_endpoint_structure_v1",
        "created_at": now_iso(),
        "endpoint": endpoint,
        "target_name": endpoint,
        "target_id": target_id,
        "selected_ablation": ablation,
        "selected_profile": raw_profile,
        "selected_profile_name": raw_profile,
        "train_file": str(train_file.resolve()),
        "external_file": str(external_file.resolve()),
        "source_cell4_result_json": str(Path(result_json_path).resolve()),
        "structure_source": (
            "legacy_final_external_result"
            if "selected_profile/final_external" in Path(result_json_path).as_posix()
            else "cell4_result"
        ),
        "external_used_for_selection": False,
        "contains_fitted_coefficients": False,
        "main_effect_vars": main_vars,
        "phaseA_gp_input_vars": gp_vars,
        "phaseA_gp_expressions": gp_rows,
        "phaseA_transferred_feature_count": int(
            data.get("relation_aware_phaseA_final_selected_n", 0) or 0
        ),
        "phaseB_time_interactions": phaseb_rows,
        "phaseC_and_rules": phasec_rows,
        "phaseC_selected_k": selected_k,
        "fixed_cox_hyperparameters": {
            "penalizer": float(cox_penalizer),
            "l1_ratio": float(cox_l1_ratio),
        },
        "fixed_search_hyperparameters": {
            key: value
            for key, value in {**ab_params, **c_params}.items()
            if key not in LOCKED_FITTED_VALUE_KEYS
        },
        "preprocessing_plan": {
            "fit_dataset": "full_eICU_only",
            "categorical_encoding": "training-only one-hot encoding with locked columns",
            "missing_values": "training-only median imputation",
            "continuous_transform": "training-only 1%/99% winsorization and z-score",
            "mimic_policy": "transform_only; never fit, select, search, or tune",
        },
    }
    if not main_vars and not gp_rows and not phaseb_rows and not phasec_rows:
        raise ValueError(f"{endpoint}: Cell 4 result contains no lockable Cox structure")
    return structure


def _write_structure_workbook(structure: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        key: value
        for key, value in structure.items()
        if not isinstance(value, (list, dict))
    }
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame({"main_effect_var": structure["main_effect_vars"]}).to_excel(
            writer, sheet_name="main_effects", index=False
        )
        pd.DataFrame({"gp_input_var": structure["phaseA_gp_input_vars"]}).to_excel(
            writer, sheet_name="gp_inputs", index=False
        )
        pd.DataFrame(structure["phaseA_gp_expressions"]).to_excel(
            writer, sheet_name="phaseA_gp", index=False
        )
        pd.DataFrame(structure["phaseB_time_interactions"]).to_excel(
            writer, sheet_name="phaseB", index=False
        )
        pd.DataFrame(structure["phaseC_and_rules"]).to_excel(
            writer, sheet_name="phaseC", index=False
        )


def write_locked_structure_manifest(
    output_root: Path,
    selected_profile: dict[str, Any],
    internal_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the Cell 4 hand-off consumed exclusively by Cell 5."""
    locked_dir = Path(output_root) / LOCKED_STRUCTURE_DIRNAME
    locked_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = locked_dir / LOCKED_STRUCTURE_MANIFEST_NAME
    if manifest_path.exists():
        manifest_path.unlink()
    endpoints = []
    pending_outputs = []
    for row in internal_rows:
        endpoint = str(row.get("target_name") or "")
        target_id = str(row.get("target_id") or ENDPOINT_IDS.get(endpoint, ""))
        result_path = Path(str(row.get("result_json_path") or ""))
        if endpoint not in ENDPOINTS or not result_path.exists():
            raise RuntimeError(f"Cannot lock {endpoint or target_id}: result.json is missing")
        structure = extract_locked_endpoint_structure(
            result_path, endpoint, target_id, selected_profile
        )
        train_file = Path(structure["train_file"])
        dry_run_audit = dry_run_validate_locked_structure_on_eicu(
            structure, pd.read_excel(train_file), max_rows=20
        )
        if dry_run_audit.get("dry_run_passed") is not True:
            raise ValueError(f"{endpoint}: locked structure dry-run did not pass")
        json_path = locked_dir / f"{target_id}_{endpoint}_locked_structure.json"
        xlsx_path = locked_dir / f"{target_id}_{endpoint}_locked_structure.xlsx"
        audit_path = locked_dir / f"{target_id}_{endpoint}_locked_structure_audit.json"
        audit = {
            "endpoint": endpoint,
            "target_name": endpoint,
            "target_id": target_id,
            "locked_structure_json": str(json_path.resolve()),
            "locked_structure_xlsx": str(xlsx_path.resolve()),
            "external_used_for_selection": False,
            "contains_fitted_coefficients": False,
            "mimic_read_in_cell4": False,
            "critical_fields_present": True,
            **dry_run_audit,
        }
        endpoints.append({
            "endpoint": endpoint,
            "target_name": endpoint,
            "target_id": target_id,
            "train_file": str(train_file.resolve()),
            "external_file": structure["external_file"],
            "locked_structure_json": str(json_path.resolve()),
            "locked_structure_xlsx": str(xlsx_path.resolve()),
            "structure_audit_json": str(audit_path.resolve()),
            "source_cell4_result_json": structure["source_cell4_result_json"],
            "dry_run_passed": True,
            "status": "success",
        })
        pending_outputs.append((structure, json_path, xlsx_path, audit, audit_path))
    if {row["endpoint"] for row in endpoints} != set(ENDPOINTS):
        raise RuntimeError("Locked structure manifest must contain all four endpoints")
    ablation = str(selected_profile.get("ablation") or "")
    profile = str(selected_profile.get("profile_name") or "")
    if ablation and profile.startswith(f"{ablation}_"):
        profile = profile.replace(f"{ablation}_", "", 1)
    manifest = {
        "schema_version": "locked_structure_manifest_v1",
        "created_at": now_iso(),
        "development_dataset": "eICU",
        "external_dataset": "MIMIC",
        "selected_profile": profile,
        "selected_ablation": ablation,
        "profile_locked_by": "eICU_internal_CV_only",
        "ablation_locked_by": "eICU_internal_CV_only",
        "endpoint_structures_locked_by": "eICU_internal_CV_only",
        "external_used_for_selection": False,
        "mimic_used_for_selection": False,
        "mimic_used_for_structure_locking": False,
        "mimic_read_in_cell4": False,
        "full_eicu_refit_done_in_cell4": False,
        "cell5_requires_full_eicu_deterministic_refit": True,
        "cell5_external_validation_after_refit": True,
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
    }
    for structure, json_path, xlsx_path, audit, audit_path in pending_outputs:
        json_path.write_text(
            json.dumps(structure, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        _write_structure_workbook(structure, xlsx_path)
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    safe_to_excel(pd.DataFrame(endpoints), locked_dir / "locked_structure_manifest.xlsx")
    return {
        "locked_structure_manifest": str(manifest_path.resolve()),
        "locked_structure_count": len(endpoints),
        "locked_structures_complete": True,
    }


def load_locked_structure_manifest(path: Path) -> dict[str, Any]:
    """Load and strictly validate the Cell 4 locked-structure manifest."""
    manifest = _read_json_object(Path(path), "locked structure manifest")
    required_values = {
        "schema_version": "locked_structure_manifest_v1",
        "development_dataset": "eICU",
        "external_dataset": "MIMIC",
        "profile_locked_by": "eICU_internal_CV_only",
        "ablation_locked_by": "eICU_internal_CV_only",
        "endpoint_structures_locked_by": "eICU_internal_CV_only",
        "external_used_for_selection": False,
        "mimic_used_for_selection": False,
        "mimic_used_for_structure_locking": False,
        "mimic_read_in_cell4": False,
        "full_eicu_refit_done_in_cell4": False,
        "cell5_requires_full_eicu_deterministic_refit": True,
        "cell5_external_validation_after_refit": True,
    }
    for field, expected in required_values.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"Locked structure manifest field {field!r} must equal {expected!r}"
            )
    if not manifest.get("selected_profile") or not manifest.get("selected_ablation"):
        raise ValueError("Locked structure manifest lacks selected profile/ablation")
    if manifest.get("external_used_for_selection") is not False:
        raise RuntimeError("Locked structures were not produced by internal-only selection")
    rows = manifest.get("endpoints")
    if (
        not isinstance(rows, list)
        or len(rows) != len(ENDPOINTS)
        or manifest.get("endpoint_count") != len(ENDPOINTS)
    ):
        raise RuntimeError("Locked structure manifest must list exactly four endpoints")
    endpoint_fields = (
        "target_id",
        "target_name",
        "endpoint",
        "train_file",
        "external_file",
        "locked_structure_json",
        "locked_structure_xlsx",
        "structure_audit_json",
        "source_cell4_result_json",
        "dry_run_passed",
        "status",
    )
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Locked structure endpoint entry must be an object")
        missing = [field for field in endpoint_fields if field not in row]
        if missing:
            raise ValueError(f"Locked structure endpoint entry missing {missing}")
        endpoint = str(row["endpoint"])
        if row["target_name"] != endpoint or row["target_id"] != ENDPOINT_IDS.get(endpoint):
            raise ValueError(f"Locked structure endpoint identity mismatch: {endpoint}")
        if row["dry_run_passed"] is not True or row["status"] != "success":
            raise ValueError(f"Locked structure endpoint did not pass dry-run: {endpoint}")
        for field in (
            "train_file",
            "external_file",
            "locked_structure_json",
            "locked_structure_xlsx",
            "structure_audit_json",
            "source_cell4_result_json",
        ):
            if not Path(str(row[field])).exists():
                raise ValueError(f"Locked structure endpoint path missing: {field}")
        structure = load_locked_endpoint_structure(
            Path(str(row["locked_structure_json"]))
        )
        if structure["target_id"] != row["target_id"]:
            raise ValueError(f"Manifest/structure target mismatch: {endpoint}")
    if {str(row.get("endpoint")) for row in rows} != set(ENDPOINTS):
        raise RuntimeError("Locked structure manifest endpoint set is invalid")
    return manifest


def load_locked_endpoint_structure(path: Path) -> dict[str, Any]:
    structure = _read_json_object(Path(path), "locked endpoint structure")
    required_fields = (
        "schema_version",
        "target_id",
        "selected_profile",
        "selected_ablation",
        "fixed_cox_hyperparameters",
        "source_cell4_result_json",
    )
    missing = [field for field in required_fields if not structure.get(field)]
    if missing:
        raise ValueError(f"Locked endpoint structure missing fields: {missing}")
    if structure.get("schema_version") != "locked_endpoint_structure_v1":
        raise ValueError("Unsupported locked endpoint structure schema_version")
    endpoint = str(structure.get("target_name") or structure.get("endpoint") or "")
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unknown locked endpoint {endpoint!r}")
    if str(structure.get("target_id")) != ENDPOINT_IDS[endpoint]:
        raise ValueError("Locked endpoint target_id does not match endpoint")
    if structure.get("contains_fitted_coefficients") is not False:
        raise ValueError("Locked endpoint structure contains fitted coefficients")
    if structure.get("external_used_for_selection") is not False:
        raise ValueError("Locked endpoint structure used external data for selection")
    if not (
        structure.get("main_effect_vars")
        or structure.get("phaseA_gp_expressions")
        or structure.get("phaseB_time_interactions")
        or structure.get("phaseC_and_rules")
    ):
        raise ValueError("Locked endpoint structure has no Cox model structure")
    for index, row in enumerate(structure.get("phaseA_gp_expressions") or [], start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Locked GP expression {index} is not an object")
        for field in (
            "expression_original",
            "expression_mapped",
            "source_feature_names",
            "target_gp_input_vars",
        ):
            if field not in row:
                raise ValueError(f"Locked GP expression {index} missing {field}")
    if not Path(str(structure["source_cell4_result_json"])).exists():
        raise ValueError("Locked endpoint source_cell4_result_json does not exist")
    if structure["selected_ablation"] not in ABLATIONS:
        raise ValueError("Locked endpoint selected_ablation is invalid")
    hyper = structure["fixed_cox_hyperparameters"]
    if not isinstance(hyper, dict) or not {"penalizer", "l1_ratio"} <= set(hyper):
        raise ValueError("Locked endpoint fixed_cox_hyperparameters is incomplete")
    train_file = Path(str(structure.get("train_file") or WORKDIR / f"{structure['target_id']}_eicu.xlsx"))
    external_file = Path(str(structure.get("external_file") or WORKDIR / f"{structure['target_id']}_mimic.xlsx"))
    if not train_file.exists() or not external_file.exists():
        raise ValueError("Locked endpoint train_file/external_file cannot be resolved")
    structure["endpoint"] = endpoint
    structure["target_name"] = endpoint
    structure["train_file"] = str(train_file.resolve())
    structure["external_file"] = str(external_file.resolve())
    return structure


def _locked_required_raw_features(structure: dict[str, Any]) -> list[str]:
    required = list(structure.get("main_effect_vars") or [])
    required.extend(structure.get("phaseA_gp_input_vars") or [])
    for row in structure.get("phaseB_time_interactions") or []:
        value = row.get("feature") or row.get("variable")
        if value and not re.fullmatch(r"gp\d+", str(value)):
            required.append(str(value))
    for row in structure.get("phaseC_and_rules") or []:
        if str(row.get("type") or "").lower() == "main":
            values = row.get("main_effects") or row.get("leaf_terms")
            if isinstance(values, list):
                required.extend(str(value) for value in values[:1] if value)
            continue
        for key in ("var1", "var2", "feature1", "feature2", "left", "right"):
            value = row.get(key)
            if value:
                required.append(str(value))
        pair = row.get("interaction_pair") or row.get("main_effects")
        if isinstance(pair, list):
            required.extend(str(value) for value in pair[:2] if value)
    return list(dict.fromkeys(str(value) for value in required if str(value)))


LOCKED_CATEGORICAL_COLUMNS = ("gender", "ethnicity", "admission_type")


def _locked_dummy_source(feature: str) -> str | None:
    for raw in LOCKED_CATEGORICAL_COLUMNS:
        if str(feature).startswith(f"{raw}_"):
            return raw
    return None


def validate_locked_required_features(
    structure: dict[str, Any],
    train_df: pd.DataFrame,
    external_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Strictly verify all raw dependencies needed by a locked structure."""
    required = _locked_required_raw_features(structure)
    train_missing = []
    external_missing = []
    one_hot_expected = []
    one_hot_external_zero = []
    train_encoded = pd.get_dummies(
        train_df.copy(),
        columns=[c for c in LOCKED_CATEGORICAL_COLUMNS if c in train_df],
        prefix=[c for c in LOCKED_CATEGORICAL_COLUMNS if c in train_df],
        dtype=float,
    )
    external_encoded = None
    if external_df is not None:
        external_encoded = pd.get_dummies(
            external_df.copy(),
            columns=[c for c in LOCKED_CATEGORICAL_COLUMNS if c in external_df],
            prefix=[c for c in LOCKED_CATEGORICAL_COLUMNS if c in external_df],
            dtype=float,
        )
    for feature in required:
        raw_categorical = _locked_dummy_source(feature)
        if raw_categorical:
            one_hot_expected.append(feature)
            if raw_categorical not in train_df or feature not in train_encoded:
                train_missing.append(feature)
            if external_df is not None:
                if raw_categorical not in external_df:
                    external_missing.append(raw_categorical)
                elif feature not in external_encoded:
                    one_hot_external_zero.append(feature)
            continue
        if feature not in train_df:
            train_missing.append(feature)
        if external_df is not None and feature not in external_df:
            external_missing.append(feature)
    audit = {
        "required_raw_features": required,
        "train_missing_raw_features": sorted(set(train_missing)),
        "external_missing_raw_features": sorted(set(external_missing)),
        "one_hot_columns_expected": sorted(set(one_hot_expected)),
        "one_hot_missing_category_filled_zero": sorted(set(one_hot_external_zero)),
        "strict_missing_feature_check_passed": not train_missing and not external_missing,
    }
    if train_missing:
        raise ValueError(f"Missing required full-eICU features: {sorted(set(train_missing))}")
    if external_missing:
        raise ValueError(f"Missing required MIMIC raw features: {sorted(set(external_missing))}")
    return audit


def _fit_locked_preprocessor(
    df: pd.DataFrame, required: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    categorical = [c for c in LOCKED_CATEGORICAL_COLUMNS if c in df]
    encoded = pd.get_dummies(df.copy(), columns=categorical, prefix=categorical, dtype=float)
    stats: dict[str, Any] = {}
    out = pd.DataFrame(index=df.index)
    for col in required:
        if col not in encoded:
            raise ValueError(f"Missing required full-eICU encoded feature {col!r}")
        series = pd.to_numeric(encoded[col], errors="coerce")
        median = float(series.median()) if series.notna().any() else 0.0
        filled = series.fillna(median).astype(float)
        binary = bool(filled.dropna().nunique() <= 2)
        lower = float(filled.quantile(0.01)) if not binary else float(filled.min())
        upper = float(filled.quantile(0.99)) if not binary else float(filled.max())
        clipped = filled.clip(lower, upper)
        mean = float(clipped.mean()) if not binary else 0.0
        std = float(clipped.std(ddof=0)) if not binary else 1.0
        if not math.isfinite(std) or std <= 0:
            std = 1.0
        out[col] = clipped if binary else (clipped - mean) / std
        stats[col] = {
            "median": median,
            "lower": lower,
            "upper": upper,
            "mean": mean,
            "std": std,
            "binary": binary,
        }
    return out, {"categorical_columns": categorical, "required_columns": required, "stats": stats}


def _transform_locked_preprocessor(
    df: pd.DataFrame, prep: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    missing_raw_categorical = [c for c in prep["categorical_columns"] if c not in df]
    if missing_raw_categorical:
        raise ValueError(f"Missing required MIMIC categorical columns: {missing_raw_categorical}")
    categorical = list(prep["categorical_columns"])
    encoded = pd.get_dummies(df.copy(), columns=categorical, prefix=categorical, dtype=float)
    out = pd.DataFrame(index=df.index)
    one_hot_zero = []
    for col in prep["required_columns"]:
        stat = prep["stats"][col]
        if col not in encoded:
            if _locked_dummy_source(col):
                series = pd.Series(0.0, index=df.index)
                one_hot_zero.append(col)
            else:
                raise ValueError(f"Missing required MIMIC feature {col!r}")
        else:
            series = pd.to_numeric(encoded[col], errors="coerce")
        clipped = series.fillna(stat["median"]).astype(float).clip(stat["lower"], stat["upper"])
        out[col] = clipped if stat["binary"] else (clipped - stat["mean"]) / stat["std"]
    return out, one_hot_zero


def _locked_expression_text(row: dict[str, Any]) -> str:
    return str(
        row.get("mapped_expression")
        or row.get("notebook_mapped_expression")
        or row.get("expression")
        or row.get("normalized_expression")
        or ""
    )


LOCKED_GP_FUNCTIONS = {
    "add",
    "sub",
    "mul",
    "div",
    "safe_div",
    "log",
    "safe_log",
    "sqrt",
    "safe_sqrt",
    "neg",
    "abs",
    "clip",
}


def _alias_gp_expression_variables(
    expression: str, frame: pd.DataFrame
) -> tuple[str, dict[str, np.ndarray]]:
    """Alias real feature names that cannot be represented as parser tokens."""
    expr = str(expression)
    alias_env: dict[str, np.ndarray] = {}
    columns = sorted((str(col) for col in frame.columns), key=len, reverse=True)
    alias_index = 0
    for col in columns:
        if (
            re.fullmatch(r"[A-Za-z_]\w*", col)
            or re.fullmatch(r"(?:x|ARG)\d+", col)
            or col in LOCKED_GP_FUNCTIONS
        ):
            continue
        if col not in expr:
            continue
        alias = f"GPVARALIAS{alias_index}"
        alias_index += 1
        expr = re.sub(re.escape(col), alias, expr)
        values = frame[col]
        if isinstance(values, pd.DataFrame):
            raise ValueError(f"Duplicate GP frame column {col!r}")
        alias_env[alias] = values.to_numpy(dtype=float)
    return expr, alias_env


def _tokenize_locked_gp_expression(expression: str) -> list[str]:
    token_pattern = re.compile(
        r"[A-Za-z_]\w*|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[(),]"
    )
    tokens = []
    position = 0
    while position < len(expression):
        if expression[position].isspace():
            position += 1
            continue
        match = token_pattern.match(expression, position)
        if not match:
            raise ValueError(
                f"Unsupported character in GP expression at position {position}: "
                f"{expression[position]!r}"
            )
        tokens.append(match.group(0))
        position = match.end()
    return tokens


def _evaluate_locked_gp_expression(expression: str, frame: pd.DataFrame) -> np.ndarray:
    """Evaluate the notebook's small GP expression language without eval()."""
    aliased_expression, alias_env = _alias_gp_expression_variables(
        expression, frame
    )
    tokens = _tokenize_locked_gp_expression(aliased_expression)
    position = 0

    def parse():
        nonlocal position
        if position >= len(tokens):
            raise ValueError(f"Invalid GP expression: {expression}")
        token = tokens[position]
        position += 1
        indexed = re.fullmatch(r"(?:x|ARG)(\d+)", token)
        if indexed and token not in frame.columns:
            raise ValueError(
                f"GP input index {token} exceeds phaseA_gp_input_vars range"
            )
        if token in alias_env:
            return alias_env[token]
        if token in frame.columns:
            return frame[token].to_numpy(dtype=float)
        if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", token):
            return np.full(len(frame), float(token))
        if position >= len(tokens) or tokens[position] != "(":
            raise ValueError(f"Unknown GP token {token!r}")
        position += 1
        if position >= len(tokens):
            raise ValueError(f"Unclosed GP expression: {expression}")
        args = [parse()]
        while position < len(tokens) and tokens[position] == ",":
            position += 1
            args.append(parse())
        if position >= len(tokens) or tokens[position] != ")":
            raise ValueError(f"Unclosed GP expression: {expression}")
        position += 1
        if token == "add":
            return args[0] + args[1]
        if token == "sub":
            return args[0] - args[1]
        if token == "mul":
            return np.clip(args[0] * args[1], -1e6, 1e6)
        if token in {"div", "safe_div"}:
            return _audited_soft_div(args[0], args[1])
        if token in {"log", "safe_log"}:
            return _audited_soft_log(args[0])
        if token in {"sqrt", "safe_sqrt"}:
            return _audited_soft_sqrt(args[0])
        if token == "neg":
            return -args[0]
        if token == "abs":
            return np.abs(args[0])
        if token == "clip":
            return np.clip(args[0], -10.0, 10.0)
        raise ValueError(f"Unsupported GP function {token!r}")

    value = np.asarray(parse(), dtype=float)
    if position != len(tokens):
        raise ValueError(f"Trailing GP tokens in {expression!r}")
    return np.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)


def _locked_rule_pair(row: dict[str, Any]) -> tuple[str, str]:
    left = row.get("var1") or row.get("feature1") or row.get("left") or row.get("a")
    right = row.get("var2") or row.get("feature2") or row.get("right") or row.get("b")
    if (not left or not right) and isinstance(row.get("variables"), list):
        values = row["variables"]
        if len(values) >= 2:
            left, right = values[:2]
    if not left or not right:
        values = row.get("interaction_pair") or row.get("main_effects")
        if isinstance(values, list) and len(values) >= 2:
            left, right = values[:2]
    return str(left or ""), str(right or "")


def _build_locked_static_design(
    transformed: pd.DataFrame, structure: dict[str, Any]
) -> pd.DataFrame:
    design = pd.DataFrame(index=transformed.index)
    for col in structure.get("main_effect_vars") or []:
        if col not in transformed:
            raise ValueError(f"Missing locked main effect variable {col!r}")
        design[f"main__{col}"] = transformed[col]
    gp_frame = transformed.copy()
    for index, col in enumerate(structure.get("phaseA_gp_input_vars") or []):
        if col not in transformed:
            raise ValueError(f"Missing GP input variable {col}")
        gp_frame[f"ARG{index}"] = transformed[col]
        gp_frame[f"x{index}"] = transformed[col]
    for index, row in enumerate(structure.get("phaseA_gp_expressions") or [], start=1):
        expression = _locked_expression_text(row)
        if not expression:
            raise ValueError(f"Locked GP expression {index} is empty")
        try:
            design[f"gp{index}"] = _evaluate_locked_gp_expression(
                expression, gp_frame
            )
        except Exception as exc:
            raise ValueError(
                f"Locked GP expression failed endpoint={structure.get('endpoint')} "
                f"gp_index={index} expression={expression!r}: {exc}"
            ) from exc
    for index, row in enumerate(structure.get("phaseC_and_rules") or [], start=1):
        if str(row.get("type") or "").lower() == "main":
            values = row.get("main_effects") or row.get("leaf_terms")
            feature = str(values[0]) if isinstance(values, list) and values else ""
            if not feature or feature not in transformed:
                raise ValueError(
                    f"Locked Phase C main rule {index} has no valid feature"
                )
            column = f"phaseC_main__{feature}"
            if column not in design:
                design[column] = transformed[feature]
            continue
        left, right = _locked_rule_pair(row)
        if not left or not right:
            raise ValueError(f"Locked Phase C rule {index} has no valid pair")
        if left not in transformed or right not in transformed:
            raise ValueError(
                f"Locked Phase C rule {index} missing variables: {left!r}, {right!r}"
            )
        for feature in (left, right):
            column = f"phaseC_main__{feature}"
            if column not in design:
                design[column] = transformed[feature]
        design[f"phaseC_and_{index}"] = (
            (transformed[left] > 0) & (transformed[right] > 0)
        ).astype(float)
    for item in structure.get("phaseB_time_interactions") or []:
        feature = str(item.get("feature") or item.get("variable") or "")
        if re.fullmatch(r"gp\d+", feature):
            if feature not in design:
                raise ValueError(f"Locked Phase B GP feature {feature!r} is unavailable")
        elif feature:
            if feature not in transformed:
                raise ValueError(f"Locked Phase B feature {feature!r} is unavailable")
            column = f"phaseB_base__{feature}"
            if column not in design and f"main__{feature}" not in design:
                design[column] = transformed[feature]
    return design.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _locked_phaseb_time_stats(time_values: np.ndarray) -> dict[str, float]:
    t = np.maximum(np.asarray(time_values, dtype=float), 0.0)
    finite = t[np.isfinite(t)]
    if not len(finite):
        finite = np.asarray([0.0])
    logt = np.log1p(finite)
    q = float(np.quantile(finite, 0.99))
    tw = np.minimum(finite, q)
    return {
        "mean_logt": float(np.mean(logt)),
        "sd_logt": max(float(np.std(logt, ddof=0)), 1e-8),
        "t_winsor_q": q,
        "mean_t_winsor": float(np.mean(tw)),
        "sd_t_winsor": max(float(np.std(tw, ddof=0)), 1e-8),
    }


def _locked_phaseb_basis(
    name: str,
    time_values: np.ndarray,
    time_stats: dict[str, float] | None = None,
) -> np.ndarray:
    t = np.maximum(np.asarray(time_values, dtype=float), 0.0)
    stats = time_stats or _locked_phaseb_time_stats(t)
    zlog = (
        np.log1p(t) - float(stats["mean_logt"])
    ) / float(stats["sd_logt"])
    tw = np.minimum(t, float(stats["t_winsor_q"]))
    t_scaled = (
        tw - float(stats["mean_t_winsor"])
    ) / float(stats["sd_t_winsor"])
    if name == "zlog":
        return zlog
    if name == "zlog2":
        return zlog ** 2
    if name == "zlog3":
        return zlog ** 3
    if name == "t_scaled":
        return t_scaled
    if name == "t_scaled2":
        return t_scaled ** 2
    if name == "t_scaled3":
        return t_scaled ** 3
    if name.startswith("early"):
        cutoff = float(name.removeprefix("early"))
        return (t <= cutoff).astype(float)
    if name.startswith("late"):
        cutoff = float(name.removeprefix("late"))
        return (t > cutoff).astype(float)
    if name.startswith("hinge_after"):
        cutoff = float(name.removeprefix("hinge_after"))
        zcut = (
            np.log1p(cutoff) - float(stats["mean_logt"])
        ) / float(stats["sd_logt"])
        return np.maximum(zlog - zcut, 0.0)
    if name.startswith("exp_decay"):
        scale = float(name.removeprefix("exp_decay"))
        return np.exp(-t / max(scale, 1e-8))
    if name.startswith("inv_decay"):
        scale = float(name.removeprefix("inv_decay"))
        return 1.0 / (1.0 + t / max(scale, 1e-8))
    raise ValueError(f"Unsupported locked Phase B basis {name!r}")


def _locked_phaseb_feature_pool(
    transformed: pd.DataFrame, design: pd.DataFrame
) -> pd.DataFrame:
    pool = transformed.copy()
    for col in design:
        pool[col] = design[col]
        if col.startswith("main__"):
            pool[col.replace("main__", "", 1)] = design[col]
        if col.startswith("phaseB_base__"):
            pool[col.replace("phaseB_base__", "", 1)] = design[col]
    return pool


def _build_locked_phaseb_long_frame(
    feature_pool: pd.DataFrame,
    base_eta: np.ndarray,
    times: pd.Series,
    events: pd.Series,
    interactions: list[dict[str, Any]],
    time_stats: dict[str, float],
) -> pd.DataFrame:
    rows = []
    fixed_cuts = (3.0, 7.0, 14.0, 30.0)
    for position, (idx, raw_time) in enumerate(times.items()):
        stop_time = float(raw_time)
        starts = [0.0] + [cut for cut in fixed_cuts if cut < stop_time]
        stops = [cut for cut in fixed_cuts if cut < stop_time] + [stop_time]
        for start, stop in zip(starts, stops):
            row = {
                "id": position,
                "start": start,
                "stop": stop,
                "event": int(events.loc[idx]) if stop == stop_time else 0,
                "base_eta": float(base_eta[position]),
            }
            eval_time = np.asarray([(start + stop) / 2.0])
            for number, item in enumerate(interactions, start=1):
                feature = str(item.get("feature") or item.get("variable") or "")
                basis = str(item.get("basis") or "")
                if feature not in feature_pool:
                    raise RuntimeError(f"Locked Phase B feature {feature!r} is unavailable")
                row[f"phaseB_{number}"] = (
                    float(feature_pool.loc[idx, feature])
                    * float(_locked_phaseb_basis(basis, eval_time, time_stats)[0])
                )
            rows.append(row)
    return pd.DataFrame(rows)


def dry_run_validate_locked_structure_on_eicu(
    structure: dict[str, Any],
    eicu_df: pd.DataFrame,
    max_rows: int = 20,
) -> dict[str, Any]:
    """Validate locked reconstruction without Cox fitting or MIMIC access."""
    warnings = []
    errors = []
    sample = eicu_df.head(max_rows).copy()
    try:
        feature_audit = validate_locked_required_features(structure, eicu_df)
        required = feature_audit["required_raw_features"]
        categorical = [c for c in LOCKED_CATEGORICAL_COLUMNS if c in eicu_df]
        category_levels = {
            col: sorted(eicu_df[col].dropna().astype(str).unique().tolist())
            for col in categorical
        }
        sample_for_encoding = sample.copy()
        for col, levels in category_levels.items():
            sample_for_encoding[col] = pd.Categorical(
                sample_for_encoding[col].astype(str), categories=levels
            )
        transformed, _ = _fit_locked_preprocessor(sample_for_encoding, required)
        design = _build_locked_static_design(transformed, structure)
        phaseb_rows = structure.get("phaseB_time_interactions") or []
        if phaseb_rows:
            time_values = pd.to_numeric(
                sample.get("time", pd.Series(np.arange(len(sample)), index=sample.index)),
                errors="coerce",
            ).fillna(0).to_numpy(dtype=float)
            time_stats = _locked_phaseb_time_stats(time_values)
            for row in phaseb_rows:
                _locked_phaseb_basis(str(row.get("basis") or ""), time_values, time_stats)
        audit = {
            "dry_run_passed": True,
            "dry_run_n": int(len(sample)),
            "static_design_columns_n": int(design.shape[1]),
            "main_effect_n": len(structure.get("main_effect_vars") or []),
            "gp_expression_n": len(structure.get("phaseA_gp_expressions") or []),
            "phaseB_interaction_n": len(phaseb_rows),
            "phaseC_rule_n": len(structure.get("phaseC_and_rules") or []),
            "missing_feature_audit": feature_audit,
            "phaseB_time_scale_definition": "cell5_locked_refit_compatible",
            "errors": errors,
            "warnings": warnings,
        }
        return audit
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        raise ValueError(
            f"{structure.get('endpoint')}: locked structure dry-run failed: {exc}"
        ) from exc


def deterministic_refit_locked_structure_on_full_eicu(
    structure: dict[str, Any],
    eicu_df: pd.DataFrame,
    time_col: str = "time",
    event_col: str = "event",
) -> dict[str, Any]:
    """Fit preprocessing and Cox coefficients on full eICU with structure fixed."""
    from lifelines import CoxPHFitter, CoxTimeVaryingFitter

    snapshot = json.dumps(structure, sort_keys=True, default=_json_default)
    feature_audit = validate_locked_required_features(structure, eicu_df)
    required = feature_audit["required_raw_features"]
    transformed, prep = _fit_locked_preprocessor(eicu_df, required)
    design = _build_locked_static_design(transformed, structure)
    if design.shape[1] == 0:
        raise RuntimeError(f"{structure.get('endpoint')}: locked static design is empty")
    fit_df = design.copy()
    fit_df[time_col] = pd.to_numeric(eicu_df[time_col], errors="coerce")
    fit_df[event_col] = pd.to_numeric(eicu_df[event_col], errors="coerce").fillna(0).astype(int)
    fit_df = fit_df.dropna(subset=[time_col])
    hyper = structure["fixed_cox_hyperparameters"]
    model = CoxPHFitter(
        penalizer=float(hyper["penalizer"]),
        l1_ratio=float(hyper["l1_ratio"]),
    )
    model.fit(fit_df, duration_col=time_col, event_col=event_col)
    phaseb_model = None
    phaseb_time_stats = _locked_phaseb_time_stats(fit_df[time_col].to_numpy(dtype=float))
    interactions = structure.get("phaseB_time_interactions") or []
    if interactions:
        aligned_design = design.loc[fit_df.index]
        transformed_aligned = transformed.loc[fit_df.index]
        feature_pool = _locked_phaseb_feature_pool(
            transformed_aligned, aligned_design
        )
        base_eta = model.predict_log_partial_hazard(aligned_design).to_numpy(dtype=float)
        phaseb_long = _build_locked_phaseb_long_frame(
            feature_pool,
            base_eta,
            fit_df[time_col],
            fit_df[event_col],
            interactions,
            phaseb_time_stats,
        )
        phaseb_model = CoxTimeVaryingFitter(penalizer=float(hyper["penalizer"]))
        phaseb_model.fit(
            phaseb_long,
            id_col="id",
            start_col="start",
            stop_col="stop",
            event_col="event",
        )
    if snapshot != json.dumps(structure, sort_keys=True, default=_json_default):
        raise RuntimeError("Locked model structure changed during full-eICU refit")
    return {
        "structure": structure,
        "preprocessor": prep,
        "design_columns": list(design.columns),
        "cox_model": model,
        "phaseb_model": phaseb_model,
        "phaseb_interactions": interactions,
        "phaseb_time_stats": phaseb_time_stats,
        "missing_feature_audit": feature_audit,
        "time_col": time_col,
        "event_col": event_col,
        "fit_dataset": "full_eICU_only",
        "full_eicu_train_n": int(len(fit_df)),
        "full_eicu_train_events": int(fit_df[event_col].sum()),
        "feature_validation_train_df": eicu_df,
        "structure_immutable": True,
    }


def validate_locked_model_on_mimic(
    fitted: dict[str, Any],
    mimic_df: pd.DataFrame,
) -> dict[str, Any]:
    """Transform MIMIC and compute C-index; no fitting or selection is permitted."""
    from lifelines.utils import concordance_index

    structure = fitted["structure"]
    snapshot = json.dumps(structure, sort_keys=True, default=_json_default)
    feature_audit = validate_locked_required_features(
        structure,
        fitted["feature_validation_train_df"],
        mimic_df,
    )
    transformed, one_hot_zero = _transform_locked_preprocessor(
        mimic_df, fitted["preprocessor"]
    )
    feature_audit["one_hot_missing_category_filled_zero"] = sorted(set(one_hot_zero))
    design = _build_locked_static_design(transformed, structure)
    missing_design = [
        col for col in fitted["design_columns"] if col not in design.columns
    ]
    extra_design = [
        col for col in design.columns if col not in fitted["design_columns"]
    ]
    if missing_design or extra_design:
        raise ValueError(
            f"Locked MIMIC design mismatch; missing={missing_design}, extra={extra_design}"
        )
    design = design[fitted["design_columns"]]
    base_eta = fitted["cox_model"].predict_log_partial_hazard(design).to_numpy(dtype=float)
    time_col = fitted["time_col"]
    event_col = fitted["event_col"]
    risk_score = base_eta.copy()
    phaseb_model = fitted.get("phaseb_model")
    interactions = fitted.get("phaseb_interactions") or []
    if phaseb_model is not None and interactions:
        feature_pool = _locked_phaseb_feature_pool(transformed, design)
        times = pd.to_numeric(mimic_df[time_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        for number, item in enumerate(interactions, start=1):
            feature = str(item.get("feature") or item.get("variable") or "")
            basis = str(item.get("basis") or "")
            parameter = f"phaseB_{number}"
            if parameter not in phaseb_model.params_:
                raise ValueError(f"Missing fitted locked Phase B parameter {parameter}")
            coefficient = float(phaseb_model.params_[parameter])
            risk_score += (
                coefficient
                * feature_pool[feature].to_numpy(dtype=float)
                * _locked_phaseb_basis(
                    basis, times, fitted["phaseb_time_stats"]
                )
            )
    cindex = concordance_index(
        pd.to_numeric(mimic_df[time_col], errors="coerce"),
        -risk_score,
        pd.to_numeric(mimic_df[event_col], errors="coerce").fillna(0).astype(int),
    )
    if snapshot != json.dumps(structure, sort_keys=True, default=_json_default):
        raise RuntimeError("Locked model structure changed during MIMIC validation")
    return {
        "endpoint": structure.get("endpoint"),
        "target_name": structure.get("target_name") or structure.get("endpoint"),
        "target_id": structure.get("target_id"),
        "external_cindex": float(cindex),
        "ABC_external_cindex": float(cindex),
        "full_eicu_train_n": int(fitted["full_eicu_train_n"]),
        "full_eicu_train_events": int(fitted["full_eicu_train_events"]),
        "mimic_external_n": int(len(mimic_df)),
        "mimic_external_events": int(pd.to_numeric(mimic_df[event_col], errors="coerce").fillna(0).sum()),
        "static_design_columns_n": len(fitted["design_columns"]),
        "final_design_matrix_n_columns": len(fitted["design_columns"]),
        "final_design_matrix_columns": list(fitted["design_columns"]),
        "n_main_effects": len(structure.get("main_effect_vars") or []),
        "main_effect_names": list(structure.get("main_effect_vars") or []),
        "gp_expression_n": len(structure.get("phaseA_gp_expressions") or []),
        "n_gp_expressions": len(structure.get("phaseA_gp_expressions") or []),
        "gp_expression_names": [
            _locked_expression_text(row)
            for row in structure.get("phaseA_gp_expressions") or []
        ],
        "n_phaseA_transferred_features": int(
            structure.get("phaseA_transferred_feature_count", 0) or 0
        ),
        "phaseB_interaction_n": len(interactions),
        "phaseC_rule_n": len(structure.get("phaseC_and_rules") or []),
        "n_phaseC_rules": len(structure.get("phaseC_and_rules") or []),
        "strict_missing_feature_check_passed": bool(
            feature_audit["strict_missing_feature_check_passed"]
        ),
        "missing_feature_audit": feature_audit,
        "mimic_fit_performed": False,
        "mimic_selection_performed": False,
        "mimic_tuning_performed": False,
        "mimic_used_for_preprocessing_fit": False,
        "locked_structure_immutable": True,
        "status": "success",
        "warnings": [],
        "errors": [],
    }


def run_locked_structure_external_validation(
    manifest_path: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    """Cell 5: fixed-structure full-eICU refit followed by MIMIC validation."""
    manifest = load_locked_structure_manifest(manifest_path)
    output_dir = Path(output_root) / LOCKED_EXTERNAL_VALIDATION_DIRNAME
    by_endpoint_dir = output_dir / "by_endpoint"
    by_endpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in manifest["endpoints"]:
        target_id = str(item["target_id"])
        target_name = str(item["target_name"])
        try:
            structure = load_locked_endpoint_structure(
                Path(item["locked_structure_json"])
            )
            eicu_df = pd.read_excel(Path(item["train_file"]))
            fitted = deterministic_refit_locked_structure_on_full_eicu(
                structure, eicu_df
            )
            mimic_df = pd.read_excel(Path(item["external_file"]))
            result = validate_locked_model_on_mimic(fitted, mimic_df)
            result["locked_structure_json"] = item["locked_structure_json"]
        except Exception as exc:
            result = {
                "target_id": target_id,
                "target_name": target_name,
                "endpoint": target_name,
                "external_cindex": None,
                "ABC_external_cindex": None,
                "full_eicu_train_n": None,
                "full_eicu_train_events": None,
                "mimic_external_n": None,
                "mimic_external_events": None,
                "static_design_columns_n": None,
                "gp_expression_n": None,
                "phaseB_interaction_n": None,
                "phaseC_rule_n": None,
                "strict_missing_feature_check_passed": False,
                "missing_feature_audit": {
                    "required_raw_features": [],
                    "train_missing_raw_features": [],
                    "external_missing_raw_features": [],
                    "one_hot_columns_expected": [],
                    "one_hot_missing_category_filled_zero": [],
                    "strict_missing_feature_check_passed": False,
                },
                "status": "failed",
                "warnings": [],
                "errors": [f"{type(exc).__name__}: {exc}"],
                "mimic_fit_performed": False,
                "mimic_selection_performed": False,
                "mimic_tuning_performed": False,
                "mimic_used_for_preprocessing_fit": False,
                "locked_structure_immutable": True,
            }
        rows.append(result)
        (by_endpoint_dir / f"{target_id}_{target_name}_external_validation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        safe_to_excel(
            pd.DataFrame([result]),
            by_endpoint_dir / f"{target_id}_{target_name}_external_validation.xlsx",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    effective_rows = [
        {
            "endpoint": row.get("endpoint"),
            "n_main_effects": row.get("n_main_effects"),
            "main_effect_names": row.get("main_effect_names"),
            "n_gp_expressions": row.get("n_gp_expressions"),
            "gp_expression_names": row.get("gp_expression_names"),
            "n_phaseA_transferred_features": row.get(
                "n_phaseA_transferred_features"
            ),
            "n_phaseC_rules": row.get("n_phaseC_rules"),
            "final_design_matrix_n_columns": row.get(
                "final_design_matrix_n_columns"
            ),
            "final_design_matrix_columns": row.get(
                "final_design_matrix_columns"
            ),
            "external_cindex": row.get("external_cindex"),
        }
        for row in rows
    ]
    (output_dir / "effective_locked_structure_summary.json").write_text(
        json.dumps(
            effective_rows, ensure_ascii=False, indent=2, default=_json_default
        ),
        encoding="utf-8",
    )
    safe_to_excel(pd.DataFrame(rows), output_dir / "summary.xlsx")
    audit = {
        "manifest_path": str(Path(manifest_path).resolve()),
        "endpoint_count": len(rows),
        "locked_structure_loaded": True,
        "full_eicu_refit_only": True,
        "deterministic_refit_only": True,
        "model_selection_disabled": True,
        "structure_selection_disabled": True,
        "parameter_tuning_disabled": True,
        "mimic_transform_only": True,
        "mimic_fit_performed": False,
        "mimic_selection_performed": False,
        "mimic_tuning_performed": False,
        "mimic_used_for_preprocessing_fit": False,
        "external_used_for_selection": False,
        "locked_structure_immutable": True,
        "phaseB_time_scale_definition": "cell5_locked_refit_compatible",
        "all_endpoints_success": all(row.get("status") == "success" for row in rows),
    }
    (output_dir / "external_validation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failed = [row for row in rows if row.get("status") != "success"]
    if failed:
        raise ValueError(
            "Locked external validation failed: "
            + "; ".join(
                f"{row.get('target_name')}: {row.get('errors')}" for row in failed
            )
        )
    return rows


def discover_targets() -> list[dict[str, Any]]:
    """Discover the four fixed endpoint files in eICU -> MIMIC direction."""
    out = []
    for endpoint in ENDPOINTS:
        target_id = ENDPOINT_IDS[endpoint]
        train = WORKDIR / f"{target_id}_eicu.xlsx"
        external = WORKDIR / f"{target_id}_mimic.xlsx"
        if not train.exists():
            raise FileNotFoundError(train)
        if not external.exists():
            raise FileNotFoundError(external)
        out.append({"target_id": target_id, "target_name": endpoint, "train_file": train, "external_file": external})
    return out


def parse_result_metrics(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Parse metrics from the existing notebook result schema."""
    return legacy.parse_result_json(path)


def parse_result_full(path: Path) -> dict[str, Any]:
    """Load result JSON and normalize missing final_selected fields."""
    data = load_result_json(path)
    for key in ["phasec_final_selected_candidates", "phasec_stgp_and_final_selected_candidates"]:
        rows = data.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and ("selected" not in row or row.get("selected") is False):
                    row["final_selected"] = True
    return data


EXTERNAL_METRIC_FIELDS = [
    "A_external_cindex",
    "AB_external_cindex",
    "AC_external_cindex",
    "ABC_external_cindex",
    "external_A_static_cindex",
    "external_AB_timevarying_cindex",
    "external_AC_static_cindex",
    "external_ABC_timevarying_cindex",
    "external_abc_cindex",
]


def sanitize_internal_placeholder_external_metrics(result_json_path: Path) -> None:
    """
    Mark notebook external fields as internal-selection placeholders.

    During profile selection the wrapper deliberately points EXTERNAL_VALID_DATA_PATH
    to eICU, not MIMIC. The original notebook still writes external_* fields, so
    this function removes/marks them to prevent accidental interpretation as final
    MIMIC validation.
    """
    path = Path(result_json_path)
    if not path.exists():
        return
    data = load_result_json(path)
    placeholder = {}
    for key in EXTERNAL_METRIC_FIELDS:
        if key in data:
            placeholder[key] = data.get(key)
            data[key] = None
    for row_key in ["summary", "external_results", "phaseC_summary"]:
        rows = data.get(row_key)
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for key in list(row.keys()):
                    key_l = str(key).lower()
                    if "external" in key_l or key in {"A_static_cindex", "AB_timevarying_cindex", "AC_static_cindex", "ABC_final_cindex", "ABC_timevarying_cindex"}:
                        placeholder[f"{row_key}.{key}"] = row.get(key)
                        row[key] = None
    data["internal_selection_external_metrics_are_placeholder"] = True
    data["external_metric_source"] = "disabled_for_internal_selection; EXTERNAL_VALID_DATA_PATH used eICU placeholder, not MIMIC"
    data["placeholder_external_metrics_original_values"] = placeholder
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def extract_phasec_generation_scores(
    result_json_path: Path,
    ablation: str,
    profile_name: str,
    target_name: str,
    pass_name: str,
) -> list[dict[str, Any]]:
    """Extract per-generation Phase C best fitness rows from one endpoint result."""
    if not Path(result_json_path).exists():
        return []
    data = load_result_json(Path(result_json_path))
    rows = data.get("phasec_stgp_and_generation_log") or []
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append({
            "ablation": ablation,
            "profile_name": profile_name,
            "target_name": target_name,
            "pass_name": pass_name,
            "fold": row.get("fold"),
            "generation": row.get("generation"),
            "avgFit": row.get("avgFit"),
            "maxFit": row.get("maxFit"),
            "hofFit": row.get("hofFit"),
            "best": row.get("best"),
            "fitness_cache_size": row.get("fitness_cache_size"),
            "candidate_fit_calls": row.get("candidate_fit_calls"),
            "cache_hits": row.get("cache_hits"),
        })
    return out


def print_phasec_generation_scores(rows: list[dict[str, Any]], prefix: str) -> None:
    """Print compact Phase C per-generation fitness summaries."""
    if not rows:
        print(f"{prefix} PhaseC generation log: none", flush=True)
        return
    df = pd.DataFrame(rows)
    if not {"generation", "maxFit", "hofFit"}.issubset(df.columns):
        print(f"{prefix} PhaseC generation log: unavailable columns", flush=True)
        return
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
    df["maxFit"] = pd.to_numeric(df["maxFit"], errors="coerce")
    df["hofFit"] = pd.to_numeric(df["hofFit"], errors="coerce")
    summary = (
        df.groupby("generation", dropna=True)
        .agg(mean_maxFit=("maxFit", "mean"), best_maxFit=("maxFit", "max"), best_hofFit=("hofFit", "max"))
        .reset_index()
        .sort_values("generation")
    )
    for r in summary.to_dict(orient="records"):
        print(
            f"{prefix} [PhaseC Gen {int(r['generation']):03d}] "
            f"mean_maxFit={legacy.to_float(r['mean_maxFit']):.6f} "
            f"best_maxFit={legacy.to_float(r['best_maxFit']):.6f} "
            f"best_hofFit={legacy.to_float(r['best_hofFit']):.6f}",
            flush=True,
        )


def _pair_key(value: Any) -> tuple[str, str] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return tuple(sorted([str(value[0]).strip(), str(value[1]).strip()]))
    text = str(value or "").replace("AND::", "")
    if " AND " in text:
        a, b = text.split(" AND ", 1)
        return tuple(sorted([a.strip(), b.strip()]))
    return None


def _candidate_records_from_result(data: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    keys = [
        "phasec_stgp_and_final_selected_candidates",
        "phasec_final_selected_candidates",
        "phaseC_selected_candidates",
        "phasec_stgp_and_selected_candidates_by_fold",
        "phasec_selected_candidates_by_fold",
        "phasec_stgp_and_rule_universe_feasible",
        "phasec_stgp_and_candidate_pool",
        "phaseC_candidate_pool",
        "phasec_all_candidates",
        "phasec_evaluated_candidates",
    ]
    for key in keys:
        rows = data.get(key)
        if isinstance(rows, str):
            try:
                rows = json.loads(rows)
            except Exception:
                rows = [rows]
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                r = dict(row)
            else:
                r = {"candidate_name": str(row), "name": str(row)}
            r["_source_result_key"] = key
            records.append(r)
    return records


def phaseC_result_key_counts(data: dict[str, Any]) -> dict[str, int]:
    """Count raw Phase C candidate sources in a result JSON for seed-pass logging."""
    keys = [
        "phasec_stgp_and_final_selected_candidates",
        "phasec_final_selected_candidates",
        "phaseC_selected_candidates",
        "phasec_stgp_and_selected_candidates_by_fold",
        "phasec_selected_candidates_by_fold",
        "phasec_stgp_and_rule_universe_feasible",
        "phasec_stgp_and_candidate_pool",
        "phaseC_candidate_pool",
        "phasec_all_candidates",
        "phasec_evaluated_candidates",
    ]
    counts = {}
    for key in keys:
        rows = data.get(key)
        if isinstance(rows, str):
            try:
                rows = json.loads(rows)
            except Exception:
                rows = [rows] if rows else []
        counts[key] = len(rows) if isinstance(rows, list) else (1 if isinstance(rows, dict) else 0)
    counts["final_selected_total"] = sum(counts[k] for k in keys[:3])
    counts["fold_selected_total"] = sum(counts[k] for k in keys[3:5])
    counts["feasible_or_pool_total"] = sum(counts[k] for k in keys[5:])
    return counts


def update_phaseC_transfer_log_after_target_run(
    phaseC_log: list[dict[str, Any]],
    result_json_path: Path,
    target_endpoint: str,
    ablation: str | None = None,
    profile_name: str | None = None,
) -> list[dict[str, Any]]:
    """Backfill transferred Phase C rules with target-run evaluation/selection details."""
    if not Path(result_json_path).exists():
        for row in phaseC_log:
            if (
                row.get("target_endpoint") == target_endpoint
                and (ablation is None or row.get("ablation") == ablation)
                and (profile_name is None or row.get("profile_name") == profile_name)
            ):
                row["found_in_result_json"] = False
                row["reason"] = "result_json_missing"
        return phaseC_log
    data = load_result_json(Path(result_json_path))
    candidates = _candidate_records_from_result(data)
    indexed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cand in candidates:
        key = _pair_key(cand.get("interaction_pair") or cand.get("candidate_name") or cand.get("name") or cand.get("expr"))
        if key:
            indexed.setdefault(key, []).append(cand)
    for row in phaseC_log:
        if row.get("target_endpoint") != target_endpoint:
            continue
        if ablation is not None and row.get("ablation") != ablation:
            continue
        if profile_name is not None and row.get("profile_name") != profile_name:
            continue
        key = _pair_key(row.get("interaction_pair") or row.get("expr"))
        matches = indexed.get(key or (), [])
        if not matches:
            row.update({
                "found_in_result_json": False,
                "reason": "transferred_rule_not_found_after_target_run",
                "whether_selected": False,
                "final_selected": False,
            })
            continue
        def _priority(c: dict[str, Any]) -> tuple[int, float]:
            key_name = str(c.get("_source_result_key", ""))
            if "final_selected" in key_name or bool(c.get("final_selected", False)):
                pri = 0
            elif "selected_candidates_by_fold" in key_name or bool(c.get("selected", False)):
                pri = 1
            elif "feasible" in key_name or "candidate_pool" in key_name or "all_candidates" in key_name or "evaluated" in key_name:
                pri = 2
            else:
                pri = 3
            return pri, -legacy.to_float(c.get("train_loglik_improvement", c.get("fitness", 0.0)))

        cand = sorted(matches, key=_priority)[0]
        selected_key = str(cand.get("_source_result_key", ""))
        is_final = "final_selected" in selected_key or bool(cand.get("final_selected", False))
        is_fold_selected = "selected_candidates_by_fold" in selected_key or (bool(cand.get("selected", False)) and not is_final)
        selected_source = "final_selected" if is_final else ("selected_by_fold" if is_fold_selected else ("feasible_universe" if matches else "not_found"))
        coef = legacy.to_float(cand.get("coefficient", cand.get("coef", cand.get("target_coefficient"))))
        row.update({
            "found_in_result_json": True,
            "target_support_inter_n": cand.get("support_inter_n", cand.get("inter_n", cand.get("target_support_inter_n"))),
            "target_support_inter_events": cand.get("support_inter_events", cand.get("inter_events", cand.get("target_support_inter_events"))),
            "min_support_n_used": row.get("min_support_n_used"),
            "min_support_events_used": row.get("min_support_events_used"),
            "passed_support_gate": row.get("passed_support_gate"),
            "target_loglik_improvement": cand.get("train_loglik_improvement", cand.get("loglik_improvement", cand.get("target_loglik_improvement"))),
            "target_cv_cindex_delta": cand.get("cv_cindex_delta", cand.get("valid_cindex_delta", cand.get("target_cv_cindex_delta"))),
            "target_coefficient": coef,
            "target_hr": math.exp(coef) if math.isfinite(coef) else math.nan,
            "converged": cand.get("converged", cand.get("fit_ok", True)),
            "reason": cand.get("reason", "found_in_result_json"),
            "whether_selected": bool(is_final or is_fold_selected),
            "final_selected": bool(is_final or cand.get("final_selected", False)),
            "matched_result_key": selected_key,
            "selected_source": selected_source,
        })
    return phaseC_log


def sanitize_existing_internal_outputs(output_root: Path) -> int:
    """Sanitize already-created internal result JSON files under profiles/."""
    root = Path(output_root) / "profiles"
    if not root.exists():
        return 0
    n = 0
    for path in root.rglob("result.json"):
        sanitize_internal_placeholder_external_metrics(path)
        n += 1
    return n


def _nan_single_endpoint_reference(endpoint: str, source: str = "missing", warning: str | None = None) -> dict[str, Any]:
    warnings = [warning] if warning else []
    return {
        "single_endpoint_ABC_internal": math.nan,
        "single_endpoint_ABC_external": math.nan,
        "single_endpoint_reference_top10_cox_internal": math.nan,
        "single_endpoint_reference_top10_cox_external": math.nan,
        "single_endpoint_gp_input_vars": [],
        "single_endpoint_reference_source": source,
        "single_endpoint_reference_status": "missing" if source == "missing" else "failed",
        "parse_warnings": warnings,
        "target_name": endpoint,
    }


def _jsonish_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [x.strip().strip("'\"") for x in value.strip("[]").split(",") if x.strip()]
    return []


def _first_finite(values: list[Any]) -> float:
    for value in values:
        v = legacy.to_float(value)
        if math.isfinite(v):
            return v
    return math.nan


def _extract_gp_input_vars_from_workbook_sheets(sheets: dict[str, pd.DataFrame]) -> list[str]:
    for sheet_name in ["summary", "full_train_audit", "leakage_audit"]:
        df = sheets.get(sheet_name)
        if df is None or df.empty:
            continue
        for col in df.columns:
            if str(col).lower() in {"gp_input_vars", "phasea_gp_input_vars", "relation_aware_phasea_current_gp_input_vars"}:
                vals = [_jsonish_list(x) for x in df[col].dropna().tolist()]
                vals = [v for v in vals if v]
                if vals:
                    return vals[-1]
    return []


def parse_single_endpoint_result_workbook(path: Path, endpoint: str) -> dict[str, Any]:
    """External reporting only; never used in internal profile selection."""
    out = _nan_single_endpoint_reference(endpoint, str(path.resolve()))
    out["single_endpoint_reference_status"] = "loaded"
    out["workbook_path"] = str(path.resolve())
    warnings_out: list[str] = []
    if not path.exists():
        return _nan_single_endpoint_reference(endpoint, str(path.resolve()), "workbook_missing")
    try:
        xl = pd.ExcelFile(path)
        sheets = {name: pd.read_excel(path, sheet_name=name) for name in xl.sheet_names}
    except Exception as exc:
        return _nan_single_endpoint_reference(endpoint, str(path.resolve()), f"workbook_read_failed:{exc}")

    summary = sheets.get("summary", pd.DataFrame())
    external_results = sheets.get("external_results", pd.DataFrame())
    fold_results = sheets.get("fold_results", pd.DataFrame())

    abc_external_candidates: list[tuple[str, str, Any]] = []
    abc_internal_candidates: list[tuple[str, str, Any]] = []
    cox_external_candidates: list[tuple[str, str, Any]] = []
    cox_internal_candidates: list[tuple[str, str, Any]] = []

    if not summary.empty:
        row = summary.iloc[0]
        for col in ["external_abc_cindex", "ABC_external_cindex", "external_ABC_timevarying_cindex", "ABC_final_cindex", "ABC_timevarying_cindex"]:
            if col in summary.columns:
                abc_external_candidates.append(("summary", col, row.get(col)))
        for col in ["ABC_internal_cindex", "internal_abc_cindex", "abc_internal_cindex"]:
            if col in summary.columns:
                abc_internal_candidates.append(("summary", col, row.get(col)))
        for col in ["reference_top10_external_cindex", "external_reference_top10_cox_cindex"]:
            if col in summary.columns:
                cox_external_candidates.append(("summary", col, row.get(col)))
        for col in ["reference_top10_internal_cindex", "internal_reference_top10_cox_cindex"]:
            if col in summary.columns:
                cox_internal_candidates.append(("summary", col, row.get(col)))

    if not external_results.empty:
        cols_l = {str(c).lower(): c for c in external_results.columns}
        model_col = next((cols_l[k] for k in ["model_name", "model", "method", "stage", "name", "phase", "config"] if k in cols_l), None)
        ext_col = next((cols_l[k] for k in ["external_cindex", "cindex_external", "abc_external_cindex"] if k in cols_l), None)
        int_col = next((cols_l[k] for k in ["internal_cindex", "abc_internal_cindex", "train_cindex"] if k in cols_l), None)
        if model_col is not None:
            labels = external_results[model_col].astype(str).str.lower()
            if ext_col is not None:
                cox_rows = external_results[labels.str.contains("reference_top10_cox", regex=False, na=False)]
                for _, r in cox_rows.iterrows():
                    cox_external_candidates.append(("external_results", str(ext_col), r.get(ext_col)))
                abc_rows = external_results[
                    labels.str.contains("abc", regex=False, na=False)
                    & ~labels.str.contains("reference_top10_cox", regex=False, na=False)
                ]
                if len(abc_rows):
                    preferred = abc_rows[labels.loc[abc_rows.index].isin(["abc", "final_abc", "gpabc", "abc_timevarying"])]
                    abc_rows = preferred if len(preferred) else abc_rows
                    for _, r in abc_rows.iterrows():
                        abc_external_candidates.append(("external_results", str(ext_col), r.get(ext_col)))
            if int_col is not None:
                cox_rows = external_results[labels.str.contains("reference_top10_cox", regex=False, na=False)]
                for _, r in cox_rows.iterrows():
                    cox_internal_candidates.append(("external_results", str(int_col), r.get(int_col)))
                abc_rows = external_results[
                    labels.str.contains("abc", regex=False, na=False)
                    & ~labels.str.contains("reference_top10_cox", regex=False, na=False)
                ]
                for _, r in abc_rows.iterrows():
                    abc_internal_candidates.append(("external_results", str(int_col), r.get(int_col)))

    if not fold_results.empty:
        if "abc_final_cindex" in fold_results.columns:
            abc_internal_candidates.append(("fold_results", "abc_final_cindex_mean", pd.to_numeric(fold_results["abc_final_cindex"], errors="coerce").mean()))

    def choose(candidates: list[tuple[str, str, Any]]) -> tuple[float, str, str]:
        finite = [(s, c, legacy.to_float(v)) for s, c, v in candidates if math.isfinite(legacy.to_float(v))]
        if not finite:
            return math.nan, "", ""
        if len(finite) > 1:
            warnings_out.append(f"multiple_candidates:{finite[:4]}")
        s, c, v = finite[0]
        return v, s, c

    abc_external, sheet_abc, col_abc = choose(abc_external_candidates)
    abc_internal, sheet_abc_int, col_abc_int = choose(abc_internal_candidates)
    cox_external, sheet_cox, col_cox = choose(cox_external_candidates)
    cox_internal, sheet_cox_int, col_cox_int = choose(cox_internal_candidates)

    expected = {
        "death_hospice": 0.772009,
        "institutional_nonhome": 0.730875,
        "icu_los_gt7d": 0.612089,
        "later_icu_same_hadm": 0.616801,
        "home": 0.746667,
        "hosp_los_gt14d": 0.661496,
    }.get(endpoint)
    if expected is not None and math.isfinite(abc_external) and abs(abc_external - expected) > 1e-4:
        warnings_out.append(f"abc_external_differs_from_sanity_check:parsed={abc_external},expected={expected}")
        print(
            f"[MT-ABC][Single endpoint reference warning] parsed single ABC external differs from expected sanity check for {endpoint}",
            flush=True,
        )

    out.update({
        "single_endpoint_ABC_internal": abc_internal,
        "single_endpoint_ABC_external": abc_external,
        "single_endpoint_reference_top10_cox_internal": cox_internal,
        "single_endpoint_reference_top10_cox_external": cox_external,
        "single_endpoint_gp_input_vars": _extract_gp_input_vars_from_workbook_sheets(sheets),
        "single_endpoint_reference_source": str(path.resolve()),
        "single_endpoint_reference_status": "loaded" if math.isfinite(abc_external) or math.isfinite(cox_external) else "parse_failed",
        "parsed_sheet_for_abc": sheet_abc,
        "parsed_sheet_for_cox": sheet_cox,
        "parsed_column_for_abc_external": col_abc,
        "parsed_column_for_cox_external": col_cox,
        "parsed_column_for_abc_internal": col_abc_int,
        "parsed_column_for_cox_internal": col_cox_int,
        "parse_warnings": warnings_out,
    })
    return out


def write_single_endpoint_reference_summary(refs: dict[str, dict[str, Any]], output_root: Path) -> None:
    """External reporting only; never used in internal profile selection."""
    rows = []
    for endpoint in ENDPOINTS:
        r = refs.get(endpoint, _nan_single_endpoint_reference(endpoint))
        rows.append({
            "target_id": ENDPOINT_IDS[endpoint],
            "target_name": endpoint,
            "single_endpoint_ABC_internal": r.get("single_endpoint_ABC_internal"),
            "single_endpoint_ABC_external": r.get("single_endpoint_ABC_external"),
            "single_endpoint_reference_top10_cox_internal": r.get("single_endpoint_reference_top10_cox_internal"),
            "single_endpoint_reference_top10_cox_external": r.get("single_endpoint_reference_top10_cox_external"),
            "single_endpoint_gp_input_vars": json.dumps(r.get("single_endpoint_gp_input_vars", []), ensure_ascii=False),
            "single_endpoint_reference_source": r.get("single_endpoint_reference_source"),
            "single_endpoint_reference_status": r.get("single_endpoint_reference_status"),
            "parsed_sheet_for_abc": r.get("parsed_sheet_for_abc"),
            "parsed_sheet_for_cox": r.get("parsed_sheet_for_cox"),
            "parsed_column_for_abc_external": r.get("parsed_column_for_abc_external"),
            "parsed_column_for_cox_external": r.get("parsed_column_for_cox_external"),
            "parsed_column_for_abc_internal": r.get("parsed_column_for_abc_internal"),
            "parsed_column_for_cox_internal": r.get("parsed_column_for_cox_internal"),
            "parse_warnings": json.dumps(r.get("parse_warnings", []), ensure_ascii=False),
        })
    summary_dir = Path(output_root) / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    safe_to_excel(pd.DataFrame(rows), summary_dir / "single_endpoint_reference_from_workbooks.xlsx")
    (summary_dir / "single_endpoint_reference_from_workbooks.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def load_single_endpoint_reference_from_result_workbooks(
    reference_root: Path | None = None,
    output_root: Path = OUTPUT_ROOT,
    disabled: bool = False,
) -> dict[str, dict[str, Any]]:
    """External reporting only; never used in internal profile selection."""
    if disabled:
        print("[MT-ABC][Single endpoint workbook reference warning] disabled by CLI.", flush=True)
        refs = {
            ep: _nan_single_endpoint_reference(ep, "disabled", "disabled_by_cli")
            for ep in ENDPOINTS
        }
        write_single_endpoint_reference_summary(refs, output_root)
        return refs
    root = Path(reference_root) if reference_root else DEFAULT_SINGLE_ENDPOINT_REFERENCE_ROOT
    refs: dict[str, dict[str, Any]] = {}
    for endpoint in ENDPOINTS:
        target_id = ENDPOINT_IDS[endpoint]
        path = root / target_id / "result.xlsx"
        try:
            parsed = parse_single_endpoint_result_workbook(path, endpoint)
            refs[endpoint] = parsed
            if parsed.get("single_endpoint_reference_status") == "loaded":
                print(f"[MT-ABC][Single endpoint workbook reference] loaded {endpoint} from {path}", flush=True)
                print(
                    f"[MT-ABC][Single endpoint workbook reference] {endpoint} "
                    f"ABC_external={legacy.to_float(parsed.get('single_endpoint_ABC_external')):.6f} "
                    f"reference_top10_cox_external={legacy.to_float(parsed.get('single_endpoint_reference_top10_cox_external')):.6f}",
                    flush=True,
                )
            else:
                print(
                    f"[MT-ABC][Single endpoint workbook reference warning] {endpoint} failed: "
                    f"{parsed.get('parse_warnings')}",
                    flush=True,
                )
        except Exception as exc:
            refs[endpoint] = _nan_single_endpoint_reference(endpoint, str(path.resolve()), f"parse_exception:{exc}")
            print(f"[MT-ABC][Single endpoint workbook reference warning] {endpoint} failed: {exc}", flush=True)
    write_single_endpoint_reference_summary(refs, output_root)
    return refs


def load_single_endpoint_reference(
    profile_name: str | None = None,
    reference_root: Path | None = None,
    output_root: Path = OUTPUT_ROOT,
    disabled: bool = False,
) -> dict[str, dict[str, Any]]:
    """External reporting only; load references from final result workbooks."""
    _ = profile_name
    return load_single_endpoint_reference_from_result_workbooks(reference_root, output_root, disabled)


def write_final_external_benchmark_outputs(
    external_rows: list[dict[str, Any]],
    single_endpoint_refs: dict[str, dict[str, Any]],
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Write descriptive final external benchmark files; never used for selection."""
    by_endpoint = {str(r.get("target_name")): r for r in external_rows}
    benchmark_rows: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        row = by_endpoint.get(endpoint, {})
        ref = single_endpoint_refs.get(endpoint, _nan_single_endpoint_reference(endpoint))
        multitarget = legacy.to_float(row.get("ABC_external_cindex", row.get("multitarget_ABC_external")))
        single_abc = legacy.to_float(ref.get("single_endpoint_ABC_external"))
        ref_cox = legacy.to_float(ref.get("single_endpoint_reference_top10_cox_external"))
        is_event4 = endpoint == "later_icu_same_hadm"
        benchmark_rows.append({
            "target_id": ENDPOINT_IDS[endpoint],
            "target_name": endpoint,
            "is_event4": is_event4,
            "multitarget_ABC_external": multitarget,
            "single_endpoint_ABC_external": single_abc,
            "single_endpoint_reference_top10_cox_external": ref_cox,
            "delta_multitarget_vs_single_endpoint_ABC": multitarget - single_abc,
            "delta_multitarget_vs_reference_top10_cox": multitarget - ref_cox,
            "above_single_endpoint_ABC_external": bool(math.isfinite(multitarget) and math.isfinite(single_abc) and multitarget > single_abc),
            "above_reference_top10_cox_external": bool(math.isfinite(multitarget) and math.isfinite(ref_cox) and multitarget > ref_cox),
            "single_endpoint_reference_source": ref.get("single_endpoint_reference_source"),
            "single_endpoint_reference_status": ref.get("single_endpoint_reference_status"),
            "multitarget_result_json_path": row.get("result_json_path", row.get("final_external_result_json_path", "")),
            "status": row.get("status", "missing_multitarget_external"),
            "errors": row.get("error_message", ""),
        })

    valid_multi = [r for r in benchmark_rows if math.isfinite(legacy.to_float(r.get("multitarget_ABC_external")))]
    valid_single = [r for r in benchmark_rows if math.isfinite(legacy.to_float(r.get("single_endpoint_ABC_external")))]
    valid_cox = [r for r in benchmark_rows if math.isfinite(legacy.to_float(r.get("single_endpoint_reference_top10_cox_external")))]
    n_above_single = sum(bool(r.get("above_single_endpoint_ABC_external")) for r in benchmark_rows)
    n_above_cox = sum(bool(r.get("above_reference_top10_cox_external")) for r in benchmark_rows)
    event4 = next((r for r in benchmark_rows if r.get("target_name") == "later_icu_same_hadm"), {})

    def finite_delta_rows(key: str) -> list[dict[str, Any]]:
        return [r for r in benchmark_rows if math.isfinite(legacy.to_float(r.get(key)))]

    def mean_delta(key: str) -> float:
        vals = [legacy.to_float(r.get(key)) for r in finite_delta_rows(key)]
        return float(np.mean(vals)) if vals else math.nan

    def min_delta(key: str) -> float:
        vals = [legacy.to_float(r.get(key)) for r in finite_delta_rows(key)]
        return float(np.min(vals)) if vals else math.nan

    def max_delta(key: str) -> float:
        vals = [legacy.to_float(r.get(key)) for r in finite_delta_rows(key)]
        return float(np.max(vals)) if vals else math.nan

    def endpoint_for_extreme(key: str, choose_min: bool) -> str:
        rows = finite_delta_rows(key)
        if not rows:
            return ""
        fn = min if choose_min else max
        chosen = fn(rows, key=lambda r: legacy.to_float(r.get(key)))
        return str(chosen.get("target_name", ""))

    report_summary = {
        "n_endpoints_total": 6,
        "n_endpoints_with_valid_multitarget_external": len(valid_multi),
        "n_endpoints_with_valid_single_reference": len(valid_single),
        "n_endpoints_with_valid_reference_top10_cox": len(valid_cox),
        "n_above_single_endpoint_ABC_external": n_above_single,
        "n_above_reference_top10_cox_external": n_above_cox,
        "mean_delta_multitarget_vs_single_endpoint_ABC": mean_delta("delta_multitarget_vs_single_endpoint_ABC"),
        "min_delta_multitarget_vs_single_endpoint_ABC": min_delta("delta_multitarget_vs_single_endpoint_ABC"),
        "max_delta_multitarget_vs_single_endpoint_ABC": max_delta("delta_multitarget_vs_single_endpoint_ABC"),
        "worst_delta_vs_single_endpoint": endpoint_for_extreme("delta_multitarget_vs_single_endpoint_ABC", choose_min=True),
        "best_delta_vs_single_endpoint": endpoint_for_extreme("delta_multitarget_vs_single_endpoint_ABC", choose_min=False),
        "mean_delta_multitarget_vs_reference_top10_cox": mean_delta("delta_multitarget_vs_reference_top10_cox"),
        "min_delta_multitarget_vs_reference_top10_cox": min_delta("delta_multitarget_vs_reference_top10_cox"),
        "max_delta_multitarget_vs_reference_top10_cox": max_delta("delta_multitarget_vs_reference_top10_cox"),
        "worst_delta_vs_reference_top10_cox_endpoint": endpoint_for_extreme("delta_multitarget_vs_reference_top10_cox", choose_min=True),
        "best_delta_vs_reference_top10_cox_endpoint": endpoint_for_extreme("delta_multitarget_vs_reference_top10_cox", choose_min=False),
        "event4_multitarget_ABC_external": event4.get("multitarget_ABC_external"),
        "event4_single_endpoint_ABC_external": event4.get("single_endpoint_ABC_external"),
        "event4_reference_top10_cox_external": event4.get("single_endpoint_reference_top10_cox_external"),
        "event4_delta_vs_single_endpoint_ABC": event4.get("delta_multitarget_vs_single_endpoint_ABC"),
        "event4_delta_vs_reference_top10_cox": event4.get("delta_multitarget_vs_reference_top10_cox"),
        "external_metric_source_note": "These benchmark indicators are descriptive reporting metrics computed only after final external validation. They are not used for internal model/profile/parameter selection.",
    }
    summary_dir = Path(output_root) / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    safe_to_excel(pd.DataFrame(benchmark_rows), summary_dir / "final_external_benchmark_by_endpoint.xlsx")
    (summary_dir / "final_external_benchmark_by_endpoint.json").write_text(
        json.dumps(benchmark_rows, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    safe_to_excel(pd.DataFrame([report_summary]), summary_dir / "final_external_performance_summary.xlsx")
    (summary_dir / "final_external_performance_summary.json").write_text(
        json.dumps(report_summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"[MT-ABC][Final external performance] n_above_single={n_above_single}/6", flush=True)
    print(f"[MT-ABC][Final external performance] n_above_reference_top10_cox={n_above_cox}/6", flush=True)
    print(
        f"[MT-ABC][Final external performance] mean_delta_vs_single="
        f"{legacy.to_float(report_summary.get('mean_delta_multitarget_vs_single_endpoint_ABC')):.4f}",
        flush=True,
    )
    print(
        f"[MT-ABC][Final external performance] min_delta_vs_single="
        f"{legacy.to_float(report_summary.get('min_delta_multitarget_vs_single_endpoint_ABC')):.4f}",
        flush=True,
    )
    print(
        f"[MT-ABC][Final external performance] event4_delta_vs_single="
        f"{legacy.to_float(report_summary.get('event4_delta_vs_single_endpoint_ABC')):.4f}",
        flush=True,
    )
    print(
        f"[MT-ABC][Final external performance] event4_delta_vs_reference_top10_cox="
        f"{legacy.to_float(report_summary.get('event4_delta_vs_reference_top10_cox')):.4f}",
        flush=True,
    )
    print("[MT-ABC][Final external performance] reporting_only=True", flush=True)
    return benchmark_rows, report_summary


def complexity_from_row(row: dict[str, Any]) -> float:
    """Compute a simple linear complexity penalty from parsed endpoint outputs."""
    result_path = Path(row.get("result_json_path") or "")
    n_trees = 0
    tree_height_sum = 0
    if result_path.exists():
        data = load_result_json(result_path)
        for r in data.get("final_gp_expressions", []) or []:
            if isinstance(r, dict):
                n_trees += 1
                tree_height_sum += int(legacy.to_float(r.get("tree_height")) if math.isfinite(legacy.to_float(r.get("tree_height"))) else 0)
    phaseb_n = int(legacy.to_float(row.get("phaseB_candidate_n")) if math.isfinite(legacy.to_float(row.get("phaseB_candidate_n"))) else 0)
    phasec_n = int(legacy.to_float(row.get("phaseC_candidate_n")) if math.isfinite(legacy.to_float(row.get("phaseC_candidate_n"))) else 0)
    return float(0.2 * n_trees + 0.05 * tree_height_sum + 0.2 * phaseb_n + 0.2 * phasec_n)


def phaseA_injection_info(result_json_path: Path) -> dict[str, Any]:
    """Read Phase A transferred-tree injection counters from target result JSON."""
    if not Path(result_json_path).exists():
        return {"phaseA_attempted_n": math.nan, "phaseA_injected_n": math.nan, "phaseA_parse_failed_n": math.nan}
    data = load_result_json(Path(result_json_path))
    final_exprs = []
    for row in data.get("final_gp_expressions", []) or []:
        if isinstance(row, dict) and row.get("expression"):
            final_exprs.append({"expression": str(row.get("expression")), "coefficient": row.get("coefficient")})
    return {
        "phaseA_attempted_n": data.get("relation_aware_phaseA_attempted_n", 0),
        "phaseA_injected_n": data.get("relation_aware_phaseA_injected_n", 0),
        "phaseA_parse_failed_n": data.get("relation_aware_phaseA_parse_failed_n", 0),
        "phaseA_parse_failed_items": data.get("relation_aware_phaseA_parse_failed_items", []),
        "phaseA_injected_items": data.get("relation_aware_phaseA_injected_items", []),
        "phaseA_current_gp_input_vars": data.get("relation_aware_phaseA_current_gp_input_vars", []),
        "phaseA_external_tree_source_json": data.get("relation_aware_phaseA_external_tree_source_json", data.get("phaseA_external_trees_json", "")),
        "phaseA_precheck_passed_n": data.get("relation_aware_phaseA_precheck_passed_n", 0),
        "phaseA_precheck_failed_n": data.get("relation_aware_phaseA_precheck_failed_n", 0),
        "phaseA_final_selected_items": data.get("relation_aware_phaseA_final_selected_items", []),
        "phaseA_final_selected_n": data.get("relation_aware_phaseA_final_selected_n", 0),
        "phaseA_injected_but_filtered_out_n": data.get("relation_aware_phaseA_injected_but_filtered_out_n", 0),
        "final_gp_expressions": final_exprs,
    }


def phaseC_injection_info(result_json_path: Path) -> dict[str, Any]:
    """Read Phase C transferred-rule injection counters from target result JSON."""
    if not Path(result_json_path).exists():
        return {"phaseC_injected_rule_n": math.nan, "phaseC_injection_failed_n": math.nan}
    data = load_result_json(Path(result_json_path))
    return {
        "phaseC_injected_rule_n": data.get("relation_aware_phaseC_injected_n", 0),
        "phaseC_injection_failed_n": data.get("relation_aware_phaseC_injection_failed_n", 0),
        "phaseC_injected_items": data.get("relation_aware_phaseC_injected_items", []),
        "phaseC_injection_failed_items": data.get("relation_aware_phaseC_injection_failed_items", []),
    }


def _norm_expr(expr: Any) -> str:
    return re.sub(r"\s+", "", str(expr or ""))


def match_transferred_trees_to_final_gp(transferred_trees: list[dict[str, Any]], final_gp_expressions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match transferred expressions against final Cox-filtered GP expressions."""
    final = {_norm_expr(row.get("expression")): (i + 1, row.get("coefficient")) for i, row in enumerate(final_gp_expressions or [])}
    out = []
    for row in transferred_trees:
        candidates = [
            row.get("notebook_named_expression"),
            row.get("named_expression"),
            row.get("notebook_mapped_expression"),
            row.get("mapped_expression"),
            row.get("expression"),
        ]
        rank_coef = None
        for expr in candidates:
            rank_coef = final.get(_norm_expr(expr))
            if rank_coef is not None:
                break
        out.append({
            **row,
            "final_selected_in_cox": rank_coef is not None,
            "survived_filtering": rank_coef is not None,
            "final_gp_rank": rank_coef[0] if rank_coef else None,
            "final_cox_coefficient": rank_coef[1] if rank_coef else None,
        })
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _empirical_percentile(values: np.ndarray) -> np.ndarray:
    """Map one endpoint's OOF score to a comparable [0, 1] rank scale."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise RuntimeError("exclusive-risk vector must be a finite one-dimensional array")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # Average tied ranks.  This avoids arbitrary conflict changes when many
    # patients have the same risk score.
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return (ranks + 0.5) / float(len(values))


def exclusive_high_risk_overlap(
    aligned_risks: list[np.ndarray],
    high_quantile: float = DEFAULT_INTERNAL_EXCLUSIVE_HIGH_QUANTILE,
) -> dict[str, float]:
    """Patient-level overlap of high relative risks for exclusive outcomes.

    Cox linear predictors from separate endpoints have arbitrary scales.  We
    therefore compare within-endpoint OOF percentiles.  A patient is rewarded
    implicitly when only one endpoint is high and is penalised when two or
    more mutually-exclusive endpoints are simultaneously high.
    """
    if len(aligned_risks) < 2:
        raise RuntimeError("exclusive-risk overlap requires at least two endpoints")
    q = float(high_quantile)
    if not (0.5 < q < 1.0):
        raise ValueError("exclusive high-risk quantile must be between 0.5 and 1")
    percentiles = np.column_stack([_empirical_percentile(x) for x in aligned_risks])
    # Soft excess above the high-risk threshold.  This is zero for low/medium
    # risk, increases smoothly inside the top-risk region, and cannot be made
    # smaller by changing an unrelated coefficient after fitting.
    high = np.clip((percentiles - q) / (1.0 - q), 0.0, 1.0)
    pairwise = 0.0
    for left in range(high.shape[1]):
        for right in range(left + 1, high.shape[1]):
            pairwise += float(np.mean(high[:, left] * high[:, right]))
    hard_count = (percentiles >= q).sum(axis=1)
    hard_overlap = float(np.mean(hard_count >= 2))
    exclusive_single_high = float(np.mean(hard_count == 1))
    return {
        "exclusive_conflict_penalty": float(pairwise),
        "pairwise_soft_high_overlap": float(pairwise),
        "hard_multi_high_rate": hard_overlap,
        "exclusive_single_high_rate": exclusive_single_high,
        "high_quantile": q,
    }


def exclusive_conflict_penalty(target_rows: dict[str, dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    """Compute the soft terminal-outcome conflict penalty on aligned OOF rows.

    The endpoint models are separate Cox models, so their linear predictors are
    standardized before the soft overlap is calculated.  This is a selection
    penalty only; it does not alter any patient's fitted risk score.
    """
    rows = []
    penalties = []
    for group in EXCLUSIVE_GROUPS:
        endpoint_payloads: dict[str, tuple[list[str], np.ndarray]] = {}
        problems = []
        for endpoint in group:
            path = Path(target_rows.get(endpoint, {}).get("result_json_path") or "")
            if not path.exists():
                problems.append(f"{endpoint}:missing_result_json")
                continue
            sidecar = path.with_name("oof_reference_risk.json")
            data = (
                json.loads(sidecar.read_text(encoding="utf-8"))
                if sidecar.exists()
                else load_result_json(path)
            )
            arr = data.get("oof_abc_risk")
            keys = data.get("oof_row_keys")
            if not isinstance(arr, list) or not arr:
                problems.append(f"{endpoint}:missing_oof_abc_risk")
                continue
            if not isinstance(keys, list) or not keys:
                problems.append(f"{endpoint}:missing_oof_row_keys")
                continue
            risk = np.asarray(arr, dtype=float)
            row_keys = [str(x) for x in keys]
            if len(risk) != len(row_keys):
                problems.append(f"{endpoint}:risk_key_length_mismatch")
                continue
            if len(set(row_keys)) != len(row_keys):
                problems.append(f"{endpoint}:duplicate_oof_row_keys")
                continue
            if not np.isfinite(risk).all():
                problems.append(f"{endpoint}:nonfinite_oof_abc_risk")
                continue
            endpoint_payloads[endpoint] = (row_keys, risk)
        if problems or len(endpoint_payloads) != len(group):
            raise RuntimeError(
                "exclusive-conflict penalty cannot be evaluated: " + ";".join(problems)
            )

        reference_keys = endpoint_payloads[group[0]][0]
        reference_set = set(reference_keys)
        aligned_risks = []
        for endpoint in group:
            keys, risk = endpoint_payloads[endpoint]
            if set(keys) != reference_set:
                raise RuntimeError(
                    f"exclusive-conflict OOF key set mismatch for {endpoint}"
                )
            risk_by_key = dict(zip(keys, risk))
            aligned = np.asarray([risk_by_key[key] for key in reference_keys], dtype=float)
            sd = float(np.std(aligned, ddof=0))
            if not np.isfinite(sd) or sd <= 1e-12:
                raise RuntimeError(
                    f"exclusive-conflict OOF risk is constant for {endpoint}"
                )
            aligned_risks.append(aligned)
        overlap = exclusive_high_risk_overlap(aligned_risks)
        total = float(overlap["exclusive_conflict_penalty"])
        rows.append({
            "exclusive_group": json.dumps(group),
            "risk_source": "OOF locked-model linear predictor at ICU day 1",
            "n_aligned_rows": int(len(reference_keys)),
            "pairwise_penalty": float(overlap["pairwise_soft_high_overlap"]),
            "sum_penalty": 0.0,
            "exclusive_conflict_penalty": total,
            "hard_multi_high_rate": float(overlap["hard_multi_high_rate"]),
            "exclusive_single_high_rate": float(overlap["exclusive_single_high_rate"]),
            "high_quantile": float(overlap["high_quantile"]),
            "penalty_definition": "within-endpoint OOF percentile soft overlap above the fixed high-risk quantile",
            "warning": "",
        })
        penalties.append(total)
    return float(np.sum(penalties)), rows


def select_joint_exclusive_phasec_k(
    target_rows: dict[str, dict[str, Any]],
    fixed_weight: float = DEFAULT_INTERNAL_EXCLUSIVE_WEIGHT,
    high_quantile: float = DEFAULT_INTERNAL_EXCLUSIVE_HIGH_QUANTILE,
) -> dict[str, Any]:
    """Jointly select Phase-C K values using eICU OOF performance and overlap.

    The endpoint notebook has already fitted every K candidate.  This function
    enumerates the eligible terminal-endpoint combinations, rewards their mean
    internal OOF discrimination, and subtracts a fixed patient-level high-risk
    overlap penalty.  The selected K values must subsequently be replayed and
    locked before external validation.
    """
    if fixed_weight <= 0 or not math.isfinite(float(fixed_weight)):
        raise ValueError("internal exclusive selection weight must be fixed and positive")
    if len(EXCLUSIVE_GROUPS) != 1:
        raise RuntimeError("joint Phase-C selector currently expects exactly one exclusive group")
    group = list(EXCLUSIVE_GROUPS[0])
    payloads: dict[str, dict[str, Any]] = {}
    local_selected: dict[str, int] = {}
    reference_keys: list[str] | None = None
    for endpoint in group:
        result_path = Path(target_rows.get(endpoint, {}).get("result_json_path") or "")
        if not result_path.exists():
            raise RuntimeError(f"missing result JSON for exclusive endpoint {endpoint}")
        sidecar = result_path.with_name("oof_reference_risk.json")
        data = load_result_json(sidecar if sidecar.exists() else result_path)
        keys = [str(x) for x in data.get("oof_row_keys", [])]
        risk_by_k = data.get("oof_abc_risk_by_k", {})
        result_data = load_result_json(result_path)
        local_selected[endpoint] = int(
            result_data.get("phasec_selected_k", result_data.get("selected_k_cat", 0))
        )
        k_rows = result_data.get("phasec_abc_unified_k_summary", [])
        if not keys or not isinstance(risk_by_k, dict) or not risk_by_k or not k_rows:
            raise RuntimeError(f"missing all-K OOF candidates for exclusive endpoint {endpoint}")
        if len(set(keys)) != len(keys):
            raise RuntimeError(f"duplicate OOF row keys for exclusive endpoint {endpoint}")
        if reference_keys is None:
            reference_keys = list(keys)
        elif set(keys) != set(reference_keys):
            raise RuntimeError(f"OOF patient set mismatch for exclusive endpoint {endpoint}")
        key_index = {key: i for i, key in enumerate(keys)}
        candidates: dict[int, dict[str, Any]] = {}
        for row in k_rows:
            k = int(row.get("K_interaction"))
            risk = np.asarray(risk_by_k.get(str(k), risk_by_k.get(k, [])), dtype=float)
            if len(risk) != len(keys) or not np.isfinite(risk).all():
                raise RuntimeError(f"incomplete K={k} OOF risk for exclusive endpoint {endpoint}")
            if reference_keys != keys:
                risk = np.asarray([risk[key_index[key]] for key in reference_keys], dtype=float)
            cindex = legacy.to_float(row.get("mean_abc"))
            eligible = bool(k == 0 or int(row.get("eligible", 0) or 0) == 1)
            if math.isfinite(cindex) and eligible:
                candidates[k] = {"risk": risk, "cindex": float(cindex)}
        if 0 not in candidates:
            raise RuntimeError(f"K=0 fallback is unavailable for exclusive endpoint {endpoint}")
        payloads[endpoint] = candidates

    rows = []
    for combo in itertools.product(*(sorted(payloads[endpoint]) for endpoint in group)):
        risks = [payloads[endpoint][k]["risk"] for endpoint, k in zip(group, combo)]
        overlap = exclusive_high_risk_overlap(risks, high_quantile=high_quantile)
        cindices = [payloads[endpoint][k]["cindex"] for endpoint, k in zip(group, combo)]
        mean_cindex = float(np.mean(cindices))
        score = mean_cindex - float(fixed_weight) * float(overlap["exclusive_conflict_penalty"])
        rows.append({
            **{f"{endpoint}_k": int(k) for endpoint, k in zip(group, combo)},
            "terminal_internal_macro_cindex": mean_cindex,
            "exclusive_conflict_penalty": float(overlap["exclusive_conflict_penalty"]),
            "hard_multi_high_rate": float(overlap["hard_multi_high_rate"]),
            "exclusive_single_high_rate": float(overlap["exclusive_single_high_rate"]),
            "fixed_exclusive_weight": float(fixed_weight),
            "joint_internal_score": float(score),
        })
    rows.sort(
        key=lambda row: (
            -row["joint_internal_score"],
            row["exclusive_conflict_penalty"],
            sum(row[f"{endpoint}_k"] for endpoint in group),
            tuple(row[f"{endpoint}_k"] for endpoint in group),
        )
    )
    best = dict(rows[0])
    local_row = next(
        (
            dict(row)
            for row in rows
            if all(int(row[f"{endpoint}_k"]) == local_selected[endpoint] for endpoint in group)
        ),
        None,
    )
    selected = {endpoint: int(best[f"{endpoint}_k"]) for endpoint in group}
    # The non-terminal endpoint is not part of the constraint and keeps its
    # locally selected K on the replay pass.
    for endpoint in ENDPOINTS:
        if endpoint not in selected:
            result_path = Path(target_rows[endpoint]["result_json_path"])
            result = load_result_json(result_path)
            selected[endpoint] = int(result.get("phasec_selected_k", result.get("selected_k_cat", 0)))
    return {
        "selected_k_by_endpoint": selected,
        "local_selected_k_by_endpoint": local_selected,
        "exclusive_group": group,
        "fixed_weight": float(fixed_weight),
        "high_quantile": float(high_quantile),
        "selected_combination": best,
        "local_selection_combination": local_row,
        "combination_rows": rows,
        "external_used_for_selection": False,
        "selection_data": "eICU internal OOF only",
    }


def _patch_reference_oof_risk(nb: dict[str, Any]) -> None:
    """Patch generated notebooks to export a common-time final ABC OOF score."""
    phaseb_patched = unified_patched = cv_patched = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        text = legacy.source_text(cell.get("source", ""))
        original_text = text
        if (
            "def run_phaseb_timebasis_pool_for_train_valid(" in text
            and '"reference_linear_predictor": _ra_fixed_reference_linear_predictor(' not in text
        ):
            needle = '        "coefficients": coefs,\n'
            replacement = (
                needle
                + '        "reference_time": 1.0,\n'
                + '        "reference_linear_predictor": _ra_fixed_reference_linear_predictor(\n'
                + '            eta_va, feature_va, selected, coefs, time_stats, None, 1.0\n'
                + '        ),  # relation_aware_reference_linear_predictor\n'
            )
            if text.count(needle) != 1:
                raise RuntimeError("unable to patch Phase-B reference predictor")
            text = text.replace(needle, replacement, 1)
            phaseb_patched += 1
        if (
            "def fit_eval_unified_abc_timevarying(" in text
        ):
            needle = '        "coefficients": coefficients,\n'
            replacement = (
                needle
                + '        "reference_time": 1.0,\n'
                + '        "reference_linear_predictor": _ra_fixed_reference_linear_predictor(\n'
                + '            eta_A_apply, feature_apply, selected_phaseB_candidates, coefficients,\n'
                + '            time_stats, phasec_apply, 1.0\n'
                + '        ),  # relation_aware_reference_linear_predictor\n'
            )
            if text.count(needle) != 1:
                raise RuntimeError("unable to patch unified ABC reference predictor")
            text = text.replace(needle, replacement, 1)
            unified_patched += 1
        if (
            "def run_internal_fixed_cv_select_phasec(" in text
            and "oof_abc_risk_by_k" in text
            and "unified_ABC" in text
            and "relation_aware_oof_assignment" not in text
        ):
            needle = "            dlt = c_abc - c_ab if np.isfinite(c_abc) and np.isfinite(c_ab) else np.nan\n"
            replacement = '''            # relation_aware_oof_assignment: one comparable score per validation row.
            if fallback_mode == "unified_ABC":
                risk_abc_va = unified.get("reference_linear_predictor")
            elif fallback_mode == "equals_AB":
                risk_abc_va = tb.get("reference_linear_predictor")
            elif fallback_mode == "equals_AC":
                risk_abc_va = _ra_log_positive_hazard(risk_ac_va)
            else:
                risk_abc_va = _ra_log_positive_hazard(fc.get("risk_maingp_va", fc.get("risk_ab_va")))
            risk_abc_va = np.asarray(risk_abc_va, dtype=float)
            if len(risk_abc_va) != len(fc["va_idx"]) or not np.isfinite(risk_abc_va).all():
                raise RuntimeError(f"invalid final ABC OOF reference risk: fold={fold}, K={k}, mode={fallback_mode}")
            oof_abc_risk_by_k[int(k)][fc["va_idx"]] = risk_abc_va
            dlt = c_abc - c_ab if np.isfinite(c_abc) and np.isfinite(c_ab) else np.nan
'''
            if text.count(needle) != 1:
                raise RuntimeError("unable to patch final ABC OOF fold assignment")
            text = text.replace(needle, replacement, 1)
            old = '''    oof_abc_risk = np.asarray(oof_abc_risk_by_k.get(int(selected_k), oof_ab_risk), dtype=float)
    if int(selected_k) == 0:
        oof_abc_risk = np.asarray(oof_ab_risk, dtype=float).copy()
'''
            new = '''    oof_abc_risk = np.asarray(oof_abc_risk_by_k.get(int(selected_k), oof_ab_risk), dtype=float)
    if len(oof_abc_risk) != len(df_train_raw) or not np.isfinite(oof_abc_risk).all():
        raise RuntimeError("selected final ABC OOF reference risk is incomplete")
'''
            if text.count(old) != 1:
                raise RuntimeError("unable to patch selected ABC OOF validation")
            text = text.replace(old, new, 1)
            # A first internal pass exports every K candidate.  A joint
            # terminal-outcome selector can then choose one K per endpoint and
            # a deterministic second pass locks those choices before MIMIC is
            # read.  This makes the exclusivity rule affect retained structure.
            force_lines = '''    _ra_force_k = globals().get("FORCE_PHASEC_SELECTED_K", None)
    if _ra_force_k is not None:
        _ra_force_k = int(_ra_force_k)
        if _ra_force_k not in [int(x) for x in k_grid]:
            raise RuntimeError(f"FORCE_PHASEC_SELECTED_K={_ra_force_k} is outside k_grid={list(k_grid)}")
        _ra_force_risk = np.asarray(oof_abc_risk_by_k.get(_ra_force_k, []), dtype=float)
        if len(_ra_force_risk) != len(df_train_raw) or not np.isfinite(_ra_force_risk).all():
            raise RuntimeError(f"forced Phase-C K={_ra_force_k} has incomplete OOF risk")
        selected_k = _ra_force_k
'''
            force_needles = [
                '    df_k_summary["selected"] = (df_k_summary["K_interaction"] == selected_k).astype(int)\n',
                '    df_k_summary["selected"] = (df_k_summary["K_interaction"] == int(selected_k)).astype(int)\n',
            ]
            force_patched = 0
            for force_needle in force_needles:
                if force_needle in text:
                    text = text.replace(force_needle, force_lines + force_needle, 1)
                    force_patched += 1
            if force_patched != 2:
                raise RuntimeError(f"unable to patch both Phase-C forced-K selectors: {force_patched}")
            return_needle = '    return {\n        "selected_k": int(selected_k),'
            return_replacement = '''    _ra_key_cols = [c for c in ["subject_id", "hadm_id", "stay_id"] if c in df_train_raw.columns]
    if not _ra_key_cols:
        raise RuntimeError("no stable patient/stay identifiers available for OOF alignment")
    _ra_oof_row_keys = df_train_raw[_ra_key_cols].astype(str).agg("|".join, axis=1).tolist()
    if len(set(_ra_oof_row_keys)) != len(_ra_oof_row_keys):
        raise RuntimeError("OOF alignment identifiers are not unique")

    return {
        "selected_k": int(selected_k),'''
            return_search_start = text.find("relation_aware_oof_assignment")
            return_position = text.find(return_needle, return_search_start)
            if return_search_start < 0 or return_position < 0:
                raise RuntimeError("unable to patch OOF row-key export")
            text = (
                text[:return_position]
                + return_replacement
                + text[return_position + len(return_needle):]
            )
            risk_needle = '        "oof_main_gp_risk": np.asarray(oof_main_gp_risk, dtype=float), "oof_ab_risk": np.asarray(oof_ab_risk, dtype=float), "oof_abc_risk": np.asarray(oof_abc_risk, dtype=float),\n'
            risk_replacement = (
                risk_needle
                + '        "oof_abc_risk_by_k": {str(int(_k)): np.asarray(_v, dtype=float) for _k, _v in oof_abc_risk_by_k.items()},\n'
                + '        "oof_row_keys": list(_ra_oof_row_keys),\n'
            )
            if text.count(risk_needle) != 1:
                raise RuntimeError("unable to add OOF row keys to CV result")
            text = text.replace(risk_needle, risk_replacement, 1)
            cv_patched += 1
        if text != original_text:
            cell["source"] = legacy.source_like(text, cell.get("source", ""))
    if (phaseb_patched, unified_patched, cv_patched) != (1, 1, 1):
        raise RuntimeError(
            "reference OOF patch count mismatch: "
            f"phaseB={phaseb_patched}, unified={unified_patched}, CV={cv_patched}"
        )


def relation_patch_notebook() -> None:
    """Monkeypatch legacy notebook patching with relation-aware Phase C and OOF-risk hooks."""
    if getattr(legacy, "_relation_aware_patch_installed", False):
        return
    original = legacy.patch_notebook_for_profile

    def patched(input_ipynb, output_ipynb, train_path, external_path, output_dir, profile_dict, seed):
        info = original(input_ipynb, output_ipynb, train_path, external_path, output_dir, profile_dict, seed)
        nb = legacy.read_notebook(output_ipynb)
        override = [
            "# Auto-injected relation-aware multitarget hooks\n",
            f"PHASEA_EXTERNAL_TREES_JSON = {legacy.py_literal(profile_dict.get('PHASEA_EXTERNAL_TREES_JSON', ''))}\n",
            f"PHASEC_EXTERNAL_RULES_JSON = {legacy.py_literal(profile_dict.get('PHASEC_EXTERNAL_RULES_JSON', ''))}\n",
            f"RELATION_AWARE_ABLATION = {legacy.py_literal(profile_dict.get('RELATION_AWARE_ABLATION', 'M0'))}\n",
            f"RELATION_AWARE_PHASEA_ENABLED = {bool(profile_dict.get('RELATION_AWARE_PHASEA_ENABLED', False))}\n",
            f"RELATION_AWARE_PHASEC_ENABLED = {bool(profile_dict.get('RELATION_AWARE_PHASEC_ENABLED', False))}\n",
            "\n",
            "def _ra_log_positive_hazard(values):\n",
            "    values = np.asarray(values, dtype=float)\n",
            "    if not np.isfinite(values).all() or np.any(values <= 0):\n",
            "        raise RuntimeError('OOF partial hazards must be finite and positive')\n",
            "    return np.log(np.clip(values, 1e-300, None))\n",
            "\n",
            "def _ra_fixed_reference_linear_predictor(eta, feature_df, selected_b, coefficients, time_stats, phasec_df=None, reference_time=1.0):\n",
            "    values = float(coefficients.get('eta_ab', 1.0)) * np.asarray(eta, dtype=float).copy()\n",
            "    feature_df = feature_df.reset_index(drop=True)\n",
            "    for cand in list(selected_b or []):\n",
            "        feat = cand.get('feature')\n",
            "        if feat not in feature_df.columns:\n",
            "            raise RuntimeError(f'missing Phase-B feature for reference risk: {feat}')\n",
            "        basis_value = float(phaseb_eval_time_basis([reference_time], cand.get('basis'), time_stats)[0])\n",
            "        feature_value = pd.to_numeric(feature_df[feat], errors='coerce').fillna(0.0).values\n",
            "        values += float(coefficients.get(cand.get('interaction_col'), 0.0)) * feature_value * basis_value\n",
            "    if phasec_df is not None:\n",
            "        phasec_df = phasec_df.reset_index(drop=True)\n",
            "        for col in phasec_df.columns:\n",
            "            values += float(coefficients.get(col, 0.0)) * pd.to_numeric(phasec_df[col], errors='coerce').fillna(0.0).values\n",
            "    if not np.isfinite(values).all():\n",
            "        raise RuntimeError('non-finite fixed-reference linear predictor')\n",
            "    return values\n",
        ]
        nb.setdefault("cells", []).insert(2, {"cell_type": "code", "execution_count": None, "id": "relation-aware-overrides", "metadata": {}, "outputs": [], "source": override})
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            text = legacy.source_text(cell.get("source", ""))
            text = text.replace(
                'PHASEC_K_VALUES if mode == "stgp_and" else [0]) if 0 <= int(k) <= 3',
                'PHASEC_K_VALUES if mode == "stgp_and" else [0]) if 0 <= int(k) <= 4',
            )
            needle = "    return rows\n\n\ndef phasec_seeded_build_candidate_pool"
            if needle in text and "PHASEC_EXTERNAL_RULES_JSON" not in text[text.find("def phasec_stgp_and_build_rule_universe"):text.find("def phasec_seeded_build_candidate_pool")]:
                repl = """    try:
        _rules_path = str(globals().get("PHASEC_EXTERNAL_RULES_JSON", "") or "")
        globals()["RELATION_AWARE_PHASEC_ATTEMPTED_N"] = 0
        globals()["RELATION_AWARE_PHASEC_INJECTED_N"] = 0
        globals()["RELATION_AWARE_PHASEC_INJECTION_FAILED_N"] = 0
        globals()["RELATION_AWARE_PHASEC_INJECTED_ITEMS"] = []
        globals()["RELATION_AWARE_PHASEC_INJECTION_FAILED_ITEMS"] = []
        if _rules_path:
            import json as _ra_json
            from pathlib import Path as _RAPath
            _rules = _ra_json.loads(_RAPath(_rules_path).read_text(encoding="utf-8"))
            for _rule in _rules:
                globals()["RELATION_AWARE_PHASEC_ATTEMPTED_N"] += 1
                _pair = _rule.get("interaction_pair") if isinstance(_rule, dict) else None
                if isinstance(_pair, (list, tuple)) and len(_pair) >= 2:
                    try:
                        if str(_pair[0]) not in term_meta or str(_pair[1]) not in term_meta:
                            raise ValueError("missing_terms_in_target")
                        _cand = _phasec_make_stgp_and_candidate(str(_pair[0]), str(_pair[1]), term_meta)
                        _cand["source_endpoint"] = _rule.get("source_endpoint")
                        _cand["source_endpoints"] = _rule.get("source_endpoints", [])
                        _cand["source_count"] = _rule.get("source_count")
                        _cand["target_endpoint"] = _rule.get("target_endpoint")
                        _cand["transfer_weight"] = _rule.get("transfer_weight")
                        _cand["mean_transfer_weight"] = _rule.get("mean_transfer_weight")
                        _cand["relation_score"] = _rule.get("relation_score")
                        _cand["transfer_type"] = "relation_aware_phaseC_rule_transfer"
                        _cand["reestimated_in_target"] = True
                        _cand["source_coefficient_transferred"] = False
                        _cand["source_rule_aggregated"] = _rule.get("source_rule_aggregated", False)
                        _sig = phasec_candidate_signature(_cand)
                        if _sig not in seen:
                            seen.add(_sig)
                            rows.append(_cand)
                            globals()["RELATION_AWARE_PHASEC_INJECTED_N"] += 1
                            globals()["RELATION_AWARE_PHASEC_INJECTED_ITEMS"].append(dict(_rule))
                    except Exception as _phasec_rule_exc:
                        globals()["RELATION_AWARE_PHASEC_INJECTION_FAILED_N"] += 1
                        _failed = dict(_rule)
                        _failed["injection_failed_reason"] = str(_phasec_rule_exc)
                        globals()["RELATION_AWARE_PHASEC_INJECTION_FAILED_ITEMS"].append(_failed)
    except Exception as _ra_exc:
        print(f"[RelationAware PhaseC warning] external rule injection failed: {_ra_exc}")
    return rows


def phasec_seeded_build_candidate_pool"""
                text = text.replace(needle, repl)
            write_needle = "    Path(RESULT_JSON).parent.mkdir(parents=True, exist_ok=True)\n"
            if write_needle in text and "relation_aware_multitarget_oof_export" not in text:
                inject = """    # relation_aware_multitarget_oof_export
    try:
        for _risk_key in ["oof_main_only_risk", "oof_main_gp_risk", "oof_ab_risk", "oof_abc_risk"]:
            _risk_val = cv_res.get(_risk_key, None)
            if _risk_val is not None:
                result_payload[_risk_key] = np.asarray(_risk_val, dtype=float).tolist()
        result_payload["oof_abc_risk_by_k"] = {
            str(int(_k)): np.asarray(_risk, dtype=float).tolist()
            for _k, _risk in (cv_res.get("oof_abc_risk_by_k", {}) or {}).items()
        }
        result_payload["oof_row_keys"] = [str(x) for x in cv_res.get("oof_row_keys", [])]
        _ra_oof_sidecar = Path(RESULT_JSON).with_name("oof_reference_risk.json")
        _ra_oof_sidecar.write_text(json.dumps({
            "risk_estimand": "OOF locked-model linear predictor at ICU day 1",
            "reference_time_days": 1.0,
            "oof_abc_risk": result_payload.get("oof_abc_risk", []),
            "oof_abc_risk_by_k": result_payload.get("oof_abc_risk_by_k", {}),
            "oof_row_keys": result_payload.get("oof_row_keys", []),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        result_payload["relation_aware_ablation"] = globals().get("RELATION_AWARE_ABLATION", "")
        result_payload["relation_aware_phaseA_enabled"] = bool(globals().get("RELATION_AWARE_PHASEA_ENABLED", False))
        result_payload["relation_aware_phaseC_enabled"] = bool(globals().get("RELATION_AWARE_PHASEC_ENABLED", False))
        result_payload["phaseA_external_trees_json"] = globals().get("PHASEA_EXTERNAL_TREES_JSON", "")
        result_payload["phaseC_external_rules_json"] = globals().get("PHASEC_EXTERNAL_RULES_JSON", "")
        result_payload["relation_aware_phaseC_attempted_n"] = int(globals().get("RELATION_AWARE_PHASEC_ATTEMPTED_N", 0))
        result_payload["relation_aware_phaseC_injected_n"] = int(globals().get("RELATION_AWARE_PHASEC_INJECTED_N", 0))
        result_payload["relation_aware_phaseC_injection_failed_n"] = int(globals().get("RELATION_AWARE_PHASEC_INJECTION_FAILED_N", 0))
        result_payload["relation_aware_phaseC_injected_items"] = globals().get("RELATION_AWARE_PHASEC_INJECTED_ITEMS", [])
        result_payload["relation_aware_phaseC_injection_failed_items"] = globals().get("RELATION_AWARE_PHASEC_INJECTION_FAILED_ITEMS", [])
        _ra_gp_input_vars = [str(c) for c in locals().get("gp_input_vars", [])]
        if not _ra_gp_input_vars:
            _ra_gp_input_vars = [str(c) for c in globals().get("gp_input_vars", [])]
        if not _ra_gp_input_vars:
            _ra_gp_input_vars = [str(c) for c in globals().get("GP_INPUT_VARS", [])]
        result_payload["phaseA_gp_input_vars"] = list(_ra_gp_input_vars)
        result_payload["gp_input_vars"] = list(_ra_gp_input_vars)
        if not _ra_gp_input_vars:
            result_payload.setdefault("warnings", []).append("phaseA_gp_input_vars_missing_or_empty")
        result_payload["relation_aware_phaseA_attempted_n"] = int(globals().get("RELATION_AWARE_PHASEA_ATTEMPTED_N", 0))
        result_payload["relation_aware_phaseA_injected_n"] = int(globals().get("RELATION_AWARE_PHASEA_INJECTED_N", 0))
        result_payload["relation_aware_phaseA_parse_failed_n"] = int(globals().get("RELATION_AWARE_PHASEA_PARSE_FAILED_N", 0))
        result_payload["relation_aware_phaseA_parse_failed_items"] = globals().get("RELATION_AWARE_PHASEA_PARSE_FAILED_ITEMS", [])
        result_payload["relation_aware_phaseA_injected_items"] = globals().get("RELATION_AWARE_PHASEA_INJECTED_ITEMS", [])
        result_payload["relation_aware_phaseA_current_gp_input_vars"] = globals().get("RELATION_AWARE_PHASEA_CURRENT_GP_INPUT_VARS", list(_ra_gp_input_vars))
        result_payload["relation_aware_phaseA_external_tree_source_json"] = globals().get("PHASEA_EXTERNAL_TREES_JSON", "")
        result_payload["relation_aware_phaseA_precheck_passed_n"] = int(globals().get("RELATION_AWARE_PHASEA_PRECHECK_PASSED_N", 0))
        result_payload["relation_aware_phaseA_precheck_failed_n"] = int(globals().get("RELATION_AWARE_PHASEA_PRECHECK_FAILED_N", 0))
        try:
            _ra_final_exprs = result_payload.get("final_gp_expressions", []) or []
            _ra_final_names = set()
            for _ra_row in _ra_final_exprs:
                if isinstance(_ra_row, dict) and _ra_row.get("expression") is not None:
                    _ra_final_names.add(str(_ra_row.get("expression")).replace(" ", ""))
            _ra_selected_items = []
            for _ra_item in globals().get("RELATION_AWARE_PHASEA_INJECTED_ITEMS", []) or []:
                _ra_named = str(_ra_item.get("named_expression", "")).replace(" ", "") if isinstance(_ra_item, dict) else ""
                if _ra_named and _ra_named in _ra_final_names:
                    _ra_selected_items.append(_ra_item)
            result_payload["relation_aware_phaseA_final_selected_items"] = _ra_selected_items
            result_payload["relation_aware_phaseA_final_selected_n"] = int(len(_ra_selected_items))
            result_payload["relation_aware_phaseA_injected_but_filtered_out_n"] = int(max(0, len(globals().get("RELATION_AWARE_PHASEA_INJECTED_ITEMS", []) or []) - len(_ra_selected_items)))
        except Exception as _ra_sel_exc:
            result_payload["relation_aware_phaseA_final_selected_items"] = []
            result_payload["relation_aware_phaseA_final_selected_n"] = 0
            result_payload.setdefault("warnings", []).append(f"relation_aware_phaseA_final_selected_match_failed: {_ra_sel_exc}")
        if bool(globals().get("RELATION_AWARE_PHASEA_ENABLED", False)) and int(globals().get("RELATION_AWARE_PHASEA_INJECTED_N", 0)) == 0:
            result_payload.setdefault("warnings", []).append("RELATION_AWARE_PHASEA_ENABLED but no transferred trees were injected.")
    except Exception as _ra_oof_exc:
        print(f"[RelationAware warning] OOF risk export failed: {_ra_oof_exc}")
"""
                text = text.replace(write_needle, inject + write_needle)
            phasea_needle = "        best_ind_x, F, gp_filter_diag = filter_gp_trees_after_evolution(best_ind_x, toolbox_x, X, T, E)\n"
            if phasea_needle in text and "relation_aware_phaseA_tree_injection" not in text:
                phasea_inject = """        # relation_aware_phaseA_tree_injection
        try:
            globals()["RELATION_AWARE_PHASEA_ATTEMPTED_N"] = 0
            globals()["RELATION_AWARE_PHASEA_INJECTED_N"] = 0
            globals()["RELATION_AWARE_PHASEA_PARSE_FAILED_N"] = 0
            globals()["RELATION_AWARE_PHASEA_PARSE_FAILED_ITEMS"] = []
            globals()["RELATION_AWARE_PHASEA_INJECTED_ITEMS"] = []
            _phasea_path = str(globals().get("PHASEA_EXTERNAL_TREES_JSON", "") or "")
            if _phasea_path:
                import json as _ra_json
                import re as _ra_re
                from pathlib import Path as _RAPath
                _pset_x = getattr(toolbox_x.compile, "keywords", {}).get("pset", None)
                _records = _ra_json.loads(_RAPath(_phasea_path).read_text(encoding="utf-8"))
                _current_gp_input_vars = [str(c) for c in locals().get("gp_input_vars", [])]
                if not _current_gp_input_vars:
                    _current_gp_input_vars = [str(c) for c in globals().get("gp_input_vars", [])]
                if not _current_gp_input_vars:
                    _current_gp_input_vars = [str(c) for c in globals().get("GP_INPUT_VARS", [])]
                if not _current_gp_input_vars:
                    try:
                        _current_gp_input_vars = [f"x{i}" for i in range(int(getattr(X, "shape", [0, 0])[1]))]
                        globals().setdefault("RELATION_AWARE_PHASEA_PARSE_FAILED_ITEMS", []).append({
                            "expression": "__current_gp_input_vars__",
                            "reason": "gp_input_vars_missing_used_x_index_fallback",
                        })
                    except Exception:
                        _current_gp_input_vars = []
                globals()["RELATION_AWARE_PHASEA_CURRENT_GP_INPUT_VARS"] = list(_current_gp_input_vars)
                globals()["RELATION_AWARE_PHASEA_PRECHECK_PASSED_N"] = int(sum(1 for _r in _records if isinstance(_r, dict) and bool(_r.get("precheck_injectable", True))))
                globals()["RELATION_AWARE_PHASEA_PRECHECK_FAILED_N"] = int(sum(1 for _r in _records if isinstance(_r, dict) and not bool(_r.get("precheck_injectable", True))))
                def _ra_remap_expr(_expr_text, _source_names, _target_names):
                    _target_idx = {str(v): i for i, v in enumerate(_target_names)}
                    _missing = []
                    def _replace(_match):
                        _old_i = int(_match.group(1))
                        if _old_i >= len(_source_names):
                            _missing.append(f"x{_old_i}:source_index_out_of_range")
                            return _match.group(0)
                        _name = str(_source_names[_old_i])
                        if _name not in _target_idx:
                            _missing.append(_name)
                            return _match.group(0)
                        return f"x{_target_idx[_name]}"
                    _mapped = _ra_re.sub(r"\\bx(\\d+)\\b", _replace, str(_expr_text))
                    return _mapped, sorted(set(_missing))
                def _ra_named_expr(_expr_text, _target_names):
                    def _replace(_match):
                        _i = int(_match.group(1))
                        return str(_target_names[_i]) if _i < len(_target_names) else _match.group(0)
                    return _ra_re.sub(r"\\bx(\\d+)\\b", _replace, str(_expr_text))
                _added = 0
                for _rec in _records:
                    _expr = _rec.get("expression") if isinstance(_rec, dict) else None
                    _source_feature_names = (_rec.get("source_feature_names") or _rec.get("feature_names", [])) if isinstance(_rec, dict) else []
                    globals()["RELATION_AWARE_PHASEA_ATTEMPTED_N"] += 1
                    if _expr and _pset_x is not None:
                        try:
                            if not isinstance(_source_feature_names, list) or not _source_feature_names:
                                raise ValueError("missing_source_feature_names_for_x_index_remap")
                            _mapped_expr, _missing_features = _ra_remap_expr(str(_expr), [str(x) for x in _source_feature_names], _current_gp_input_vars)
                            if _missing_features:
                                raise ValueError("missing_target_features_for_transfer:" + ",".join(_missing_features))
                            _tree = gp.PrimitiveTree.from_string(str(_mapped_expr), _pset_x)
                            if int(getattr(_tree, "height", 0)) <= int(MAX_TREE_HEIGHT_X):
                                best_ind_x.append(_tree)
                                _added += 1
                                globals()["RELATION_AWARE_PHASEA_INJECTED_N"] += 1
                                globals()["RELATION_AWARE_PHASEA_INJECTED_ITEMS"].append({
                                    "expression": str(_expr),
                                    "mapped_expression": str(_mapped_expr),
                                    "named_expression": _ra_named_expr(str(_mapped_expr), _current_gp_input_vars),
                                    "feature_names": [str(x) for x in _source_feature_names],
                                    "source_feature_names": [str(x) for x in _source_feature_names],
                                    "used_features": [str(x) for x in (_rec.get("used_features", []) if isinstance(_rec, dict) else [])],
                                    "current_gp_input_vars": _current_gp_input_vars,
                                    "tree_group_id": _rec.get("tree_group_id") if isinstance(_rec, dict) else None,
                                    "tree_id": _rec.get("tree_id", _rec.get("candidate_id")) if isinstance(_rec, dict) else None,
                                    "candidate_id": _rec.get("candidate_id", _rec.get("tree_id")) if isinstance(_rec, dict) else None,
                                    "source_endpoint": _rec.get("source_endpoint") if isinstance(_rec, dict) else None,
                                    "target_endpoint": _rec.get("target_endpoint") if isinstance(_rec, dict) else None,
                                    "transfer_weight": _rec.get("transfer_weight") if isinstance(_rec, dict) else None,
                                    "generation": _rec.get("generation") if isinstance(_rec, dict) else None,
                                    "reestimated_in_target": True,
                                    "remap_strategy": "feature_name_exact",
                                    "remap_missing_features": _missing_features,
                                    "precheck_injectable": _rec.get("precheck_injectable") if isinstance(_rec, dict) else None,
                                    "precheck_reason": _rec.get("precheck_reason") if isinstance(_rec, dict) else None,
                                })
                            else:
                                globals()["RELATION_AWARE_PHASEA_PARSE_FAILED_N"] += 1
                                globals()["RELATION_AWARE_PHASEA_PARSE_FAILED_ITEMS"].append({
                                    "expression": str(_expr),
                                    "reason": "tree_height_exceeds_MAX_TREE_HEIGHT_X",
                                    "mapped_expression": str(_mapped_expr),
                                    "source_feature_names": [str(x) for x in _source_feature_names],
                                    "target_gp_input_vars": _current_gp_input_vars,
                                    "missing_features": _missing_features,
                                    "tree_group_id": _rec.get("tree_group_id") if isinstance(_rec, dict) else None,
                                    "tree_id": _rec.get("tree_id", _rec.get("candidate_id")) if isinstance(_rec, dict) else None,
                                    "source_endpoint": _rec.get("source_endpoint") if isinstance(_rec, dict) else None,
                                    "target_endpoint": _rec.get("target_endpoint") if isinstance(_rec, dict) else None,
                                })
                        except Exception as _tree_exc:
                            globals()["RELATION_AWARE_PHASEA_PARSE_FAILED_N"] += 1
                            globals()["RELATION_AWARE_PHASEA_PARSE_FAILED_ITEMS"].append({
                                "expression": str(_expr),
                                "reason": str(_tree_exc),
                                "mapped_expression": locals().get("_mapped_expr", ""),
                                "source_feature_names": [str(x) for x in _source_feature_names] if isinstance(_source_feature_names, list) else [],
                                "target_gp_input_vars": _current_gp_input_vars,
                                "missing_features": locals().get("_missing_features", []),
                                "tree_group_id": _rec.get("tree_group_id") if isinstance(_rec, dict) else None,
                                "tree_id": _rec.get("tree_id", _rec.get("candidate_id")) if isinstance(_rec, dict) else None,
                                "source_endpoint": _rec.get("source_endpoint") if isinstance(_rec, dict) else None,
                                "target_endpoint": _rec.get("target_endpoint") if isinstance(_rec, dict) else None,
                            })
                            print(f"[RelationAware PhaseA warning] skipped transferred tree {_expr}: {_tree_exc}")
                print(f"[RelationAware PhaseA] attempted={globals()['RELATION_AWARE_PHASEA_ATTEMPTED_N']} injected={globals()['RELATION_AWARE_PHASEA_INJECTED_N']} parse_failed={globals()['RELATION_AWARE_PHASEA_PARSE_FAILED_N']} before Cox filtering")
                if bool(globals().get("RELATION_AWARE_PHASEA_ENABLED", False)) and globals()["RELATION_AWARE_PHASEA_INJECTED_N"] == 0:
                    print("[RelationAware PhaseA warning] enabled but injected_n == 0")
        except Exception as _ra_phasea_exc:
            globals()["RELATION_AWARE_PHASEA_PARSE_FAILED_N"] = int(globals().get("RELATION_AWARE_PHASEA_PARSE_FAILED_N", 0)) + 1
            globals().setdefault("RELATION_AWARE_PHASEA_PARSE_FAILED_ITEMS", []).append({"expression": "__phaseA_external_trees_json__", "reason": str(_ra_phasea_exc)})
            print(f"[RelationAware PhaseA warning] external tree injection failed: {_ra_phasea_exc}")
"""
                text = text.replace(phasea_needle, phasea_inject + phasea_needle)
            cell["source"] = legacy.source_like(text, cell.get("source", ""))
        _patch_reference_oof_risk(nb)
        legacy.write_notebook(output_ipynb, nb)
        return info

    legacy.patch_notebook_for_profile = patched
    legacy._relation_aware_patch_installed = True


def _expression_x_indices(expr: str) -> list[int]:
    return sorted({int(x) for x in re.findall(r"\bx(\d+)\b", str(expr or ""))})


def precheck_phaseA_external_trees(
    target_endpoint: str,
    trees: list[dict[str, Any]],
    target_gp_input_vars: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter Phase A transfer candidates before notebook injection."""
    target_names = [str(x) for x in target_gp_input_vars]
    target_set = set(target_names)
    target_index = {name: i for i, name in enumerate(target_names)}
    injectable: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for rec in trees:
        expr = str(rec.get("expression", "")) if isinstance(rec, dict) else ""
        feature_names = [str(x) for x in (rec.get("source_feature_names") or rec.get("feature_names", []))] if isinstance(rec, dict) else []
        used = []
        missing = []
        reason = ""
        mapped_expression = expr
        for idx_value in _expression_x_indices(expr):
            if idx_value >= len(feature_names):
                missing.append(f"x{idx_value}:source_index_out_of_range")
                continue
            name = feature_names[idx_value]
            used.append(name)
            if name not in target_set:
                missing.append(name)
        if not missing and expr and feature_names:
            def repl(match):
                source_i = int(match.group(1))
                name = feature_names[source_i]
                return f"x{target_index[name]}"
            mapped_expression = re.sub(r"\bx(\d+)\b", repl, expr)
        if not expr:
            reason = "missing_expression"
        elif not feature_names:
            reason = "missing_source_feature_names_for_x_index_remap"
        elif missing:
            reason = "missing_target_features_for_transfer:" + ",".join(sorted(set(missing)))
        injectable_flag = not reason
        row = {
            "target_endpoint": target_endpoint,
            "source_endpoint": rec.get("source_endpoint") if isinstance(rec, dict) else None,
            "tree_group_id": rec.get("tree_group_id") if isinstance(rec, dict) else None,
            "tree_id": rec.get("tree_id", rec.get("candidate_id")) if isinstance(rec, dict) else None,
            "expression": expr,
            "mapped_expression": mapped_expression,
            "source_feature_names": json.dumps(feature_names, ensure_ascii=False),
            "used_features": json.dumps(sorted(set(used)), ensure_ascii=False),
            "target_gp_input_vars": json.dumps(target_names, ensure_ascii=False),
            "missing_features": json.dumps(sorted(set(missing)), ensure_ascii=False),
            "injectable": injectable_flag,
            "reason": reason,
        }
        rows.append(row)
        if injectable_flag:
            copied = dict(rec)
            copied["used_features"] = sorted(set(used)) if used else list(copied.get("used_features", []))
            copied["source_feature_names"] = feature_names
            copied["feature_names"] = feature_names
            copied["mapped_expression"] = mapped_expression
            copied["target_gp_input_vars"] = target_names
            copied["precheck_injectable"] = True
            copied["precheck_reason"] = ""
            injectable.append(copied)
    return injectable, rows


class MultiTargetABCCompetitionRunner:
    """Run relation-aware multitarget ABC ablations and internal-only profile selection."""

    def __init__(
        self,
        output_root: Path = OUTPUT_ROOT,
        random_seed: int = 20260523,
        timeout: int = 24 * 60 * 60,
        force: bool = False,
        single_endpoint_reference_root: Path | None = None,
        disable_single_endpoint_workbook_reference: bool = False,
        profile_override_json: Path | None = None,
        trial_id: str | None = None,
        notebook_template: Path | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.summary_dir = self.output_root / "summaries"
        self.selected_dir = self.output_root / "selected_profile"
        self.random_seed = int(random_seed)
        self.timeout = int(timeout)
        self.force = bool(force)
        self.single_endpoint_reference_root = Path(single_endpoint_reference_root) if single_endpoint_reference_root else DEFAULT_SINGLE_ENDPOINT_REFERENCE_ROOT
        self.disable_single_endpoint_workbook_reference = bool(disable_single_endpoint_workbook_reference)
        self.profile_override_json = Path(profile_override_json) if profile_override_json else None
        self.trial_id = str(trial_id or "")
        self.notebook_template = Path(notebook_template) if notebook_template else None
        if self.notebook_template:
            if not self.notebook_template.exists():
                raise FileNotFoundError(self.notebook_template)
            legacy.NOTEBOOK_FILE = self.notebook_template
        self.profile_override_payload: dict[str, Any] = {}
        self.multitarget_params: dict[str, Any] = {}
        self.matrix_override_payload: dict[str, Any] = {}
        if self.profile_override_json:
            payload = _read_json_object(self.profile_override_json, "profile override")
            if "scope" in payload and payload.get("scope") != "multi_target_only":
                raise ValueError("profile override scope must be 'multi_target_only'")
            if (
                "single_target_hyperparameters_modified" in payload
                and payload.get("single_target_hyperparameters_modified") is not False
            ):
                raise ValueError("profile override must not modify single-target hyperparameters")
            if (
                "single_target_algorithm_rerun" in payload
                and payload.get("single_target_algorithm_rerun") is not False
            ):
                raise ValueError("profile override must not rerun the single-target algorithm")
            params = payload.get("params")
            if params is None:
                metadata = {
                    "trial_id",
                    "scope",
                    "base_profile",
                    "ablation",
                    "single_target_hyperparameters_modified",
                    "single_target_algorithm_rerun",
                }
                params = {
                    key: value for key, value in payload.items() if key not in metadata
                }
            if not isinstance(params, dict):
                raise ValueError("profile override params must be a JSON object")
            self.profile_override_payload = payload
            self.multitarget_params = dict(params)
            self.matrix_override_payload = {
                key: self.multitarget_params.pop(key)
                for key in (
                    "phaseA_relation_matrix_override",
                    "phaseC_relation_matrix_override",
                )
                if key in self.multitarget_params
            }
        self.exclusive_conflict_matrix = build_exclusive_conflict_matrix(ENDPOINTS)
        self.phaseA_transfer_matrix = build_phaseA_transfer_matrix(ENDPOINTS)
        self.phaseC_transfer_matrix = build_phaseC_transfer_matrix(ENDPOINTS)
        self.phaseA_matrix_override_applied = "phaseA_relation_matrix_override" in self.matrix_override_payload
        self.phaseC_matrix_override_applied = "phaseC_relation_matrix_override" in self.matrix_override_payload
        if "phaseA_relation_matrix_override" in self.matrix_override_payload:
            self.phaseA_transfer_matrix = self._coerce_matrix_override(
                self.matrix_override_payload["phaseA_relation_matrix_override"],
                "phaseA_relation_matrix_override",
            )
        if "phaseC_relation_matrix_override" in self.matrix_override_payload:
            self.phaseC_transfer_matrix = self._coerce_matrix_override(
                self.matrix_override_payload["phaseC_relation_matrix_override"],
                "phaseC_relation_matrix_override",
            )
        event4_index = (
            ENDPOINTS.index("later_icu_same_hadm")
            if "later_icu_same_hadm" in ENDPOINTS
            else None
        )
        resource_sources = tuple(
            name
            for name in (
                "icu_los_gt7d",
                "hosp_los_gt7d",
                "hosp_los_gt14d",
            )
            if name in ENDPOINTS
        )
        event4_boost = float(
            self.multitarget_params.get(
                "EVENT4_RELATION_BOOST_TO_ICU_RESOURCE_ENDPOINTS", 1.0
            )
        )
        if event4_index is not None:
            for source in resource_sources:
                source_index = ENDPOINTS.index(source)
                self.phaseA_transfer_matrix[event4_index, source_index] = min(
                    1.0,
                    self.phaseA_transfer_matrix[event4_index, source_index]
                    * event4_boost,
                )
                self.phaseC_transfer_matrix[event4_index, source_index] = min(
                    1.0,
                    self.phaseC_transfer_matrix[event4_index, source_index]
                    * event4_boost,
                )
        self.phaseA_transfer_matrix_final = self.phaseA_transfer_matrix.copy()
        self.phaseC_transfer_matrix_final = self.phaseC_transfer_matrix.copy()
        self.phaseC_legacy_transfer_matrix = build_legacy_phaseC_transfer_matrix(ENDPOINTS)
        self.pipeline = SingleEndpointABCPipeline({})
        self.baseline_by_endpoint: dict[str, dict[str, Any]] = {}
        self.single_endpoint_reference_by_endpoint: dict[str, dict[str, Any]] = {}
        self.locked_candidate_registry: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        self.knowledge_audit: dict[str, Any] = {}
        self.knowledge_audit_rows: list[dict[str, Any]] = []
        print(f"[MT-ABC] random_seed={self.random_seed} output_root={self.output_root}", flush=True)
        print("[MT-ABC] external_used_for_selection=False", flush=True)
        print(f"[MT-ABC][Matrix] phaseA_matrix_override_applied={self.phaseA_matrix_override_applied}", flush=True)
        print(f"[MT-ABC][Matrix] phaseC_matrix_override_applied={self.phaseC_matrix_override_applied}", flush=True)
        print(f"[MT-ABC][Matrix] event4_relation_boost_applied_after_override={event4_boost}", flush=True)
        print(f"[MT-ABC][Matrix] exclusive_conflict_matrix={self.exclusive_conflict_matrix.tolist()}", flush=True)
        print(f"[MT-ABC][Matrix] phaseA_transfer_matrix final_values={self.phaseA_transfer_matrix.tolist()}", flush=True)
        print(f"[MT-ABC][Matrix] phaseC_transfer_matrix final_values={self.phaseC_transfer_matrix.tolist()}", flush=True)
        print(f"[MT-ABC][Matrix] legacy_phaseC_transfer_matrix debug_only={self.phaseC_legacy_transfer_matrix.tolist()}", flush=True)
        relation_patch_notebook()

    @staticmethod
    def _coerce_matrix_override(value: Any, label: str) -> np.ndarray:
        """Validate a target-row/source-column matrix override from profile JSON."""
        if isinstance(value, dict):
            value = [
                [value.get(target, {}).get(source, 0.0) for source in ENDPOINTS]
                for target in ENDPOINTS
            ]
        mat = np.asarray(value, dtype=float)
        expected_shape = (len(ENDPOINTS), len(ENDPOINTS))
        if mat.shape != expected_shape:
            raise ValueError(f"{label} must have shape {expected_shape}, got {mat.shape}")
        if not np.isfinite(mat).all():
            raise ValueError(f"{label} contains NaN or infinite values")
        if (mat < 0.0).any() or (mat > 1.0).any():
            raise ValueError(f"{label} values must all be in [0, 1]")
        if not np.allclose(np.diag(mat), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"{label} diagonal must be 0")
        return mat

    def profiles(self, profile_set: str, only_profile: str | None = None) -> list[dict[str, Any]]:
        if profile_set == "optuna_search":
            base_profile = str(
                self.profile_override_payload.get("base_profile")
                or "profile_007_high_mutation"
            )
            base = next(
                (p for p in legacy.build_profiles("full") if p["profile_name"] == base_profile),
                None,
            )
            if base is None:
                raise ValueError(f"Unknown base profile: {base_profile}")
            params = dict(base["params"])
            params.update(self.multitarget_params)
            k_profile = params.pop("PHASEC_K_PROFILE", None)
            if k_profile is not None:
                k_values = {
                    "k_0_1_2": [0, 1, 2],
                    "k_0_1_2_3": [0, 1, 2, 3],
                    "k_1_2_3": [1, 2, 3],
                    "k_0_1_2_3_4": [0, 1, 2, 3, 4],
                }.get(str(k_profile))
                if k_values is None:
                    raise ValueError(f"Unknown PHASEC_K_PROFILE: {k_profile}")
                params["PHASEC_K_VALUES"] = k_values
            profile_name = only_profile or (
                f"optuna_{self.trial_id}" if self.trial_id else "optuna_trial"
            )
            profiles = [{
                "profile_id": int(base["profile_id"]),
                "profile_name": profile_name,
                "seed_profile_name": base_profile,
                "params": params,
            }]
        else:
            profiles = legacy.build_profiles(profile_set)
        if only_profile:
            profiles = [p for p in profiles if p["profile_name"] == only_profile]
        return profiles

    def run(self, profile_set: str = "pilot", only_profile: str | None = None, only_ablation: str | None = None, skip_final_external: bool = False) -> dict[str, Any]:
        started = now_iso()
        targets = discover_targets()
        profiles = self.profiles(profile_set, only_profile)
        if not USE_SINGLE_ENDPOINT_KNOWLEDGE_CACHE:
            raise RuntimeError("The default multitarget flow requires USE_SINGLE_ENDPOINT_KNOWLEDGE_CACHE=True")
        if not SKIP_MULTITARGET_SELF_PRE_DISCOVERY:
            raise RuntimeError("The default multitarget flow requires SKIP_MULTITARGET_SELF_PRE_DISCOVERY=True")
        knowledge_cache = load_single_endpoint_knowledge_cache(SINGLE_ENDPOINT_KNOWLEDGE_CACHE)
        knowledge_packages = load_single_endpoint_knowledge_packages(knowledge_cache)
        knowledge_validation = validate_single_endpoint_knowledge_for_multitarget(knowledge_packages)
        self.baseline_by_endpoint = load_baseline_summary_from_single_endpoint_knowledge(
            knowledge_packages
        )
        phaseA_source_pool = build_phaseA_source_pool_from_single_endpoint_knowledge(
            knowledge_packages,
            max_per_endpoint=int(
                self.multitarget_params.get(
                    "MAX_PHASEA_SOURCE_TREES_PER_ENDPOINT",
                    MAX_PHASEA_SOURCE_TREES_PER_ENDPOINT,
                )
            ),
        )
        phaseC_source_pool = build_phaseC_source_pool_from_single_endpoint_knowledge(
            knowledge_packages,
            max_per_endpoint=int(
                self.multitarget_params.get(
                    "MAX_PHASEC_SOURCE_RULES_PER_ENDPOINT",
                    MAX_PHASEC_SOURCE_RULES_PER_ENDPOINT,
                )
            ),
        )
        target_gp_input_vars = {
            bundle["target_name"]: _phasea_input_vars(bundle)
            for bundle in knowledge_packages.values()
        }
        phaseA_cached_candidates, phaseA_cached_transfer_log = (
            build_phaseA_transfer_candidates_from_source_pool(
                phaseA_source_pool,
                target_gp_input_vars,
                self.phaseA_transfer_matrix,
                min_transfer_weight=float(
                    self.multitarget_params.get("PHASEA_MIN_TRANSFER_WEIGHT", 0.0)
                ),
                relation_weight_power=float(
                    self.multitarget_params.get("PHASEA_RELATION_WEIGHT_POWER", 1.0)
                ),
                max_trees_per_target=int(
                    self.multitarget_params.get(
                        "MULTITARGET_PHASEA_MAX_TRANSFER_TREES_PER_TARGET", 100
                    )
                ),
            )
        )
        phaseC_cached_raw = [
            row
            for endpoint in ENDPOINTS
            for row in phaseC_source_pool.get(endpoint, [])
        ]
        phaseC_cached_aggregated = aggregate_phaseC_rule_pool(phaseC_cached_raw)
        self.single_endpoint_reference_by_endpoint = {
            bundle["target_name"]: {
                "single_endpoint_ABC_internal": bundle["single_endpoint_knowledge"].get("ABC_internal_cindex"),
                "single_endpoint_ABC_external": bundle["single_endpoint_knowledge"].get("ABC_external_cindex"),
                "single_endpoint_reference_top10_cox_internal": self.baseline_by_endpoint[bundle["target_name"]]["baseline_internal_for_report"],
                "single_endpoint_reference_top10_cox_external": self.baseline_by_endpoint[bundle["target_name"]]["baseline_external_for_report"],
                "single_endpoint_reference_source": bundle["paths"]["single_endpoint_knowledge"],
                "single_endpoint_reference_status": "loaded_from_single_endpoint_knowledge",
            }
            for bundle in knowledge_packages.values()
        }
        validation_by_id = {
            row["target_id"]: row for row in knowledge_validation["rows"]
        }
        self.knowledge_audit_rows = []
        for target_id in sorted(knowledge_packages):
            bundle = knowledge_packages[target_id]
            target_name = bundle["target_name"]
            phasec_rows = phaseC_source_pool[target_name]
            validation_row = validation_by_id[target_id]
            baseline = self.baseline_by_endpoint[target_name]
            phasec_counts = _phasec_knowledge_counts(bundle)
            phasea_raw_n = int(validation_row["phaseA_candidate_raw_n"])
            phasea_valid_n = len(phaseA_source_pool[target_name])
            phasec_unique_n = len({
                tuple(row.get("interaction_pair") or [])
                for row in phasec_rows
            })
            self.knowledge_audit_rows.append({
                "target_id": target_id,
                "target_name": target_name,
                "phaseA_gp_input_vars_n": len(target_gp_input_vars[target_name]),
                "phaseA_candidate_source_used": validation_row["phaseA_candidate_source"],
                "phaseA_source_tree_raw_n": phasea_raw_n,
                "phaseA_source_tree_valid_n": phasea_valid_n,
                "phaseA_source_tree_dropped_n": max(0, phasea_raw_n - phasea_valid_n),
                **phasec_counts,
                "phaseC_source_rule_n": len(phasec_rows),
                "phaseC_unique_rule_n": phasec_unique_n,
                "baseline_internal_for_report": baseline["baseline_internal_for_report"],
                "baseline_external_for_report": baseline["baseline_external_for_report"],
                "baseline_source_for_report": baseline["baseline_source_for_report"],
                "single_endpoint_ABC_internal": bundle["single_endpoint_knowledge"].get("ABC_internal_cindex"),
                "single_endpoint_ABC_external": bundle["single_endpoint_knowledge"].get("ABC_external_cindex"),
                "external_used_for_selection": bundle["single_endpoint_knowledge"].get("external_used_for_selection"),
                "knowledge_validation_passed": validation_row["knowledge_validation_passed"],
                "warnings": json.dumps(bundle["warnings"], ensure_ascii=False),
                "errors": json.dumps(bundle["errors"], ensure_ascii=False),
            })
        self.knowledge_audit = {
            "single_endpoint_knowledge_cache_path": str(SINGLE_ENDPOINT_KNOWLEDGE_CACHE),
            "single_endpoint_knowledge_loaded": True,
            "single_endpoint_knowledge_endpoint_n": len(knowledge_packages),
            "single_endpoint_pre_discovery_enabled": False,
            "single_endpoint_algorithm_rerun": False,
            "baseline_source_mode": "single_endpoint_knowledge",
            "baseline_loaded_from_single_endpoint_knowledge": True,
            "baseline_generation_disabled": True,
            "phaseA_source_mode": "read_from_single_endpoint_knowledge",
            "phaseC_source_mode": "read_from_single_endpoint_knowledge",
            "phaseA_pre_discovery_skipped": True,
            "phaseC_pre_discovery_skipped": True,
            "target_internal_evolution_enabled": True,
            "target_internal_cv_enabled": True,
            "target_structure_locking_enabled": True,
            "phaseA_source_tree_n_by_endpoint": {
                endpoint: len(rows) for endpoint, rows in phaseA_source_pool.items()
            },
            "phaseC_source_rule_n_by_endpoint": {
                endpoint: len(rows) for endpoint, rows in phaseC_source_pool.items()
            },
            "external_metrics_used_for_selection": False,
            "legacy_multitarget_self_feature_generation_deleted": True,
            "legacy_selected_profile_final_external_reading_disabled_for_internal": True,
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "single_endpoint_knowledge_load_audit.json").write_text(
            json.dumps(
                {**self.knowledge_audit, "endpoints": self.knowledge_audit_rows},
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        safe_to_excel(
            pd.DataFrame(self.knowledge_audit_rows),
            self.output_root / "single_endpoint_knowledge_load_audit.xlsx",
        )
        print(
            f"[MT-ABC][Single endpoint knowledge] loaded={len(knowledge_packages)} "
            f"cache={SINGLE_ENDPOINT_KNOWLEDGE_CACHE}",
            flush=True,
        )
        for r in self.knowledge_audit_rows:
            print(
                f"[MT-ABC][Single endpoint knowledge] {r['target_name']} "
                f"phaseA_trees={r['phaseA_source_tree_valid_n']} "
                f"phaseC_rules={r['phaseC_source_rule_n']}",
                flush=True,
            )
        for endpoint in ENDPOINTS:
            baseline = self.baseline_by_endpoint[endpoint]
            print(
                f"[MT-ABC][Baseline knowledge] {endpoint}: "
                f"internal={legacy.to_float(baseline['baseline_internal_for_report']):.4f} "
                f"external_reporting={legacy.to_float(baseline['baseline_external_for_report']):.4f}",
                flush=True,
            )
        print(f"[MT-ABC] START timestamp={started} profile_set={profile_set} only_profile={only_profile} only_ablation={only_ablation}", flush=True)
        for t in targets:
            try:
                train_df = pd.read_excel(t["train_file"], usecols=["event"])
                print(
                    f"[MT-ABC][Target] {t['target_name']} train_file_exists={Path(t['train_file']).exists()} "
                    f"external_file_exists={Path(t['external_file']).exists()} train_n={len(train_df)} "
                    f"train_events={int(pd.to_numeric(train_df['event'], errors='coerce').fillna(0).sum())} "
                    f"external_n=not_read_before_profile_lock external_events=not_read_before_profile_lock",
                    flush=True,
                )
            except Exception as exc:
                print(f"[MT-ABC][Target warning] {t['target_name']} metadata read failed: {exc}", flush=True)
        ablations = {only_ablation: ABLATIONS[only_ablation]} if only_ablation else ABLATIONS
        need_phaseA = any(bool(cfg["relation_aware_phaseA"]) for cfg in ablations.values())
        if not need_phaseA:
            print("[MT-ABC] Phase A sidecar skipped: selected ablations do not use relation-aware Phase A.", flush=True)

        long_rows: list[dict[str, Any]] = []
        phaseC_log: list[dict[str, Any]] = []
        phaseC_rule_pool_rows: list[dict[str, Any]] = []
        phaseC_rule_pool_aggregated_rows: list[dict[str, Any]] = []
        phaseC_rule_injection_log: list[dict[str, Any]] = []
        phaseC_rule_selection_log: list[dict[str, Any]] = []
        phaseC_generation_log: list[dict[str, Any]] = []
        phaseA_generation_log: list[dict[str, Any]] = []
        phaseA_transfer_event_log: list[dict[str, Any]] = []
        phaseA_top_candidate_log = []
        phaseA_tree_candidate_log = []
        phaseA_expected_gp_input_vars_by_endpoint: dict[str, list[str]] = {}
        phaseA_precheck_log: list[dict[str, Any]] = []
        phaseA_injection_log: list[dict[str, Any]] = []
        warnings: list[str] = []
        failed_targets: list[dict[str, Any]] = []

        for ab_name, ab_cfg in ablations.items():
            for profile in profiles:
                print("=" * 80, flush=True)
                print(
                    f"[MT-ABC][Competition Iteration] ablation={ab_name} "
                    f"profile={profile['profile_name']} "
                    f"relation_phaseA={ab_cfg['relation_aware_phaseA']} "
                    f"relation_phaseC={ab_cfg['relation_aware_phaseC']}",
                    flush=True,
                )
                print("=" * 80, flush=True)
                phaseA_trees: dict[str, list[dict[str, Any]]] = {}
                if ab_cfg["relation_aware_phaseA"]:
                    phaseA_trees = {
                        endpoint: [dict(row) for row in rows]
                        for endpoint, rows in phaseA_cached_candidates.items()
                    }
                    phaseA_expected_gp_input_vars_by_endpoint.update(target_gp_input_vars)
                    phaseA_transfer_event_log.extend([
                        {**row, "ablation": ab_name, "profile_name": profile["profile_name"]}
                        for row in phaseA_cached_transfer_log
                    ])
                    for ep, trees in phaseA_trees.items():
                        for tr in trees:
                            phaseA_tree_candidate_log.append({
                                **tr,
                                "target_endpoint": ep,
                                "ablation": ab_name,
                                "profile_name": profile["profile_name"],
                                "survived_to_top_trees": True,
                            })
                rule_pool_raw: list[dict[str, Any]] = []
                rule_pool: list[dict[str, Any]] = []
                if ab_cfg["relation_aware_phaseC"]:
                    rule_pool_raw = [dict(row) for row in phaseC_cached_raw]
                    rule_pool = [dict(row) for row in phaseC_cached_aggregated]
                    phaseC_rule_pool_rows.extend([{**r, "ablation": ab_name, "profile_name": profile["profile_name"]} for r in rule_pool_raw])
                    phaseC_rule_pool_aggregated_rows.extend([{**r, "ablation": ab_name, "profile_name": profile["profile_name"]} for r in rule_pool])
                    by_source = Counter(str(r.get("source_endpoint")) for r in rule_pool_raw)
                    source_dist = Counter(int(r.get("source_count", 0)) for r in rule_pool)
                    multi_source_n = sum(1 for r in rule_pool if int(r.get("source_count", 0)) > 1)
                    print(f"[MT-ABC][{ab_name}][PhaseC rule pool raw] source_counts={dict(by_source)} raw_n={len(rule_pool_raw)}", flush=True)
                    print(
                        f"[MT-ABC][{ab_name}][PhaseC rule pool aggregated] unique_pairs={len(rule_pool)} "
                        f"multi_source_n={multi_source_n} source_count_dist={dict(source_dist)}",
                        flush=True,
                    )
                    for r in sorted(rule_pool, key=lambda x: (-int(x.get("source_count", 0)), -legacy.to_float(x.get("max_source_fitness"))))[:20]:
                        print(
                            f"[MT-ABC][{ab_name}][PhaseC top multisource] pair={r.get('interaction_pair')} "
                            f"sources={r.get('source_endpoints')} source_count={r.get('source_count')}",
                            flush=True,
                        )
                    print(
                        f"[MT-ABC][{ab_name}][{profile['profile_name']}] "
                        f"Phase C cache pool ready raw_n={len(rule_pool_raw)} "
                        f"aggregated_n={len(rule_pool)}",
                        flush=True,
                    )

                for t in targets:
                    print(f"[MT-ABC][{ab_name}][Main pass] START {t['target_id']} / {t['target_name']}", flush=True)
                    ext_rules = []
                    phasec_rules_for_log = []
                    if ab_cfg["relation_aware_phaseC"]:
                        phasec_weight_power = float(
                            self.multitarget_params.get(
                                "PHASEC_RELATION_WEIGHT_POWER", 1.0
                            )
                        )
                        phasec_matrix = np.power(
                            np.clip(self.phaseC_transfer_matrix, 0.0, None),
                            phasec_weight_power,
                        )
                        phasec_max_rules = int(
                            self.multitarget_params.get(
                                "PHASEC_MAX_RULES_PER_TARGET",
                                PHASEC_MAX_RULES_PER_TARGET,
                            )
                        )
                        if t["target_name"] == "later_icu_same_hadm":
                            phasec_max_rules = max(
                                1,
                                int(round(
                                    phasec_max_rules
                                    * float(self.multitarget_params.get(
                                        "EVENT4_PHASEC_MAX_RULES_MULTIPLIER", 1.0
                                    ))
                                )),
                            )
                        ext_rules = get_transferable_phaseC_rules(
                            t["target_name"],
                            rule_pool,
                            ENDPOINTS,
                            phasec_matrix,
                            min_transfer_weight=float(
                                self.multitarget_params.get(
                                    "PHASEC_MIN_TRANSFER_WEIGHT",
                                    PHASEC_MIN_TRANSFER_WEIGHT,
                                )
                            ),
                            max_rules=phasec_max_rules,
                        )
                        try:
                            train_df_for_gate = pd.read_excel(t["train_file"])
                            n_events_for_gate = int(pd.to_numeric(train_df_for_gate.get("event", pd.Series([], dtype=float)), errors="coerce").fillna(0).sum())
                            gated_rules = [
                                evaluate_transferred_phaseC_rule_in_target(
                                    r,
                                    t["target_name"],
                                    train_df_for_gate,
                                    cox_config={"n_train": len(train_df_for_gate), "n_events": n_events_for_gate, "event_col": "event"},
                                )
                                for r in ext_rules
                            ]
                        except Exception as exc:
                            print(f"[MT-ABC][{ab_name}][PhaseC support warning] target={t['target_name']} support gate failed to run: {exc}", flush=True)
                            gated_rules = ext_rules
                        phasec_rules_for_log = list(gated_rules)
                        ext_rules = [r for r in gated_rules if bool(r.get("passed_support_gate", True))]
                        relation_scores = [legacy.to_float(r.get("relation_score")) for r in ext_rules if math.isfinite(legacy.to_float(r.get("relation_score")))]
                        source_distribution = Counter(s for r in ext_rules for s in (r.get("source_endpoints") if isinstance(r.get("source_endpoints"), list) else [r.get("source_endpoint")]))
                        print(
                            f"[MT-ABC][{ab_name}][PhaseC transfer] target={t['target_name']} received_rules={len(ext_rules)} "
                            f"mean_relation_score={(float(np.mean(relation_scores)) if relation_scores else math.nan):.4f} "
                            f"sources={dict(source_distribution)}",
                            flush=True,
                        )
                        for r in phasec_rules_for_log:
                            print(
                                f"[MT-ABC][{ab_name}][PhaseC transfer] {r.get('source_endpoints')} -> {t['target_name']} "
                                f"pair={r.get('interaction_pair')} weight={r.get('transfer_weight')} relation_score={r.get('relation_score')} "
                                f"support_pass={r.get('passed_support_gate')}",
                                flush=True,
                            )
                            phaseC_log.append({
                                **r,
                                "ablation": ab_name,
                                "profile_name": profile["profile_name"],
                                "whether_selected": None,
                                "target_loglik_improvement": None,
                                "target_cv_cindex_delta": None,
                                "target_coefficient": None,
                            })
                    raw_ext_trees = phaseA_trees.get(t["target_name"], []) if ab_cfg["relation_aware_phaseA"] else []
                    phasea_max_trees = int(
                        self.multitarget_params.get(
                            "MULTITARGET_PHASEA_MAX_TRANSFER_TREES_PER_TARGET", 100
                        )
                    )
                    if t["target_name"] == "later_icu_same_hadm":
                        phasea_max_trees = max(
                            1,
                            int(round(
                                phasea_max_trees
                                * float(self.multitarget_params.get(
                                    "EVENT4_PHASEA_MAX_TREES_MULTIPLIER", 1.0
                                ))
                            )),
                        )
                    raw_ext_trees = raw_ext_trees[:phasea_max_trees]
                    if raw_ext_trees:
                        expected_gp_input_vars = phaseA_expected_gp_input_vars_by_endpoint.get(t["target_name"], [])
                        ext_trees, precheck_rows = precheck_phaseA_external_trees(t["target_name"], raw_ext_trees, expected_gp_input_vars)
                        for pre_row in precheck_rows:
                            pre_row.update({"ablation": ab_name, "profile_name": profile["profile_name"]})
                        phaseA_precheck_log.extend(precheck_rows)
                        precheck_by_key = {
                            (str(r.get("tree_group_id", "")), str(r.get("tree_id", "")), str(r.get("expression", ""))): r
                            for r in precheck_rows
                        }
                        for log_row in phaseA_tree_candidate_log:
                            if log_row.get("target_endpoint") != t["target_name"]:
                                continue
                            key = (
                                str(log_row.get("tree_group_id", "")),
                                str(log_row.get("tree_id", "")),
                                str(log_row.get("expression", "")),
                            )
                            pre = precheck_by_key.get(key)
                            if pre:
                                log_row["phaseA_precheck_injectable"] = bool(pre.get("injectable"))
                                log_row["phaseA_precheck_reason"] = pre.get("reason")
                                log_row["phaseA_precheck_mapped_expression"] = pre.get("mapped_expression")
                                log_row["phaseA_precheck_missing_features"] = pre.get("missing_features")
                        if not ext_trees:
                            print(
                                f"[MT-ABC][PhaseA precheck warning] target={t['target_name']} no injectable transferred trees after strict feature-name mapping.",
                                flush=True,
                            )
                        else:
                            print(
                                f"[MT-ABC][PhaseA precheck] target={t['target_name']} "
                                f"injectable={len(ext_trees)}/{len(raw_ext_trees)}",
                                flush=True,
                            )
                    else:
                        ext_trees = []
                    run_dir = self.output_root / "profiles" / ab_name / profile["profile_name"] / t["target_id"]
                    cand_entry = {
                        "phaseA_external_trees": ext_trees,
                        "phaseC_external_rules": ext_rules,
                        "phaseA_external_trees_json": str((run_dir / "phaseA_external_trees.json").resolve()) if ext_trees else "",
                        "phaseC_external_rules_json": str((run_dir / "phaseC_external_rules.json").resolve()) if ext_rules else "",
                        "phaseA_external_trees_n": len(ext_trees),
                        "phaseC_external_rules_n": len(ext_rules),
                    }
                    self.locked_candidate_registry.setdefault(ab_name, {}).setdefault(profile["profile_name"], {})[t["target_name"]] = cand_entry
                    print(
                        f"[MT-ABC][{ab_name}][Locked candidates] {t['target_name']} "
                        f"phaseA_trees={len(ext_trees)} phaseC_rules={len(ext_rules)}",
                        flush=True,
                    )
                    seed = self._seed(
                        ab_name,
                        str(profile.get("seed_profile_name") or profile["profile_name"]),
                        t["target_id"],
                        "main",
                    )
                    status = self.pipeline.fit_internal_cv(
                        t["train_file"], t, self._profile_params(
                            profile, ab_name, ab_cfg, t["target_name"]
                        ), run_dir,
                        phaseA_external_trees=ext_trees, phaseC_external_rules=ext_rules,
                        timeout=self.timeout, force=self.force, seed=seed,
                    )
                    sanitize_internal_placeholder_external_metrics(Path(status["result_json"]))
                    main_gen_rows = extract_phasec_generation_scores(Path(status["result_json"]), ab_name, profile["profile_name"], t["target_name"], "main")
                    phaseC_generation_log.extend(main_gen_rows)
                    print_phasec_generation_scores(main_gen_rows, f"[MT-ABC][{ab_name}][Main pass][{t['target_name']}]")
                    phaseC_log = update_phaseC_transfer_log_after_target_run(
                        phaseC_log,
                        Path(status["result_json"]),
                        t["target_name"],
                        ablation=ab_name,
                        profile_name=profile["profile_name"],
                    )
                    phasec_inj = phaseC_injection_info(Path(status["result_json"]))
                    row = legacy.build_long_row({"profile_id": profile["profile_id"], "profile_name": f"{ab_name}_{profile['profile_name']}"}, t, status)
                    row["ablation"] = ab_name
                    b = self.baseline_by_endpoint.get(t["target_name"], {})
                    row["baseline_internal_for_report"] = b.get("baseline_internal_for_report")
                    row["baseline_external_for_report"] = b.get("baseline_external_for_report")
                    row["baseline_source_for_report"] = b.get("baseline_source_for_report")
                    row["baseline_internal_cindex"] = row["baseline_internal_for_report"]
                    sref = self.single_endpoint_reference_by_endpoint.get(t["target_name"], {})
                    row["single_endpoint_ABC_internal"] = sref.get("single_endpoint_ABC_internal")
                    row["single_endpoint_ABC_external"] = sref.get("single_endpoint_ABC_external")
                    row["single_endpoint_reference_source"] = sref.get("single_endpoint_reference_source")
                    row["delta_ABC_minus_cox_baseline"] = legacy.to_float(row.get("ABC_internal_cindex")) - legacy.to_float(row.get("baseline_internal_for_report"))
                    row["delta_ABC_minus_baseline"] = row["delta_ABC_minus_cox_baseline"]
                    row["delta_ABC_minus_single_endpoint_ABC"] = legacy.to_float(row.get("ABC_internal_cindex")) - legacy.to_float(row.get("single_endpoint_ABC_internal"))
                    row["phaseA_transfer_n"] = len(ext_trees)
                    inj = phaseA_injection_info(Path(status["result_json"]))
                    row.update(inj)
                    failed_by_expr: dict[str, str] = {}
                    failed_by_key: dict[tuple[str, str], str] = {}
                    for x in (inj.get("phaseA_parse_failed_items") or []):
                        if not isinstance(x, dict):
                            continue
                        expr = str(x.get("expression", ""))
                        reason = str(x.get("reason", ""))
                        failed_by_expr[expr] = reason
                        gid = str(x.get("tree_group_id", ""))
                        if gid:
                            failed_by_key[(gid, expr)] = reason

                    injected_by_expr: dict[str, dict[str, Any]] = {}
                    injected_by_key: dict[tuple[str, str], dict[str, Any]] = {}
                    for x in (inj.get("phaseA_injected_items") or []):
                        if not isinstance(x, dict):
                            continue
                        expr = str(x.get("expression", ""))
                        injected_by_expr[expr] = x
                        gid = str(x.get("tree_group_id", ""))
                        if gid:
                            injected_by_key[(gid, expr)] = x
                    for log_row in phaseA_tree_candidate_log:
                        if log_row.get("target_endpoint") == t["target_name"]:
                            expr = str(log_row.get("expression", ""))
                            gid = str(log_row.get("tree_group_id", ""))
                            key = (gid, expr)
                            injected_item = injected_by_key.get(key) if gid else None
                            if injected_item is None:
                                injected_item = injected_by_expr.get(expr)
                            failed_reason = failed_by_key.get(key, "") if gid else ""
                            if not failed_reason:
                                failed_reason = failed_by_expr.get(expr, "")
                            log_row["injected_to_notebook"] = injected_item is not None
                            log_row["parse_success"] = injected_item is not None
                            log_row["parse_failed_reason"] = failed_reason
                            injected_item = injected_item or {}
                            log_row["notebook_mapped_expression"] = injected_item.get("mapped_expression")
                            log_row["notebook_named_expression"] = injected_item.get("named_expression")
                            log_row["current_gp_input_vars"] = json.dumps(injected_item.get("current_gp_input_vars", []), ensure_ascii=False)
                    matched_phaseA = match_transferred_trees_to_final_gp(
                        [r for r in phaseA_tree_candidate_log if r.get("target_endpoint") == t["target_name"]],
                        inj.get("final_gp_expressions") or [],
                    )
                    selected_by_expr = {_norm_expr(r.get("expression")): r for r in matched_phaseA if r.get("final_selected_in_cox")}
                    selected_by_key = {
                        (str(r.get("tree_group_id", "")), _norm_expr(r.get("expression"))): r
                        for r in matched_phaseA
                        if r.get("final_selected_in_cox") and str(r.get("tree_group_id", ""))
                    }
                    for log_row in phaseA_tree_candidate_log:
                        if log_row.get("target_endpoint") == t["target_name"]:
                            norm_expr = _norm_expr(log_row.get("expression"))
                            gid = str(log_row.get("tree_group_id", ""))
                            match = selected_by_key.get((gid, norm_expr)) if gid else None
                            if match is None:
                                match = selected_by_expr.get(norm_expr)
                            log_row["final_selected_in_cox"] = match is not None
                            log_row["survived_filtering"] = match is not None
                            log_row["final_cox_coefficient"] = match.get("final_cox_coefficient") if match else None
                            log_row["final_gp_rank"] = match.get("final_gp_rank") if match else None
                    tree_rows_by_group: dict[str, list[dict[str, Any]]] = {}
                    for tree_row in phaseA_tree_candidate_log:
                        if tree_row.get("target_endpoint") == t["target_name"]:
                            tree_rows_by_group.setdefault(str(tree_row.get("tree_group_id")), []).append(tree_row)
                    for group_row in phaseA_top_candidate_log:
                        if group_row.get("target_endpoint") != t["target_name"]:
                            continue
                        group_trees = tree_rows_by_group.get(str(group_row.get("tree_group_id")), [])
                        group_row["injected_to_notebook"] = bool(group_trees) and all(bool(x.get("injected_to_notebook")) for x in group_trees)
                        group_row["parse_success"] = bool(group_trees) and all(bool(x.get("parse_success")) for x in group_trees)
                        group_row["final_selected_in_cox_n"] = sum(1 for x in group_trees if x.get("final_selected_in_cox"))
                        group_row["final_selected_in_cox_any"] = any(bool(x.get("final_selected_in_cox")) for x in group_trees)
                    final_selected_items = [
                        {
                            "expression": r.get("expression"),
                            "mapped_expression": r.get("notebook_mapped_expression"),
                            "named_expression": r.get("notebook_named_expression"),
                            "final_cox_coefficient": r.get("final_cox_coefficient"),
                            "final_gp_rank": r.get("final_gp_rank"),
                        }
                        for r in phaseA_tree_candidate_log
                        if r.get("target_endpoint") == t["target_name"] and r.get("final_selected_in_cox")
                    ]
                    target_phasea = [r for r in phaseA_tree_candidate_log if r.get("target_endpoint") == t["target_name"]]
                    row["phaseA_top_candidate_tree_n"] = len(target_phasea)
                    row["phaseA_generated_by_transfer_tree_n"] = sum(1 for r in target_phasea if bool(r.get("generated_by_transfer")))
                    row["phaseA_generated_transfer_n"] = row["phaseA_generated_by_transfer_tree_n"]
                    notebook_matched_n = int(inj.get("phaseA_final_selected_n", 0) or 0)
                    runner_matched_n = len(final_selected_items)
                    row["phaseA_final_selected_n_notebook"] = notebook_matched_n
                    row["phaseA_final_selected_n_runner"] = runner_matched_n
                    row["phaseA_final_selected_transferred_n"] = max(notebook_matched_n, runner_matched_n)
                    injected_n = legacy.to_float(inj.get("phaseA_injected_n"))
                    row["phaseA_injected_but_filtered_out_n"] = int(inj.get("phaseA_injected_but_filtered_out_n", max(0, int(injected_n if math.isfinite(injected_n) else 0) - len(final_selected_items))) or 0)
                    deltas = [legacy.to_float(r.get("cox_cv_mean_delta_cindex")) for r in target_phasea if math.isfinite(legacy.to_float(r.get("cox_cv_mean_delta_cindex")))]
                    minfolds = [legacy.to_float(r.get("cox_cv_min_fold_cindex")) for r in target_phasea if math.isfinite(legacy.to_float(r.get("cox_cv_min_fold_cindex")))]
                    row["phaseA_best_cox_cv_delta_cindex"] = max(deltas) if deltas else math.nan
                    row["phaseA_best_min_fold_cindex"] = max(minfolds) if minfolds else math.nan
                    phaseA_injection_log.append({
                        "ablation": ab_name,
                        "profile_name": profile["profile_name"],
                        "target_endpoint": t["target_name"],
                        "phaseA_attempted_n": inj.get("phaseA_attempted_n"),
                        "phaseA_injected_n": inj.get("phaseA_injected_n"),
                        "phaseA_parse_failed_n": inj.get("phaseA_parse_failed_n"),
                        "phaseA_precheck_passed_n": inj.get("phaseA_precheck_passed_n"),
                        "phaseA_precheck_failed_n": inj.get("phaseA_precheck_failed_n"),
                        "phaseA_current_gp_input_vars": json.dumps(inj.get("phaseA_current_gp_input_vars", []), ensure_ascii=False),
                        "phaseA_external_tree_source_json": inj.get("phaseA_external_tree_source_json"),
                        "phaseA_final_selected_n_notebook": row.get("phaseA_final_selected_n_notebook"),
                        "phaseA_final_selected_n_runner": row.get("phaseA_final_selected_n_runner"),
                        "phaseA_final_selected_transferred_n": row.get("phaseA_final_selected_transferred_n"),
                        "phaseA_top_candidate_tree_n": row.get("phaseA_top_candidate_tree_n"),
                        "phaseA_generated_by_transfer_tree_n": row.get("phaseA_generated_by_transfer_tree_n"),
                        "phaseA_generated_transfer_n": row.get("phaseA_generated_transfer_n"),
                        "phaseA_injected_but_filtered_out_n": row.get("phaseA_injected_but_filtered_out_n"),
                        "phaseA_best_cox_cv_delta_cindex": row.get("phaseA_best_cox_cv_delta_cindex"),
                        "phaseA_best_min_fold_cindex": row.get("phaseA_best_min_fold_cindex"),
                        "phaseA_final_selected_items": json.dumps(final_selected_items, ensure_ascii=False),
                        "result_json_path": status.get("result_json"),
                    })
                    print(
                        f"[MT-ABC][{ab_name}][PhaseA injection] {t['target_name']} "
                        f"attempted={inj.get('phaseA_attempted_n')} injected={inj.get('phaseA_injected_n')} "
                        f"parse_failed={inj.get('phaseA_parse_failed_n')} "
                        f"final_selected_transferred={row.get('phaseA_final_selected_transferred_n')} "
                        f"filtered_out={row.get('phaseA_injected_but_filtered_out_n')}",
                        flush=True,
                    )
                    row["phaseC_transferred_rule_n"] = len(ext_rules)
                    target_phasec_log = [
                        r for r in phaseC_log
                        if r.get("target_endpoint") == t["target_name"]
                        and r.get("ablation") == ab_name
                        and r.get("profile_name") == profile["profile_name"]
                    ]
                    relation_scores = [legacy.to_float(r.get("relation_score")) for r in target_phasec_log if math.isfinite(legacy.to_float(r.get("relation_score")))]
                    row["phaseC_transferred_rule_n"] = len(target_phasec_log)
                    row["phaseC_raw_rule_pool_n"] = len(rule_pool_raw)
                    row["phaseC_aggregated_rule_pool_n"] = len(rule_pool)
                    row["phaseC_injected_rule_n"] = phasec_inj.get("phaseC_injected_rule_n")
                    row["phaseC_found_in_result_json_n"] = sum(1 for r in target_phasec_log if bool(r.get("found_in_result_json")))
                    row["phaseC_selected_transferred_rule_n"] = sum(1 for r in target_phasec_log if bool(r.get("whether_selected")))
                    row["phaseC_final_selected_transferred_rule_n"] = sum(1 for r in target_phasec_log if bool(r.get("final_selected")))
                    row["phaseC_transferred_selected_n"] = row["phaseC_selected_transferred_rule_n"]
                    row["phaseC_mean_relation_score"] = float(np.mean(relation_scores)) if relation_scores else math.nan
                    row["phaseC_max_relation_score"] = max(relation_scores) if relation_scores else math.nan
                    row["phaseC_multi_source_rule_n"] = sum(1 for r in target_phasec_log if int(legacy.to_float(r.get("source_count")) if math.isfinite(legacy.to_float(r.get("source_count"))) else 0) > 1)
                    row["phaseC_support_gate_pass_n"] = sum(1 for r in target_phasec_log if bool(r.get("passed_support_gate")))
                    row["phaseC_support_gate_fail_n"] = sum(1 for r in target_phasec_log if r.get("passed_support_gate") is False)
                    phaseC_rule_injection_log.append({
                        "ablation": ab_name,
                        "profile_name": profile["profile_name"],
                        "target_endpoint": t["target_name"],
                        "phaseC_attempted_rule_n": len(ext_rules),
                        "phaseC_transfer_candidate_n": len(target_phasec_log),
                        "phaseC_injected_rule_n": phasec_inj.get("phaseC_injected_rule_n"),
                        "phaseC_injection_failed_n": phasec_inj.get("phaseC_injection_failed_n"),
                        "phaseC_injection_failed_items": json.dumps(phasec_inj.get("phaseC_injection_failed_items", []), ensure_ascii=False, default=str),
                        "result_json_path": status.get("result_json"),
                    })
                    phaseC_rule_selection_log.append({
                        "ablation": ab_name,
                        "profile_name": profile["profile_name"],
                        "target_endpoint": t["target_name"],
                        "phaseC_transferred_rule_n": row["phaseC_transferred_rule_n"],
                        "phaseC_injected_rule_n": row["phaseC_injected_rule_n"],
                        "phaseC_found_in_result_json_n": row["phaseC_found_in_result_json_n"],
                        "phaseC_selected_transferred_rule_n": row["phaseC_selected_transferred_rule_n"],
                        "phaseC_final_selected_transferred_rule_n": row["phaseC_final_selected_transferred_rule_n"],
                        "phaseC_mean_relation_score": row["phaseC_mean_relation_score"],
                        "phaseC_support_gate_pass_n": row["phaseC_support_gate_pass_n"],
                        "phaseC_support_gate_fail_n": row["phaseC_support_gate_fail_n"],
                        "result_json_path": status.get("result_json"),
                    })
                    row["complexity_penalty_endpoint"] = complexity_from_row(row)
                    long_rows.append(row)
                    print(
                        f"[MT-ABC][{ab_name}][Main pass] DONE {t['target_name']} "
                        f"status={row['status']} A={legacy.to_float(row.get('A_internal_cindex')):.4f} "
                        f"AB={legacy.to_float(row.get('AB_internal_cindex')):.4f} "
                        f"AC={legacy.to_float(row.get('AC_internal_cindex')):.4f} "
                        f"ABC={legacy.to_float(row.get('ABC_internal_cindex')):.4f} "
                        f"baseline={legacy.to_float(row.get('baseline_internal_cindex')):.4f} "
                        f"single_ref={legacy.to_float(row.get('single_endpoint_ABC_internal')):.4f} "
                        f"delta_vs_cox={legacy.to_float(row.get('delta_ABC_minus_cox_baseline')):.4f} "
                        f"delta_vs_single={legacy.to_float(row.get('delta_ABC_minus_single_endpoint_ABC')):.4f} "
                        f"phaseA_transfer_n={row['phaseA_transfer_n']} phaseA_injected_n={row.get('phaseA_injected_n')} "
                        f"phaseC_transfer_n={row['phaseC_transferred_rule_n']} phaseC_injected_n={row.get('phaseC_injected_rule_n')} "
                        f"phaseC_selected_n={row.get('phaseC_transferred_selected_n')} "
                        f"phaseB_allowed={row.get('phaseB_allowed')}",
                        flush=True,
                    )
                    if row["status"] != "success":
                        failed_targets.append({"ablation": ab_name, "profile_name": profile["profile_name"], "target_name": t["target_name"], "error": row.get("error_message")})
                    self._save_running_outputs(
                        long_rows,
                        phaseA_transfer_event_log,
                        phaseA_top_candidate_log,
                        phaseA_tree_candidate_log,
                        phaseC_log,
                        warnings,
                        phaseA_injection_log,
                        phaseC_rule_pool_rows,
                        phaseC_rule_pool_aggregated_rows,
                        phaseC_rule_injection_log,
                        phaseC_rule_selection_log,
                        phaseA_precheck_log,
                    )

        # Pass 1 has fitted every Phase-C K candidate for each endpoint.  When
        # terminal exclusivity is enabled, jointly choose K on aligned eICU OOF
        # patients and replay once with those K values forced.  Only the replay
        # is allowed to lock structures for external validation.
        joint_selection = None
        if (
            bool(self.multitarget_params.get("INTERNAL_EXCLUSIVE_SELECTION_ENABLED", False))
            and not self.multitarget_params.get("INTERNAL_EXCLUSIVE_FORCE_K_BY_ENDPOINT")
        ):
            if len(profiles) != 1 or len(ablations) != 1:
                raise RuntimeError("internal exclusive joint selection requires one profile and one ablation")
            current_map = {str(row["target_name"]): row for row in long_rows}
            joint_selection = select_joint_exclusive_phasec_k(
                current_map,
                fixed_weight=float(self.multitarget_params.get(
                    "INTERNAL_EXCLUSIVE_SELECTION_WEIGHT",
                    DEFAULT_INTERNAL_EXCLUSIVE_WEIGHT,
                )),
                high_quantile=float(self.multitarget_params.get(
                    "INTERNAL_EXCLUSIVE_HIGH_QUANTILE",
                    DEFAULT_INTERNAL_EXCLUSIVE_HIGH_QUANTILE,
                )),
            )
            joint_path = self.output_root / "internal_exclusive_joint_selection.json"
            joint_path.write_text(
                json.dumps(joint_selection, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
            safe_to_excel(
                pd.DataFrame(joint_selection["combination_rows"]),
                self.output_root / "internal_exclusive_joint_selection.xlsx",
            )
            replay_params = dict(self.multitarget_params)
            # Matrix overrides are deliberately separated from
            # ``multitarget_params`` during runner initialisation.  They are
            # nevertheless part of the fitted candidate trajectory and must
            # be restored verbatim for a forced-K replay; otherwise replay
            # silently falls back to the default relation matrices and can
            # reorder Phase-A trees / Phase-C rules.
            replay_params.update(
                json.loads(json.dumps(self.matrix_override_payload, default=_json_default))
            )
            for matrix_key in (
                "phaseA_relation_matrix_override",
                "phaseC_relation_matrix_override",
            ):
                if matrix_key not in replay_params:
                    raise RuntimeError(
                        f"Internal exclusivity replay lost required {matrix_key}"
                    )
                expected_matrix = self.matrix_override_payload.get(matrix_key)
                if json.dumps(
                    replay_params[matrix_key], sort_keys=True, default=_json_default
                ) != json.dumps(expected_matrix, sort_keys=True, default=_json_default):
                    raise RuntimeError(
                        f"Internal exclusivity replay changed {matrix_key}"
                    )
            replay_params["INTERNAL_EXCLUSIVE_FORCE_K_BY_ENDPOINT"] = dict(
                joint_selection["selected_k_by_endpoint"]
            )
            replay_params["INTERNAL_EXCLUSIVE_SELECTION_ENABLED"] = True
            replay_params["INTERNAL_EXCLUSIVE_REPLAY"] = True
            replay_config = self.output_root / "internal_exclusive_replay_config.json"
            replay_config.write_text(
                json.dumps({
                    "scope": "multi_target_only",
                    "base_profile": self.profile_override_payload.get("base_profile", "profile_007_high_mutation"),
                    "single_target_hyperparameters_modified": False,
                    "single_target_algorithm_rerun": False,
                    "params": replay_params,
                }, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
            local_selected_all = dict(joint_selection.get("local_selected_k_by_endpoint", {}))
            for endpoint in ENDPOINTS:
                if endpoint not in local_selected_all:
                    result_path = Path(current_map[endpoint]["result_json_path"])
                    result_payload = load_result_json(result_path)
                    local_selected_all[endpoint] = int(
                        result_payload.get(
                            "phasec_selected_k", result_payload.get("selected_k_cat", 0)
                        )
                    )
            if all(
                int(joint_selection["selected_k_by_endpoint"][endpoint])
                == int(local_selected_all[endpoint])
                for endpoint in ENDPOINTS
            ):
                # The fixed exclusivity rule genuinely participated but chose
                # the already fitted local combination.  Lock the first pass
                # directly; refitting an identical stochastic GP trajectory is
                # unnecessary and can only introduce tie-order drift.
                long_df, profile_df, conflict_df, ablation_df = self._summarize(long_rows)
                selected = profile_df.loc[
                    profile_df.get("selected_profile", False) == True
                ].head(1)
                selected_record = selected.iloc[0].to_dict() if len(selected) else {}
                self._write_final_outputs(
                    long_df, profile_df, conflict_df, ablation_df, [], selected_record,
                    failed_targets, warnings, started, phaseA_generation_log,
                    phaseC_generation_log, phaseA_injection_log, phaseC_rule_pool_rows,
                    phaseC_log, phaseC_rule_pool_aggregated_rows,
                    phaseC_rule_injection_log, phaseC_rule_selection_log,
                    phaseA_transfer_event_log, phaseA_top_candidate_log,
                    phaseA_tree_candidate_log, phaseA_precheck_log,
                )
                identity_audit = {
                    "status": "passed",
                    "mode": "identity_selection_no_replay_needed",
                    "local_selected_k_by_endpoint": local_selected_all,
                    "joint_selected_k_by_endpoint": joint_selection["selected_k_by_endpoint"],
                    "first_pass_structures_locked_directly": True,
                    "external_used_for_selection": False,
                }
                (self.output_root / "internal_exclusive_replay_verification.json").write_text(
                    json.dumps(identity_audit, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                audit_path = self.output_root / "run_audit.json"
                audit = load_result_json(audit_path)
                audit["internal_exclusive_joint_selection"] = joint_selection
                audit["internal_exclusive_replay_verification"] = identity_audit
                audit_path.write_text(
                    json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default),
                    encoding="utf-8",
                )
                return {
                    "selected_profile": selected_record,
                    "failed_targets": failed_targets,
                    "warnings": warnings,
                    "internal_exclusive_joint_selection": joint_selection,
                    "internal_exclusive_replay_verification": identity_audit,
                    "external_used_for_selection": False,
                }
            replay_root = self.output_root / "internal_exclusive_locked_replay"
            replay_runner = MultiTargetABCCompetitionRunner(
                replay_root,
                random_seed=self.random_seed,
                timeout=self.timeout,
                force=True,
                single_endpoint_reference_root=self.single_endpoint_reference_root,
                disable_single_endpoint_workbook_reference=self.disable_single_endpoint_workbook_reference,
                profile_override_json=replay_config,
                trial_id=(self.trial_id + "_exclusive_replay"),
                notebook_template=self.notebook_template,
            )
            replay_result = replay_runner.run(
                profile_set="optuna_search",
                only_profile=profiles[0]["profile_name"],
                only_ablation=next(iter(ablations)),
                skip_final_external=True,
            )
            # A replay is valid only if forcing K does not regenerate a
            # different candidate universe.  Verify every patient-level OOF
            # risk for every K before promoting any structure.
            replay_verification = []
            for target in targets:
                endpoint = str(target["target_name"])
                original_result_path = Path(current_map[endpoint]["result_json_path"])
                replay_result_path = (
                    replay_root
                    / "profiles"
                    / next(iter(ablations))
                    / profiles[0]["profile_name"]
                    / target["target_id"]
                    / "result.json"
                )
                original_result = load_result_json(original_result_path)
                forced_result = load_result_json(replay_result_path)
                expected_k = int(joint_selection["selected_k_by_endpoint"][endpoint])
                actual_k = int(
                    forced_result.get(
                        "phasec_selected_k", forced_result.get("selected_k_cat", -1)
                    )
                )
                if actual_k != expected_k:
                    raise RuntimeError(
                        f"exclusive replay K mismatch for {endpoint}: "
                        f"expected={expected_k}, actual={actual_k}"
                    )
                original_keys = [str(x) for x in original_result.get("oof_row_keys", [])]
                replay_keys = [str(x) for x in forced_result.get("oof_row_keys", [])]
                if original_keys != replay_keys:
                    raise RuntimeError(f"exclusive replay row-key mismatch for {endpoint}")
                original_by_k = original_result.get("oof_abc_risk_by_k", {}) or {}
                replay_by_k = forced_result.get("oof_abc_risk_by_k", {}) or {}
                if set(original_by_k) != set(replay_by_k):
                    raise RuntimeError(f"exclusive replay K inventory mismatch for {endpoint}")
                max_abs = 0.0
                for k in original_by_k:
                    left = np.asarray(original_by_k[k], dtype=float)
                    right = np.asarray(replay_by_k[k], dtype=float)
                    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
                        raise RuntimeError(f"exclusive replay invalid K={k} OOF risk for {endpoint}")
                    max_abs = max(max_abs, float(np.max(np.abs(left - right))))
                if max_abs > 1e-12:
                    raise RuntimeError(
                        f"exclusive replay changed the K-candidate trajectory for {endpoint}: "
                        f"max_abs_oof_difference={max_abs}"
                    )
                replay_verification.append({
                    "endpoint": endpoint,
                    "expected_k": expected_k,
                    "actual_k": actual_k,
                    "n_rows": len(original_keys),
                    "k_values": sorted(int(k) for k in original_by_k),
                    "max_abs_oof_difference": max_abs,
                    "passed": True,
                })
            (self.output_root / "internal_exclusive_replay_verification.json").write_text(
                json.dumps(replay_verification, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # Promote only the locked replay artifacts.  This keeps the public
            # output interface unchanged, so the normal external-only command
            # evaluates exactly the jointly selected structures.
            for artifact in ("selected_profile", "locked_structures", "summaries"):
                source = replay_root / artifact
                destination = self.output_root / artifact
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            shutil.copy2(replay_root / "run_audit.json", self.output_root / "run_audit.json")
            promoted_audit_path = self.output_root / "run_audit.json"
            promoted_audit = load_result_json(promoted_audit_path)
            promoted_audit["internal_exclusive_joint_selection"] = joint_selection
            promoted_audit["internal_exclusive_locked_replay_root"] = str(replay_root.resolve())
            promoted_audit["external_used_for_internal_exclusive_selection"] = False
            promoted_audit["internal_exclusive_replay_verification"] = replay_verification
            promoted_audit_path.write_text(
                json.dumps(promoted_audit, ensure_ascii=False, indent=2, default=_json_default),
                encoding="utf-8",
            )
            return {
                "selected_profile": replay_result.get("selected_profile", {}),
                "failed_targets": replay_result.get("failed_targets", []),
                "warnings": replay_result.get("warnings", []),
                "internal_exclusive_joint_selection": joint_selection,
                "internal_exclusive_replay_root": str(replay_root.resolve()),
                "internal_exclusive_replay_verification": replay_verification,
                "external_used_for_selection": False,
            }

        long_df, profile_df, conflict_df, ablation_df = self._summarize(long_rows)
        if len(profile_df):
            print("[MT-ABC][Competition Score Summary]", flush=True)
            cols = [
                "ablation", "profile_name", "eligible_for_selection", "n_targets_success",
                "mean_delta_vs_cox", "min_delta_vs_cox", "n_positive_vs_cox",
                "mean_delta_vs_single", "min_delta_vs_single", "n_positive_vs_single",
                "exclusive_conflict_penalty", "complexity_penalty", "failure_penalty", "profile_score",
            ]
            print(profile_df[[c for c in cols if c in profile_df.columns]].to_string(index=False), flush=True)
        selected = profile_df.loc[profile_df.get("selected_profile", False) == True].head(1) if len(profile_df) else pd.DataFrame()
        selected_record = selected.iloc[0].to_dict() if len(selected) else {}
        external_rows = []
        if selected_record and not skip_final_external:
            warnings.append(
                "Automatic external validation is disabled. Run Cell 5 from the "
                "locked-structure manifest after Cell 4 completes."
            )

        self._write_final_outputs(
            long_df,
            profile_df,
            conflict_df,
            ablation_df,
            external_rows,
            selected_record,
            failed_targets,
            warnings,
            started,
            phaseA_generation_log,
            phaseC_generation_log,
            phaseA_injection_log,
            phaseC_rule_pool_rows,
            phaseC_log,
            phaseC_rule_pool_aggregated_rows,
            phaseC_rule_injection_log,
            phaseC_rule_selection_log,
            phaseA_transfer_event_log,
            phaseA_top_candidate_log,
            phaseA_tree_candidate_log,
            phaseA_precheck_log,
        )
        return {"selected_profile": selected_record, "failed_targets": failed_targets, "warnings": warnings}

    def _seed(self, *parts: str) -> int:
        h = hashlib.md5("::".join(parts).encode("utf-8")).hexdigest()
        return self.random_seed + int(h[:6], 16) % 100000

    def _profile_params(
        self,
        profile: dict[str, Any],
        ablation: str,
        cfg: dict[str, Any],
        target_name: str | None = None,
    ) -> dict[str, Any]:
        params = dict(profile["params"])
        forced_k_by_endpoint = params.pop(
            "INTERNAL_EXCLUSIVE_FORCE_K_BY_ENDPOINT", {}
        )
        # Selection-only settings are consumed by the multitarget controller,
        # not injected into the endpoint notebook.
        params.pop("INTERNAL_EXCLUSIVE_SELECTION_WEIGHT", None)
        params.pop("INTERNAL_EXCLUSIVE_HIGH_QUANTILE", None)
        params.pop("INTERNAL_EXCLUSIVE_SELECTION_ENABLED", None)
        params.pop("INTERNAL_EXCLUSIVE_REPLAY", None)
        if forced_k_by_endpoint:
            if target_name not in forced_k_by_endpoint:
                raise RuntimeError(
                    f"Missing internally selected Phase-C K for {target_name}"
                )
            params["FORCE_PHASEC_SELECTED_K"] = int(
                forced_k_by_endpoint[target_name]
            )
        if target_name == "later_icu_same_hadm":
            params["COX_PENALIZER"] = float(
                params.get("COX_PENALIZER", 0.01)
            ) * float(
                self.multitarget_params.get(
                    "EVENT4_COX_PENALIZER_MULTIPLIER", 1.0
                )
            )
        params.update({
            "RELATION_AWARE_ABLATION": ablation,
            "RELATION_AWARE_PHASEA_ENABLED": bool(cfg["relation_aware_phaseA"]),
            "RELATION_AWARE_PHASEC_ENABLED": bool(cfg["relation_aware_phaseC"]),
        })
        return params

    def _save_running_outputs(
        self,
        long_rows,
        phaseA_transfer_event_log,
        phaseA_top_candidate_log,
        phaseA_tree_candidate_log,
        phaseC_log,
        warnings,
        phaseA_injection_log=None,
        phaseC_rule_pool_rows=None,
        phaseC_rule_pool_aggregated_rows=None,
        phaseC_rule_injection_log=None,
        phaseC_rule_selection_log=None,
        phaseA_precheck_log=None,
    ) -> None:
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        if long_rows:
            safe_to_excel(pd.DataFrame(long_rows), self.summary_dir / "profile_target_long.xlsx")
        safe_to_excel(pd.DataFrame(phaseA_transfer_event_log), self.summary_dir / "phaseA_transfer_event_log.xlsx")
        safe_to_excel(pd.DataFrame(phaseA_top_candidate_log), self.summary_dir / "phaseA_top_candidate_log.xlsx")
        safe_to_excel(pd.DataFrame(phaseA_tree_candidate_log), self.summary_dir / "phaseA_transfer_log.xlsx")
        (self.summary_dir / "phaseA_transfer_log.json").write_text(
            json.dumps(phaseA_tree_candidate_log, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        safe_to_excel(pd.DataFrame(phaseA_precheck_log or []), self.summary_dir / "phaseA_transfer_precheck_log.xlsx")
        safe_to_excel(pd.DataFrame(phaseA_injection_log or []), self.summary_dir / "phaseA_injection_summary.xlsx")
        safe_to_excel(pd.DataFrame(phaseA_injection_log or []), self.summary_dir / "phaseA_injection_log.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_rule_pool_rows or []), self.summary_dir / "phaseC_rule_pool_raw.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_rule_pool_aggregated_rows or []), self.summary_dir / "phaseC_rule_pool_aggregated.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_rule_pool_aggregated_rows or []), self.summary_dir / "phaseC_rule_pool.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_log), self.summary_dir / "phaseC_rule_transfer_log.xlsx")
        (self.summary_dir / "phaseC_rule_transfer_log.json").write_text(
            json.dumps(phaseC_log, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        safe_to_excel(pd.DataFrame(phaseC_rule_injection_log or []), self.summary_dir / "phaseC_rule_injection_summary.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_rule_injection_log or []), self.summary_dir / "phaseC_rule_injection_log.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_rule_selection_log or []), self.summary_dir / "phaseC_rule_selection_summary.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_rule_selection_log or []), self.summary_dir / "phaseC_rule_selection_log.xlsx")

    def _summarize(self, long_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        long_df = pd.DataFrame(long_rows)
        if len(long_df) and "ablation" in long_df.columns:
            m0 = long_df[long_df["ablation"].astype(str) == "M0"]
            if len(m0):
                m0_ref = {
                    str(r["target_name"]): legacy.to_float(r.get("ABC_internal_cindex"))
                    for r in m0.to_dict(orient="records")
                }
                for idx, row in long_df.iterrows():
                    if not math.isfinite(legacy.to_float(row.get("single_endpoint_ABC_internal"))):
                        ep = str(row.get("target_name"))
                        if ep in m0_ref:
                            long_df.at[idx, "single_endpoint_ABC_internal"] = m0_ref[ep]
                            long_df.at[idx, "single_endpoint_reference_source"] = "current_run_M0_ABC_internal"
                            long_df.at[idx, "delta_ABC_minus_single_endpoint_ABC"] = legacy.to_float(row.get("ABC_internal_cindex")) - m0_ref[ep]
        rows, conflict_rows = [], []
        if len(long_df):
            for (ablation, pname), g in long_df.groupby(["ablation", "profile_name"], dropna=False):
                deltas_cox = pd.to_numeric(g.get("delta_ABC_minus_cox_baseline", g.get("delta_ABC_minus_baseline")), errors="coerce")
                deltas_single = pd.to_numeric(g.get("delta_ABC_minus_single_endpoint_ABC"), errors="coerce")
                target_map = {str(r["target_name"]): r for r in g.to_dict(orient="records")}
                conflict, crows = exclusive_conflict_penalty(target_map)
                for cr in crows:
                    cr.update({"ablation": ablation, "profile_name": pname})
                conflict_rows.extend(crows)
                finite_cox = deltas_cox.dropna()
                finite_single = deltas_single.dropna()
                mean_delta = float(finite_cox.mean()) if len(finite_cox) else math.nan
                min_delta = float(finite_cox.min()) if len(finite_cox) else math.nan
                n_pos = int((finite_cox > 0).sum())
                neg_pen = float(np.sum(np.maximum(0, -finite_cox.to_numpy(float)) ** 2)) if len(finite_cox) else math.nan
                mean_single = float(finite_single.mean()) if len(finite_single) else math.nan
                min_single = float(finite_single.min()) if len(finite_single) else math.nan
                n_pos_single = int((finite_single > 0).sum())
                neg_single = float(np.sum(np.maximum(0, -finite_single.to_numpy(float)) ** 2)) if len(finite_single) else math.nan
                worst_cox_idx = deltas_cox.astype(float).idxmin() if len(finite_cox) else None
                worst_single_idx = deltas_single.astype(float).idxmin() if len(finite_single) else None
                worst_delta_vs_cox_endpoint = str(g.loc[worst_cox_idx, "target_name"]) if worst_cox_idx is not None else ""
                worst_delta_vs_single_endpoint = str(g.loc[worst_single_idx, "target_name"]) if worst_single_idx is not None else ""
                complexity = float(pd.to_numeric(g["complexity_penalty_endpoint"], errors="coerce").fillna(0).sum())
                n_success = int((g["status"] == "success").sum())
                n_failed = int(len(ENDPOINTS) - n_success)
                all_success = n_success == len(ENDPOINTS)
                all_above_cox = all_success and len(finite_cox) == len(ENDPOINTS) and bool((finite_cox > 0).all())
                all_above_single = all_success and len(finite_single) == len(ENDPOINTS) and bool((finite_single > 0).all())
                baseline_missing = len(finite_cox) < len(ENDPOINTS)
                eligible = all_success and not baseline_missing
                not_reason = ""
                if not all_success:
                    not_reason = "not_all_targets_success"
                elif baseline_missing:
                    not_reason = "missing_ordinary_cox_baseline"
                failure_penalty = 0.0 if eligible else abs(SCORE_WEIGHTS["failure_penalty"])
                if not eligible:
                    score = -1e9
                else:
                    score = (
                        SCORE_WEIGHTS["mean_delta_vs_cox"] * mean_delta
                        + SCORE_WEIGHTS["min_delta_vs_cox"] * min_delta
                        + SCORE_WEIGHTS["n_positive_vs_cox"] * n_pos
                        + SCORE_WEIGHTS["negative_delta_vs_cox_penalty"] * neg_pen
                        + SCORE_WEIGHTS["mean_delta_vs_single"] * (mean_single if math.isfinite(mean_single) else 0.0)
                        + SCORE_WEIGHTS["min_delta_vs_single"] * (min_single if math.isfinite(min_single) else 0.0)
                        + SCORE_WEIGHTS["n_positive_vs_single"] * n_pos_single
                        + SCORE_WEIGHTS["negative_delta_vs_single_penalty"] * (neg_single if math.isfinite(neg_single) else 0.0)
                        + SCORE_WEIGHTS["exclusive_conflict_penalty"] * conflict
                        + SCORE_WEIGHTS["complexity_penalty"] * complexity
                    )
                rows.append({
                    "ablation": ablation,
                    "profile_name": pname,
                    "n_targets_success": n_success,
                    "n_targets_failed": n_failed,
                    "all_6_targets_success": all_success,
                    "all_6_above_cox": all_above_cox,
                    "all_6_above_single": all_above_single,
                    "eligible_for_selection": eligible,
                    "not_eligible_reason": not_reason,
                    "failure_penalty": failure_penalty,
                    "mean_delta_vs_cox": mean_delta,
                    "min_delta_vs_cox": min_delta,
                    "worst_delta_vs_cox_endpoint": worst_delta_vs_cox_endpoint,
                    "worst_delta_vs_cox": min_delta,
                    "n_positive_vs_cox": n_pos,
                    "negative_delta_vs_cox_penalty": neg_pen,
                    "mean_delta_vs_single": mean_single,
                    "min_delta_vs_single": min_single,
                    "worst_delta_vs_single_endpoint": worst_delta_vs_single_endpoint,
                    "worst_delta_vs_single": min_single,
                    "n_positive_vs_single": n_pos_single,
                    "negative_delta_vs_single_penalty": neg_single,
                    "mean_delta_vs_baseline": mean_delta,
                    "min_delta_vs_baseline": min_delta,
                    "n_positive_targets": n_pos,
                    "negative_delta_penalty": neg_pen,
                    "exclusive_conflict_penalty": conflict,
                    "complexity_penalty": complexity,
                    "profile_score": float(score),
                })
        profile_df = pd.DataFrame(rows)
        if len(profile_df):
            profile_df = profile_df.sort_values(["eligible_for_selection", "profile_score"], ascending=[False, False]).reset_index(drop=True)
            profile_df["selected_profile"] = False
            eligible_idx = profile_df.index[profile_df["eligible_for_selection"] == True].tolist()
            if eligible_idx:
                profile_df.loc[eligible_idx[0], "selected_profile"] = True
        conflict_df = pd.DataFrame(conflict_rows)
        ablation_df = profile_df.groupby("ablation", as_index=False).agg(best_profile_score=("profile_score", "max"), mean_profile_score=("profile_score", "mean"), n_profiles=("profile_name", "nunique")) if len(profile_df) else pd.DataFrame()
        return long_df, profile_df, conflict_df, ablation_df

    def run_external_only_from_locked(self, profile_set: str = "pilot", only_profile: str | None = None, only_ablation: str | None = None) -> dict[str, Any]:
        """Run Cell 5 from the immutable Cell 4 structure manifest."""
        started = now_iso()
        manifest_path = (
            self.output_root
            / LOCKED_STRUCTURE_DIRNAME
            / LOCKED_STRUCTURE_MANIFEST_NAME
        )
        audit_path = self.output_root / "run_audit.json"
        manifest = load_locked_structure_manifest(manifest_path)
        if audit_path.exists():
            audit = load_result_json(audit_path)
            if audit.get("external_used_for_selection") is not False:
                raise RuntimeError("external_used_for_selection is not False; refusing final external validation.")
        locked_ablation = str(manifest.get("selected_ablation") or "")
        locked_profile = str(manifest.get("selected_profile") or "")
        if only_ablation and locked_ablation != only_ablation:
            raise RuntimeError(f"Locked ablation {locked_ablation} does not match requested {only_ablation}.")
        if only_profile and only_profile not in {locked_profile, locked_profile.replace(f"{locked_ablation}_", "", 1)}:
            raise RuntimeError(f"Locked profile {locked_profile} does not match requested {only_profile}.")
        print("[MT-ABC] Loaded immutable endpoint structures from Cell 4.", flush=True)
        external_rows = run_locked_structure_external_validation(
            manifest_path, self.output_root
        )
        # Cell 4 intentionally writes empty external summary artifacts because
        # MIMIC is not read until the immutable structures are locked.  Cell 5
        # must therefore replace those placeholders with the locked-model
        # evaluation rows; otherwise downstream reporting sees an empty file
        # even though the endpoint evaluations succeeded.
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.selected_dir.mkdir(parents=True, exist_ok=True)
        safe_to_excel(
            pd.DataFrame(external_rows),
            self.summary_dir / "final_external_validation.xlsx",
        )
        (self.selected_dir / "selected_profile_external.json").write_text(
            json.dumps(
                external_rows,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        audit = load_result_json(audit_path) if audit_path.exists() else {}
        audit.update({
            "external_only_from_locked": True,
            "external_only_started_at": started,
            "external_only_finished_at": now_iso(),
            "external_used_for_selection": False,
            "locked_structure_manifest": str(manifest_path.resolve()),
            "locked_structure_loaded": True,
            "full_eicu_refit_only": True,
            "deterministic_refit_only": True,
            "model_selection_disabled": True,
            "structure_selection_disabled": True,
            "parameter_tuning_disabled": True,
            "mimic_transform_only": True,
            "mimic_fit_performed": False,
            "mimic_selection_performed": False,
            "mimic_tuning_performed": False,
            "mimic_used_for_preprocessing_fit": False,
            "locked_structure_immutable": True,
            "mimic_preprocessing_fit": False,
            "mimic_model_fit": False,
            "mimic_selection": False,
            "mimic_tuning": False,
            "final_external_rows_n": len(external_rows),
        })
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        return {
            "locked_structure_manifest": str(manifest_path.resolve()),
            "external_rows_n": len(external_rows),
            "external_used_for_selection": False,
        }

    def _write_final_outputs(
        self,
        long_df,
        profile_df,
        conflict_df,
        ablation_df,
        external_rows,
        selected_record,
        failed_targets,
        warnings,
        started,
        phaseA_generation_log,
        phaseC_generation_log,
        phaseA_injection_log,
        phaseC_rule_pool_rows,
        phaseC_log,
        phaseC_rule_pool_aggregated_rows=None,
        phaseC_rule_injection_log=None,
        phaseC_rule_selection_log=None,
        phaseA_transfer_event_log=None,
        phaseA_top_candidate_log=None,
        phaseA_tree_candidate_log=None,
        phaseA_precheck_log=None,
    ) -> None:
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.selected_dir.mkdir(parents=True, exist_ok=True)
        safe_to_excel(long_df, self.summary_dir / "profile_target_long.xlsx")
        safe_to_excel(profile_df, self.summary_dir / "profile_summary.xlsx")
        safe_to_excel(conflict_df, self.summary_dir / "exclusive_conflict_summary.xlsx")
        safe_to_excel(ablation_df, self.summary_dir / "ablation_summary.xlsx")
        safe_to_excel(pd.DataFrame(phaseA_generation_log), self.summary_dir / "phaseA_generation_score_log.xlsx")
        safe_to_excel(pd.DataFrame(phaseA_generation_log), self.summary_dir / "phaseA_coxaware_generation_log.xlsx")
        safe_to_excel(pd.DataFrame(phaseA_transfer_event_log or []), self.summary_dir / "phaseA_transfer_event_log.xlsx")
        safe_to_excel(pd.DataFrame(phaseA_top_candidate_log or []), self.summary_dir / "phaseA_top_candidate_log.xlsx")
        safe_to_excel(pd.DataFrame(phaseA_tree_candidate_log or []), self.summary_dir / "phaseA_transfer_log.xlsx")
        (self.summary_dir / "phaseA_transfer_log.json").write_text(
            json.dumps(phaseA_tree_candidate_log or [], ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        safe_to_excel(pd.DataFrame(phaseA_precheck_log or []), self.summary_dir / "phaseA_transfer_precheck_log.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_generation_log), self.summary_dir / "phaseC_generation_score_log.xlsx")
        safe_to_excel(pd.DataFrame(phaseA_injection_log), self.summary_dir / "phaseA_injection_summary.xlsx")
        safe_to_excel(pd.DataFrame(phaseA_injection_log), self.summary_dir / "phaseA_injection_log.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_rule_pool_rows), self.summary_dir / "phaseC_rule_pool_raw.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_rule_pool_aggregated_rows or []), self.summary_dir / "phaseC_rule_pool_aggregated.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_rule_pool_aggregated_rows or []), self.summary_dir / "phaseC_rule_pool.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_log), self.summary_dir / "phaseC_rule_transfer_log.xlsx")
        (self.summary_dir / "phaseC_rule_transfer_log.json").write_text(
            json.dumps(phaseC_log, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        safe_to_excel(pd.DataFrame(phaseC_rule_injection_log or []), self.summary_dir / "phaseC_rule_injection_summary.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_rule_injection_log or []), self.summary_dir / "phaseC_rule_injection_log.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_rule_selection_log or []), self.summary_dir / "phaseC_rule_selection_summary.xlsx")
        safe_to_excel(pd.DataFrame(phaseC_rule_selection_log or []), self.summary_dir / "phaseC_rule_selection_log.xlsx")
        safe_to_excel(pd.DataFrame(external_rows), self.summary_dir / "final_external_validation.xlsx")
        internal_rows = long_df[(long_df.get("profile_name") == selected_record.get("profile_name")) & (long_df.get("ablation") == selected_record.get("ablation"))].to_dict(orient="records") if len(long_df) and selected_record else []
        locked_structure_audit = {
            "locked_structure_manifest": "",
            "locked_structure_count": 0,
            "locked_structures_complete": False,
        }
        if selected_record:
            locked_structure_audit = write_locked_structure_manifest(
                self.output_root, selected_record, internal_rows
            )
        selected_config = dict(selected_record or {})
        selected_config.update({
            "phaseA_matrix_override_applied": self.phaseA_matrix_override_applied,
            "phaseC_matrix_override_applied": self.phaseC_matrix_override_applied,
            "phaseA_transfer_matrix_final_values": self.phaseA_transfer_matrix_final.tolist(),
            "phaseC_transfer_matrix_final_values": self.phaseC_transfer_matrix_final.tolist(),
            "matrix_direction": "rows are target endpoints; columns are source endpoints; matrix[target, source] means source -> target",
        })
        if selected_record:
            ab = str(selected_record.get("ablation"))
            pname = str(selected_record.get("profile_name", ""))
            raw_name = pname.replace(f"{ab}_", "", 1)
            endpoint_locked = {}
            for ep, reg in self.locked_candidate_registry.get(ab, {}).get(raw_name, {}).items():
                endpoint_locked[ep] = {
                    "phaseA_external_trees_json": reg.get("phaseA_external_trees_json", ""),
                    "phaseC_external_rules_json": reg.get("phaseC_external_rules_json", ""),
                    "phaseA_external_trees_n": reg.get("phaseA_external_trees_n", 0),
                    "phaseC_external_rules_n": reg.get("phaseC_external_rules_n", 0),
                }
            selected_config["endpoint_locked_candidates"] = endpoint_locked
            selected_config["locked_candidate_registry"] = self.locked_candidate_registry.get(ab, {}).get(raw_name, {})
            selected_config.update(locked_structure_audit)
        (self.selected_dir / "selected_profile_config.json").write_text(json.dumps(selected_config, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        (self.selected_dir / "selected_profile_internal.json").write_text(json.dumps(internal_rows, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        (self.selected_dir / "selected_profile_external.json").write_text(json.dumps(external_rows, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        audit = {
            "started_at": started,
            "finished_at": now_iso(),
            "random_seed": self.random_seed,
            "endpoint_names": ENDPOINTS,
            "endpoint_groups": ENDPOINT_GROUPS,
            "exclusive_groups": EXCLUSIVE_GROUPS,
            "exclusive_conflict_matrix": self.exclusive_conflict_matrix.tolist(),
            "phaseA_transfer_matrix": self.phaseA_transfer_matrix.tolist(),
            "phaseC_transfer_matrix": self.phaseC_transfer_matrix.tolist(),
            "phaseA_matrix_override_applied": self.phaseA_matrix_override_applied,
            "phaseC_matrix_override_applied": self.phaseC_matrix_override_applied,
            "phaseA_transfer_matrix_final_values": self.phaseA_transfer_matrix_final.tolist(),
            "phaseC_transfer_matrix_final_values": self.phaseC_transfer_matrix_final.tolist(),
            "matrix_direction": "rows are target endpoints; columns are source endpoints; matrix[target, source] means source -> target",
            "phaseC_legacy_transfer_matrix": self.phaseC_legacy_transfer_matrix.tolist(),
            "phaseA_transfer_matrix_type": "severity_aware_v1_nonhome_bridge",
            "phaseC_transfer_matrix_type": "rule_aware_v1_nonhome_bridge",
            "exclusive_conflict_matrix_type": "terminal_disposition_exclusive",
            "phaseA_fitness_mode": "cox_cv",
            "phaseC_min_transfer_weight": PHASEC_MIN_TRANSFER_WEIGHT,
            "phaseC_max_rules": PHASEC_MAX_RULES_PER_TARGET,
            "phaseC_pool_mode": PHASEC_POOL_MODE,
            "phaseC_support_gate_mode": PHASEC_SUPPORT_GATE_MODE,
            "score_weights": SCORE_WEIGHTS,
            "baseline_file_path": next((v.get("baseline_file_path") for v in self.baseline_by_endpoint.values() if v.get("baseline_file_path")), ""),
            "single_endpoint_reference_source": next((v.get("single_endpoint_reference_source") for v in self.single_endpoint_reference_by_endpoint.values() if v.get("single_endpoint_reference_source")), ""),
            "single_endpoint_workbook_reference_root": str(self.single_endpoint_reference_root),
            "single_endpoint_workbook_reference_disabled": self.disable_single_endpoint_workbook_reference,
            "notebook_cell_structure_version": "relation_aware_multitarget_abc_v2",
            "external_used_for_selection": False,
            "selection_rule": "profile_score uses only eICU internal CV metrics; MIMIC external is final validation only.",
            "failed_targets": failed_targets,
            "warnings": warnings,
            "selected_profile": selected_record,
            "trial_id": self.trial_id,
            "profile_override_json": (
                str(self.profile_override_json.resolve())
                if self.profile_override_json else ""
            ),
            "notebook_template": (
                str(self.notebook_template.resolve()) if self.notebook_template else ""
            ),
            "multitarget_profile_override": self.multitarget_params,
            "single_target_hyperparameters_modified": False,
            "single_target_algorithm_rerun": False,
        }
        audit.update(locked_structure_audit)
        audit.update(self.knowledge_audit)
        if audit.get("external_metrics_used_for_selection") is not False:
            raise RuntimeError("External metrics were used for internal selection")
        if len(profile_df) and selected_record:
            sel = profile_df[
                (profile_df.get("profile_name") == selected_record.get("profile_name"))
                & (profile_df.get("ablation") == selected_record.get("ablation"))
            ]
            if len(sel):
                audit.update({
                    "worst_delta_vs_cox_endpoint": sel.iloc[0].get("worst_delta_vs_cox_endpoint"),
                    "worst_delta_vs_cox": sel.iloc[0].get("worst_delta_vs_cox"),
                    "worst_delta_vs_single_endpoint": sel.iloc[0].get("worst_delta_vs_single_endpoint"),
                    "worst_delta_vs_single": sel.iloc[0].get("worst_delta_vs_single"),
                })
        (self.output_root / "run_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Relation-aware multitarget GP-Cox ABC runner")
    p.add_argument(
        "--profile-set",
        choices=["pilot", "full", "b_sensitive", "optuna_search"],
        default="pilot",
    )
    p.add_argument("--only-profile", default=None)
    p.add_argument("--only-ablation", choices=list(ABLATIONS), default=None)
    p.add_argument("--output-root", default=str(OUTPUT_ROOT))
    p.add_argument("--timeout", type=int, default=24 * 60 * 60)
    p.add_argument("--random-seed", type=int, default=20260523)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-final-external", action="store_true")
    p.add_argument("--external-only-from-locked", action="store_true")
    p.add_argument("--sanitize-internal-external-fields", action="store_true")
    p.add_argument("--single-endpoint-reference-root", default=str(DEFAULT_SINGLE_ENDPOINT_REFERENCE_ROOT))
    p.add_argument("--disable-single-endpoint-workbook-reference", action="store_true")
    p.add_argument("--profile-override-json", default=None)
    p.add_argument("--trial-id", default=None)
    p.add_argument("--notebook-template", default=None)
    args = p.parse_args(argv)
    runner = MultiTargetABCCompetitionRunner(
        Path(args.output_root),
        random_seed=args.random_seed,
        timeout=args.timeout,
        force=args.force,
        single_endpoint_reference_root=Path(args.single_endpoint_reference_root),
        disable_single_endpoint_workbook_reference=args.disable_single_endpoint_workbook_reference,
        profile_override_json=(
            Path(args.profile_override_json) if args.profile_override_json else None
        ),
        trial_id=args.trial_id,
        notebook_template=(
            Path(args.notebook_template) if args.notebook_template else None
        ),
    )
    try:
        if args.sanitize_internal_external_fields:
            n = sanitize_existing_internal_outputs(Path(args.output_root))
            print(f"Sanitized internal placeholder external metrics in {n} result.json files.")
            return 0
        if args.external_only_from_locked:
            result = runner.run_external_only_from_locked(args.profile_set, args.only_profile, args.only_ablation)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
            return 0
        result = runner.run(args.profile_set, args.only_profile, args.only_ablation, args.skip_final_external)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
        return 0
    except Exception:
        traceback.print_exc()
        return 1
