from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Sequence

import numpy as np
import pandas as pd


MIN_SUPPORT_N = 15
MIN_SUPPORT_EVENTS = 5
MIN_LOGLIK_IMPROVEMENT = 0.1
MIN_CV_DELTA = 0.0

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

FINAL_SELECTED_KEYS = {
    "phasec_stgp_and_final_selected_candidates",
    "phasec_final_selected_candidates",
    "phaseC_selected_candidates",
}
FOLD_SELECTED_KEYS = {
    "phasec_stgp_and_selected_candidates_by_fold",
    "phasec_selected_candidates_by_fold",
}
FEASIBLE_KEYS = {
    "phasec_stgp_and_rule_universe_feasible",
    "phasec_stgp_and_candidate_pool",
    "phaseC_candidate_pool",
    "phasec_all_candidates",
    "phasec_evaluated_candidates",
}


def _as_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows = []
        for x in value:
            if isinstance(x, dict):
                rows.append(x)
            elif isinstance(x, (list, tuple)):
                rows.extend(_as_records(list(x)))
            elif x is not None:
                rows.append({"candidate_name": str(x), "expr": str(x)})
        return rows
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        try:
            return _as_records(json.loads(value))
        except Exception:
            return [{"candidate_name": value, "expr": value}] if value.strip() else []
    return []


def _parse_pair(obj: Any) -> tuple[str, str] | None:
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return str(obj[0]).strip(), str(obj[1]).strip()
    text = str(obj or "")
    text = text.replace("AND::", "")
    m = re.search(r"(.+?)\s+AND\s+(.+)", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def _pair_key(value: Any) -> tuple[str, str] | None:
    pair = _parse_pair(value)
    if not pair:
        return None
    a, b = pair
    if not a or not b or a == b:
        return None
    return tuple(sorted([a, b]))


def _to_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _metric(row: dict[str, Any], names: Sequence[str]) -> float:
    for name in names:
        val = _to_float(row.get(name))
        if math.isfinite(val):
            return val
    return math.nan


def _dedupe_keep_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = (
            row.get("source_endpoint"),
            tuple(row.get("interaction_pair") or []),
            row.get("source_result_key"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def collect_phaseC_rule_pool(
    results_by_endpoint: dict[str, dict[str, Any]],
    max_rules_per_endpoint: int = 100,
    top_k_by_fitness: int = 30,
    top_k_by_loglik: int = 30,
    top_k_by_support_events: int = 30,
    include_final_selected: bool = True,
    include_selected_by_fold: bool = True,
    include_feasible_universe: bool = True,
) -> list[dict[str, Any]]:
    """Collect broad Phase C rule candidates from eICU seed/main result JSONs."""
    pool: list[dict[str, Any]] = []
    for endpoint, result in results_by_endpoint.items():
        endpoint_rows: list[dict[str, Any]] = []
        source_key_counts: Counter[str] = Counter()
        for key in PHASEC_CANDIDATE_KEYS:
            if key in FINAL_SELECTED_KEYS and not include_final_selected:
                continue
            if key in FOLD_SELECTED_KEYS and not include_selected_by_fold:
                continue
            if key in FEASIBLE_KEYS and not include_feasible_universe:
                continue
            rows = _as_records(result.get(key))
            source_key_counts[key] += len(rows)
            for raw in rows:
                cand = dict(raw)
                pair = _pair_key(cand.get("interaction_pair") or cand.get("candidate_name") or cand.get("name") or cand.get("expr"))
                if not pair:
                    continue
                expr = cand.get("candidate_name") or cand.get("name") or cand.get("expr") or f"AND::{pair[0]} AND {pair[1]}"
                source_support_events = _metric(cand, ["support_inter_events", "inter_events", "source_support_events", "target_support_inter_events"])
                endpoint_rows.append({
                    "source_endpoint": endpoint,
                    "operator": "AND",
                    "interaction_pair": list(pair),
                    "expr": str(expr),
                    "source_result_key": key,
                    "source_support_n": _metric(cand, ["support_inter_n", "inter_n", "source_support_n", "target_support_inter_n"]),
                    "source_support_events": source_support_events,
                    "source_fitness": _metric(cand, ["fitness", "maxFit", "hofFit", "source_fitness"]),
                    "source_loglik_improvement": _metric(cand, ["train_loglik_improvement", "loglik_improvement", "source_loglik_improvement"]),
                    "source_cv_cindex_delta": _metric(cand, ["cv_cindex_delta", "valid_cindex_delta", "target_cv_cindex_delta"]),
                    "semantic_valid": cand.get("semantic_valid", True),
                    "appeared_as_final_selected": key in FINAL_SELECTED_KEYS or bool(cand.get("final_selected", False)),
                    "appeared_as_fold_selected": key in FOLD_SELECTED_KEYS or bool(cand.get("selected_by_fold", False)),
                    "appeared_in_feasible_universe": key in FEASIBLE_KEYS,
                    "raw_candidate": json.dumps(cand, ensure_ascii=False, default=str),
                })
        selected = [r for r in endpoint_rows if r["appeared_as_final_selected"]]
        fold_selected = [r for r in endpoint_rows if r["appeared_as_fold_selected"]]
        by_fit = sorted(endpoint_rows, key=lambda r: _to_float(r.get("source_fitness"), -1e9), reverse=True)[:top_k_by_fitness]
        by_loglik = sorted(endpoint_rows, key=lambda r: _to_float(r.get("source_loglik_improvement"), -1e9), reverse=True)[:top_k_by_loglik]
        by_events = sorted(endpoint_rows, key=lambda r: _to_float(r.get("source_support_events"), -1e9), reverse=True)[:top_k_by_support_events]
        merged = _dedupe_keep_order((selected if include_final_selected else []) + (fold_selected if include_selected_by_fold else []) + by_fit + by_loglik + by_events)
        for row in merged[:max_rules_per_endpoint]:
            row["source_key_counts"] = json.dumps(dict(source_key_counts), ensure_ascii=False)
            pool.append(row)
    return pool


def aggregate_phaseC_rule_pool(rule_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate identical AND interaction pairs across source endpoints."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rule_pool:
        key = _pair_key(row.get("interaction_pair") or row.get("expr"))
        if key:
            grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        source_eps = sorted({str(r.get("source_endpoint")) for r in rows if r.get("source_endpoint")})
        fitness = [_to_float(r.get("source_fitness")) for r in rows if math.isfinite(_to_float(r.get("source_fitness")))]
        loglik = [_to_float(r.get("source_loglik_improvement")) for r in rows if math.isfinite(_to_float(r.get("source_loglik_improvement")))]
        events = [_to_float(r.get("source_support_events")) for r in rows if math.isfinite(_to_float(r.get("source_support_events")))]
        supports = [_to_float(r.get("source_support_n")) for r in rows if math.isfinite(_to_float(r.get("source_support_n")))]
        out.append({
            "operator": "AND",
            "interaction_pair": list(key),
            "expr": f"AND::{key[0]} AND {key[1]}",
            "source_endpoint": source_eps[0] if source_eps else "",
            "source_endpoints": source_eps,
            "source_count": len(source_eps),
            "source_rows_n": len(rows),
            "max_source_fitness": max(fitness) if fitness else math.nan,
            "mean_source_fitness": float(np.mean(fitness)) if fitness else math.nan,
            "max_source_loglik_improvement": max(loglik) if loglik else math.nan,
            "mean_source_loglik_improvement": float(np.mean(loglik)) if loglik else math.nan,
            "total_source_support_events": float(np.sum(events)) if events else 0.0,
            "max_source_support_events": max(events) if events else math.nan,
            "max_source_support_n": max(supports) if supports else math.nan,
            "appeared_as_final_selected_n": sum(1 for r in rows if bool(r.get("appeared_as_final_selected"))),
            "appeared_as_fold_selected_n": sum(1 for r in rows if bool(r.get("appeared_as_fold_selected"))),
            "appeared_in_feasible_universe_n": sum(1 for r in rows if bool(r.get("appeared_in_feasible_universe"))),
            "semantic_valid": all(r.get("semantic_valid", True) is not False for r in rows),
            "source_evidence_summary": json.dumps({
                "source_endpoints": source_eps,
                "source_result_keys": sorted({str(r.get("source_result_key")) for r in rows if r.get("source_result_key")}),
                "source_count": len(source_eps),
                "source_rows_n": len(rows),
            }, ensure_ascii=False),
            "source_rule_aggregated": True,
        })
    return sorted(out, key=lambda r: (-int(r.get("source_count", 0)), -_to_float(r.get("max_source_fitness"), -1e9), str(r.get("expr", ""))))


def _quality_score(rule: dict[str, Any], all_rules: list[dict[str, Any]]) -> float:
    vals = []
    for key in ["max_source_fitness", "max_source_loglik_improvement", "total_source_support_events"]:
        nums = [_to_float(r.get(key)) for r in all_rules if math.isfinite(_to_float(r.get(key)))]
        val = _to_float(rule.get(key))
        if nums and math.isfinite(val):
            lo, hi = min(nums), max(nums)
            vals.append(0.0 if hi <= lo else (val - lo) / (hi - lo))
    return float(np.mean(vals)) if vals else 0.0


def get_transferable_phaseC_rules(
    target_endpoint: str,
    rule_pool: list[dict[str, Any]],
    endpoint_names: Sequence[str],
    transfer_matrix: np.ndarray,
    min_transfer_weight: float = 0.30,
    max_rules: int = 40,
) -> list[dict[str, Any]]:
    """Select multi-source relation-weighted Phase C rule structures for a target endpoint."""
    idx = {name: i for i, name in enumerate(endpoint_names)}
    if target_endpoint not in idx:
        raise KeyError(target_endpoint)
    aggregated = rule_pool
    if not aggregated or not any(isinstance(r.get("source_endpoints"), list) for r in aggregated):
        aggregated = aggregate_phaseC_rule_pool(rule_pool)
    out = []
    for rule in aggregated:
        if rule.get("semantic_valid", True) is False:
            continue
        pair = _pair_key(rule.get("interaction_pair") or rule.get("expr"))
        if not pair:
            continue
        sources = rule.get("source_endpoints")
        if not isinstance(sources, list):
            src = rule.get("source_endpoint")
            sources = [src] if src else []
        sources = [str(s) for s in sources if str(s) in idx and str(s) != target_endpoint]
        if not sources:
            continue
        weights = [float(transfer_matrix[idx[target_endpoint], idx[source]]) for source in sources]
        if not weights or max(weights) < min_transfer_weight:
            continue
        max_w = float(max(weights))
        mean_w = float(np.mean(weights))
        source_count = int(rule.get("source_count") or len(set(sources)))
        quality = _quality_score(rule, aggregated)
        relation_score = 0.50 * max_w + 0.25 * mean_w + 0.15 * min(1.0, source_count / 3.0) + 0.10 * quality
        new_rule = dict(rule)
        new_rule.update({
            "target_endpoint": target_endpoint,
            "source_endpoint": sources[0],
            "source_endpoints": sources,
            "source_count": source_count,
            "interaction_pair": list(pair),
            "expr": f"AND::{pair[0]} AND {pair[1]}",
            "transfer_type": "relation_aware_phaseC_rule_transfer",
            "transfer_weight": max_w,
            "mean_transfer_weight": mean_w,
            "weighted_source_count": float(sum(weights)),
            "relation_score": float(relation_score),
            "reestimated_in_target": True,
            "source_coefficient_transferred": False,
            "source_rule_aggregated": True,
            "source_evidence_summary": rule.get("source_evidence_summary", json.dumps({"source_endpoints": sources}, ensure_ascii=False)),
        })
        out.append(new_rule)
    return sorted(
        out,
        key=lambda r: (
            -_to_float(r.get("relation_score"), -1e9),
            -int(r.get("source_count", 0)),
            -_to_float(r.get("max_source_fitness"), -1e9),
            -_to_float(r.get("max_source_loglik_improvement"), -1e9),
            -_to_float(r.get("total_source_support_events"), -1e9),
            str(r.get("expr", "")),
        ),
    )[:max_rules]


def adaptive_phaseC_support_gate(n_train: int, n_events: int) -> dict[str, Any]:
    """Same support/events formula for all endpoints; no endpoint-specific threshold."""
    min_support_n = min(30, max(10, int(np.ceil(0.03 * max(0, int(n_train))))))
    min_support_events = min(8, max(3, int(np.ceil(0.03 * max(0, int(n_events))))))
    return {
        "min_support_n": int(min_support_n),
        "min_support_events": int(min_support_events),
        "support_gate_mode": "adaptive_uniform",
    }


def evaluate_transferred_phaseC_rule_in_target(
    rule: dict[str, Any],
    target_endpoint: str,
    train_fold_df: pd.DataFrame,
    val_fold_df: pd.DataFrame | None = None,
    base_design_info: dict[str, Any] | None = None,
    cox_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Re-evaluate transferred rule support/events in target eICU data.

    Cox loglik/C-index and coefficients are still re-estimated by the target
    notebook pipeline. No MIMIC fields are read here.
    """
    pair = _parse_pair(rule.get("interaction_pair") or rule.get("expr"))
    out = dict(rule)
    out.update({
        "target_endpoint": target_endpoint,
        "operator": "AND",
        "reestimated_in_target": True,
        "source_coefficient_transferred": False,
        "target_support_inter_n": 0,
        "target_support_inter_events": 0,
        "target_loglik_improvement": math.nan,
        "target_cv_cindex_delta": math.nan,
        "target_coefficient": math.nan,
        "target_hr": math.nan,
        "converged": False,
        "whether_selected": False,
        "final_selected": False,
        "reason": "",
    })
    gate_cfg = cox_config or {}
    if "n_train" in gate_cfg or "n_events" in gate_cfg:
        gate = adaptive_phaseC_support_gate(int(gate_cfg.get("n_train", len(train_fold_df))), int(gate_cfg.get("n_events", 0)))
        min_n = gate["min_support_n"]
        min_e = gate["min_support_events"]
        gate_mode = gate["support_gate_mode"]
    else:
        min_n = int(gate_cfg.get("MIN_SUPPORT_N", MIN_SUPPORT_N))
        min_e = int(gate_cfg.get("MIN_SUPPORT_EVENTS", MIN_SUPPORT_EVENTS))
        gate_mode = "static_legacy_fallback"
    out["min_support_n_used"] = min_n
    out["min_support_events_used"] = min_e
    out["support_gate_mode"] = gate_mode
    if not pair:
        out["reason"] = "invalid_rule_pair"
        out["passed_support_gate"] = False
        return out
    a, b = pair
    if a not in train_fold_df.columns or b not in train_fold_df.columns:
        out["reason"] = "missing_terms_in_target"
        out["injection_failed_reason"] = "missing_terms_in_target"
        out["passed_support_gate"] = False
        return out
    event_col = gate_cfg.get("event_col", "event")
    if event_col not in train_fold_df.columns:
        for cand in ["event", "EVENT", "target", target_endpoint]:
            if cand in train_fold_df.columns:
                event_col = cand
                break
    xa = pd.to_numeric(train_fold_df[a], errors="coerce").fillna(0).to_numpy(float) > 0
    xb = pd.to_numeric(train_fold_df[b], errors="coerce").fillna(0).to_numpy(float) > 0
    mask = xa & xb
    events = pd.to_numeric(train_fold_df[event_col], errors="coerce").fillna(0).to_numpy(int) if event_col in train_fold_df.columns else np.zeros(len(train_fold_df), dtype=int)
    out["target_support_inter_n"] = int(mask.sum())
    out["target_support_inter_events"] = int(events[mask].sum())
    passed = out["target_support_inter_n"] >= min_n and out["target_support_inter_events"] >= min_e
    out["passed_support_gate"] = bool(passed)
    out["converged"] = bool(passed)
    out["reason"] = "passed_support_gate_pending_notebook_cox_reestimate" if passed else "failed_support_gate"
    return out
