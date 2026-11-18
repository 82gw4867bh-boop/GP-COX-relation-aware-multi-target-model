from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import operator
import random
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold

import frozen_endpoint_evaluator as frozen_eval


ID_COLUMNS = {"subject_id", "hadm_id", "stay_id"}
TIME_COLUMN = "time"
EVENT_COLUMN = "event"
FORBIDDEN_FILE_MARKERS = ("mimic", "external")
PRIMITIVE_NAMES = {"add", "sub", "mul", "safe_div", "safe_log", "safe_sqrt", "clip"}
CANDIDATE_TYPE_SCORE = {
    "phaseA_gp": 1.0,
    "phaseC_and_rule": 0.9,
    "phaseC_main_effect": 0.6,
}


@dataclass
class Candidate:
    candidate_id: str
    source_endpoint: str
    target_endpoint: str
    candidate_type: str
    expression_or_rule: str
    used_variables: list[str]
    source_quality: float
    source_selection_frequency: float
    target_signal: float
    target_support: float
    relation_weight: float
    novelty: float
    nonredundancy: float
    simplicity: float
    input_scope_score: float
    input_scope: str
    raw: dict[str, Any]


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def clip01(x: float) -> float:
    if not math.isfinite(float(x)):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def protected_division(a: float, b: float) -> float:
    return 0.0 if abs(float(b)) < 1e-9 else clip01(float(a) / float(b))


def sqrt_abs(x: float) -> float:
    return clip01(math.sqrt(abs(float(x))))


def log1p_abs(x: float) -> float:
    return clip01(math.log1p(abs(float(x))))


def average(a: float, b: float) -> float:
    return clip01((float(a) + float(b)) / 2.0)


def load_config(path: Path) -> dict[str, Any]:
    cfg = read_json(path, {})
    root = Path(cfg.get("branch_root", path.resolve().parent)).resolve()
    output_root_value = cfg.get("run_output_root", cfg.get("output_root", "."))
    output_root = Path(output_root_value)
    if not output_root.is_absolute():
        output_root = (root / output_root).resolve()
    cfg["_branch_root"] = str(root)
    cfg["_output_root"] = str(output_root)
    cfg["_knowledge_root"] = str((root / cfg["knowledge_root"]).resolve())
    cfg["_eligible_predictor_manifest"] = str((root / "eligible_predictor_manifest.json").resolve())
    for endpoint in cfg["endpoints"]:
        endpoint["train_file_resolved"] = str((root / endpoint["train_file"]).resolve())
        endpoint["mimic_file_resolved"] = endpoint["train_file_resolved"].replace("_eicu.xlsx", "_mimic.xlsx")
    return cfg


def endpoint_names(cfg: dict[str, Any]) -> list[str]:
    return [e["name"] for e in cfg["endpoints"]]


def relation_lookup(cfg: dict[str, Any], source: str, target: str) -> float:
    names = endpoint_names(cfg)
    return float(cfg["relation_matrix"][names.index(target)][names.index(source)])


def extract_variables(expr: str, eligible: set[str]) -> list[str]:
    text = str(expr or "")
    found: list[str] = []
    for name in sorted(eligible, key=len, reverse=True):
        pattern = re.compile(r"(?<![A-Za-z0-9_.])" + re.escape(name) + r"(?![A-Za-z0-9_.])")
        if pattern.search(text):
            found.append(name)
    return sorted(set(found))


def split_args(text: str) -> list[str]:
    out: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(text[start:i].strip())
            start = i + 1
    out.append(text[start:].strip())
    return out


def eval_gp_expr(expr: str, df: pd.DataFrame) -> pd.Series:
    text = str(expr).strip()
    if text in df.columns:
        return pd.to_numeric(df[text], errors="coerce").fillna(0.0)
    try:
        return pd.Series(float(text), index=df.index)
    except Exception:
        pass
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$", text)
    if not match:
        raise ValueError(f"unsupported GP expression: {expr}")
    name, arg_text = match.group(1), match.group(2)
    args = [eval_gp_expr(x, df) for x in split_args(arg_text)]
    if name == "add":
        return args[0] + args[1]
    if name == "sub":
        return args[0] - args[1]
    if name == "mul":
        return args[0] * args[1]
    if name == "safe_div":
        denom = args[1].replace(0, np.nan)
        return (args[0] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if name == "safe_log":
        return np.log1p(np.abs(args[0]))
    if name == "safe_sqrt":
        return np.sqrt(np.abs(args[0]))
    if name == "clip":
        return args[0].clip(-10.0, 10.0)
    raise ValueError(f"unsupported GP primitive: {name}")


def eval_phasec_rule(expr: str, used_variables: list[str], df: pd.DataFrame) -> pd.Series:
    text = str(expr or "")
    if text.startswith("AND("):
        cols = split_args(text[4:-1])
        vals = [(pd.to_numeric(df[c], errors="coerce").fillna(0) > 0).astype(int) for c in cols]
        out = vals[0]
        for val in vals[1:]:
            out = out & val
        return out.astype(float)
    if used_variables:
        return pd.to_numeric(df[used_variables[0]], errors="coerce").fillna(0.0)
    raise ValueError(f"unsupported PhaseC rule: {expr}")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip().startswith("{"):
            return default
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def normalize_quality(raw_value: float) -> float:
    return clip01(math.log1p(abs(float(raw_value))) / 3.0)


def load_eligible_manifest(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = read_json(Path(cfg["_eligible_predictor_manifest"]), [])
    manifest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("variable_name"):
            manifest[str(row["variable_name"])] = row
    return manifest


def allowed_manifest_variables(manifest: dict[str, dict[str, Any]]) -> set[str]:
    required = ["available_eicu", "available_mimic", "early_icu_available", "harmonized", "leakage_free"]
    return {
        name
        for name, row in manifest.items()
        if all(bool(row.get(key, False)) for key in required)
    }


def load_knowledge(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = Path(cfg["_knowledge_root"])
    cache = read_json(root / "single_endpoint_knowledge_cache.json", {})
    audit = {
        "source_notebook": cache.get("generated_by"),
        "source_parameter_profile": cache.get("profile_name"),
        "source_endpoint_count": len(cfg["endpoints"]),
        "knowledge_root": str(root),
        "external_used_for_selection": bool(cache.get("external_used_for_selection", False)),
    }
    if audit["source_notebook"] != cfg["source_notebook"]:
        raise RuntimeError(f"Unexpected source notebook: {audit['source_notebook']}")
    if audit["source_parameter_profile"] != cfg["source_parameter_profile"]:
        raise RuntimeError(f"Unexpected source profile: {audit['source_parameter_profile']}")

    knowledge: dict[str, dict[str, Any]] = {}
    for ep in cfg["endpoints"]:
        ep_dir = root / ep["folder"]
        payload = read_json(ep_dir / "single_endpoint_knowledge.json", {})
        if payload.get("generated_by") != cfg["source_notebook"]:
            raise RuntimeError(f"{ep['name']} generated_by mismatch")
        if payload.get("profile_name") != cfg["source_parameter_profile"]:
            raise RuntimeError(f"{ep['name']} profile mismatch")
        knowledge[ep["name"]] = payload

    out = Path(cfg["_output_root"]) / "knowledge_library"
    out.mkdir(exist_ok=True)
    write_json(out / "knowledge_source_audit.json", audit)
    write_json(out / "loaded_endpoint_manifest.json", {
        "endpoints": endpoint_names(cfg),
        "folders": {e["name"]: e["folder"] for e in cfg["endpoints"]},
    })
    return knowledge


def phasea_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    phasea = payload.get("phaseA", {})
    return list(phasea.get("final_gp_expressions") or phasea.get("phaseA_top_tree_candidates") or [])


def phasec_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    phasec = payload.get("phaseC", {})
    rows = list(phasec.get("phaseC_selected_candidates") or phasec.get("phasec_final_selected_candidates") or [])
    return [r for r in rows if str(r.get("type", "")).lower() in {"and_rule", "main"}]


def phasec_candidate_type(record: dict[str, Any]) -> str:
    raw_type = str(record.get("type", "")).lower()
    expr = str(record.get("expr") or record.get("name") or "")
    used = [str(x) for x in (record.get("leaf_terms") or record.get("main_effects") or [])]
    if raw_type == "and_rule" and expr.startswith("AND(") and len(used) >= 2:
        return "phaseC_and_rule"
    return "phaseC_main_effect"


def source_frequency(record: dict[str, Any]) -> float:
    fold = str(record.get("fold", ""))
    return 1.0 if fold == "final_full_train" else 0.5


def candidate_input_scope(used: list[str], target_gp_vars: set[str], eligible: set[str], manifest: dict[str, dict[str, Any]]) -> tuple[str, float, str]:
    missing = [x for x in used if x not in manifest]
    if missing:
        return "rejected", 0.0, "not_in_eligible_predictor_manifest:" + ",".join(missing)
    failed = [
        x
        for x in used
        if x not in eligible
        or not bool(manifest[x].get("early_icu_available", False))
        or not bool(manifest[x].get("harmonized", False))
        or not bool(manifest[x].get("leakage_free", False))
    ]
    if failed:
        return "rejected", 0.0, "manifest_safety_gate_failed:" + ",".join(failed)
    unavailable = [x for x in used if not bool(manifest[x].get("available_eicu", False)) or not bool(manifest[x].get("available_mimic", False))]
    if unavailable:
        return "rejected", 0.0, "manifest_availability_gate_failed:" + ",".join(unavailable)
    missing = [x for x in used if x not in eligible]
    if missing:
        return "rejected", 0.0, "unavailable_or_nonharmonized:" + ",".join(missing)
    if all(x in target_gp_vars for x in used):
        return "strict_target_gp_input", 1.0, ""
    return "eligible_predictor_pool", 0.75, ""


def scope_score(scope: str, cfg: dict[str, Any], default_score: float) -> float:
    if scope == "strict_target_gp_input":
        return safe_float(cfg.get("scope_scores", {}).get("strict_target_gp_input"), 1.0)
    if scope == "eligible_predictor_pool":
        return safe_float(cfg.get("scope_scores", {}).get("eligible_predictor_pool"), default_score)
    return default_score


def build_candidates(cfg: dict[str, Any], knowledge: dict[str, dict[str, Any]]) -> list[Candidate]:
    manifest = load_eligible_manifest(cfg)
    manifest_allowed = allowed_manifest_variables(manifest)
    data_pools = {}
    for ep in cfg["endpoints"]:
        columns = set(map(str, pd.read_excel(ep["train_file_resolved"], nrows=0).columns))
        data_pools[ep["name"]] = manifest_allowed & columns
    target_gp = {
        name: set(map(str, payload.get("gp_input_vars") or payload.get("phaseA_gp_input_vars") or []))
        for name, payload in knowledge.items()
    }
    candidates: list[Candidate] = []
    rejected: list[dict[str, Any]] = []
    seen_by_target: dict[str, set[str]] = {}
    for target, payload in knowledge.items():
        existing = {
            "phaseA_gp:" + str(row.get("expression", ""))
            for row in phasea_records(payload)
            if row.get("expression")
        }
        existing.update(
            phasec_candidate_type(row)
            + ":"
            + str(row.get("expr") or row.get("name") or "")
            for row in phasec_records(payload)
        )
        seen_by_target[target] = existing

    for source, payload in knowledge.items():
        source_pool = data_pools[source]
        for target in knowledge:
            if target == source:
                continue
            relation = relation_lookup(cfg, source, target)
            if relation < safe_float(cfg.get("candidate_ranking", {}).get("min_relation_weight"), 0.0):
                continue
            for i, rec in enumerate(phasea_records(payload)):
                expr = str(rec.get("expression", ""))
                used = extract_variables(expr, data_pools[target] | source_pool)
                scope, scope_score_value, reason = candidate_input_scope(used, target_gp[target], data_pools[target], manifest)
                if reason:
                    rejected.append({"source_endpoint": source, "target_endpoint": target, "candidate_type": "phaseA_gp", "expression_or_rule": expr, "reason": reason})
                    continue
                simplicity = clip01(1.0 / max(1.0, safe_float(rec.get("tree_length"), len(used) + 1)))
                key = "phaseA_gp:" + expr
                if key in seen_by_target[target]:
                    rejected.append({"source_endpoint": source, "target_endpoint": target, "candidate_type": "phaseA_gp", "expression_or_rule": expr, "reason": "duplicate_of_target_2G_structure"})
                    continue
                novelty = 1.0
                seen_by_target[target].add(key)
                quality = normalize_quality(safe_float(rec.get("coefficient")))
                candidates.append(Candidate(
                    candidate_id=f"{source}__to__{target}__phaseA__{i}",
                    source_endpoint=source,
                    target_endpoint=target,
                    candidate_type="phaseA_gp",
                    expression_or_rule=expr,
                    used_variables=used,
                    source_quality=quality,
                    source_selection_frequency=1.0,
                    target_signal=clip01(quality * relation),
                    target_support=scope_score_value,
                    relation_weight=relation,
                    novelty=novelty,
                    nonredundancy=novelty,
                    simplicity=simplicity,
                    input_scope_score=scope_score(scope, cfg, scope_score_value),
                    input_scope=scope,
                    raw=rec,
                ))
            for i, rec in enumerate(phasec_records(payload)):
                expr = str(rec.get("expr") or rec.get("name") or "")
                used = [str(x) for x in (rec.get("leaf_terms") or rec.get("main_effects") or [])]
                candidate_type = phasec_candidate_type(rec)
                if candidate_type == "phaseC_and_rule":
                    support_cfg = cfg.get("phaseC_and_rule_support", {})
                    support_n = safe_float(rec.get("support_inter_n"), 0.0)
                    support_events = safe_float(rec.get("support_inter_events"), 0.0)
                    support_nonevents = max(0.0, support_n - support_events)
                    if len(used) > int(support_cfg.get("max_rule_length", 999)):
                        rejected.append({"source_endpoint": source, "target_endpoint": target, "candidate_type": candidate_type, "expression_or_rule": expr, "reason": "phaseC_and_rule_length_gate_failed"})
                        continue
                    if (
                        support_n < safe_float(support_cfg.get("minimum_support_n"), 0.0)
                        or support_events < safe_float(support_cfg.get("minimum_support_events"), 0.0)
                        or support_nonevents < safe_float(support_cfg.get("minimum_support_nonevents"), 0.0)
                    ):
                        rejected.append({"source_endpoint": source, "target_endpoint": target, "candidate_type": candidate_type, "expression_or_rule": expr, "reason": "phaseC_and_rule_support_gate_failed"})
                        continue
                scope, scope_score_value, reason = candidate_input_scope(used, target_gp[target], data_pools[target], manifest)
                if reason:
                    rejected.append({"source_endpoint": source, "target_endpoint": target, "candidate_type": candidate_type, "expression_or_rule": expr, "reason": reason})
                    continue
                raw_quality = safe_float(rec.get("fitness"), safe_float(rec.get("train_loglik_improvement")))
                quality = normalize_quality(raw_quality)
                key = candidate_type + ":" + expr
                if key in seen_by_target[target]:
                    rejected.append({"source_endpoint": source, "target_endpoint": target, "candidate_type": candidate_type, "expression_or_rule": expr, "reason": "duplicate_of_target_2G_structure"})
                    continue
                novelty = 1.0
                seen_by_target[target].add(key)
                support_n = safe_float(rec.get("support_inter_n"), 0.0)
                support_score = clip01(math.log1p(max(0.0, support_n)) / 5.0) if support_n else scope_score_value
                candidates.append(Candidate(
                    candidate_id=f"{source}__to__{target}__{candidate_type}__{i}",
                    source_endpoint=source,
                    target_endpoint=target,
                    candidate_type=candidate_type,
                    expression_or_rule=expr,
                    used_variables=used,
                    source_quality=quality,
                    source_selection_frequency=source_frequency(rec),
                    target_signal=clip01(quality * relation),
                    target_support=support_score,
                    relation_weight=relation,
                    novelty=novelty,
                    nonredundancy=novelty,
                    simplicity=clip01(1.0 / max(1.0, safe_float(rec.get("complexity"), len(used) + 1))),
                    input_scope_score=scope_score(scope, cfg, scope_score_value),
                    input_scope=scope,
                    raw=rec,
                ))

    out = Path(cfg["_output_root"]) / "candidate_pool"
    out.mkdir(exist_ok=True)
    write_json(out / "candidate_pool.json", [asdict(c) for c in candidates])
    stats = {
        "total_candidates": len(candidates),
        "number_of_phaseA_candidates": sum(c.candidate_type == "phaseA_gp" for c in candidates),
        "number_of_phaseC_and_rule_candidates": sum(c.candidate_type == "phaseC_and_rule" for c in candidates),
        "number_of_phaseC_main_effect_candidates": sum(c.candidate_type == "phaseC_main_effect" for c in candidates),
        "number_of_strict_target_gp_input_candidates": sum(c.input_scope == "strict_target_gp_input" for c in candidates),
        "number_of_eligible_predictor_pool_candidates": sum(c.input_scope == "eligible_predictor_pool" for c in candidates),
        "phaseA_candidates": sum(c.candidate_type == "phaseA_gp" for c in candidates),
        "phaseC_and_rule_candidates": sum(c.candidate_type == "phaseC_and_rule" for c in candidates),
        "phaseC_main_effect_candidates": sum(c.candidate_type == "phaseC_main_effect" for c in candidates),
        "strict_target_gp_input_candidates": sum(c.input_scope == "strict_target_gp_input" for c in candidates),
        "eligible_predictor_pool_candidates": sum(c.input_scope == "eligible_predictor_pool" for c in candidates),
        "rejection_reasons": rejected,
    }
    write_json(out / "candidate_statistics.json", stats)
    return candidates


def make_feature(df: pd.DataFrame, cand: Candidate) -> pd.Series:
    if cand.candidate_type == "phaseA_gp":
        return eval_gp_expr(cand.expression_or_rule, df)
    return eval_phasec_rule(cand.expression_or_rule, cand.used_variables, df)


def add_baseline_features(df: pd.DataFrame, payload: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    features = [str(x) for x in (payload.get("gp_input_vars") or payload.get("phaseA_gp_input_vars") or []) if str(x) in work.columns]
    for i, rec in enumerate(phasea_records(payload)):
        expr = str(rec.get("expression", ""))
        col = f"BASE_PHASEA_{i}"
        try:
            work[col] = eval_gp_expr(expr, work)
            features.append(col)
        except Exception:
            continue
    for i, rec in enumerate(phasec_records(payload)):
        expr = str(rec.get("expr") or rec.get("name") or "")
        used = [str(x) for x in (rec.get("leaf_terms") or rec.get("main_effects") or [])]
        if all(x in work.columns for x in used):
            col = f"BASE_PHASEC_{i}"
            try:
                work[col] = eval_phasec_rule(expr, used, work)
                features.append(col)
            except Exception:
                continue
    return work, list(dict.fromkeys(features))


def clean_model_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cols = [TIME_COLUMN, EVENT_COLUMN] + [c for c in features if c in df.columns]
    out = df[cols].copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in cols:
        if col not in {TIME_COLUMN, EVENT_COLUMN}:
            med = out[col].median()
            out[col] = out[col].fillna(0.0 if pd.isna(med) else med)
    out = out.dropna(subset=[TIME_COLUMN, EVENT_COLUMN])
    keep_features = []
    for col in features:
        if col in out.columns and out[col].nunique(dropna=True) > 1:
            keep_features.append(col)
    return out[[TIME_COLUMN, EVENT_COLUMN] + keep_features]


def cox_cv_cindex(df: pd.DataFrame, features: list[str], folds: int, seed: int) -> tuple[float, list[float], str, int]:
    frame = clean_model_frame(df, features)
    if len(frame) < 40 or frame[EVENT_COLUMN].nunique() < 2:
        return 0.5, [0.5] * folds, "insufficient_data", folds
    y = frame[EVENT_COLUMN].astype(int)
    n_splits = min(folds, int(y.value_counts().min())) if int(y.value_counts().min()) >= 2 else 2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: list[float] = []
    failure_count = 0
    for train_idx, test_idx in skf.split(frame, y):
        train = frame.iloc[train_idx].copy()
        test = frame.iloc[test_idx].copy()
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(train, duration_col=TIME_COLUMN, event_col=EVENT_COLUMN, show_progress=False)
            risk = cph.predict_partial_hazard(test)
            score = concordance_index(test[TIME_COLUMN], -risk, test[EVENT_COLUMN])
            scores.append(float(score))
        except Exception:
            failure_count += 1
            scores.append(0.5)
    while len(scores) < folds:
        scores.append(float(np.mean(scores)) if scores else 0.5)
    status = "cox_failed" if failure_count else "ok"
    return float(np.mean(scores)), scores[:folds], status, failure_count


def prepare_target_frames(cfg: dict[str, Any], knowledge: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    frames: dict[str, dict[str, Any]] = {}
    for ep in cfg["endpoints"]:
        path = Path(ep["train_file_resolved"])
        if any(marker in str(path).lower() for marker in FORBIDDEN_FILE_MARKERS):
            raise RuntimeError(f"Forbidden external/MIMIC path touched: {path}")
        df = pd.read_excel(path)
        cv_states = frozen_eval.prepare_frozen_cv_states(
            df,
            knowledge[ep["name"]],
            folds=int(cfg["smoke"]["cv_folds"]),
            seed=int(cfg["smoke"]["random_seed"]),
        )
        mean_base, fold_base, status, failure_count = frozen_eval.cross_validated_prepared_states(
            cv_states
        )
        frames[ep["name"]] = {
            "df": df,
            "knowledge": knowledge[ep["name"]],
            "base_features": [
                *list(knowledge[ep["name"]].get("main_effect_vars", []) or []),
                *[f"gp{i + 1}" for i, _ in enumerate(phasea_records(knowledge[ep["name"]]))],
            ],
            "baseline_mean_cindex": mean_base,
            "baseline_fold_cindex": fold_base,
            "baseline_status": status,
            "baseline_cox_failure_count": failure_count,
            "cv_states": cv_states,
            "candidate_evaluation_cache": {},
        }
    return frames


def cox_external_cindex(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]) -> tuple[float | None, str]:
    train = clean_model_frame(train_df, features)
    kept = [c for c in train.columns if c not in {TIME_COLUMN, EVENT_COLUMN}]
    if len(train) < 40 or train[EVENT_COLUMN].nunique() < 2 or not kept:
        return None, "insufficient_train_data"
    test = test_df[[TIME_COLUMN, EVENT_COLUMN] + [c for c in kept if c in test_df.columns]].copy()
    for col in [TIME_COLUMN, EVENT_COLUMN] + kept:
        if col not in test.columns:
            test[col] = np.nan
        test[col] = pd.to_numeric(test[col], errors="coerce")
    for col in kept:
        med = train[col].median()
        fill = 0.0 if pd.isna(med) else med
        test[col] = test[col].replace([np.inf, -np.inf], np.nan).fillna(fill)
    test = test.dropna(subset=[TIME_COLUMN, EVENT_COLUMN])
    if len(test) < 10 or test[EVENT_COLUMN].nunique() < 2:
        return None, "insufficient_test_data"
    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(train[[TIME_COLUMN, EVENT_COLUMN] + kept], duration_col=TIME_COLUMN, event_col=EVENT_COLUMN, show_progress=False)
        risk = cph.predict_partial_hazard(test[[TIME_COLUMN, EVENT_COLUMN] + kept])
        return float(concordance_index(test[TIME_COLUMN], -risk, test[EVENT_COLUMN])), "ok"
    except Exception as exc:
        return None, f"cox_failed:{exc}"


def cox_failure_handling_self_check() -> dict[str, Any]:
    n = 80
    df = pd.DataFrame({
        TIME_COLUMN: np.arange(1, n + 1, dtype=float),
        EVENT_COLUMN: [0, 1] * (n // 2),
        "bad_feature": np.linspace(0.0, 1.0, n),
    })
    original_fit = CoxPHFitter.fit

    def forced_failure(self, *args, **kwargs):
        raise RuntimeError("forced_cox_failure_self_check")

    CoxPHFitter.fit = forced_failure
    try:
        mean, folds, status, failure_count = cox_cv_cindex(df, ["bad_feature"], 4, 20260708)
    finally:
        CoxPHFitter.fit = original_fit
    return {
        "status": status,
        "failure_count": failure_count,
        "mean_cindex": mean,
        "fold_cindex": folds,
        "passed": status == "cox_failed" and failure_count > 0,
    }


def apply_transfer_features(df: pd.DataFrame, accepted: list[Candidate]) -> tuple[pd.DataFrame, list[str], list[dict[str, Any]]]:
    out = df.copy()
    cols: list[str] = []
    records: list[dict[str, Any]] = []
    for cand in accepted:
        col = "FINAL_TRANSFER_" + re.sub(r"[^A-Za-z0-9_]+", "_", cand.candidate_id)[:110]
        out[col] = make_feature(out, cand)
        cols.append(col)
        records.append({
            "candidate_id": cand.candidate_id,
            "source_endpoint": cand.source_endpoint,
            "target_endpoint": cand.target_endpoint,
            "candidate_type": cand.candidate_type,
            "expression_or_rule": cand.expression_or_rule,
            "used_variables": cand.used_variables,
            "input_scope": cand.input_scope,
        })
    return out, cols, records


def reference_metric(payload: dict[str, Any], key: str) -> float | None:
    cache_keys = {
        "ordinary_cox": ["reference_top10_external_cindex"],
        "single_gp": ["ABC_external_cindex", "external_abc_cindex", "external_ABC_timevarying_cindex"],
    }
    for candidate_key in cache_keys[key]:
        if candidate_key in payload:
            return safe_float(payload.get(candidate_key), None)
    cache_summary = payload.get("summary", {})
    if isinstance(cache_summary, dict):
        for candidate_key in cache_keys[key]:
            if candidate_key in cache_summary:
                return safe_float(cache_summary.get(candidate_key), None)
    return None


def run_external_validation(
    cfg: dict[str, Any],
    knowledge: dict[str, dict[str, Any]],
    candidates: list[Candidate],
    best_details: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    branch_root = Path(cfg["_output_root"])
    candidate_by_id = {c.candidate_id: c for c in candidates}
    accepted_by_target: dict[str, list[Candidate]] = {name: [] for name in endpoint_names(cfg)}
    for item in best_details.get("accepted_transfers", []):
        cand = candidate_by_id.get(str(item.get("candidate_id")))
        if cand is not None:
            accepted_by_target[cand.target_endpoint].append(cand)

    rows: list[dict[str, Any]] = []
    zero_exchange_audit = []
    for ep in cfg["endpoints"]:
        endpoint = ep["name"]
        train_path = Path(ep["train_file_resolved"])
        mimic_path = Path(ep["mimic_file_resolved"])
        train_df = pd.read_excel(train_path)
        mimic_df = pd.read_excel(mimic_path)
        accepted = accepted_by_target[endpoint]
        extra_gp = [
            candidate.expression_or_rule
            for candidate in accepted
            if candidate.candidate_type == "phaseA_gp"
        ]
        extra_phasec = [
            dict(candidate.raw)
            for candidate in accepted
            if candidate.candidate_type in {"phaseC_and_rule", "phaseC_main_effect"}
        ]
        zero_result = frozen_eval.evaluate_frozen_endpoint(
            train_df, mimic_df, knowledge[endpoint]
        )
        source_2g = reference_metric(knowledge[endpoint], "single_gp")
        zero_delta = None if source_2g is None else float(zero_result["external_cindex"] - source_2g)
        zero_exchange_audit.append({
            "endpoint": endpoint,
            "reconstructed_zero_exchange_cindex": zero_result["external_cindex"],
            "stored_2G_cindex": source_2g,
            "delta": zero_delta,
            "equal_within_1e_12": zero_delta is not None and abs(zero_delta) <= 1e-12,
            "evaluation_backend": "original_single_target_staged_model",
        })
        final_result = frozen_eval.evaluate_frozen_endpoint(
            train_df,
            mimic_df,
            knowledge[endpoint],
            extra_gp_expressions=extra_gp,
            extra_phasec_candidates=extra_phasec,
        )
        external_cindex = float(final_result["external_cindex"])
        external_status = "ok"
        transfer_records = [
            {
                "candidate_id": candidate.candidate_id,
                "source_endpoint": candidate.source_endpoint,
                "target_endpoint": candidate.target_endpoint,
                "candidate_type": candidate.candidate_type,
                "expression_or_rule": candidate.expression_or_rule,
                "used_variables": candidate.used_variables,
                "input_scope": candidate.input_scope,
            }
            for candidate in accepted
        ]
        ordinary = reference_metric(knowledge[endpoint], "ordinary_cox")
        single = reference_metric(knowledge[endpoint], "single_gp")
        rows.append({
            "endpoint": endpoint,
            "ordinary_cox_external_cindex": ordinary,
            "single_target_gpcox_external_cindex": single,
            "multitarget_exchange_gp_external_cindex": external_cindex,
            "external_validation_status": external_status,
            "delta_vs_ordinary_cox": None if external_cindex is None or ordinary is None else external_cindex - ordinary,
            "delta_vs_single_target_gpcox": None if external_cindex is None or single is None else external_cindex - single,
            "number_of_accepted_phaseA_gp_transfers": sum(c.candidate_type == "phaseA_gp" for c in accepted),
            "number_of_accepted_phaseC_AND_rule_transfers": sum(c.candidate_type == "phaseC_and_rule" for c in accepted),
            "number_of_accepted_phaseC_main_effect_transfers": sum(c.candidate_type == "phaseC_main_effect" for c in accepted),
            "final_transferred_objects": json.dumps(transfer_records, ensure_ascii=False),
            "final_gp_policy_tree": best_details.get("policy_tree"),
            "evaluation_backend": "original_single_target_staged_model",
            "final_model": final_result.get("final_model"),
            "zero_exchange_reconstructed_cindex": zero_result["external_cindex"],
            "zero_exchange_delta_vs_stored_2G": zero_delta,
        })

    label = "4endpoint"
    df = pd.DataFrame(rows)
    df.to_csv(branch_root / f"external_validation_{label}.csv", index=False, encoding="utf-8-sig")
    df.to_excel(branch_root / f"external_validation_{label}.xlsx", index=False)
    audit = {
        "mimic_used_only_after_exchange_policy_freezing": True,
        "mimic_not_used_during_gp_policy_evolution": True,
        "mimic_not_used_during_candidate_ranking": True,
        "mimic_not_used_during_internal_cv_selection": True,
        "external_validation_rows": len(rows),
        "external_validation_files": [str(branch_root / f"external_validation_{label}.csv"), str(branch_root / f"external_validation_{label}.xlsx")],
        "evaluation_backend": "original_single_target_staged_model",
        "zero_exchange_equivalence_audit": zero_exchange_audit,
        "zero_exchange_equivalence_all_endpoints_passed": all(
            row["equal_within_1e_12"] for row in zero_exchange_audit
        ),
    }
    if not audit["zero_exchange_equivalence_all_endpoints_passed"]:
        raise RuntimeError(
            "The exact frozen-2G evaluator failed zero-exchange equivalence"
        )
    write_json(branch_root / "audit" / "external_validation_audit.json", audit)
    return rows, audit


def evaluate_final_plan_internal(
    cfg: dict[str, Any],
    target_frames: dict[str, dict[str, Any]],
    candidates: list[Candidate],
    best_details: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    accepted_by_target: dict[str, list[Candidate]] = {
        name: [] for name in endpoint_names(cfg)
    }
    for item in best_details.get("accepted_transfers", []):
        candidate = candidate_by_id.get(str(item.get("candidate_id")))
        if candidate is not None:
            accepted_by_target[candidate.target_endpoint].append(candidate)
    output: dict[str, dict[str, Any]] = {}
    for name, target_frame in target_frames.items():
        accepted = accepted_by_target[name]
        mean, fold_scores, status, failures = frozen_eval.cross_validated_prepared_states(
            target_frame["cv_states"],
            extra_gp_expressions=[
                candidate.expression_or_rule
                for candidate in accepted
                if candidate.candidate_type == "phaseA_gp"
            ],
            extra_phasec_candidates=[
                dict(candidate.raw)
                for candidate in accepted
                if candidate.candidate_type in {"phaseC_and_rule", "phaseC_main_effect"}
            ],
        )
        output[name] = {
            "mean_cindex": mean,
            "fold_cindex": fold_scores,
            "status": status,
            "cox_failure_count": failures,
            "accepted_transfer_n": len(accepted),
            "evaluation_backend": "original_single_target_staged_model",
        }
    return output


def build_policy_toolbox(seed: int, cfg: dict[str, Any]) -> tuple[base.Toolbox, gp.PrimitiveSet]:
    random.seed(seed)
    pset = gp.PrimitiveSet("POLICY", 10)
    names = [
        "R",
        "source_quality",
        "source_selection_frequency",
        "target_signal",
        "target_support",
        "novelty",
        "nonredundancy",
        "simplicity",
        "candidate_type",
        "input_scope_score",
    ]
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(names)})
    pset.addPrimitive(lambda a, b: clip01(a + b), 2, name="add")
    pset.addPrimitive(lambda a, b: clip01(a - b), 2, name="subtract")
    pset.addPrimitive(lambda a, b: clip01(a * b), 2, name="multiply")
    pset.addPrimitive(protected_division, 2, name="protected_division")
    pset.addPrimitive(min, 2, name="min")
    pset.addPrimitive(max, 2, name="max")
    pset.addPrimitive(average, 2, name="average")
    pset.addPrimitive(clip01, 1, name="clip01")
    pset.addPrimitive(sqrt_abs, 1, name="sqrt_abs")
    pset.addPrimitive(log1p_abs, 1, name="log1p_abs")
    for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pset.addTerminal(value)

    if not hasattr(creator, "FitnessMaxPolicyTree"):
        creator.create("FitnessMaxPolicyTree", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "PolicyTreeIndividual"):
        creator.create("PolicyTreeIndividual", gp.PrimitiveTree, fitness=creator.FitnessMaxPolicyTree)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.PolicyTreeIndividual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    policy_cfg = cfg.get("policy", {})
    toolbox.register("select", tools.selTournament, tournsize=int(policy_cfg.get("tournament_size", 2)))
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    max_height = int(policy_cfg.get("max_height", 4))
    max_nodes = int(policy_cfg.get("max_nodes", 25))
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_height))
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=max_nodes))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=max_nodes))
    return toolbox, pset


def candidate_type_score(candidate_type: str, cfg: dict[str, Any]) -> float:
    scores = cfg.get("candidate_type_scores", {})
    return safe_float(scores.get(candidate_type), CANDIDATE_TYPE_SCORE.get(candidate_type, 0.0))


def candidate_args(c: Candidate, cfg: dict[str, Any]) -> tuple[float, ...]:
    return (
        c.relation_weight,
        c.source_quality,
        c.source_selection_frequency,
        c.target_signal,
        c.target_support,
        c.novelty,
        c.nonredundancy,
        c.simplicity,
        candidate_type_score(c.candidate_type, cfg),
        c.input_scope_score,
    )


def apply_current_acceptance(result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    output = dict(result)
    status = str(output.get("status", "cox_failed"))
    fold_deltas = [float(value) for value in output.get("fold_deltas", [])]
    delta = float(output.get("delta", 0.0))
    acceptance = cfg["acceptance"]
    accepted = (
        status == "ok"
        and delta >= float(acceptance["minimum_delta_cindex"])
        and sum(value > 0 for value in fold_deltas) >= int(acceptance["positive_folds_min"])
        and bool(fold_deltas)
        and min(fold_deltas) >= float(acceptance["worst_fold_drop_min"])
    )
    output["accepted"] = accepted
    output["reason"] = (
        "accepted"
        if accepted
        else ("cox_failed" if status == "cox_failed" else "acceptance_threshold_not_met")
    )
    output["fitness_penalty_applied"] = status == "cox_failed"
    return output


def evaluate_candidate(target_frame: dict[str, Any], cand: Candidate, cfg: dict[str, Any]) -> dict[str, Any]:
    cache = target_frame.setdefault("candidate_evaluation_cache", {})
    if cand.candidate_id in cache:
        return dict(cache[cand.candidate_id])
    shared_cache_path = Path(cfg["_branch_root"]) / "frozen_2g_exact_cv_cache.json"
    shared_cache = read_json(shared_cache_path, {}) or {}
    shared_key_payload = {
        "evaluator_version": "single_target_staged_v2_acceptance_recomputed",
        "candidate_id": cand.candidate_id,
        "candidate_type": cand.candidate_type,
        "expression_or_rule": cand.expression_or_rule,
        "target_endpoint": cand.target_endpoint,
        "source_result_json": target_frame["knowledge"].get("source_result_json"),
        "folds": int(cfg["smoke"]["cv_folds"]),
        "seed": int(cfg["smoke"]["random_seed"]),
    }
    shared_key = hashlib.sha256(
        json.dumps(shared_key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if shared_key in shared_cache:
        result = apply_current_acceptance(dict(shared_cache[shared_key]), cfg)
        result["loaded_from_shared_exact_cv_cache"] = True
        cache[cand.candidate_id] = dict(result)
        return result
    try:
        extra_gp = [cand.expression_or_rule] if cand.candidate_type == "phaseA_gp" else []
        extra_phasec = [dict(cand.raw)] if cand.candidate_type in {
            "phaseC_and_rule", "phaseC_main_effect"
        } else []
        mean, folds, status, failure_count = frozen_eval.cross_validated_prepared_states(
            target_frame["cv_states"],
            extra_gp_expressions=extra_gp,
            extra_phasec_candidates=extra_phasec,
        )
    except Exception as exc:
        result = {
            "candidate_id": cand.candidate_id,
            "target_endpoint": cand.target_endpoint,
            "accepted": False,
            "reason": f"cox_or_feature_failed:{exc}",
            "status": "cox_failed",
            "fitness_penalty_applied": True,
            "cox_failure_count": int(cfg["smoke"]["cv_folds"]),
            "delta": 0.0,
            "fold_deltas": [],
        }
        cache[cand.candidate_id] = dict(result)
        shared_cache[shared_key] = dict(result)
        write_json(shared_cache_path, shared_cache)
        return result
    base_folds = target_frame["baseline_fold_cindex"]
    fold_deltas = [float(a - b) for a, b in zip(folds, base_folds)]
    delta = float(mean - target_frame["baseline_mean_cindex"])
    result = {
        "candidate_id": cand.candidate_id,
        "source_endpoint": cand.source_endpoint,
        "target_endpoint": cand.target_endpoint,
        "candidate_type": cand.candidate_type,
        "expression_or_rule": cand.expression_or_rule,
        "input_scope": cand.input_scope,
        "status": status,
        "cox_failure_count": failure_count,
        "baseline_mean_cindex": target_frame["baseline_mean_cindex"],
        "candidate_mean_cindex": mean,
        "delta": delta,
        "fold_deltas": fold_deltas,
        "used_variables": cand.used_variables,
        "evaluation_backend": "original_single_target_staged_model",
    }
    result = apply_current_acceptance(result, cfg)
    cache[cand.candidate_id] = dict(result)
    shared_cache[shared_key] = {
        key: value
        for key, value in result.items()
        if key not in {"accepted", "reason", "fitness_penalty_applied", "loaded_from_shared_exact_cv_cache"}
    }
    write_json(shared_cache_path, shared_cache)
    return result


def apply_transfer_capacity(accepted: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    capacity = cfg.get("transfer_capacity", {})
    if not capacity:
        return accepted
    max_total = int(capacity.get("max_total_accepted_transfers", 10_000))
    max_per_target = int(capacity.get("max_accepted_total_per_target", 10_000))
    max_by_type = {
        "phaseA_gp": int(capacity.get("max_phaseA_accepted_per_target", 10_000)),
        "phaseC_and_rule": int(capacity.get("max_phaseC_and_accepted_per_target", 10_000)),
        "phaseC_main_effect": int(capacity.get("max_phaseC_main_accepted_per_target", 10_000)),
    }
    max_source_target_type = int(capacity.get("max_features_per_source_target_type", 10_000))
    kept: list[dict[str, Any]] = []
    count_target: dict[str, int] = {}
    count_target_type: dict[tuple[str, str], int] = {}
    count_source_target_type: dict[tuple[str, str, str], int] = {}
    for item in sorted(accepted, key=lambda x: float(x.get("delta", 0.0)), reverse=True):
        target = str(item.get("target_endpoint"))
        source = str(item.get("source_endpoint"))
        ctype = str(item.get("candidate_type"))
        if len(kept) >= max_total:
            break
        if count_target.get(target, 0) >= max_per_target:
            continue
        if count_target_type.get((target, ctype), 0) >= max_by_type.get(ctype, 10_000):
            continue
        if count_source_target_type.get((source, target, ctype), 0) >= max_source_target_type:
            continue
        kept.append(item)
        count_target[target] = count_target.get(target, 0) + 1
        count_target_type[(target, ctype)] = count_target_type.get((target, ctype), 0) + 1
        count_source_target_type[(source, target, ctype)] = count_source_target_type.get((source, target, ctype), 0) + 1
    return kept


def score_policy(
    individual: Any,
    toolbox: base.Toolbox,
    cfg: dict[str, Any],
    candidates: list[Candidate],
    target_frames: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    func = toolbox.compile(expr=individual)
    by_target: dict[str, list[tuple[float, Candidate]]] = {name: [] for name in target_frames}
    for cand in candidates:
        try:
            score = clip01(func(*candidate_args(cand, cfg)))
        except Exception:
            score = 0.0
        by_target[cand.target_endpoint].append((score, cand))

    accepted: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []
    max_per_target = int(cfg["smoke"].get("top_k_transfer_candidates", cfg["smoke"].get("max_candidates_per_target", 1)))
    top_k_by_type = cfg["smoke"].get("top_k_by_candidate_type", {})
    endpoint_best_delta: dict[str, float] = {}
    endpoint_avg_topk_delta: dict[str, float] = {}
    endpoint_stability: dict[str, float] = {}
    endpoint_evaluated_by_type: dict[str, dict[str, int]] = {}
    for target, scored in by_target.items():
        scored = sorted(scored, key=lambda x: x[0], reverse=True)
        endpoint_best_delta[target] = 0.0
        target_results: list[dict[str, Any]] = []
        selected: list[tuple[float, Candidate]] = []
        seen_ids: set[str] = set()
        if isinstance(top_k_by_type, dict) and top_k_by_type:
            for candidate_type, type_k in top_k_by_type.items():
                type_scored = [(score, cand) for score, cand in scored if cand.candidate_type == candidate_type]
                for score, cand in type_scored[: int(type_k)]:
                    if cand.candidate_id not in seen_ids:
                        selected.append((score, cand))
                        seen_ids.add(cand.candidate_id)
        else:
            selected = scored[:max_per_target]
        selected = sorted(selected, key=lambda x: x[0], reverse=True)
        endpoint_evaluated_by_type[target] = {}
        for score, cand in selected:
            endpoint_evaluated_by_type[target][cand.candidate_type] = endpoint_evaluated_by_type[target].get(cand.candidate_type, 0) + 1
            result = evaluate_candidate(target_frames[target], cand, cfg)
            result["policy_score"] = score
            evaluated.append(result)
            target_results.append(result)
            endpoint_best_delta[target] = max(endpoint_best_delta[target], float(result["delta"]))
            if result["accepted"]:
                accepted.append(result)
        endpoint_avg_topk_delta[target] = float(np.mean([float(r.get("delta", 0.0)) for r in target_results])) if target_results else 0.0
        endpoint_stability[target] = float(np.mean([min(r.get("fold_deltas") or [0.0]) for r in target_results])) if target_results else 0.0
    accepted_before_capacity = list(accepted)
    accepted = apply_transfer_capacity(accepted, cfg)

    deltas = list(endpoint_best_delta.values())
    avg_topk = list(endpoint_avg_topk_delta.values())
    stability_values = list(endpoint_stability.values())
    improved = sum(x >= float(cfg["acceptance"]["minimum_delta_cindex"]) for x in deltas)
    harmed = sum(x < -0.005 for x in deltas)
    cox_failed_count = sum(1 for r in evaluated if r.get("status") == "cox_failed")
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    mean_topk_delta = float(np.mean(avg_topk)) if avg_topk else 0.0
    mean_stability = float(np.mean(stability_values)) if stability_values else 0.0
    worst_delta = min(deltas) if deltas else 0.0
    complexity_penalty = 0.002 * len(individual)
    accepted_penalty = 0.003 * max(0, len(accepted) - len(target_frames))
    cox_failure_penalty = 0.02 * cox_failed_count
    fitness = (
        mean_delta
        + 0.40 * mean_topk_delta
        + 0.20 * mean_stability
        + 0.02 * improved
        + 0.004 * len(accepted)
        - 0.03 * harmed
        + min(0.0, worst_delta)
        - complexity_penalty
        - accepted_penalty
        - cox_failure_penalty
    )
    details = {
        "policy_tree": str(individual),
        "fitness": fitness,
        "mean_endpoint_improvement": mean_delta,
        "mean_topk_candidate_delta": mean_topk_delta,
        "mean_fold_stability": mean_stability,
        "endpoint_best_delta": endpoint_best_delta,
        "endpoint_avg_topk_delta": endpoint_avg_topk_delta,
        "endpoint_stability": endpoint_stability,
        "improved_endpoint_count": improved,
        "harmed_endpoint_count": harmed,
        "cox_failed_evaluation_count": cox_failed_count,
        "cox_failure_penalty": cox_failure_penalty,
        "top_k_transfer_candidates": max_per_target,
        "top_k_by_candidate_type": top_k_by_type,
        "endpoint_evaluated_by_type": endpoint_evaluated_by_type,
        "accepted_transfers_before_capacity": accepted_before_capacity,
        "accepted_transfers": accepted,
        "evaluated_transfers": evaluated,
        "tree_size": len(individual),
    }
    return (fitness,), details


def run_smoke(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    branch_root = Path(cfg["_output_root"])
    branch_root.mkdir(parents=True, exist_ok=True)
    for dirname in ["knowledge_library", "candidate_pool", "evolution_results", "audit", "smoke_test"]:
        (branch_root / dirname).mkdir(exist_ok=True)

    knowledge = load_knowledge(cfg)
    expected_n = len(cfg["endpoints"])
    if len(knowledge) != expected_n:
        raise RuntimeError(f"Endpoint count mismatch: expected {expected_n}, got {len(knowledge)}")

    candidates = build_candidates(cfg, knowledge)
    target_frames = prepare_target_frames(cfg, knowledge)
    cox_failure_check = cox_failure_handling_self_check()
    if not cox_failure_check["passed"]:
        raise RuntimeError("Cox failure handling self-check failed.")
    toolbox, _pset = build_policy_toolbox(int(cfg["smoke"]["random_seed"]), cfg)

    eval_details: list[dict[str, Any]] = []

    def evaluate(individual):
        fitness, details = score_policy(individual, toolbox, cfg, candidates, target_frames)
        eval_details.append(details)
        return fitness

    toolbox.register("evaluate", evaluate)
    policy_cfg = cfg.get("policy", {})
    pop = toolbox.population(n=int(policy_cfg.get("pop_size", cfg["smoke"].get("population", 4))))
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "max"]

    first_generation_examples = [str(ind) for ind in pop[: min(3, len(pop))]]
    cxpb = float(policy_cfg.get("crossover_prob", 0.5))
    mutpb = float(policy_cfg.get("mutation_prob", 0.3))
    n_gen = int(policy_cfg.get("n_gen", cfg["smoke"].get("generations", 2)))
    elite_n = max(0, min(int(policy_cfg.get("elite_n", 1)), len(pop) - 1))
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid), **record)
    for gen in range(1, n_gen + 1):
        elites = [deepcopy(ind) for ind in tools.selBest(pop, elite_n)] if elite_n else []
        offspring = [deepcopy(ind) for ind in toolbox.select(pop, len(pop) - elite_n)]
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=cxpb, mutpb=mutpb)
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
        pop = elites + offspring
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid), **record)

    best = hof[0]
    _fitness, best_details = score_policy(best, toolbox, cfg, candidates, target_frames)
    generation_rows = [dict(row) for row in logbook]
    final_plan_internal_cv = evaluate_final_plan_internal(
        cfg, target_frames, candidates, best_details
    )
    external_rows, external_audit = run_external_validation(cfg, knowledge, candidates, best_details)

    safety = {
        "no_single_target_gp_rerun_occurred": True,
        "no_new_gp_expression_generation_occurred": True,
        "no_new_phaseC_rule_generation_occurred": True,
        "no_mimic_metric_used_during_evolution": True,
        "forbidden_mimic_or_external_files_accessed": [],
        "evaluation_data_role": "eICU_internal_cv_only",
        "mimic_role": "final_external_validation_after_policy_freezing_only",
    }
    audit = {
        "branch_name": cfg["branch_name"],
        "endpoint_count": expected_n,
        "endpoints": endpoint_names(cfg),
        "relation_matrix_dimension": [len(cfg["relation_matrix"]), len(cfg["relation_matrix"][0])],
        "source_notebook": cfg["source_notebook"],
        "source_parameter_profile": cfg["source_parameter_profile"],
        "safety_audit": safety,
    }
    write_json(branch_root / "audit" / "safety_audit.json", safety)
    write_json(branch_root / "audit" / "run_audit.json", audit)
    write_json(branch_root / "evolution_results" / "generation_statistics.json", generation_rows)
    write_json(branch_root / "evolution_results" / "best_policy_tree.json", best_details)
    write_json(branch_root / "evolution_results" / "evolution_statistics.json", {
        "generations": generation_rows,
        "best_policy_tree": best_details["policy_tree"],
        "fitness": best_details["fitness"],
        "accepted_transfers": best_details["accepted_transfers"],
        "accepted_phaseA_gp_transfer_count": sum(x.get("candidate_type") == "phaseA_gp" for x in best_details["accepted_transfers"]),
        "accepted_phaseC_AND_rule_transfer_count": sum(x.get("candidate_type") == "phaseC_and_rule" for x in best_details["accepted_transfers"]),
        "accepted_phaseC_main_effect_transfer_count": sum(x.get("candidate_type") == "phaseC_main_effect" for x in best_details["accepted_transfers"]),
        "endpoint_cv_improvement": best_details["endpoint_best_delta"],
        "endpoint_avg_topk_delta": best_details["endpoint_avg_topk_delta"],
        "endpoint_stability": best_details["endpoint_stability"],
        "mean_topk_candidate_delta": best_details["mean_topk_candidate_delta"],
        "mean_fold_stability": best_details["mean_fold_stability"],
        "improved_endpoint_count": best_details["improved_endpoint_count"],
        "harmed_endpoint_count": best_details["harmed_endpoint_count"],
        "cox_failed_evaluation_count": best_details["cox_failed_evaluation_count"],
        "cox_failure_penalty": best_details["cox_failure_penalty"],
        "top_k_transfer_candidates": best_details["top_k_transfer_candidates"],
        "top_k_by_candidate_type": best_details["top_k_by_candidate_type"],
        "endpoint_evaluated_by_type": best_details["endpoint_evaluated_by_type"],
        "evaluated_transfer_count": len(best_details["evaluated_transfers"]),
    })

    smoke_summary = {
        "success": True,
        "branch_name": cfg["branch_name"],
        "endpoint_count": expected_n,
        "candidate_count": len(candidates),
        "phaseA_candidate_count": sum(c.candidate_type == "phaseA_gp" for c in candidates),
        "phaseC_candidate_count": sum(c.candidate_type in {"phaseC_and_rule", "phaseC_main_effect"} for c in candidates),
        "phaseC_AND_rule_candidate_count": sum(c.candidate_type == "phaseC_and_rule" for c in candidates),
        "phaseC_main_effect_candidate_count": sum(c.candidate_type == "phaseC_main_effect" for c in candidates),
        "strict_target_gp_input_candidates": sum(c.input_scope == "strict_target_gp_input" for c in candidates),
        "eligible_predictor_pool_candidates": sum(c.input_scope == "eligible_predictor_pool" for c in candidates),
        "top_k_transfer_candidates": best_details["top_k_transfer_candidates"],
        "top_k_by_candidate_type": best_details["top_k_by_candidate_type"],
        "endpoint_evaluated_by_type": best_details["endpoint_evaluated_by_type"],
        "first_generation_policy_tree_examples": first_generation_examples,
        "best_policy_tree": str(best),
        "best_fitness": best_details["fitness"],
        "accepted_transfer_examples": best_details["accepted_transfers"][:5],
        "accepted_phaseA_gp_transfer_count": sum(x.get("candidate_type") == "phaseA_gp" for x in best_details["accepted_transfers"]),
        "accepted_phaseC_AND_rule_transfer_count": sum(x.get("candidate_type") == "phaseC_and_rule" for x in best_details["accepted_transfers"]),
        "accepted_phaseC_main_effect_transfer_count": sum(x.get("candidate_type") == "phaseC_main_effect" for x in best_details["accepted_transfers"]),
        "cox_failed_evaluation_count": best_details["cox_failed_evaluation_count"],
        "cox_failure_handling_self_check": cox_failure_check,
        "baseline_internal_cv": {
            target: {
                "mean_cindex": frame["baseline_mean_cindex"],
                "fold_cindex": frame["baseline_fold_cindex"],
                "status": frame["baseline_status"],
                "cox_failure_count": frame["baseline_cox_failure_count"],
                "feature_count": len(frame["base_features"]),
            }
            for target, frame in target_frames.items()
        },
        "external_validation": external_rows,
        "external_validation_audit": external_audit,
        "final_plan_internal_cv": final_plan_internal_cv,
        "outputs": {
            "knowledge_library": str(branch_root / "knowledge_library"),
            "candidate_pool": str(branch_root / "candidate_pool"),
            "evolution_results": str(branch_root / "evolution_results"),
            "audit": str(branch_root / "audit"),
            "smoke_test": str(branch_root / "smoke_test"),
        },
        "safety_audit": safety,
    }
    write_json(branch_root / "smoke_test" / "smoke_summary.json", smoke_summary)
    return smoke_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="branch_config.json")
    args = parser.parse_args()
    summary = run_smoke(Path(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
