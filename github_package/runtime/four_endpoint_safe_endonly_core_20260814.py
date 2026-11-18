"""Fixed-structure safe-primitive reconstruction of four-endpoint End-only trial 3.

The module reuses the frozen data, preprocessing, structures, time bases and
Phase-C rules from ``four_endpoint_results_core_20260814``.  It changes only
the numerical semantics of the three historically unsafe GP primitives,
refits the GP scaler and Cox/CTV coefficients on eICU, and applies the fitted
model to MIMIC-IV without refitting or selection.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

import four_endpoint_results_core_20260814 as legacy


TAU = 0.25

# Public constants/functions used by the figure renderer.
ROOT = legacy.ROOT
LOCKED = legacy.LOCKED
DATA_ROOT = legacy.DATA_ROOT
ENDPOINT_ORDER = legacy.ENDPOINT_ORDER
ENDPOINT_SHORT = legacy.ENDPOINT_SHORT
ENDPOINT_TITLE = legacy.ENDPOINT_TITLE
ENDPOINT_LETTER = legacy.ENDPOINT_LETTER
ENDPOINT_COLOR = legacy.ENDPOINT_COLOR
DOMAIN_ORDER = legacy.DOMAIN_ORDER
COMPONENT_ORDER = legacy.COMPONENT_ORDER
EXPECTED_EXTERNAL = legacy.EXPECTED_EXTERNAL
FROZEN = legacy.FROZEN
EndpointModel = legacy.EndpointModel
Term = legacy.Term
Family = legacy.Family
readable_name = legacy.readable_name
expression_leaves = legacy.expression_leaves
domain_for_leaves = legacy.domain_for_leaves
aggregate_shap = legacy.aggregate_shap
relative_importance = legacy.relative_importance


def _soft_abs(value: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(value) + TAU**2) - TAU


def eval_gp_expression(expression: str, frame: pd.DataFrame) -> np.ndarray:
    """Evaluate a locked expression with the tau=0.25 safe semantics."""
    text = str(expression).strip()
    if text in frame:
        output = pd.to_numeric(frame[text], errors="coerce").fillna(0.0).to_numpy(float)
    else:
        try:
            output = np.full(len(frame), float(text))
        except Exception:
            match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", text)
            if not match:
                raise ValueError(f"unsupported GP expression: {expression}")
            name, arguments = match.groups()
            args = [eval_gp_expression(value, frame) for value in legacy.split_args(arguments)]
            if name == "add":
                output = args[0] + args[1]
            elif name == "sub":
                output = args[0] - args[1]
            elif name == "mul":
                output = args[0] * args[1]
            elif name == "safe_div":
                output = args[0] * args[1] / (np.square(args[1]) + TAU**2)
            elif name == "safe_log":
                output = np.log1p(_soft_abs(args[0]))
            elif name == "safe_sqrt":
                output = np.sqrt(_soft_abs(args[0]) + TAU) - np.sqrt(TAU)
            elif name == "clip":
                output = np.clip(args[0], -10.0, 10.0)
            else:
                raise ValueError(f"unsupported GP primitive: {name}")
    output = np.asarray(output, dtype=float)
    if not np.all(np.isfinite(output)):
        raise FloatingPointError(f"non-finite safe GP output: {expression}")
    return output


def readable_safe_expression(expression: str) -> str:
    value = str(expression).strip()
    match = re.fullmatch(r"([A-Za-z_]+)\((.*)\)", value)
    if not match:
        return readable_name(value)
    fn, raw = match.groups()
    args = [readable_safe_expression(item) for item in legacy.split_args(raw)]
    if fn == "safe_sqrt":
        return f"ssqrt_tau({args[0]})"
    if fn == "safe_log":
        return f"slog_tau({args[0]})"
    if fn == "safe_div":
        return f"sdiv_tau({args[0]}, {args[1]})"
    if fn == "sub":
        return f"{args[0]} - {args[1]}"
    if fn == "add":
        return f"{args[0]} + {args[1]}"
    if fn == "mul":
        return f"{args[0]} x {args[1]}"
    if fn == "clip":
        return f"clip({args[0]})"
    return f"{fn}({', '.join(args)})"


def _build_safe_final_fit(endpoint: str) -> dict[str, Any]:
    payload = legacy._knowledge_payload(endpoint)
    stem = legacy.DATA_FILES[endpoint]
    train_raw = pd.read_excel(DATA_ROOT / f"{stem}_eicu.xlsx")
    mimic_raw = pd.read_excel(DATA_ROOT / f"{stem}_mimic.xlsx")
    train, prep = FROZEN.fit_preprocess(train_raw)
    mimic = FROZEN.transform_preprocess(mimic_raw, prep)

    main_names = [str(value) for value in payload.get("main_effect_vars", []) or []]
    expressions = FROZEN.source_gp_expressions(payload) + list(legacy.ACCEPTED_EXTRA_GP[endpoint])
    gp_train_raw = np.column_stack([eval_gp_expression(expr, train) for expr in expressions])
    gp_mimic_raw = np.column_stack([eval_gp_expression(expr, mimic) for expr in expressions])
    scaler = StandardScaler().fit(gp_train_raw)
    gp_train = scaler.transform(gp_train_raw)
    gp_mimic = scaler.transform(gp_mimic_raw)
    main_train = train[main_names].to_numpy(float) if main_names else np.empty((len(train), 0))
    main_mimic = mimic[main_names].to_numpy(float) if main_names else np.empty((len(mimic), 0))
    phasea_names = [*main_names, *[f"gp{i+1}" for i in range(len(expressions))]]
    x_train = pd.DataFrame(np.column_stack([main_train, gp_train]), columns=phasea_names)
    x_mimic = pd.DataFrame(np.column_stack([main_mimic, gp_mimic]), columns=phasea_names)
    fit = x_train.copy()
    fit[FROZEN.TIME_COL] = train[FROZEN.TIME_COL].to_numpy(float)
    fit[FROZEN.EVENT_COL] = train[FROZEN.EVENT_COL].to_numpy(int)
    cox = CoxPHFitter(penalizer=FROZEN.COX_PENALIZER, l1_ratio=FROZEN.COX_L1_RATIO)
    cox.fit(fit, duration_col=FROZEN.TIME_COL, event_col=FROZEN.EVENT_COL)
    eta_train = cox.predict_log_partial_hazard(x_train).to_numpy(float)
    eta_mimic = cox.predict_log_partial_hazard(x_mimic).to_numpy(float)

    phaseb = FROZEN.source_phaseb(payload)
    phasec = FROZEN.source_phasec(payload)
    static_train = FROZEN._phasec_frame(train, phasec)
    static_mimic = FROZEN._phasec_frame(mimic, phasec).reindex(columns=static_train.columns, fill_value=0.0)
    feature_train = FROZEN._feature_frame(main_train, main_names, gp_train_raw)
    feature_mimic = FROZEN._feature_frame(main_mimic, main_names, gp_mimic_raw)
    times_train = train[FROZEN.TIME_COL].to_numpy(float)
    events_train = train[FROZEN.EVENT_COL].to_numpy(int)
    times_mimic = mimic[FROZEN.TIME_COL].to_numpy(float)
    events_mimic = mimic[FROZEN.EVENT_COL].to_numpy(int)
    time_stats = FROZEN._time_stats(times_train)

    if phaseb:
        long = FROZEN._long_frame(
            times_train, events_train, eta_train, feature_train, phaseb, time_stats, static_train
        )
        design = ["eta_ab", *[str(row["interaction_col"]) for row in phaseb], *list(static_train.columns)]
        usable = [
            name for name in design
            if name in long and float(pd.to_numeric(long[name], errors="coerce").std(ddof=0)) > 1e-8
        ]
        final_model = CoxTimeVaryingFitter(penalizer=FROZEN.PHASEB_PENALIZER)
        final_model.fit(
            long[["id", "start", "stop", "event", *usable]],
            id_col="id", start_col="start", stop_col="stop", event_col="event",
        )
        final_coefficients = {str(k): float(v) for k, v in final_model.params_.to_dict().items()}
        external_cindex = FROZEN._time_dependent_cindex(
            times_mimic, events_mimic, eta_mimic, feature_mimic,
            phaseb, time_stats, static_mimic, final_coefficients,
        )
    elif len(static_train.columns):
        phasec_train = pd.DataFrame({"eta_ab": eta_train})
        phasec_mimic = pd.DataFrame({"eta_ab": eta_mimic})
        for name in static_train:
            phasec_train[name] = static_train[name].to_numpy(float)
            phasec_mimic[name] = static_mimic[name].to_numpy(float)
        phasec_fit = phasec_train.copy()
        phasec_fit[FROZEN.TIME_COL] = times_train
        phasec_fit[FROZEN.EVENT_COL] = events_train
        final_model = CoxPHFitter(
            penalizer=FROZEN.PHASEC_PENALIZER, l1_ratio=FROZEN.PHASEC_L1_RATIO
        )
        final_model.fit(phasec_fit, duration_col=FROZEN.TIME_COL, event_col=FROZEN.EVENT_COL)
        final_coefficients = {str(k): float(v) for k, v in final_model.params_.to_dict().items()}
        risk = final_model.predict_partial_hazard(phasec_mimic).to_numpy(float)
        external_cindex = concordance_index(times_mimic, -risk, events_mimic)
    else:
        final_coefficients = {"eta_ab": 1.0}
        external_cindex = concordance_index(times_mimic, -eta_mimic, events_mimic)

    return {
        "payload": payload,
        "train_raw": train_raw, "mimic_raw": mimic_raw,
        "train": train, "mimic": mimic, "prep": prep,
        "main_names": main_names, "expressions": expressions,
        "gp_train_raw": gp_train_raw, "gp_mimic_raw": gp_mimic_raw,
        "gp_train": gp_train, "gp_mimic": gp_mimic, "gp_scaler": scaler,
        "phasea_names": phasea_names,
        "phasea_coefficients": {str(k): float(v) for k, v in cox.params_.to_dict().items()},
        "final_coefficients": final_coefficients,
        "eta_train": eta_train, "eta_mimic": eta_mimic,
        "phaseb": phaseb, "phasec": phasec,
        "feature_train": feature_train, "feature_mimic": feature_mimic,
        "static_train": static_train, "static_mimic": static_mimic,
        "time_stats": time_stats,
        "external_cindex": float(external_cindex),
    }


def _safe_term_labels(terms: list[Term]) -> list[Term]:
    output: list[Term] = []
    for term in terms:
        if term.expression is None or term.gp_index is None:
            output.append(term)
            continue
        base = f"GP{term.gp_index}: {readable_safe_expression(term.expression)}"
        label = f"{base} x {legacy._time_basis_label(term.time_basis)}" if term.time_basis else base
        output.append(replace(term, label=label))
    return output


def load_endpoint_model(endpoint: str) -> EndpointModel:
    state = _build_safe_final_fit(endpoint)
    values = list(legacy._make_terms_and_design(state))
    values[0] = _safe_term_labels(values[0])
    model = EndpointModel(
        endpoint=endpoint, title=ENDPOINT_TITLE[endpoint], payload=state["payload"],
        train_raw=state["train_raw"], mimic_raw=state["mimic_raw"],
        train_encoded=state["train"], mimic_encoded=state["mimic"],
        preprocessor=state["prep"], gp_expressions=state["expressions"],
        gp_scaler=state["gp_scaler"], gp_train_raw=state["gp_train_raw"],
        gp_mimic_raw=state["gp_mimic_raw"], phasea_names=state["phasea_names"],
        phasea_coefficients=state["phasea_coefficients"],
        final_coefficients=state["final_coefficients"],
        backbone_multiplier=float(state["final_coefficients"].get("eta_ab", 1.0)),
        phaseb=state["phaseb"], phasec=state["phasec"], terms=values[0],
        families=values[1], train_design=values[2], mimic_design=values[3],
        background_mean=values[4], train_shap=values[5], mimic_shap=values[6],
        train_score=values[7], mimic_score=values[8],
        external_cindex=state["external_cindex"], max_additivity_error=values[9],
    )
    arrays = [
        model.gp_train_raw, model.gp_mimic_raw, model.train_design,
        model.mimic_design, model.train_shap, model.mimic_shap,
        model.train_score, model.mimic_score,
    ]
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError(f"{endpoint}: non-finite safe model asset")
    return model


def load_all_models() -> dict[str, EndpointModel]:
    return {endpoint: load_endpoint_model(endpoint) for endpoint in ENDPOINT_ORDER}


def audit_payload(
    safe_models: dict[str, EndpointModel],
    legacy_models: dict[str, EndpointModel] | None = None,
) -> dict[str, Any]:
    legacy_models = legacy_models or legacy.load_all_models()
    rows = []
    for endpoint in ENDPOINT_ORDER:
        old = legacy_models[endpoint]
        new = safe_models[endpoint]
        old_inventory = [(term.component, term.family_id, term.expression, term.time_basis) for term in old.terms]
        new_inventory = [(term.component, term.family_id, term.expression, term.time_basis) for term in new.terms]
        rows.append({
            "endpoint": endpoint,
            "legacy_external_cindex": old.external_cindex,
            "locked_expected_external_cindex": EXPECTED_EXTERNAL[endpoint],
            "legacy_reproduction_abs_error": abs(old.external_cindex - EXPECTED_EXTERNAL[endpoint]),
            "safe_external_cindex": new.external_cindex,
            "safe_minus_legacy_external_cindex": new.external_cindex - old.external_cindex,
            "term_inventory_identical": old_inventory == new_inventory,
            "term_count": len(new.terms),
            "gp_expression_count": len(new.gp_expressions),
            "phaseB_count": len(new.phaseb),
            "phaseC_count": len(new.phasec),
            "max_shap_additivity_error": new.max_additivity_error,
            "all_arrays_finite": True,
            "phaseA_coefficients": new.phasea_coefficients,
            "final_coefficients": new.final_coefficients,
        })
    return {
        "schema": "four_endpoint_end_only_fixed_structure_safe_repair_v1",
        "model": "End-only trial 3",
        "tau": TAU,
        "formulas": {
            "soft_abs": "sqrt(x^2 + tau^2) - tau",
            "safe_log": "log(1 + soft_abs(x))",
            "safe_sqrt": "sqrt(soft_abs(x) + tau) - sqrt(tau)",
            "safe_div": "a*b/(b^2 + tau^2)",
        },
        "structure_and_terms_locked": True,
        "model_search_performed": False,
        "mimic_refit_performed": False,
        "models": rows,
        "legacy_reproduction_all_within_1e_10": all(
            row["legacy_reproduction_abs_error"] <= 1e-10 for row in rows
        ),
        "term_inventory_all_identical": all(row["term_inventory_identical"] for row in rows),
        "safe_all_finite": all(row["all_arrays_finite"] for row in rows),
        "shap_additivity_all_within_1e_10": all(
            row["max_shap_additivity_error"] <= 1e-10 for row in rows
        ),
    }
