"""Reconstruct the locked four-endpoint End-only trial-3 models for display.

This module performs no search or model selection.  It reads the frozen 2G
knowledge, adds only the four transfers recorded for locked End-only trial 3,
refits the prescribed Cox/CTV coefficients on eICU, and checks the published
locked MIMIC C-indices before exposing additive day-1 attributions.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import importlib.util
import json
import os
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
LOCKED = ROOT.parent / "config"
KNOWLEDGE = LOCKED / "locked_structures"
END_ONLY = ROOT / "end_only_backend"
FROZEN_EVALUATOR = END_ONLY / "frozen_endpoint_evaluator.py"


ENDPOINT_ORDER = (
    "event_1_death_hospice",
    "event_2_institutional_nonhome",
    "event_5_home",
    "event_3_icu_los_gt7d",
)
ENDPOINT_SHORT = {
    "event_1_death_hospice": "death_hospice",
    "event_2_institutional_nonhome": "institutional_nonhome",
    "event_5_home": "home",
    "event_3_icu_los_gt7d": "icu_los_gt7d",
}
ENDPOINT_FOLDER = {
    "event_1_death_hospice": "event_1_death_hospice",
    "event_2_institutional_nonhome": "event_2_institutional_nonhome",
    "event_5_home": "event_5_home",
    "event_3_icu_los_gt7d": "event_3_icu_los_gt7d",
}
DATA_FILES = {
    "event_1_death_hospice": "event_death_hospice_corrected_time",
    "event_2_institutional_nonhome": "event_care_dependent_survival_after_index_icu",
    "event_5_home": "event_favourable_outcome_after_index_icu",
    "event_3_icu_los_gt7d": "event_3_icu_time_to_exit",
}
DATA_ROOT = Path(os.environ.get("COXGP_DATA_DIR", ROOT.parent / "data"))
ENDPOINT_TITLE = {
    "event_1_death_hospice": "Death or hospice",
    "event_2_institutional_nonhome": "High-dependency institutional discharge",
    "event_5_home": "Favourable disposition after index ICU",
    "event_3_icu_los_gt7d": "Remaining in ICU at day 7",
}
ENDPOINT_LETTER = dict(zip(ENDPOINT_ORDER, "ABCD"))
ENDPOINT_COLOR = {
    "event_1_death_hospice": "#28576A",
    "event_2_institutional_nonhome": "#2A9D8F",
    "event_5_home": "#E76F51",
    "event_3_icu_los_gt7d": "#D49A15",
}

EXPECTED_EXTERNAL = {
    "event_1_death_hospice": 0.8155731225296443,
    "event_2_institutional_nonhome": 0.667590027700831,
    "event_5_home": 0.6924088833862106,
    "event_3_icu_los_gt7d": 0.6845923056473359,
}

ACCEPTED_EXTRA_GP = {
    "event_1_death_hospice": [
        "safe_log(Arterial.BP..Systolic.)",
        "clip(GCS.total)",
    ],
    "event_2_institutional_nonhome": [],
    "event_5_home": [
        "sub(Sodium, age)",
        "sub(Anion.Gap, Total.Ca)",
    ],
    "event_3_icu_los_gt7d": [],
}

DOMAIN_ORDER = (
    "Laboratory",
    "Physiology / neurologic",
    "Patient / admission context",
    "Treatment / support",
    "Cross-domain joint",
)
COMPONENT_ORDER = (
    "Raw clinical",
    "GP transform",
    "Time interaction",
    "Indicator",
    "Logical / joint",
)

LAB_VARS = {
    "Albumin", "Anion.Gap", "BE", "BUN", "Chloride", "CO2.Blood",
    "creatinine", "Eosinophils", "Glucose.Blood", "hematocrit",
    "Hemoglobin", "INR", "Ionized.Ca", "lactate", "Lymphocytes",
    "Magnesium", "PH", "platelets", "Potassium", "PT", "PTT", "RBC",
    "RDW", "Sodium", "Total.Ca", "WBC",
}
PHYS_VARS = {
    "Arterial.BP..Diastolic.", "Arterial.BP..mean.",
    "Arterial.BP..Systolic.", "GCS.total", "HR", "respiration", "PO2",
}
CONTEXT_VARS = {
    "age", "gender_Male", "ethnicity_Caucasian", "Height", "Weight", "BMI",
    "sum_diagnosis", "Los.before.ICU.admission", "admission_type_unplanned_admission",
}
TREATMENT_VARS = {
    "Mechanical.ventilation", "Norepinephrine", "Epinephrine", "Dopamine",
    "Dobutamine", "Paralysis", "Kcl", "cefazolin", "Morphine",
}

FUNCTION_NAMES = {
    "safe_sqrt", "safe_log", "safe_div", "sub", "add", "mul", "clip",
}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FROZEN = _load_module(FROZEN_EVALUATOR, "four_endpoint_frozen_evaluator")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def unique(values) -> tuple[str, ...]:
    return tuple(OrderedDict((str(value), None) for value in values).keys())


def expression_leaves(expression: str) -> tuple[str, ...]:
    return unique(
        token for token in TOKEN_RE.findall(str(expression))
        if token not in FUNCTION_NAMES
    )


def readable_name(name: str) -> str:
    mapping = {
        "BE": "Base excess",
        "BUN": "Blood urea nitrogen",
        "Glucose.Blood": "Blood glucose",
        "PH": "pH",
        "PT": "Prothrombin time",
        "PTT": "Partial thromboplastin time",
        "lactate": "Lactate",
        "GCS.total": "GCS total",
        "Arterial.BP..Systolic.": "Arterial systolic BP",
        "Arterial.BP..mean.": "Mean arterial BP",
        "sum_diagnosis": "Diagnosis count",
        "Total.Ca": "Total calcium",
        "Ionized.Ca": "Ionized calcium",
        "Los.before.ICU.admission": "Pre-ICU LOS",
        "Mechanical.ventilation": "Mechanical ventilation",
        "admission_type_unplanned_admission": "Unplanned admission",
        "ethnicity_Caucasian": "Caucasian ethnicity",
        "gender_Male": "Male sex",
        "cefazolin": "Cefazolin",
        "Kcl": "Potassium chloride",
    }
    return mapping.get(str(name), str(name).replace("_", " ").replace(".", " "))


def split_args(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def readable_expression(expression: str) -> str:
    value = str(expression).strip()
    match = re.fullmatch(r"([A-Za-z_]+)\((.*)\)", value)
    if not match:
        return readable_name(value)
    fn, raw = match.groups()
    args = [readable_expression(item) for item in split_args(raw)]
    if fn == "safe_sqrt":
        return f"sqrt({args[0]})"
    if fn == "safe_log":
        return f"log({args[0]})"
    if fn == "safe_div":
        return f"{args[0]} / {args[1]}"
    if fn == "sub":
        return f"{args[0]} - {args[1]}"
    if fn == "add":
        return f"{args[0]} + {args[1]}"
    if fn == "mul":
        return f"{args[0]} x {args[1]}"
    if fn == "clip":
        return f"clip({args[0]})"
    return f"{fn}({', '.join(args)})"


def domain_for_leaves(leaves: tuple[str, ...]) -> str:
    domains = []
    for value in leaves:
        if value in LAB_VARS:
            domains.append("Laboratory")
        elif value in PHYS_VARS:
            domains.append("Physiology / neurologic")
        elif value in TREATMENT_VARS:
            domains.append("Treatment / support")
        else:
            domains.append("Patient / admission context")
    found = unique(domains)
    return found[0] if len(found) == 1 else "Cross-domain joint"


@dataclass(frozen=True)
class Term:
    label: str
    component: str
    family_id: str
    family_label: str
    family_kind: str
    leaves: tuple[str, ...]
    domain: str
    coefficient: float
    source_feature: str | None = None
    expression: str | None = None
    gp_index: int | None = None
    time_basis: str | None = None


@dataclass(frozen=True)
class Family:
    family_id: str
    label: str
    kind: str
    leaves: tuple[str, ...]
    domain: str
    term_indices: tuple[int, ...]
    has_raw: bool
    has_gp: bool
    has_time: bool


@dataclass
class EndpointModel:
    endpoint: str
    title: str
    payload: dict[str, Any]
    train_raw: pd.DataFrame
    mimic_raw: pd.DataFrame
    train_encoded: pd.DataFrame
    mimic_encoded: pd.DataFrame
    preprocessor: dict[str, Any]
    gp_expressions: list[str]
    gp_scaler: StandardScaler
    gp_train_raw: np.ndarray
    gp_mimic_raw: np.ndarray
    phasea_names: list[str]
    phasea_coefficients: dict[str, float]
    final_coefficients: dict[str, float]
    backbone_multiplier: float
    phaseb: list[dict[str, Any]]
    phasec: list[dict[str, Any]]
    terms: list[Term]
    families: list[Family]
    train_design: np.ndarray
    mimic_design: np.ndarray
    background_mean: np.ndarray
    train_shap: np.ndarray
    mimic_shap: np.ndarray
    train_score: np.ndarray
    mimic_score: np.ndarray
    external_cindex: float
    max_additivity_error: float


def _knowledge_payload(endpoint: str) -> dict[str, Any]:
    return read_json(KNOWLEDGE / ENDPOINT_FOLDER[endpoint] / "result.json")


def _time_basis_label(basis: str) -> str:
    return {
        "zlog3": "zlog(t)", "exp_decay3": "exp-decay(t)",
        "early7": "early(t)", "late7": "late(t)",
        "log1p": "log(1+t)", "linear": "t",
    }.get(str(basis), str(basis))


def _build_final_fit(endpoint: str):
    payload = _knowledge_payload(endpoint)
    stem = DATA_FILES[endpoint]
    train_raw = pd.read_excel(DATA_ROOT / f"{stem}_eicu.xlsx")
    mimic_raw = pd.read_excel(DATA_ROOT / f"{stem}_mimic.xlsx")
    train, prep = FROZEN.fit_preprocess(train_raw)
    mimic = FROZEN.transform_preprocess(mimic_raw, prep)

    main_names = [str(value) for value in payload.get("main_effect_vars", []) or []]
    expressions = FROZEN.source_gp_expressions(payload) + list(ACCEPTED_EXTRA_GP[endpoint])
    gp_train_raw = np.column_stack([FROZEN.eval_gp_expression(expr, train) for expr in expressions])
    gp_mimic_raw = np.column_stack([FROZEN.eval_gp_expression(expr, mimic) for expr in expressions])
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
            times_train, events_train, eta_train, feature_train,
            phaseb, time_stats, static_train,
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
        final_model = CoxPHFitter(penalizer=FROZEN.PHASEC_PENALIZER, l1_ratio=FROZEN.PHASEC_L1_RATIO)
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


def _make_terms_and_design(state: dict[str, Any], reference_time: float = 1.0):
    expressions = state["expressions"]
    main_names = state["main_names"]
    phasea_names = state["phasea_names"]
    backbone = float(state["final_coefficients"].get("eta_ab", 1.0))
    phasea_train = np.column_stack([
        state["train"][main_names].to_numpy(float) if main_names else np.empty((len(state["train"]), 0)),
        state["gp_train"],
    ])
    phasea_mimic = np.column_stack([
        state["mimic"][main_names].to_numpy(float) if main_names else np.empty((len(state["mimic"]), 0)),
        state["gp_mimic"],
    ])
    terms: list[Term] = []
    train_columns: list[np.ndarray] = []
    mimic_columns: list[np.ndarray] = []

    for index, name in enumerate(phasea_names):
        if name.startswith("gp"):
            gp_index = int(name[2:])
            expression = expressions[gp_index - 1]
            leaves = expression_leaves(expression)
            if len(leaves) == 1:
                family_id = f"parent::{leaves[0]}"
                family_label = readable_name(leaves[0])
                family_kind = "parent"
            else:
                family_id = f"joint_gp::{expression}"
                family_label = readable_expression(expression)
                family_kind = "joint_gp"
            label = f"GP{gp_index}: {readable_expression(expression)}"
            component = "GP transform"
            source_feature = None
        else:
            gp_index = None
            expression = None
            leaves = (name,)
            family_id = f"parent::{name}"
            family_label = readable_name(name)
            family_kind = "parent"
            label = readable_name(name)
            component = "Raw clinical"
            source_feature = name
        coefficient = backbone * float(state["phasea_coefficients"][name])
        terms.append(Term(
            label=label, component=component, family_id=family_id,
            family_label=family_label, family_kind=family_kind,
            leaves=leaves, domain=domain_for_leaves(leaves), coefficient=coefficient,
            source_feature=source_feature, expression=expression, gp_index=gp_index,
        ))
        train_columns.append(phasea_train[:, index])
        mimic_columns.append(phasea_mimic[:, index])

    phaseb_by_col = {str(row["interaction_col"]): row for row in state["phaseb"]}
    interaction_pairs = {
        str(row["interaction"]): tuple(str(v) for v in row["interaction_pair"])
        for row in state["phasec"] if row.get("interaction")
    }
    for feature, coefficient in state["final_coefficients"].items():
        if feature == "eta_ab":
            continue
        if feature in phaseb_by_col:
            row = phaseb_by_col[feature]
            base = str(row["feature"])
            basis = str(row["basis"])
            basis_value = float(FROZEN._time_basis(np.asarray([reference_time]), basis, state["time_stats"])[0])
            tr = state["feature_train"][base].to_numpy(float) * basis_value
            mi = state["feature_mimic"][base].to_numpy(float) * basis_value
            if base.startswith("gp"):
                gp_index = int(base[2:])
                expression = expressions[gp_index - 1]
                leaves = expression_leaves(expression)
                source_feature = None
                base_label = f"GP{gp_index}: {readable_expression(expression)}"
            else:
                gp_index = None
                expression = None
                leaves = (base,)
                source_feature = base
                base_label = readable_name(base)
            label = f"{base_label} x {_time_basis_label(basis)}"
            family_id = f"time::{base}::{basis}"
            family_kind = "time"
            component = "Time interaction"
            family_label = label
        else:
            tr = state["static_train"][feature].to_numpy(float)
            mi = state["static_mimic"][feature].to_numpy(float)
            leaves = interaction_pairs.get(feature, (feature,))
            if feature in interaction_pairs:
                label = " AND ".join(readable_name(v) for v in leaves)
                component = "Logical / joint"
                family_id = f"logical::{feature}"
                family_kind = "logical"
            else:
                label = readable_name(feature)
                component = "Indicator"
                family_id = f"indicator::{feature}"
                family_kind = "indicator"
            family_label = label
            source_feature = feature
            expression = None
            gp_index = None
            basis = None
        terms.append(Term(
            label=label, component=component, family_id=family_id,
            family_label=family_label, family_kind=family_kind,
            leaves=tuple(leaves), domain=domain_for_leaves(tuple(leaves)),
            coefficient=float(coefficient), source_feature=source_feature,
            expression=expression, gp_index=gp_index,
            time_basis=basis if feature in phaseb_by_col else None,
        ))
        train_columns.append(tr)
        mimic_columns.append(mi)

    train_design = np.column_stack(train_columns)
    mimic_design = np.column_stack(mimic_columns)
    beta = np.asarray([term.coefficient for term in terms])
    background = train_design.mean(axis=0)
    train_shap = (train_design - background) * beta
    mimic_shap = (mimic_design - background) * beta
    train_score = train_design @ beta
    mimic_score = mimic_design @ beta
    error = max(
        float(np.max(np.abs(background @ beta + train_shap.sum(axis=1) - train_score))),
        float(np.max(np.abs(background @ beta + mimic_shap.sum(axis=1) - mimic_score))),
    )
    grouped: OrderedDict[str, list[int]] = OrderedDict()
    for index, term in enumerate(terms):
        grouped.setdefault(term.family_id, []).append(index)
    families = []
    for family_id, indices in grouped.items():
        first = terms[indices[0]]
        families.append(Family(
            family_id=family_id, label=first.family_label, kind=first.family_kind,
            leaves=unique(leaf for i in indices for leaf in terms[i].leaves),
            domain=first.domain, term_indices=tuple(indices),
            has_raw=any(terms[i].component == "Raw clinical" for i in indices),
            has_gp=any(terms[i].component == "GP transform" for i in indices),
            has_time=any(terms[i].component == "Time interaction" for i in indices),
        ))
    return terms, families, train_design, mimic_design, background, train_shap, mimic_shap, train_score, mimic_score, error


def load_endpoint_model(endpoint: str) -> EndpointModel:
    state = _build_final_fit(endpoint)
    expected = EXPECTED_EXTERNAL[endpoint]
    if abs(state["external_cindex"] - expected) > 1e-12:
        raise AssertionError(
            f"{endpoint}: reconstructed external C-index {state['external_cindex']} != {expected}"
        )
    values = _make_terms_and_design(state)
    return EndpointModel(
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


def load_all_models() -> dict[str, EndpointModel]:
    return {endpoint: load_endpoint_model(endpoint) for endpoint in ENDPOINT_ORDER}


def aggregate_shap(model: EndpointModel, cohort: str = "mimic") -> np.ndarray:
    shap = model.mimic_shap if cohort == "mimic" else model.train_shap
    return np.column_stack([shap[:, family.term_indices].sum(axis=1) for family in model.families])


def relative_importance(matrix: np.ndarray) -> np.ndarray:
    values = np.mean(np.abs(np.asarray(matrix, float)), axis=0)
    total = float(values.sum())
    return values / total if total > 0 else np.zeros_like(values)


def audit_payload(models: dict[str, EndpointModel]) -> dict[str, Any]:
    rows = []
    for endpoint in ENDPOINT_ORDER:
        model = models[endpoint]
        rows.append({
            "endpoint": endpoint,
            "title": model.title,
            "locked_model": "End-only trial 3",
            "external_cindex": model.external_cindex,
            "expected_external_cindex": EXPECTED_EXTERNAL[endpoint],
            "external_abs_error": abs(model.external_cindex - EXPECTED_EXTERNAL[endpoint]),
            "max_shap_additivity_error": model.max_additivity_error,
            "main_effect_vars": model.payload.get("main_effect_vars", []),
            "gp_expressions": model.gp_expressions,
            "phaseB": model.phaseb,
            "phaseC": model.phasec,
            "phaseA_coefficients": model.phasea_coefficients,
            "final_coefficients": model.final_coefficients,
            "terms": [term.__dict__ for term in model.terms],
            "families": [family.__dict__ for family in model.families],
        })
    return {
        "schema": "four_endpoint_locked_end_only_display_audit_v1",
        "model": "End-only trial 3",
        "reference_time_days": 1.0,
        "models": rows,
        "all_external_matches_within_1e_12": all(row["external_abs_error"] <= 1e-12 for row in rows),
    }
