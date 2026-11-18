from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.utils import concordance_index
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


TIME_COL = "time"
EVENT_COL = "event"
MIN_TIME = 0.001
CATEGORICAL_CANDIDATES = ("gender", "ethnicity", "admission_type")
ID_EXCLUDE_CANDIDATES = {
    "subject_id", "hadm_id", "stay_id", "subjectid", "hadmid", "stayid",
    "patientunitstayid",
}
COX_PENALIZER = 0.01
COX_L1_RATIO = 0.0
PHASEB_PENALIZER = 0.05
PHASEB_INTERVAL_CUTS = (3.0, 7.0, 14.0)
PHASEB_T_WINSOR_Q = 0.95
PHASEC_PENALIZER = 0.05
PHASEC_L1_RATIO = 0.5


def _is_id_col(name: str) -> bool:
    value = str(name).strip().lower()
    return value in ID_EXCLUDE_CANDIDATES or bool(
        re.search(r"(^id$|^id_|_id$)", value)
    )


def _is_binary(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().unique()
    return bool(len(values)) and set(np.asarray(values, dtype=float)).issubset({0.0, 1.0})


def fit_preprocess(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = df_raw.copy()
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").clip(lower=MIN_TIME)
    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce").fillna(0).astype(int)
    cat_cols = [name for name in CATEGORICAL_CANDIDATES if name in df]
    id_cols = [name for name in df if _is_id_col(name)]
    numeric = [name for name in df if name not in [TIME_COL, EVENT_COL, *cat_cols, *id_cols]]
    for name in numeric:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    binary = [name for name in numeric if _is_binary(df[name])]
    continuous = [name for name in numeric if name not in binary]
    defaults = {
        name: float(df[name].median()) if df[name].notna().any() else 0.0
        for name in continuous
    }
    imputer = None
    stats: dict[str, dict[str, float]] = {}
    if continuous:
        imputer = IterativeImputer(
            random_state=42,
            max_iter=20,
            initial_strategy="median",
            skip_complete=True,
            sample_posterior=False,
        )
        values = pd.DataFrame(
            imputer.fit_transform(df[continuous]),
            columns=continuous,
            index=df.index,
        )
        for name in continuous:
            series = values[name].astype(float)
            low = float(np.nanquantile(series, 0.01))
            high = float(np.nanquantile(series, 0.99))
            clipped = series.clip(low, high)
            mean = float(np.nanmean(clipped))
            std = float(np.nanstd(clipped, ddof=0))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            df[name] = (clipped - mean) / std
            stats[name] = {"low": low, "high": high, "mean": mean, "std": std}
    binary_fill: dict[str, float] = {}
    for name in binary:
        mode = df[name].dropna().mode()
        fill = float(mode.iloc[0]) if len(mode) else 0.0
        binary_fill[name] = fill
        df[name] = df[name].fillna(fill)
    for name in cat_cols:
        df[name] = df[name].astype("object").fillna("UNKNOWN")
    encoded = pd.get_dummies(
        df.drop(columns=id_cols, errors="ignore"),
        columns=cat_cols,
        drop_first=True,
    )
    model_cols = [name for name in encoded if name not in {TIME_COL, EVENT_COL}]
    prep = {
        "cat_cols": cat_cols,
        "continuous": continuous,
        "binary": binary,
        "defaults": defaults,
        "imputer": imputer,
        "stats": stats,
        "binary_fill": binary_fill,
        "model_cols": model_cols,
    }
    return encoded[[TIME_COL, EVENT_COL, *model_cols]], prep


def transform_preprocess(df_raw: pd.DataFrame, prep: dict[str, Any]) -> pd.DataFrame:
    df = df_raw.copy()
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").clip(lower=MIN_TIME)
    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce").fillna(0).astype(int)
    for name in [*prep["continuous"], *prep["binary"]]:
        if name not in df:
            df[name] = np.nan
        df[name] = pd.to_numeric(df[name], errors="coerce")
    if prep["continuous"]:
        try:
            values = pd.DataFrame(
                prep["imputer"].transform(df[prep["continuous"]]),
                columns=prep["continuous"],
                index=df.index,
            )
        except Exception:
            values = df[prep["continuous"]].copy()
            for name in prep["continuous"]:
                values[name] = values[name].fillna(prep["defaults"][name])
        for name in prep["continuous"]:
            stat = prep["stats"][name]
            clipped = values[name].astype(float).clip(stat["low"], stat["high"])
            df[name] = (clipped - stat["mean"]) / stat["std"]
    for name in prep["binary"]:
        df[name] = df[name].fillna(prep["binary_fill"][name])
    cat_cols = []
    for name in prep["cat_cols"]:
        if name not in df:
            df[name] = "UNKNOWN"
        df[name] = df[name].astype("object").fillna("UNKNOWN")
        cat_cols.append(name)
    encoded = pd.get_dummies(
        df.drop(columns=[name for name in df if _is_id_col(name)], errors="ignore"),
        columns=cat_cols,
        drop_first=True,
    )
    for name in prep["model_cols"]:
        if name not in encoded:
            encoded[name] = 0.0
    return encoded[[TIME_COL, EVENT_COL, *prep["model_cols"]]]


def _split_args(text: str) -> list[str]:
    output: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            output.append(text[start:index].strip())
            start = index + 1
    output.append(text[start:].strip())
    return output


def eval_gp_expression(expression: str, frame: pd.DataFrame) -> np.ndarray:
    text = str(expression).strip()
    if text in frame:
        return pd.to_numeric(frame[text], errors="coerce").fillna(0.0).to_numpy(float)
    try:
        return np.full(len(frame), float(text))
    except Exception:
        pass
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", text)
    if not match:
        raise ValueError(f"unsupported GP expression: {expression}")
    name, arguments = match.groups()
    args = [eval_gp_expression(value, frame) for value in _split_args(arguments)]
    if name == "add":
        return args[0] + args[1]
    if name == "sub":
        return args[0] - args[1]
    if name == "mul":
        return args[0] * args[1]
    if name == "safe_div":
        return args[0] / (args[1] + 1e-9)
    if name == "safe_log":
        return np.log(np.abs(args[0]) + 1e-9)
    if name == "safe_sqrt":
        return np.sqrt(np.abs(args[0]) + 1e-9)
    if name == "clip":
        return np.clip(args[0], -10.0, 10.0)
    raise ValueError(f"unsupported GP primitive: {name}")


def _records(payload: dict[str, Any], key: str, nested: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not value and isinstance(payload.get(nested), dict):
        value = payload[nested].get(key)
    return [dict(row) for row in (value or []) if isinstance(row, dict)]


def source_gp_expressions(payload: dict[str, Any]) -> list[str]:
    rows = _records(payload, "final_gp_expressions", "phaseA")
    return [str(row.get("expression")) for row in rows if row.get("expression")]


def source_phaseb(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return _records(payload, "phaseb_timebasis_final_selected_candidates", "phaseB")


def source_phasec(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return _records(payload, "phasec_stgp_and_final_selected_candidates", "phaseC")


def _phasec_frame(encoded: pd.DataFrame, candidates: list[dict[str, Any]]) -> pd.DataFrame:
    output = pd.DataFrame(index=np.arange(len(encoded)))
    data = encoded.reset_index(drop=True)
    for candidate in candidates:
        for name in candidate.get("main_effects", []) or []:
            if name not in data:
                raise ValueError(f"missing PhaseC main effect {name}")
            if name not in output:
                output[name] = pd.to_numeric(data[name], errors="coerce").fillna(0.0).to_numpy(float)
        interaction = candidate.get("interaction")
        pair = candidate.get("interaction_pair")
        if interaction and isinstance(pair, list) and len(pair) >= 2:
            left, right = pair[:2]
            if left not in data or right not in data:
                raise ValueError(f"missing PhaseC interaction pair {left}, {right}")
            if interaction not in output:
                output[interaction] = (
                    pd.to_numeric(data[left], errors="coerce").fillna(0.0).astype(bool)
                    & pd.to_numeric(data[right], errors="coerce").fillna(0.0).astype(bool)
                ).astype(float)
    return output.loc[:, ~output.columns.duplicated()].copy()


def _time_stats(times: np.ndarray) -> dict[str, float]:
    values = np.clip(np.asarray(times, float), MIN_TIME, None)
    log_values = np.log1p(values)
    quantile = float(np.nanquantile(values, PHASEB_T_WINSOR_Q))
    winsor = np.minimum(values, quantile)
    return {
        "mean_logt": float(np.mean(log_values)),
        "sd_logt": max(float(np.std(log_values, ddof=0)), 1e-8),
        "t_winsor_q": quantile,
        "mean_t_winsor": float(np.mean(winsor)),
        "sd_t_winsor": max(float(np.std(winsor, ddof=0)), 1e-8),
    }


def _time_basis(times: np.ndarray, basis: str, stats: dict[str, float]) -> np.ndarray:
    values = np.clip(np.asarray(times, float), MIN_TIME, None)
    zlog = (np.log1p(values) - stats["mean_logt"]) / stats["sd_logt"]
    scaled = (np.minimum(values, stats["t_winsor_q"]) - stats["mean_t_winsor"]) / stats["sd_t_winsor"]
    mapping = {
        "zlog": zlog, "zlog2": zlog ** 2, "zlog3": zlog ** 3,
        "early3": (values <= 3).astype(float), "early7": (values <= 7).astype(float),
        "late7": (values > 7).astype(float), "exp_decay3": np.exp(-values / 3),
        "exp_decay7": np.exp(-values / 7), "inv_decay3": 1 / (1 + values / 3),
        "inv_decay7": 1 / (1 + values / 7), "t_scaled": scaled,
        "t_scaled2": scaled ** 2, "t_scaled3": scaled ** 3,
    }
    if basis == "hinge_after3":
        return np.maximum(0.0, zlog - (np.log1p(3.0) - stats["mean_logt"]) / stats["sd_logt"])
    if basis == "hinge_after7":
        return np.maximum(0.0, zlog - (np.log1p(7.0) - stats["mean_logt"]) / stats["sd_logt"])
    if basis not in mapping:
        raise ValueError(f"unknown PhaseB basis {basis}")
    return mapping[basis]


def _feature_frame(main: np.ndarray, main_names: list[str], gp: np.ndarray) -> pd.DataFrame:
    output = pd.DataFrame(index=np.arange(len(gp)))
    for index, name in enumerate(main_names):
        output[name] = main[:, index]
    for index in range(gp.shape[1]):
        output[f"gp{index + 1}"] = gp[:, index]
    return output


def _long_frame(
    times: np.ndarray,
    events: np.ndarray,
    eta: np.ndarray,
    features: pd.DataFrame,
    phaseb: list[dict[str, Any]],
    stats: dict[str, float],
    static: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for index, raw_time in enumerate(times):
        end = float(np.clip(raw_time, MIN_TIME, None))
        start = 0.0
        for stop in [cut for cut in PHASEB_INTERVAL_CUTS if cut < end] + [end]:
            row = {
                "id": index,
                "start": start,
                "stop": float(stop),
                "event": int(events[index] == 1 and abs(stop - end) < 1e-12),
                "eta_ab": float(eta[index]) if np.isfinite(eta[index]) else 0.0,
            }
            for candidate in phaseb:
                feature = str(candidate["feature"])
                value = float(features.loc[index, feature]) if feature in features else 0.0
                row[str(candidate["interaction_col"])] = value * float(
                    _time_basis(np.asarray([stop]), str(candidate["basis"]), stats)[0]
                )
            for name in static:
                row[name] = float(static.loc[index, name])
            rows.append(row)
            start = float(stop)
    return pd.DataFrame(rows)


def _time_dependent_cindex(
    times: np.ndarray,
    events: np.ndarray,
    eta: np.ndarray,
    features: pd.DataFrame,
    phaseb: list[dict[str, Any]],
    stats: dict[str, float],
    static: pd.DataFrame,
    coefficients: dict[str, float],
) -> float:
    concordant = ties = permissible = 0.0
    for index in np.where(np.asarray(events, int) == 1)[0]:
        comparable = np.where(np.asarray(times, float) > float(times[index]))[0]
        if not len(comparable):
            continue
        risk = float(coefficients.get("eta_ab", 1.0)) * eta.copy()
        for candidate in phaseb:
            feature = str(candidate["feature"])
            basis = float(_time_basis(np.asarray([times[index]]), str(candidate["basis"]), stats)[0])
            risk += float(coefficients.get(str(candidate["interaction_col"]), 0.0)) * features[feature].to_numpy(float) * basis
        for name in static:
            risk += float(coefficients.get(name, 0.0)) * static[name].to_numpy(float)
        other = risk[comparable]
        concordant += float(np.sum(risk[index] > other))
        ties += float(np.sum(np.isclose(risk[index], other)))
        permissible += float(len(comparable))
    return float((concordant + 0.5 * ties) / permissible) if permissible else math.nan


def prepare_frozen_state(
    train_raw: pd.DataFrame,
    apply_raw: pd.DataFrame,
    payload: dict[str, Any],
) -> dict[str, Any]:
    train, prep = fit_preprocess(train_raw)
    apply = transform_preprocess(apply_raw, prep)
    return {"train": train, "apply": apply, "payload": payload}


def evaluate_prepared_frozen_endpoint(
    state: dict[str, Any],
    extra_gp_expressions: list[str] | None = None,
    extra_phasec_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    train = state["train"]
    apply = state["apply"]
    payload = state["payload"]
    main_names = [str(name) for name in payload.get("main_effect_vars", []) or []]
    for name in main_names:
        if name not in train:
            raise ValueError(f"missing main effect {name}")
    expressions = source_gp_expressions(payload) + list(extra_gp_expressions or [])
    if not expressions:
        raise ValueError("frozen endpoint has no GP expressions")
    gp_train = np.column_stack([eval_gp_expression(expr, train) for expr in expressions])
    gp_apply = np.column_stack([eval_gp_expression(expr, apply) for expr in expressions])
    scaler = StandardScaler()
    gp_train_z = scaler.fit_transform(gp_train)
    gp_apply_z = scaler.transform(gp_apply)
    main_train = train[main_names].to_numpy(float) if main_names else np.empty((len(train), 0))
    main_apply = apply[main_names].to_numpy(float) if main_names else np.empty((len(apply), 0))
    columns = [*main_names, *[f"gp{i + 1}" for i in range(gp_train.shape[1])]]
    x_train = pd.DataFrame(np.column_stack([main_train, gp_train_z]), columns=columns)
    x_apply = pd.DataFrame(np.column_stack([main_apply, gp_apply_z]), columns=columns)
    fit = x_train.copy()
    fit[TIME_COL] = pd.to_numeric(train[TIME_COL], errors="coerce").to_numpy(float)
    fit[EVENT_COL] = pd.to_numeric(train[EVENT_COL], errors="coerce").fillna(0).to_numpy(int)
    model = CoxPHFitter(penalizer=COX_PENALIZER, l1_ratio=COX_L1_RATIO)
    model.fit(fit, duration_col=TIME_COL, event_col=EVENT_COL)
    eta_train = model.predict_log_partial_hazard(x_train).to_numpy(float)
    eta_apply = model.predict_log_partial_hazard(x_apply).to_numpy(float)
    times_train = fit[TIME_COL].to_numpy(float)
    events_train = fit[EVENT_COL].to_numpy(int)
    times_apply = pd.to_numeric(apply[TIME_COL], errors="coerce").to_numpy(float)
    events_apply = pd.to_numeric(apply[EVENT_COL], errors="coerce").fillna(0).to_numpy(int)
    a_cindex = float(concordance_index(times_apply, -eta_apply, events_apply))

    phaseb = source_phaseb(payload)
    phasec = source_phasec(payload) + list(extra_phasec_candidates or [])
    static_train = _phasec_frame(train, phasec)
    static_apply = _phasec_frame(apply, phasec).reindex(columns=static_train.columns, fill_value=0.0)
    feature_train = _feature_frame(main_train, main_names, gp_train)
    feature_apply = _feature_frame(main_apply, main_names, gp_apply)
    if phaseb:
        stats = _time_stats(times_train)
        long = _long_frame(
            times_train, events_train, eta_train, feature_train,
            phaseb, stats, static_train,
        )
        design = ["eta_ab", *[str(row["interaction_col"]) for row in phaseb], *list(static_train.columns)]
        usable = [name for name in design if name in long and float(pd.to_numeric(long[name], errors="coerce").std(ddof=0)) > 1e-8]
        ctv = CoxTimeVaryingFitter(penalizer=PHASEB_PENALIZER)
        ctv.fit(
            long[["id", "start", "stop", "event", *usable]],
            id_col="id", start_col="start", stop_col="stop", event_col="event",
        )
        coefficients = {str(name): float(value) for name, value in ctv.params_.to_dict().items()}
        final_cindex = _time_dependent_cindex(
            times_apply, events_apply, eta_apply, feature_apply,
            phaseb, stats, static_apply, coefficients,
        )
        backend = "ABC_timevarying" if phasec else "AB_timevarying"
    elif len(static_train.columns):
        phasec_train = pd.DataFrame({"eta_ab": eta_train})
        phasec_apply = pd.DataFrame({"eta_ab": eta_apply})
        for name in static_train:
            phasec_train[name] = static_train[name].to_numpy(float)
            phasec_apply[name] = static_apply[name].to_numpy(float)
        phasec_fit = phasec_train.copy()
        phasec_fit[TIME_COL] = times_train
        phasec_fit[EVENT_COL] = events_train
        phasec_model = CoxPHFitter(penalizer=PHASEC_PENALIZER, l1_ratio=PHASEC_L1_RATIO)
        phasec_model.fit(phasec_fit, duration_col=TIME_COL, event_col=EVENT_COL)
        risk = phasec_model.predict_partial_hazard(phasec_apply).to_numpy(float)
        final_cindex = float(concordance_index(times_apply, -risk, events_apply))
        backend = "AC_static"
    else:
        final_cindex = a_cindex
        backend = "A_static"
    return {
        "external_cindex": float(final_cindex),
        "A_static_cindex": a_cindex,
        "final_model": backend,
        "gp_expression_n": len(expressions),
        "phaseB_interaction_n": len(phaseb),
        "phaseC_candidate_n": len(phasec),
        "preprocessing": "MICE + training-only 1%/99% winsorized z-score",
    }


def evaluate_frozen_endpoint(
    train_raw: pd.DataFrame,
    apply_raw: pd.DataFrame,
    payload: dict[str, Any],
    extra_gp_expressions: list[str] | None = None,
    extra_phasec_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    state = prepare_frozen_state(train_raw, apply_raw, payload)
    return evaluate_prepared_frozen_endpoint(
        state,
        extra_gp_expressions=extra_gp_expressions,
        extra_phasec_candidates=extra_phasec_candidates,
    )


def prepare_frozen_cv_states(
    raw: pd.DataFrame,
    payload: dict[str, Any],
    folds: int = 4,
    seed: int = 20260715,
) -> list[dict[str, Any]]:
    data = raw.reset_index(drop=True)
    events = pd.to_numeric(data[EVENT_COL], errors="coerce").fillna(0).astype(int)
    splitter = (
        KFold(n_splits=folds, shuffle=True, random_state=seed)
        if events.nunique() == 1
        else StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    )
    split_args = (data,) if events.nunique() == 1 else (data, events)
    return [
        prepare_frozen_state(
            data.iloc[train_index].reset_index(drop=True),
            data.iloc[valid_index].reset_index(drop=True),
            payload,
        )
        for train_index, valid_index in splitter.split(*split_args)
    ]


def cross_validated_prepared_states(
    states: list[dict[str, Any]],
    extra_gp_expressions: list[str] | None = None,
    extra_phasec_candidates: list[dict[str, Any]] | None = None,
) -> tuple[float, list[float], str, int]:
    scores = []
    failures = 0
    for state in states:
        try:
            result = evaluate_prepared_frozen_endpoint(
                state,
                extra_gp_expressions=extra_gp_expressions,
                extra_phasec_candidates=extra_phasec_candidates,
            )
            scores.append(float(result["external_cindex"]))
        except Exception:
            failures += 1
            scores.append(0.5)
    return float(np.mean(scores)), scores, "cox_failed" if failures else "ok", failures


def cross_validated_frozen_endpoint(
    raw: pd.DataFrame,
    payload: dict[str, Any],
    extra_gp_expressions: list[str] | None = None,
    extra_phasec_candidates: list[dict[str, Any]] | None = None,
    folds: int = 4,
    seed: int = 20260715,
) -> tuple[float, list[float], str, int]:
    states = prepare_frozen_cv_states(raw, payload, folds=folds, seed=seed)
    return cross_validated_prepared_states(
        states,
        extra_gp_expressions=extra_gp_expressions,
        extra_phasec_candidates=extra_phasec_candidates,
    )
