#!/usr/bin/env python
"""DeepSurv and Random Survival Forest baselines for five endpoint datasets.

This script intentionally mirrors the project Cox-GP preprocessing behavior and
keeps MIMIC as external-only validation. It is not an aggressive benchmark search.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path


TIME_COL = "time"
EVENT_COL = "event"
MIN_TIME = 0.001
CATEGORICAL_CANDIDATES = ["gender", "ethnicity", "admission_type"]
ID_EXCLUDE_CANDIDATES = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "subjectid",
    "hadmid",
    "stayid",
    "patientunitstayid",
]
MICE_MAX_ITER = 20
MICE_RANDOM_STATE = 42
MICE_INITIAL_STRATEGY = "median"
MICE_SKIP_COMPLETE = True
WINSOR_Q_LOW = 0.01
WINSOR_Q_HIGH = 0.99
STANDARDIZE_CONTINUOUS = True
STANDARDIZE_EPS = 1e-8
DEFAULT_SEED = 42
OUTPUT_ROOT = Path("deepsurv_rsf_baseline_outputs")

ENDPOINTS = [{'name': 'event_1_death_hospice', 'eicu_path': 'event_1_death_hospice_eicu.xlsx', 'mimic_path': 'event_1_death_hospice_mimic.xlsx', 'expected_eicu_n': 313, 'expected_eicu_events': 25, 'expected_mimic_n': 340, 'expected_mimic_events': 50}, {'name': 'event_2_institutional_nonhome', 'eicu_path': 'event_2_institutional_nonhome_eicu.xlsx', 'mimic_path': 'event_2_institutional_nonhome_mimic.xlsx', 'expected_eicu_n': 313, 'expected_eicu_events': 174, 'expected_mimic_n': 340, 'expected_mimic_events': 233}, {'name': 'event_3_icu_los_gt7d', 'eicu_path': 'event_3_icu_los_gt7d_eicu.xlsx', 'mimic_path': 'event_3_icu_los_gt7d_mimic.xlsx', 'expected_eicu_n': 313, 'expected_eicu_events': 71, 'expected_mimic_n': 340, 'expected_mimic_events': 100}, {'name': 'event_5_home', 'eicu_path': 'event_5_home_eicu.xlsx', 'mimic_path': 'event_5_home_mimic.xlsx', 'expected_eicu_n': 313, 'expected_eicu_events': 111, 'expected_mimic_n': 340, 'expected_mimic_events': 55}]

REQUIREMENTS = [
    "pandas",
    "numpy",
    "scikit-learn",
    "scikit-survival",
    "torch",
    "pycox",
    "torchtuples",
    "openpyxl",
    "joblib",
]


def write_requirements(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "requirements_deepsurv_rsf.txt").write_text(
        "\n".join(REQUIREMENTS) + "\n", encoding="utf-8"
    )


def check_required_packages(out_dir: Path) -> None:
    import_names = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "scikit-survival": "sksurv",
        "torch": "torch",
        "pycox": "pycox",
        "torchtuples": "torchtuples",
        "openpyxl": "openpyxl",
        "joblib": "joblib",
    }
    write_requirements(out_dir)
    missing = []
    for package, module in import_names.items():
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing.append(f"{package} ({type(exc).__name__}: {exc})")
    if missing:
        raise RuntimeError(
            "Required baseline packages are missing. Install packages from "
            f"{out_dir / 'requirements_deepsurv_rsf.txt'} and rerun. Missing: "
            + "; ".join(missing)
        )


def import_runtime_dependencies():
    global np, pd, joblib, torch, tt, CoxPH, RandomSurvivalForest, Surv
    global concordance_index_censored, IterativeImputer, StratifiedKFold
    global StratifiedShuffleSplit

    import joblib
    import numpy as np
    import pandas as pd
    import torch
    import torchtuples as tt
    from pycox.models import CoxPH
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if "np" in globals():
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return str(obj)


def log_line(out_dir: Path, msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with (out_dir / "run_log.txt").open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg, flush=True)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# The following preprocessing functions are copied from
# newcox_ABC_unified_FINAL_CLEAN.ipynb, with only audit-friendly surrounding code
# added elsewhere in this script.
def _coerce_time_event(df):
    df = df.copy()
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce").clip(lower=MIN_TIME)
    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce").fillna(0).astype(int)
    return df


def _is_id_col(c: str) -> bool:
    s = str(c).strip().lower()
    if s in {x.lower() for x in ID_EXCLUDE_CANDIDATES}:
        return True
    return bool(re.search(r"(^id$|^id_|_id$)", s))


def _is_binary_series(s) -> bool:
    vals = pd.to_numeric(s, errors="coerce").dropna().unique()
    if len(vals) == 0:
        return False
    return set(np.asarray(vals, dtype=float)).issubset({0.0, 1.0})


def _fit_mice_on_continuous(df_num):
    if df_num.shape[1] == 0:
        return None, df_num.copy()
    imp = IterativeImputer(
        random_state=MICE_RANDOM_STATE,
        max_iter=MICE_MAX_ITER,
        initial_strategy=MICE_INITIAL_STRATEGY,
        skip_complete=MICE_SKIP_COMPLETE,
        sample_posterior=False,
    )
    arr = imp.fit_transform(df_num)
    return imp, pd.DataFrame(arr, columns=df_num.columns, index=df_num.index)


def _transform_mice_on_continuous(df_num, imp):
    if df_num.shape[1] == 0:
        return df_num.copy()
    if imp is None:
        return df_num.fillna(0.0)
    arr = imp.transform(df_num)
    return pd.DataFrame(arr, columns=df_num.columns, index=df_num.index)


def _fit_winsor_zscore_on_continuous(df_cont):
    stats = {}
    out = df_cont.copy()
    for c in df_cont.columns:
        s = pd.to_numeric(df_cont[c], errors="coerce").astype(float)
        if np.isfinite(s).any():
            q_low = float(np.nanquantile(s, WINSOR_Q_LOW))
            q_high = float(np.nanquantile(s, WINSOR_Q_HIGH))
        else:
            q_low, q_high = 0.0, 0.0
        if q_high < q_low:
            q_low, q_high = q_high, q_low
        sw = s.clip(lower=q_low, upper=q_high)
        if np.isfinite(sw).any():
            mu = float(np.nanmean(sw))
            sd = float(np.nanstd(sw, ddof=0))
        else:
            mu, sd = 0.0, 1.0
        if not np.isfinite(sd) or sd < STANDARDIZE_EPS:
            sd = 1.0
        out[c] = (sw - mu) / sd
        stats[c] = {"q_low": q_low, "q_high": q_high, "mean": mu, "sd": sd}
    return out, stats


def _transform_winsor_zscore_on_continuous(df_cont, stats: dict):
    out = df_cont.copy()
    for c in df_cont.columns:
        s = pd.to_numeric(df_cont[c], errors="coerce").astype(float)
        st = stats.get(c)
        if st is None:
            out[c] = s
            continue
        sd = float(st.get("sd", 1.0))
        if not np.isfinite(sd) or sd < STANDARDIZE_EPS:
            sd = 1.0
        s = s.clip(lower=float(st.get("q_low", np.nan)), upper=float(st.get("q_high", np.nan)))
        out[c] = (s - float(st.get("mean", 0.0))) / sd
    return out


def _apply_cat_fill(df, cat_cols):
    for c in cat_cols:
        df[c] = df[c].astype("object").fillna("UNKNOWN")
    return df


def fit_preprocess(df_raw, cat_candidates=None):
    if cat_candidates is None:
        cat_candidates = CATEGORICAL_CANDIDATES
    df = _coerce_time_event(df_raw).copy()
    cat_cols = [c for c in cat_candidates if c in df.columns]
    id_cols = [c for c in df.columns if _is_id_col(c)]
    base_drop = [TIME_COL, EVENT_COL] + cat_cols + id_cols
    num_cols_all = [c for c in df.columns if c not in base_drop]
    for c in num_cols_all:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    binary_num_cols = [c for c in num_cols_all if _is_binary_series(df[c])]
    cont_cols = [c for c in num_cols_all if c not in binary_num_cols]

    mice_fill_defaults = {}
    for c in cont_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        mice_fill_defaults[c] = float(s.median(skipna=True)) if s.notna().any() else 0.0

    imp, cont_imputed = _fit_mice_on_continuous(df[cont_cols]) if cont_cols else (None, pd.DataFrame(index=df.index))
    cont_standardize_stats = {}
    if cont_cols and STANDARDIZE_CONTINUOUS:
        cont_imputed, cont_standardize_stats = _fit_winsor_zscore_on_continuous(cont_imputed)
    for c in cont_cols:
        df[c] = cont_imputed[c]

    binary_fill = {}
    for c in binary_num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        mode = s.dropna().mode()
        fill_v = float(mode.iloc[0]) if len(mode) else 0.0
        binary_fill[c] = fill_v
        df[c] = s.fillna(fill_v)

    df = _apply_cat_fill(df, cat_cols)
    df_model = df.drop(columns=id_cols, errors="ignore")
    df_enc = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    model_cols = [c for c in df_enc.columns if c not in (TIME_COL, EVENT_COL)]
    categorical_dummy_cols = [c for c in model_cols if c not in cont_cols and c not in binary_num_cols]
    prep = {
        "cat_candidates": list(cat_candidates),
        "cat_cols": cat_cols,
        "id_cols": id_cols,
        "cont_cols": cont_cols,
        "binary_num_cols": binary_num_cols,
        "binary_fill": binary_fill,
        "mice_fill_defaults": mice_fill_defaults,
        "mice_imputer": imp,
        "cont_standardize_stats": cont_standardize_stats,
        "standardize_continuous": STANDARDIZE_CONTINUOUS,
        "winsor_q_low": WINSOR_Q_LOW,
        "winsor_q_high": WINSOR_Q_HIGH,
        "dummy_columns": model_cols,
        "categorical_dummy_cols": categorical_dummy_cols,
    }
    return df_enc[[TIME_COL, EVENT_COL] + model_cols], prep


def transform_preprocess(df_raw, prep):
    df = _coerce_time_event(df_raw).copy()
    cat_cols = [c for c in prep["cat_candidates"] if c in df.columns]
    id_cols = [c for c in df.columns if _is_id_col(c)]
    for c in prep["cont_cols"] + prep["binary_num_cols"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if prep["cont_cols"]:
        cont_df = df[prep["cont_cols"]].copy()
        try:
            cont_imputed = _transform_mice_on_continuous(cont_df, prep["mice_imputer"])
        except Exception:
            cont_imputed = cont_df.copy()
            for c in prep["cont_cols"]:
                cont_imputed[c] = cont_imputed[c].fillna(prep["mice_fill_defaults"].get(c, 0.0))
        if prep.get("standardize_continuous", False):
            cont_imputed = _transform_winsor_zscore_on_continuous(cont_imputed, prep.get("cont_standardize_stats", {}))
        for c in prep["cont_cols"]:
            df[c] = cont_imputed[c]

    for c in prep["binary_num_cols"]:
        fill_v = prep["binary_fill"].get(c, 0.0)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(fill_v)

    df = _apply_cat_fill(df, cat_cols)
    df_model = df.drop(columns=id_cols, errors="ignore")
    df_enc = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
    for col in prep["dummy_columns"]:
        if col not in df_enc.columns:
            df_enc[col] = 0.0
    keep = set(prep["dummy_columns"]) | {TIME_COL, EVENT_COL}
    df_enc = df_enc.drop(columns=[c for c in df_enc.columns if c not in keep])
    return df_enc[[TIME_COL, EVENT_COL] + prep["dummy_columns"]]


def extract_xy(df_encoded):
    feature_cols = [c for c in df_encoded.columns if c not in {TIME_COL, EVENT_COL}]
    bad_features = [c for c in feature_cols if _is_id_col(c) or c in {TIME_COL, EVENT_COL}]
    if bad_features:
        raise ValueError(f"Forbidden feature columns detected: {bad_features}")
    x_df = df_encoded[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_df = x_df.replace([np.inf, -np.inf], np.nan)
    if x_df.isna().any().any():
        raise ValueError("Missing values remain after preprocessing.")
    x = x_df.to_numpy(dtype=np.float32)
    time = pd.to_numeric(df_encoded[TIME_COL], errors="coerce").to_numpy(dtype=np.float32)
    event = pd.to_numeric(df_encoded[EVENT_COL], errors="coerce").fillna(0).astype(int).to_numpy(dtype=np.int32)
    if np.any(~np.isfinite(time)) or np.any(time <= 0):
        raise ValueError("Invalid survival times after preprocessing.")
    if not set(np.unique(event)).issubset({0, 1}):
        raise ValueError("Event column is not binary after preprocessing.")
    if not np.isfinite(x).all():
        raise ValueError("Feature matrix contains non-finite values.")
    return x, time, event, feature_cols


def make_deepsurv_data(x, time, event):
    return {
        "x": np.asarray(x, dtype=np.float32),
        "t": np.asarray(time, dtype=np.float32),
        "e": np.asarray(event, dtype=np.int32),
    }


def make_surv_y(time, event):
    return Surv.from_arrays(
        event=np.asarray(event).astype(bool),
        time=np.asarray(time, dtype=float),
    )


def evaluate_cindex(time, event, risk_score):
    return float(
        concordance_index_censored(
            np.asarray(event).astype(bool),
            np.asarray(time, dtype=float),
            np.asarray(risk_score, dtype=float).reshape(-1),
        )[0]
    )


def get_deepsurv_configs(mode: str):
    if mode == "smoke":
        return [
            {
                "hidden_layers": [16],
                "dropout": 0.1,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "batch_size": 64,
                "max_epochs": 25,
                "patience": 7,
                "batch_norm": False,
            }
        ]
    return [
        {"hidden_layers": [16], "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 64, "max_epochs": 300, "patience": 30, "batch_norm": False},
        {"hidden_layers": [32], "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 64, "max_epochs": 300, "patience": 30, "batch_norm": False},
        {"hidden_layers": [32, 16], "dropout": 0.2, "learning_rate": 0.0003, "weight_decay": 0.0001, "batch_size": 64, "max_epochs": 300, "patience": 30, "batch_norm": False},
        {"hidden_layers": [16], "dropout": 0.2, "learning_rate": 0.0003, "weight_decay": 0.001, "batch_size": 64, "max_epochs": 300, "patience": 30, "batch_norm": False},
    ]


def get_rsf_configs(mode: str):
    if mode == "smoke":
        return [
            {
                "n_estimators": 50,
                "max_depth": 3,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "n_jobs": -1,
            }
        ]
    return [
        {"n_estimators": 300, "max_depth": 3, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": "sqrt", "n_jobs": -1},
        {"n_estimators": 500, "max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": "sqrt", "n_jobs": -1},
        {"n_estimators": 300, "max_depth": None, "min_samples_split": 20, "min_samples_leaf": 10, "max_features": 0.5, "n_jobs": -1},
        {"n_estimators": 500, "max_depth": None, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": 0.5, "n_jobs": -1},
    ]


def make_splits(raw_df, mode: str, seed: int):
    y = _coerce_time_event(raw_df)[EVENT_COL].astype(int).to_numpy()
    if mode == "smoke":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        return list(splitter.split(np.arange(len(y)), y))
    n_splits = min(4, int(np.sum(y == 1)), int(np.sum(y == 0)))
    if n_splits < 2:
        raise ValueError("Not enough events/censored samples for stratified validation.")
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(len(y)), y))


def build_deepsurv_net(n_features: int, params: dict):
    layers = []
    in_features = n_features
    for width in params["hidden_layers"]:
        layers.append(torch.nn.Linear(in_features, int(width)))
        if params.get("batch_norm", False):
            layers.append(torch.nn.BatchNorm1d(int(width)))
        layers.append(torch.nn.ReLU())
        if float(params.get("dropout", 0.0)) > 0:
            layers.append(torch.nn.Dropout(float(params["dropout"])))
        in_features = int(width)
    layers.append(torch.nn.Linear(in_features, 1))
    return torch.nn.Sequential(*layers)


def fit_deepsurv(train_data, valid_data, params: dict, seed: int):
    set_all_seeds(seed)
    net = build_deepsurv_net(train_data["x"].shape[1], params)
    optimizer = tt.optim.Adam(
        lr=float(params["learning_rate"]),
        weight_decay=float(params.get("weight_decay", 0.0)),
    )
    model = CoxPH(net, optimizer)
    callbacks = []
    val_data = None
    if valid_data is not None:
        callbacks = [tt.callbacks.EarlyStopping(patience=int(params.get("patience", 20)))]
        val_data = (valid_data["x"], (valid_data["t"], valid_data["e"].astype("float32")))
    log = model.fit(
        train_data["x"],
        (train_data["t"], train_data["e"].astype("float32")),
        batch_size=int(params.get("batch_size", 64)),
        epochs=int(params.get("max_epochs", 100)),
        callbacks=callbacks,
        val_data=val_data,
        verbose=False,
    )
    try:
        epochs_run = int(len(log.to_pandas()))
    except Exception:
        epochs_run = int(params.get("max_epochs", 100))
    return model, int(epochs_run)


def predict_deepsurv_risk(model, x):
    return np.asarray(model.predict(np.asarray(x, dtype=np.float32))).reshape(-1)


def fit_rsf(x, time, event, params: dict, seed: int):
    model = RandomSurvivalForest(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        n_jobs=int(params.get("n_jobs", -1)),
        random_state=seed,
    )
    model.fit(np.asarray(x, dtype=np.float32), make_surv_y(time, event))
    return model


def predict_rsf_risk(model, x):
    return np.asarray(model.predict(np.asarray(x, dtype=np.float32))).reshape(-1)


def cross_validate_model(raw_eicu, model_name: str, configs: list[dict], mode: str, seed: int):
    rows = []
    splits = make_splits(raw_eicu, mode, seed)
    for config_id, params in enumerate(configs):
        for fold, (tr_idx, va_idx) in enumerate(splits, 1):
            tr_raw = raw_eicu.iloc[tr_idx].reset_index(drop=True)
            va_raw = raw_eicu.iloc[va_idx].reset_index(drop=True)
            tr_enc, prep = fit_preprocess(tr_raw)
            va_enc = transform_preprocess(va_raw, prep)
            x_tr, t_tr, e_tr, _ = extract_xy(tr_enc)
            x_va, t_va, e_va, _ = extract_xy(va_enc)
            cindex = float("nan")
            epochs = None
            err = ""
            try:
                if model_name == "DeepSurv":
                    train_data = make_deepsurv_data(x_tr, t_tr, e_tr)
                    valid_data = make_deepsurv_data(x_va, t_va, e_va)
                    model, epochs = fit_deepsurv(train_data, valid_data, params, seed + fold + config_id)
                    risk = predict_deepsurv_risk(model, x_va)
                elif model_name == "RSF":
                    model = fit_rsf(x_tr, t_tr, e_tr, params, seed + fold + config_id)
                    risk = predict_rsf_risk(model, x_va)
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                cindex = evaluate_cindex(t_va, e_va, risk)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "model": model_name,
                    "config_id": int(config_id),
                    "fold": int(fold),
                    "train_n": int(len(tr_idx)),
                    "valid_n": int(len(va_idx)),
                    "valid_events": int(np.sum(e_va)),
                    "valid_cindex": cindex,
                    "params": json.dumps(jsonable(params), ensure_ascii=False),
                    "epochs": epochs,
                    "error": err,
                }
            )
    return pd.DataFrame(rows)


def select_best_params(cv_df, configs):
    grouped = cv_df.groupby("config_id", dropna=False)["valid_cindex"].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped[np.isfinite(grouped["mean"])]
    if grouped.empty:
        errors = "; ".join([str(x) for x in cv_df.get("error", []) if str(x)])
        raise RuntimeError(f"No valid CV C-index values. {errors}")
    best_id = int(grouped.sort_values(["mean", "count"], ascending=[False, False]).iloc[0]["config_id"])
    return dict(configs[best_id]), best_id


def summarize_cv(cv_df, config_id: int):
    vals = cv_df.loc[cv_df["config_id"] == config_id, "valid_cindex"].astype(float)
    vals = vals[np.isfinite(vals)]
    return {
        "validation_cindex_mean": float(vals.mean()) if len(vals) else float("nan"),
        "validation_cindex_std": float(vals.std(ddof=0)) if len(vals) else float("nan"),
    }


def load_endpoint_data(cfg: dict):
    eicu_path = Path(cfg["eicu_path"])
    mimic_path = Path(cfg["mimic_path"])
    if not eicu_path.exists():
        raise FileNotFoundError(eicu_path)
    if not mimic_path.exists():
        raise FileNotFoundError(mimic_path)
    eicu = _coerce_time_event(pd.read_excel(eicu_path))
    mimic = _coerce_time_event(pd.read_excel(mimic_path))
    return eicu, mimic


def make_data_audit(cfg, raw_eicu, raw_mimic):
    warnings = []
    counts = {
        "eicu_n": int(len(raw_eicu)),
        "eicu_events": int(raw_eicu[EVENT_COL].sum()),
        "mimic_n": int(len(raw_mimic)),
        "mimic_events": int(raw_mimic[EVENT_COL].sum()),
    }
    expected_map = {
        "eicu_n": "expected_eicu_n",
        "eicu_events": "expected_eicu_events",
        "mimic_n": "expected_mimic_n",
        "mimic_events": "expected_mimic_events",
    }
    for got_key, exp_key in expected_map.items():
        if int(cfg[exp_key]) != counts[got_key]:
            warnings.append(f"{got_key}_expected_{cfg[exp_key]}_got_{counts[got_key]}")
    return {
        "endpoint": cfg["name"],
        "eicu_path": cfg["eicu_path"],
        "mimic_path": cfg["mimic_path"],
        **counts,
        "raw_columns": [str(c) for c in raw_eicu.columns],
        "raw_column_count": int(raw_eicu.shape[1]),
        "warnings": warnings,
    }, warnings


def preprocessing_audit(raw_eicu, raw_mimic, enc_eicu, enc_mimic, prep):
    x_eicu = enc_eicu.drop(columns=[TIME_COL, EVENT_COL], errors="ignore")
    x_mimic = enc_mimic.drop(columns=[TIME_COL, EVENT_COL], errors="ignore")
    return {
        "train_sample_size": int(len(raw_eicu)),
        "train_event_count": int(raw_eicu[EVENT_COL].sum()),
        "external_sample_size": int(len(raw_mimic)),
        "external_event_count": int(raw_mimic[EVENT_COL].sum()),
        "raw_columns": [str(c) for c in raw_eicu.columns],
        "raw_column_count": int(raw_eicu.shape[1]),
        "dropped_id_columns": list(prep.get("id_cols", [])),
        "categorical_columns": list(prep.get("cat_cols", [])),
        "continuous_columns": list(prep.get("cont_cols", [])),
        "binary_numeric_columns": list(prep.get("binary_num_cols", [])),
        "categorical_dummy_columns": list(prep.get("categorical_dummy_cols", [])),
        "final_feature_columns": list(prep.get("dummy_columns", [])),
        "encoded_eicu_shape": list(enc_eicu.shape),
        "encoded_mimic_shape": list(enc_mimic.shape),
        "missing_after_preprocessing": {
            "eicu": int(x_eicu.isna().sum().sum()),
            "mimic": int(x_mimic.isna().sum().sum()),
        },
        "infinite_after_preprocessing": {
            "eicu": int(np.isinf(x_eicu.to_numpy(dtype=float)).sum()),
            "mimic": int(np.isinf(x_mimic.to_numpy(dtype=float)).sum()),
        },
        "mimic_transform_used_only_eicu_fitted_preprocessor": True,
        "preprocessing_constants": {
            "TIME_COL": TIME_COL,
            "EVENT_COL": EVENT_COL,
            "MIN_TIME": MIN_TIME,
            "CATEGORICAL_CANDIDATES": CATEGORICAL_CANDIDATES,
            "ID_EXCLUDE_CANDIDATES": ID_EXCLUDE_CANDIDATES,
            "MICE_MAX_ITER": MICE_MAX_ITER,
            "MICE_RANDOM_STATE": MICE_RANDOM_STATE,
            "MICE_INITIAL_STRATEGY": MICE_INITIAL_STRATEGY,
            "MICE_SKIP_COMPLETE": MICE_SKIP_COMPLETE,
            "WINSOR_Q_LOW": WINSOR_Q_LOW,
            "WINSOR_Q_HIGH": WINSOR_Q_HIGH,
            "STANDARDIZE_CONTINUOUS": STANDARDIZE_CONTINUOUS,
            "STANDARDIZE_EPS": STANDARDIZE_EPS,
        },
    }


def save_package_versions(out_dir: Path):
    modules = {
        "python": sys.version,
        "pandas": "pandas",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "scikit-survival": "sksurv",
        "torch": "torch",
        "pycox": "pycox",
        "torchtuples": "torchtuples",
        "openpyxl": "openpyxl",
        "joblib": "joblib",
    }
    versions = {}
    for key, module in modules.items():
        if key == "python":
            versions[key] = module
            continue
        mod = importlib.import_module(module)
        versions[key] = getattr(mod, "__version__", "unknown")
    (out_dir / "package_versions.json").write_text(
        json.dumps(jsonable(versions), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def final_fit_predict(raw_eicu, raw_mimic, prep, enc_eicu, enc_mimic, feature_cols, model_name, params, seed):
    x_eicu, t_eicu, e_eicu, _ = extract_xy(enc_eicu)
    x_mimic, t_mimic, e_mimic, _ = extract_xy(enc_mimic)
    if model_name == "DeepSurv":
        train_data = make_deepsurv_data(x_eicu, t_eicu, e_eicu)
        model, epochs = fit_deepsurv(train_data, None, params, seed)
        risk = predict_deepsurv_risk(model, x_mimic)
        return model, risk, evaluate_cindex(t_mimic, e_mimic, risk), epochs
    if model_name == "RSF":
        model = fit_rsf(x_eicu, t_eicu, e_eicu, params, seed)
        risk = predict_rsf_risk(model, x_mimic)
        return model, risk, evaluate_cindex(t_mimic, e_mimic, risk), None
    raise ValueError(model_name)


def run_endpoint(cfg: dict, mode: str, seed: int, mode_dir: Path, force: bool):
    endpoint = cfg["name"]
    endpoint_dir = mode_dir / endpoint
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    if (endpoint_dir / "result.json").exists() and not force:
        cached = json.loads((endpoint_dir / "result.json").read_text(encoding="utf-8"))
        return cached["summary_rows"]

    log_line(mode_dir, f"[{endpoint}] loading data")
    raw_eicu, raw_mimic = load_endpoint_data(cfg)
    data_audit, warnings = make_data_audit(cfg, raw_eicu, raw_mimic)
    save_json(endpoint_dir / "data_audit.json", data_audit)

    deepsurv_configs = get_deepsurv_configs(mode)
    rsf_configs = get_rsf_configs(mode)
    ds_cv = cross_validate_model(raw_eicu, "DeepSurv", deepsurv_configs, mode, seed)
    rsf_cv = cross_validate_model(raw_eicu, "RSF", rsf_configs, mode, seed)
    ds_cv.to_csv(endpoint_dir / "cv_results_deepsurv.csv", index=False)
    rsf_cv.to_csv(endpoint_dir / "cv_results_rsf.csv", index=False)

    best_ds, best_ds_id = select_best_params(ds_cv, deepsurv_configs)
    best_rsf, best_rsf_id = select_best_params(rsf_cv, rsf_configs)
    save_json(endpoint_dir / "best_params_deepsurv.json", best_ds)
    save_json(endpoint_dir / "best_params_rsf.json", best_rsf)

    enc_eicu, prep = fit_preprocess(raw_eicu)
    enc_mimic = transform_preprocess(raw_mimic, prep)
    x_eicu, t_eicu, e_eicu, feature_cols = extract_xy(enc_eicu)
    x_mimic, t_mimic, e_mimic, mimic_feature_cols = extract_xy(enc_mimic)
    if feature_cols != mimic_feature_cols:
        raise ValueError("MIMIC feature columns do not match eICU-fitted feature columns.")

    prep_audit = preprocessing_audit(raw_eicu, raw_mimic, enc_eicu, enc_mimic, prep)
    save_json(endpoint_dir / "preprocessing_audit.json", prep_audit)
    save_json(endpoint_dir / "feature_names.json", feature_cols)
    save_json(endpoint_dir / "feature_names_deepsurv.json", feature_cols)
    save_json(endpoint_dir / "feature_names_rsf.json", feature_cols)
    joblib.dump(prep, endpoint_dir / "preprocessor_full_eicu.joblib")

    ds_model, ds_risk, ds_external, ds_epochs = final_fit_predict(
        raw_eicu, raw_mimic, prep, enc_eicu, enc_mimic, feature_cols, "DeepSurv", best_ds, seed
    )
    rsf_model, rsf_risk, rsf_external, _ = final_fit_predict(
        raw_eicu, raw_mimic, prep, enc_eicu, enc_mimic, feature_cols, "RSF", best_rsf, seed
    )

    torch.save(ds_model.net.state_dict(), endpoint_dir / "deepsurv_model.pt")
    joblib.dump(rsf_model, endpoint_dir / "rsf_model.joblib")

    pred_cols = [c for c in ID_EXCLUDE_CANDIDATES if c in raw_mimic.columns]
    for wanted in ["subject_id", "hadm_id", "stay_id"]:
        if wanted not in raw_mimic.columns:
            warnings.append(f"missing_prediction_id_column:{wanted}")
    pred = raw_mimic[pred_cols + [TIME_COL, EVENT_COL]].copy()
    pred["risk_score_deepsurv"] = ds_risk
    pred["risk_score_rsf"] = rsf_risk
    pred.to_csv(endpoint_dir / "mimic_predictions.csv", index=False)

    ds_cv_sum = summarize_cv(ds_cv, best_ds_id)
    rsf_cv_sum = summarize_cv(rsf_cv, best_rsf_id)
    common = {
        "endpoint": endpoint,
        "mode": mode,
        "train_n": int(len(raw_eicu)),
        "train_events": int(raw_eicu[EVENT_COL].sum()),
        "mimic_n": int(len(raw_mimic)),
        "mimic_events": int(raw_mimic[EVENT_COL].sum()),
        "n_raw_columns": int(raw_eicu.shape[1]),
        "n_features_after_preprocessing": int(len(feature_cols)),
        "random_seed": int(seed),
        "output_path": str(endpoint_dir),
        "warnings": "; ".join(sorted(set(warnings))),
    }
    ds_row = {
        **common,
        "model": "DeepSurv",
        **ds_cv_sum,
        "selected_hyperparameters": json.dumps(jsonable(best_ds), ensure_ascii=False),
        "final_training_epochs_if_applicable": ds_epochs,
        "external_mimic_cindex": ds_external,
        "notes": "smoke-mode only; not formal" if mode == "smoke" else "formal limited baseline",
    }
    rsf_row = {
        **common,
        "model": "RSF",
        **rsf_cv_sum,
        "selected_hyperparameters": json.dumps(jsonable(best_rsf), ensure_ascii=False),
        "final_training_epochs_if_applicable": None,
        "external_mimic_cindex": rsf_external,
        "notes": "smoke-mode only; not formal" if mode == "smoke" else "formal limited baseline",
    }
    result_rows = [ds_row, rsf_row]
    pd.DataFrame(result_rows).to_excel(endpoint_dir / "result.xlsx", index=False)
    save_json(
        endpoint_dir / "result.json",
        {
            "endpoint": endpoint,
            "summary_rows": result_rows,
            "data_audit": data_audit,
            "preprocessing_audit": prep_audit,
        },
    )
    log_line(mode_dir, f"[{endpoint}] completed smoke/full flow")
    return result_rows


def failure_rows(cfg: dict, mode: str, seed: int, output_path: Path, exc: Exception):
    msg = f"{type(exc).__name__}: {exc}"
    base = {
        "endpoint": cfg["name"],
        "mode": mode,
        "train_n": None,
        "train_events": None,
        "mimic_n": None,
        "mimic_events": None,
        "n_features_after_preprocessing": None,
        "validation_cindex_mean": None,
        "validation_cindex_std": None,
        "external_mimic_cindex": None,
        "selected_hyperparameters": None,
        "random_seed": int(seed),
        "output_path": str(output_path / cfg["name"]),
        "warnings": msg,
    }
    return [{**base, "model": "DeepSurv"}, {**base, "model": "RSF"}]


def save_summary(rows: list[dict], mode_dir: Path):
    df = pd.DataFrame(rows)
    summary_cols = [
        "endpoint",
        "model",
        "mode",
        "train_n",
        "train_events",
        "mimic_n",
        "mimic_events",
        "n_features_after_preprocessing",
        "validation_cindex_mean",
        "validation_cindex_std",
        "external_mimic_cindex",
        "selected_hyperparameters",
        "output_path",
        "warnings",
    ]
    for c in summary_cols:
        if c not in df.columns:
            df[c] = None
    df = df[summary_cols]
    df.to_csv(mode_dir / "summary_results.csv", index=False)
    df.to_excel(mode_dir / "summary_results.xlsx", index=False)
    return df


def print_final_table(summary_df):
    rows = []
    for endpoint, sub in summary_df.groupby("endpoint", sort=False):
        ds = sub[sub["model"] == "DeepSurv"]
        rsf = sub[sub["model"] == "RSF"]
        rows.append(
            {
                "endpoint": endpoint,
                "deepsurv_external": ds["external_mimic_cindex"].iloc[0] if len(ds) else None,
                "rsf_external": rsf["external_mimic_cindex"].iloc[0] if len(rsf) else None,
                "deepsurv_validation": ds["validation_cindex_mean"].iloc[0] if len(ds) else None,
                "rsf_validation": rsf["validation_cindex_mean"].iloc[0] if len(rsf) else None,
                "warnings": "; ".join([str(x) for x in sub["warnings"].dropna().unique() if str(x)]),
            }
        )
    table = pd.DataFrame(rows)
    print("\nSmoke/full concise result table:")
    print(table.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--endpoint", choices=[cfg["name"] for cfg in ENDPOINTS])
    return parser.parse_args()


def main():
    args = parse_args()
    mode_dir = OUTPUT_ROOT / args.mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    write_requirements(mode_dir)
    check_required_packages(mode_dir)
    import_runtime_dependencies()
    set_all_seeds(args.seed)
    save_package_versions(mode_dir)
    log_line(mode_dir, f"Starting DeepSurv/RSF baseline run mode={args.mode} seed={args.seed}")
    selected = [cfg for cfg in ENDPOINTS if args.endpoint in (None, cfg["name"])]
    all_rows = []
    for cfg in selected:
        try:
            all_rows.extend(run_endpoint(cfg, args.mode, args.seed, mode_dir, args.force))
        except Exception as exc:
            log_line(mode_dir, f"[{cfg['name']}] FAILED: {type(exc).__name__}: {exc}")
            log_line(mode_dir, traceback.format_exc())
            all_rows.extend(failure_rows(cfg, args.mode, args.seed, mode_dir, exc))
    summary_df = save_summary(all_rows, mode_dir)
    print_final_table(summary_df)
    log_line(mode_dir, f"Finished run. Summary: {mode_dir / 'summary_results.xlsx'}")


if __name__ == "__main__":
    main()
