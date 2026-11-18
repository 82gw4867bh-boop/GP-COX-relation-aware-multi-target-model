#!/usr/bin/env python
"""Limited DeepSurv/RSF grid runner with external-only MIMIC reporting.

Modes:
- preflight: inspect local files/packages and write a readiness report.
- run: run the limited eICU-only model-selection grid and evaluate MIMIC once.
- report: rebuild comparison files from existing outputs; no training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import py_compile
import sys
import traceback
from datetime import datetime
from pathlib import Path

import run_deepsurv_rsf_baselines as base


OUTPUT_ROOT = Path("deepsurv_rsf_limited_grid_outputs")
DEFAULT_MY_ALGO_SUMMARY = Path(
    "multitarget_5endpoint/outputs/"
    "goal_mode_5endpoint_stage1_top3_with_deepsurv_rsf_summary.xlsx"
)

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

IMPORT_NAMES = {
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

DEEPSURV_CONFIGS = [
    {
        "config_id": "DS-1_small_standard",
        "hidden_layers": [16],
        "dropout": 0.10,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 300,
        "patience": 30,
        "batch_size": "full_or_64",
        "batch_norm": False,
    },
    {
        "config_id": "DS-2_small_wider",
        "hidden_layers": [32],
        "dropout": 0.10,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 300,
        "patience": 30,
        "batch_size": "full_or_64",
        "batch_norm": False,
    },
    {
        "config_id": "DS-3_two_layer_regularized",
        "hidden_layers": [32, 16],
        "dropout": 0.20,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "max_epochs": 300,
        "patience": 30,
        "batch_size": "full_or_64",
        "batch_norm": False,
    },
    {
        "config_id": "DS-4_conservative_regularized",
        "hidden_layers": [16],
        "dropout": 0.20,
        "lr": 3e-4,
        "weight_decay": 1e-3,
        "max_epochs": 300,
        "patience": 30,
        "batch_size": "full_or_64",
        "batch_norm": False,
    },
]

RSF_CONFIGS = [
    {
        "config_id": "RSF-1_shallow_conservative",
        "n_estimators": 300,
        "max_depth": 3,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "n_jobs": -1,
    },
    {
        "config_id": "RSF-2_medium",
        "n_estimators": 500,
        "max_depth": 5,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "n_jobs": -1,
    },
    {
        "config_id": "RSF-3_deep_regularized",
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": 0.5,
        "n_jobs": -1,
    },
    {
        "config_id": "RSF-4_doc_style_stable",
        "n_estimators": 1000,
        "max_depth": None,
        "min_samples_split": 10,
        "min_samples_leaf": 15,
        "max_features": "sqrt",
        "n_jobs": -1,
    },
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stable_offset(*parts) -> int:
    text = "::".join(str(part) for part in parts)
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % 100000


def jsonable(obj):
    return base.jsonable(obj)


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_requirements(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "requirements_deepsurv_rsf.txt").write_text(
        "\n".join(REQUIREMENTS) + "\n", encoding="utf-8"
    )


def log_line(out_dir: Path, msg: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_log.txt").open("a", encoding="utf-8") as f:
        f.write(f"[{now()}] {msg}\n")
    print(msg, flush=True)


def package_status() -> dict:
    status = {"python": sys.version, "packages": {}, "missing": []}
    for package, module_name in IMPORT_NAMES.items():
        try:
            mod = importlib.import_module(module_name)
            status["packages"][package] = {
                "available": True,
                "module": module_name,
                "version": getattr(mod, "__version__", "unknown"),
            }
        except Exception as exc:
            status["packages"][package] = {
                "available": False,
                "module": module_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            status["missing"].append(package)
    return status


def ensure_runtime(out_dir: Path) -> None:
    write_requirements(out_dir)
    status = package_status()
    save_json(out_dir / "package_versions.json", status)
    if status["missing"]:
        raise RuntimeError(
            "Missing required packages for --mode run: " + ", ".join(status["missing"])
        )
    base.import_runtime_dependencies()
    base.set_all_seeds(42)


def load_pandas():
    return importlib.import_module("pandas")


def ds_runtime_params(config: dict, n_train: int) -> dict:
    params = dict(config)
    params["learning_rate"] = float(params.pop("lr"))
    if params.get("batch_size") == "full_or_64":
        params["batch_size"] = int(min(64, max(1, n_train)))
    params.pop("config_id", None)
    return params


def model_configs(model_name: str) -> list[dict]:
    return DEEPSURV_CONFIGS if model_name == "DeepSurv" else RSF_CONFIGS


def validation_splits(raw_eicu, validation: str, seed: int):
    y = base._coerce_time_event(raw_eicu)[base.EVENT_COL].astype(int).to_numpy()
    if validation == "split":
        splitter = base.StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        return list(splitter.split(base.np.arange(len(y)), y))
    n_splits = min(4, int(base.np.sum(y == 1)), int(base.np.sum(y == 0)))
    if n_splits < 2:
        raise ValueError("Not enough events/censored samples for stratified validation.")
    splitter = base.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(base.np.arange(len(y)), y))


def fit_predict(model_name: str, x_tr, t_tr, e_tr, x_eval, params: dict, seed: int, valid=None):
    if model_name == "DeepSurv":
        train_data = base.make_deepsurv_data(x_tr, t_tr, e_tr)
        valid_data = None
        if valid is not None:
            x_va, t_va, e_va = valid
            valid_data = base.make_deepsurv_data(x_va, t_va, e_va)
        model, epochs = base.fit_deepsurv(train_data, valid_data, params, seed)
        return model, base.predict_deepsurv_risk(model, x_eval), epochs
    model = base.fit_rsf(x_tr, t_tr, e_tr, params, seed)
    return model, base.predict_rsf_risk(model, x_eval), None


def cross_validate_configs(raw_eicu, model_name: str, validation: str, seed: int):
    rows = []
    splits = validation_splits(raw_eicu, validation, seed)
    for config_index, config in enumerate(model_configs(model_name)):
        for fold, (tr_idx, va_idx) in enumerate(splits, 1):
            tr_raw = raw_eicu.iloc[tr_idx].reset_index(drop=True)
            va_raw = raw_eicu.iloc[va_idx].reset_index(drop=True)
            tr_enc, prep = base.fit_preprocess(tr_raw)
            va_enc = base.transform_preprocess(va_raw, prep)
            x_tr, t_tr, e_tr, _ = base.extract_xy(tr_enc)
            x_va, t_va, e_va, _ = base.extract_xy(va_enc)
            params = ds_runtime_params(config, len(x_tr)) if model_name == "DeepSurv" else {
                k: v for k, v in config.items() if k != "config_id"
            }
            err = ""
            cindex = float("nan")
            epochs = None
            try:
                _, risk, epochs = fit_predict(
                    model_name,
                    x_tr,
                    t_tr,
                    e_tr,
                    x_va,
                    params,
                    seed + 1000 * config_index + fold,
                    valid=(x_va, t_va, e_va) if model_name == "DeepSurv" else None,
                )
                cindex = base.evaluate_cindex(t_va, e_va, risk)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "model": model_name,
                    "config_id": config["config_id"],
                    "fold": int(fold),
                    "validation": validation,
                    "train_n": int(len(tr_idx)),
                    "valid_n": int(len(va_idx)),
                    "valid_events": int(base.np.sum(e_va)),
                    "valid_cindex": cindex,
                    "params": json.dumps(jsonable(config), ensure_ascii=False),
                    "runtime_params": json.dumps(jsonable(params), ensure_ascii=False),
                    "epochs": epochs,
                    "error": err,
                }
            )
    return base.pd.DataFrame(rows)


def summarize_config_cv(cv_df):
    grouped = (
        cv_df.groupby(["model", "config_id"], dropna=False)["valid_cindex"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped = grouped.rename(
        columns={"mean": "validation_cindex_mean", "std": "validation_cindex_std", "count": "validation_fold_count"}
    )
    grouped = grouped[base.np.isfinite(grouped["validation_cindex_mean"])]
    return grouped


def selected_config_id(cv_df) -> str:
    summary = summarize_config_cv(cv_df)
    if summary.empty:
        errors = "; ".join(str(x) for x in cv_df.get("error", []) if str(x))
        raise RuntimeError("No valid validation C-index values. " + errors)
    best = summary.sort_values(
        ["validation_cindex_mean", "validation_fold_count"], ascending=[False, False]
    ).iloc[0]
    return str(best["config_id"])


def config_by_id(model_name: str, config_id: str) -> dict:
    for config in model_configs(model_name):
        if config["config_id"] == config_id:
            return dict(config)
    raise KeyError(config_id)


def final_external_for_all_configs(raw_eicu, raw_mimic, endpoint_dir: Path, seed: int, ds_final_epochs=None):
    enc_eicu, prep = base.fit_preprocess(raw_eicu)
    enc_mimic = base.transform_preprocess(raw_mimic, prep)
    x_eicu, t_eicu, e_eicu, feature_cols = base.extract_xy(enc_eicu)
    x_mimic, t_mimic, e_mimic, mimic_features = base.extract_xy(enc_mimic)
    if feature_cols != mimic_features:
        raise ValueError("MIMIC feature columns do not match eICU-fitted feature columns.")

    prep_audit = base.preprocessing_audit(raw_eicu, raw_mimic, enc_eicu, enc_mimic, prep)
    save_json(endpoint_dir / "preprocessing_audit.json", prep_audit)
    save_json(endpoint_dir / "feature_names.json", feature_cols)

    id_cols = [c for c in base.ID_EXCLUDE_CANDIDATES if c in raw_mimic.columns]
    predictions = raw_mimic[id_cols + [base.TIME_COL, base.EVENT_COL]].copy()
    all_pred = predictions.copy()
    result_rows = []

    for model_name in ["DeepSurv", "RSF"]:
        for config in model_configs(model_name):
            params = ds_runtime_params(config, len(x_eicu)) if model_name == "DeepSurv" else {
                k: v for k, v in config.items() if k != "config_id"
            }
            if model_name == "DeepSurv" and ds_final_epochs:
                final_epochs = ds_final_epochs.get(config["config_id"])
                if final_epochs is not None and final_epochs > 0:
                    params["max_epochs"] = int(final_epochs)
            error = ""
            epochs = None
            cindex = float("nan")
            try:
                _, risk, epochs = fit_predict(
                    model_name,
                    x_eicu,
                    t_eicu,
                    e_eicu,
                    x_mimic,
                    params,
                    seed + stable_offset(model_name, config["config_id"]),
                    valid=None,
                )
                if not base.np.isfinite(risk).all():
                    raise ValueError("Non-finite MIMIC risk scores produced by final model.")
                pred_col = f"risk_{model_name}_{config['config_id']}"
                all_pred[pred_col] = risk
                cindex = base.evaluate_cindex(t_mimic, e_mimic, risk)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            result_rows.append(
                {
                    "model": model_name,
                    "config_id": config["config_id"],
                    "external_mimic_cindex": cindex,
                    "final_training_epochs_if_applicable": epochs,
                    "params": json.dumps(jsonable(config), ensure_ascii=False),
                    "runtime_params": json.dumps(jsonable(params), ensure_ascii=False),
                    "external_error": error,
                }
            )
    all_pred.to_csv(endpoint_dir / "mimic_predictions_all_configs.csv", index=False)
    return base.pd.DataFrame(result_rows), all_pred, feature_cols


def make_data_audit(cfg, raw_eicu, raw_mimic):
    audit, warnings = base.make_data_audit(cfg, raw_eicu, raw_mimic)
    audit["limited_grid_note"] = "MIMIC is external-only; selection uses eICU validation only."
    return audit, warnings


def run_endpoint(cfg: dict, args, out_dir: Path) -> tuple[list[dict], list[dict]]:
    endpoint = cfg["name"]
    endpoint_dir = out_dir / endpoint
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    selected_file = endpoint_dir / "result_selected.xlsx"
    if selected_file.exists() and not args.force:
        cached = base.pd.read_excel(selected_file)
        all_cached = base.pd.read_excel(endpoint_dir / "result_all_configs.xlsx")
        return cached.to_dict("records"), all_cached.to_dict("records")

    log_line(out_dir, f"[{endpoint}] loading data")
    raw_eicu, raw_mimic = base.load_endpoint_data(cfg)
    data_audit, warnings = make_data_audit(cfg, raw_eicu, raw_mimic)
    save_json(endpoint_dir / "data_audit.json", data_audit)

    ds_cv = cross_validate_configs(raw_eicu, "DeepSurv", args.validation, args.seed)
    rsf_cv = cross_validate_configs(raw_eicu, "RSF", args.validation, args.seed)
    ds_cv.to_csv(endpoint_dir / "cv_results_deepsurv.csv", index=False)
    rsf_cv.to_csv(endpoint_dir / "cv_results_rsf.csv", index=False)

    best_ds_id = selected_config_id(ds_cv)
    best_rsf_id = selected_config_id(rsf_cv)
    save_json(endpoint_dir / "best_params_deepsurv.json", config_by_id("DeepSurv", best_ds_id))
    save_json(endpoint_dir / "best_params_rsf.json", config_by_id("RSF", best_rsf_id))

    ds_final_epochs = {}
    for config_id, sub in ds_cv.groupby("config_id", dropna=False):
        vals = base.pd.to_numeric(sub["epochs"], errors="coerce")
        vals = vals[base.np.isfinite(vals)]
        if len(vals):
            ds_final_epochs[str(config_id)] = int(max(1, round(float(vals.median()))))
    save_json(endpoint_dir / "deepsurv_final_epochs_from_cv.json", ds_final_epochs)

    external_df, all_pred, feature_cols = final_external_for_all_configs(
        raw_eicu, raw_mimic, endpoint_dir, args.seed, ds_final_epochs=ds_final_epochs
    )
    cv_summary = base.pd.concat([summarize_config_cv(ds_cv), summarize_config_cv(rsf_cv)], ignore_index=True)
    all_config = cv_summary.merge(external_df, on=["model", "config_id"], how="outer")
    all_config["endpoint"] = endpoint
    all_config["validation"] = args.validation
    all_config["train_n"] = int(len(raw_eicu))
    all_config["train_events"] = int(raw_eicu[base.EVENT_COL].sum())
    all_config["mimic_n"] = int(len(raw_mimic))
    all_config["mimic_events"] = int(raw_mimic[base.EVENT_COL].sum())
    all_config["n_features_after_preprocessing"] = int(len(feature_cols))
    all_config["random_seed"] = int(args.seed)
    all_config["output_path"] = str(endpoint_dir)
    all_config["warnings"] = "; ".join(sorted(set(warnings)))
    all_config["selected_by_eicu_validation"] = all_config["config_id"].isin([best_ds_id, best_rsf_id])
    all_config.to_csv(endpoint_dir / "all_config_results_endpoint.csv", index=False)
    all_config.to_excel(endpoint_dir / "result_all_configs.xlsx", index=False)

    selected = all_config[all_config["selected_by_eicu_validation"]].copy()
    selected.to_excel(endpoint_dir / "result_selected.xlsx", index=False)
    selected_pred_cols = [
        c for c in all_pred.columns
        if c in [base.TIME_COL, base.EVENT_COL]
        or base._is_id_col(c)
        or c.endswith(best_ds_id)
        or c.endswith(best_rsf_id)
    ]
    all_pred[selected_pred_cols].to_csv(endpoint_dir / "mimic_predictions_selected.csv", index=False)

    log_line(out_dir, f"[{endpoint}] completed limited grid")
    return selected.to_dict("records"), all_config.to_dict("records")


def save_global_results(selected_rows: list[dict], all_rows: list[dict], out_dir: Path):
    pd = base.pd if hasattr(base, "pd") else load_pandas()
    selected_df = pd.DataFrame(selected_rows)
    all_df = pd.DataFrame(all_rows)
    selected_df.to_csv(out_dir / "summary_selected_results.csv", index=False)
    selected_df.to_excel(out_dir / "summary_selected_results.xlsx", index=False)
    all_df.to_csv(out_dir / "all_config_results.csv", index=False)
    all_df.to_excel(out_dir / "all_config_results.xlsx", index=False)
    return selected_df, all_df


def rebuild_global_results_from_endpoint_dirs(out_dir: Path):
    pd = base.pd if hasattr(base, "pd") else load_pandas()
    selected_frames = []
    all_frames = []
    for cfg in base.ENDPOINTS:
        endpoint_dir = out_dir / cfg["name"]
        selected_path = endpoint_dir / "result_selected.xlsx"
        all_path = endpoint_dir / "result_all_configs.xlsx"
        if selected_path.exists():
            selected_frames.append(pd.read_excel(selected_path))
        if all_path.exists():
            all_frames.append(pd.read_excel(all_path))
    selected_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    all_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    if len(selected_df) or len(all_df):
        save_global_results(selected_df.to_dict("records"), all_df.to_dict("records"), out_dir)
    return selected_df, all_df


def read_table(path: Path, sheet_name=None):
    pd = base.pd if hasattr(base, "pd") else load_pandas()
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        if sheet_name is None:
            sheet_name = 0
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path)


def normalize_my_algorithm_table(df, label: str, source: str):
    pd = base.pd if hasattr(base, "pd") else load_pandas()
    rows = []
    endpoint_cols = ["endpoint", "target_name", "target", "target_id"]
    cindex_cols = [
        "MT_external",
        "my_algorithm_external_cindex",
        "multitarget_external_cindex",
        "external_mimic_cindex",
        "external_cindex",
        "ABC_external_cindex",
        "c_index",
        "cindex",
    ]
    endpoint_col = next((c for c in endpoint_cols if c in df.columns), None)
    cindex_col = next((c for c in cindex_cols if c in df.columns), None)
    if endpoint_col is None or cindex_col is None:
        return pd.DataFrame(rows)
    for _, row in df.iterrows():
        try:
            value = float(row[cindex_col])
        except Exception:
            continue
        rows.append(
            {
                "endpoint": str(row[endpoint_col]),
                "comparison_model": label,
                "comparison_external_mimic_cindex": value,
                "comparison_source": source,
            }
        )
    return pd.DataFrame(rows)


def load_my_algorithm_results(path_arg: str | None, label: str):
    pd = base.pd if hasattr(base, "pd") else load_pandas()
    paths = []
    if path_arg:
        paths.append(Path(path_arg))
    elif DEFAULT_MY_ALGO_SUMMARY.exists():
        paths.append(DEFAULT_MY_ALGO_SUMMARY)

    frames = []
    notes = []
    for path in paths:
        if not path.exists():
            notes.append(f"my algorithm path missing: {path}")
            continue
        try:
            if path.is_file() and path.suffix.lower() in {".xlsx", ".xls"}:
                try:
                    df = pd.read_excel(path, sheet_name="top3_endpoint_external_compare")
                    frame = normalize_my_algorithm_table(df, label, str(path) + "#top3_endpoint_external_compare")
                    if len(frame):
                        frames.append(frame)
                        continue
                except Exception:
                    pass
                df = pd.read_excel(path)
                frame = normalize_my_algorithm_table(df, label, str(path))
                if len(frame):
                    frames.append(frame)
            elif path.is_file():
                df = pd.read_csv(path)
                frame = normalize_my_algorithm_table(df, label, str(path))
                if len(frame):
                    frames.append(frame)
            elif path.is_dir():
                for result in path.rglob("result.xlsx"):
                    try:
                        df = pd.read_excel(result)
                        frame = normalize_my_algorithm_table(df, label, str(result))
                        if len(frame):
                            frames.append(frame)
                    except Exception:
                        notes.append(f"could not parse {result}")
        except Exception as exc:
            notes.append(f"could not parse {path}: {type(exc).__name__}: {exc}")
    if frames:
        return pd.concat(frames, ignore_index=True), notes
    notes.append("No parsable my-algorithm endpoint external C-index table found.")
    return pd.DataFrame(
        columns=[
            "endpoint",
            "comparison_model",
            "comparison_external_mimic_cindex",
            "comparison_source",
        ]
    ), notes


def build_comparison(out_dir: Path, my_algorithm_summary: str | None, label: str):
    pd = base.pd if hasattr(base, "pd") else load_pandas()
    selected_path = out_dir / "summary_selected_results.xlsx"
    if not selected_path.exists():
        selected_path = out_dir / "summary_selected_results.csv"
    if selected_path.exists():
        selected = read_table(selected_path)
    else:
        selected = pd.DataFrame()

    comparison_rows = []
    if len(selected):
        for _, row in selected.iterrows():
            comparison_rows.append(
                {
                    "endpoint": row.get("endpoint"),
                    "comparison_model": row.get("model"),
                    "comparison_external_mimic_cindex": row.get("external_mimic_cindex"),
                    "comparison_source": str(selected_path),
                }
            )
    my_df, notes = load_my_algorithm_results(my_algorithm_summary, label)
    comparison = pd.concat([pd.DataFrame(comparison_rows), my_df], ignore_index=True)
    comparison["note"] = "; ".join(notes) if notes else ""
    comparison.to_csv(out_dir / "comparison_with_my_algorithm.csv", index=False)
    comparison.to_excel(out_dir / "comparison_with_my_algorithm.xlsx", index=False)
    return comparison, notes


def endpoint_short_name(endpoint: str) -> str:
    mapping = {
        "event_1_death_hospice": "death_hospice",
        "event_2_institutional_nonhome": "institutional_nonhome",
        "event_3_icu_los_gt7d": "icu_los_gt7d",
        "event_4_later_icu_same_hadm": "later_icu_same_hadm",
        "event_5_home": "home",
        "event_6_hosp_los_gt14d": "hosp_los_gt14d",
    }
    return mapping.get(str(endpoint), str(endpoint))


def build_six_endpoint_external_comparison(out_dir: Path, my_algorithm_summary: str | None = None):
    pd = base.pd if hasattr(base, "pd") else load_pandas()
    selected_path = out_dir / "summary_selected_results.xlsx"
    selected = pd.read_excel(selected_path) if selected_path.exists() else pd.DataFrame()
    baseline_path = Path(my_algorithm_summary) if my_algorithm_summary else DEFAULT_MY_ALGO_SUMMARY

    baseline = pd.DataFrame()
    if baseline_path.exists() and baseline_path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            baseline = pd.read_excel(baseline_path, sheet_name="top3_endpoint_external_compare")
        except Exception:
            baseline = pd.DataFrame()
    if len(baseline):
        baseline = baseline.copy()
        baseline = baseline.sort_values(["endpoint", "rank_trial"]).drop_duplicates("endpoint", keep="first")
        baseline = baseline.rename(
            columns={
                "endpoint": "endpoint_short",
                "MT_external": "multitarget_gpcox_external",
                "ordinary_cox_external": "ordinary_cox_external",
                "single_target_gpcox_external": "single_target_gpcox_external",
            }
        )
        baseline = baseline[
            [
                "endpoint_short",
                "ordinary_cox_external",
                "single_target_gpcox_external",
                "multitarget_gpcox_external",
                "rank_trial",
            ]
        ]

    rows = []
    for cfg in base.ENDPOINTS:
        endpoint = cfg["name"]
        short = endpoint_short_name(endpoint)
        row = {"endpoint": endpoint, "endpoint_short": short}
        if len(baseline):
            b = baseline[baseline["endpoint_short"] == short]
            if len(b):
                for col in [
                    "ordinary_cox_external",
                    "single_target_gpcox_external",
                    "multitarget_gpcox_external",
                    "rank_trial",
                ]:
                    row[col] = b.iloc[0].get(col)
        if len(selected):
            sub = selected[selected["endpoint"] == endpoint]
            for model in ["DeepSurv", "RSF"]:
                m = sub[sub["model"] == model]
                prefix = model.lower()
                if len(m):
                    row[f"{prefix}_external"] = m.iloc[0].get("external_mimic_cindex")
                    row[f"{prefix}_internal_cv_mean"] = m.iloc[0].get("validation_cindex_mean")
                    row[f"{prefix}_selected_config"] = m.iloc[0].get("config_id")
                    row[f"{prefix}_external_error"] = m.iloc[0].get("external_error")
        rows.append(row)

    external = pd.DataFrame(rows)
    metric_cols = [
        "ordinary_cox_external",
        "single_target_gpcox_external",
        "multitarget_gpcox_external",
        "deepsurv_external",
        "rsf_external",
    ]
    mean_rows = []
    for col in metric_cols:
        if col in external.columns:
            vals = pd.to_numeric(external[col], errors="coerce")
            mean_rows.append(
                {
                    "model": col.replace("_external", ""),
                    "mean_external_cindex": float(vals.mean(skipna=True)),
                    "n_endpoints_with_value": int(vals.notna().sum()),
                }
            )
    means = pd.DataFrame(mean_rows)
    external.to_csv(out_dir / "six_endpoint_external_comparison.csv", index=False)
    means.to_csv(out_dir / "six_endpoint_external_comparison_means.csv", index=False)
    with pd.ExcelWriter(out_dir / "six_endpoint_external_comparison.xlsx") as writer:
        external.to_excel(writer, sheet_name="by_endpoint", index=False)
        means.to_excel(writer, sheet_name="means", index=False)
    return external, means


def run_preflight(args) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_requirements(out_dir)
    status = package_status()
    save_json(out_dir / "package_versions.json", status)

    refs = [
        "AGENTS.md",
        "PROJECT_MEMORY.md",
        "CODEX_REFERENCE_DeepSurv_RSF_repository_pattern.md",
        "CODEX_REFERENCE_baseline_implementation_blueprint.md",
        "run_deepsurv_rsf_baselines.py",
        "newcox_ABC_unified_FINAL_CLEAN(14).ipynb",
        "newcox_ABC_unified_FINAL_CLEAN.ipynb",
    ]
    ref_status = {p: Path(p).exists() for p in refs}

    syntax_ok = True
    syntax_error = ""
    try:
        py_compile.compile(__file__, doraise=True)
    except Exception as exc:
        syntax_ok = False
        syntax_error = f"{type(exc).__name__}: {exc}"

    data_rows = []
    try:
        pd = load_pandas()
        for cfg in base.ENDPOINTS:
            row = {"endpoint": cfg["name"], "eicu_path": cfg["eicu_path"], "mimic_path": cfg["mimic_path"]}
            for cohort, key in [("eicu", "eicu_path"), ("mimic", "mimic_path")]:
                path = Path(cfg[key])
                row[f"{cohort}_exists"] = path.exists()
                if path.exists():
                    df = pd.read_excel(path, usecols=[base.TIME_COL, base.EVENT_COL])
                    row[f"{cohort}_n"] = int(len(df))
                    row[f"{cohort}_events"] = int(pd.to_numeric(df[base.EVENT_COL], errors="coerce").fillna(0).sum())
            data_rows.append(row)
    except Exception as exc:
        data_rows.append({"error": f"{type(exc).__name__}: {exc}"})

    my_algo_df, my_notes = load_my_algorithm_results(args.my_algorithm_summary, args.my_algorithm_label)
    save_json(
        out_dir / "preflight_details.json",
        {
            "created_at": now(),
            "mode": "preflight",
            "validation_default": args.validation,
            "references": ref_status,
            "packages": status,
            "cpu_count": os.cpu_count(),
            "data": data_rows,
            "syntax_ok": syntax_ok,
            "syntax_error": syntax_error,
            "my_algorithm_rows_detected": int(len(my_algo_df)),
            "my_algorithm_notes": my_notes,
            "will_not_train_in_preflight": True,
            "run_mode_command_example": (
                f"python {Path(__file__).name} --mode run --validation {args.validation} "
                f"--output-dir {out_dir}"
            ),
        },
    )

    lines = [
        "# Local preflight report",
        "",
        f"- Created at: `{now()}`",
        f"- Output directory: `{out_dir}`",
        f"- Syntax check: `{'PASS' if syntax_ok else 'FAIL'}`",
        f"- CPU logical cores: `{os.cpu_count()}`",
        f"- Missing packages for run: `{', '.join(status['missing']) if status['missing'] else 'none'}`",
        f"- My-algorithm comparison rows detected: `{len(my_algo_df)}`",
        "",
        "## Reference files",
    ]
    for p, exists in ref_status.items():
        lines.append(f"- `{p}`: `{'found' if exists else 'missing'}`")
    lines.extend(["", "## Endpoint input audit"])
    for row in data_rows:
        lines.append(f"- `{row.get('endpoint', 'unknown')}`: {json.dumps(row, ensure_ascii=False)}")
    if status["missing"]:
        lines.extend(
            [
                "",
                "## Run blocker",
                "Formal `--mode run` is blocked until missing packages are installed.",
                "Preflight did not train, preprocess, or launch any grid job.",
            ]
        )
    if syntax_error:
        lines.extend(["", "## Syntax error", syntax_error])
    if my_notes:
        lines.extend(["", "## Comparison notes", *[f"- {note}" for note in my_notes]])
    (out_dir / "local_preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Preflight complete: {out_dir / 'local_preflight_report.md'}")
    return 0 if syntax_ok else 1


def run_limited_grid(args) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_runtime(out_dir)
    base.set_all_seeds(args.seed)
    log_line(out_dir, f"Starting limited grid validation={args.validation} seed={args.seed}")
    selected = [cfg for cfg in base.ENDPOINTS if args.endpoint in (None, cfg["name"])]
    selected_rows = []
    all_rows = []
    for cfg in selected:
        try:
            endpoint_selected, endpoint_all = run_endpoint(cfg, args, out_dir)
            selected_rows.extend(endpoint_selected)
            all_rows.extend(endpoint_all)
        except Exception as exc:
            log_line(out_dir, f"[{cfg['name']}] FAILED: {type(exc).__name__}: {exc}")
            log_line(out_dir, traceback.format_exc())
    save_global_results(selected_rows, all_rows, out_dir)
    build_comparison(out_dir, args.my_algorithm_summary, args.my_algorithm_label)
    log_line(out_dir, f"Finished limited grid. Summary: {out_dir / 'summary_selected_results.xlsx'}")
    return 0


def run_report(args) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_requirements(out_dir)
    status = package_status()
    save_json(out_dir / "package_versions.json", status)
    rebuild_global_results_from_endpoint_dirs(out_dir)
    build_comparison(out_dir, args.my_algorithm_summary, args.my_algorithm_label)
    build_six_endpoint_external_comparison(out_dir, args.my_algorithm_summary)
    print(f"Report complete: {out_dir / 'comparison_with_my_algorithm.xlsx'}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "run", "report"], required=True)
    parser.add_argument("--validation", choices=["cv4", "split"], default="cv4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--endpoint", choices=[cfg["name"] for cfg in base.ENDPOINTS])
    parser.add_argument("--my-algorithm-summary")
    parser.add_argument("--my-algorithm-label", default="three-stage Cox-GP")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "preflight":
        return run_preflight(args)
    if args.mode == "report":
        return run_report(args)
    return run_limited_grid(args)


if __name__ == "__main__":
    raise SystemExit(main())
