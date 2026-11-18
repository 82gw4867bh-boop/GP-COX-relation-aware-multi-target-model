#!/usr/bin/env python
"""DeepSurv/RSF limited-grid baselines for the locked four-endpoint dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_deepsurv_rsf_baselines as base
import run_deepsurv_rsf_limited_grid as grid
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "four_endpoint_total_calcium_pipeline_20260808" / "data"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "four_endpoint_total_calcium_pipeline_20260808"
    / "harmonised_ca_initial_screen_20260809"
    / "locked_four_endpoint_results_20260812"
    / "deepsurv_rsf_cv4"
)

ENDPOINTS = [
    {
        "name": "event_1_death_hospice",
        "eicu_path": str(DATA_ROOT / "event_death_hospice_corrected_time_eicu.xlsx"),
        "mimic_path": str(DATA_ROOT / "event_death_hospice_corrected_time_mimic.xlsx"),
        "expected_eicu_n": 313,
        "expected_eicu_events": 25,
        "expected_mimic_n": 340,
        "expected_mimic_events": 50,
    },
    {
        "name": "event_2_institutional_nonhome",
        "eicu_path": str(DATA_ROOT / "event_care_dependent_survival_after_index_icu_eicu.xlsx"),
        "mimic_path": str(DATA_ROOT / "event_care_dependent_survival_after_index_icu_mimic.xlsx"),
        "expected_eicu_n": 313,
        "expected_eicu_events": 57,
        "expected_mimic_n": 340,
        "expected_mimic_events": 67,
    },
    {
        "name": "event_3_icu_los_gt7d",
        "eicu_path": str(DATA_ROOT / "event_3_icu_time_to_exit_eicu.xlsx"),
        "mimic_path": str(DATA_ROOT / "event_3_icu_time_to_exit_mimic.xlsx"),
        "expected_eicu_n": 313,
        "expected_eicu_events": 313,
        "expected_mimic_n": 340,
        "expected_mimic_events": 340,
    },
    {
        "name": "event_5_home",
        "eicu_path": str(DATA_ROOT / "event_favourable_outcome_after_index_icu_eicu.xlsx"),
        "mimic_path": str(DATA_ROOT / "event_favourable_outcome_after_index_icu_mimic.xlsx"),
        "expected_eicu_n": 313,
        "expected_eicu_events": 153,
        "expected_mimic_n": 340,
        "expected_mimic_events": 180,
    },
]


def four_endpoint_validation_splits(raw_eicu, validation: str, seed: int):
    """Use KFold for the all-event ICU-exit endpoint, stratification otherwise."""
    y = base._coerce_time_event(raw_eicu)[base.EVENT_COL].astype(int).to_numpy()
    indices = base.np.arange(len(y))
    if validation == "split":
        if len(base.np.unique(y)) == 1:
            splitter = KFold(n_splits=4, shuffle=True, random_state=seed)
            return [next(iter(splitter.split(indices)))]
        splitter = base.StratifiedShuffleSplit(
            n_splits=1, test_size=0.25, random_state=seed
        )
        return list(splitter.split(indices, y))
    if len(base.np.unique(y)) == 1:
        splitter = KFold(n_splits=4, shuffle=True, random_state=seed)
        return list(splitter.split(indices))
    splitter = base.StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    return list(splitter.split(indices, y))


def configure_runtime() -> None:
    base.ENDPOINTS = ENDPOINTS
    grid.validation_splits = four_endpoint_validation_splits
    grid.final_external_for_all_configs = selected_external_only


def selected_external_only(
    raw_eicu, raw_mimic, endpoint_dir: Path, seed: int, ds_final_epochs=None
):
    """Evaluate only the two configurations selected by eICU cross-validation."""
    ds_cv = base.pd.read_csv(endpoint_dir / "cv_results_deepsurv.csv")
    rsf_cv = base.pd.read_csv(endpoint_dir / "cv_results_rsf.csv")
    selected_ids = {
        "DeepSurv": grid.selected_config_id(ds_cv),
        "RSF": grid.selected_config_id(rsf_cv),
    }

    enc_eicu, prep = base.fit_preprocess(raw_eicu)
    enc_mimic = base.transform_preprocess(raw_mimic, prep)
    x_eicu, t_eicu, e_eicu, feature_cols = base.extract_xy(enc_eicu)
    x_mimic, t_mimic, e_mimic, mimic_features = base.extract_xy(enc_mimic)
    if feature_cols != mimic_features:
        raise ValueError("MIMIC feature columns do not match eICU-fitted columns.")

    grid.save_json(
        endpoint_dir / "preprocessing_audit.json",
        base.preprocessing_audit(raw_eicu, raw_mimic, enc_eicu, enc_mimic, prep),
    )
    grid.save_json(endpoint_dir / "feature_names.json", feature_cols)
    id_cols = [c for c in base.ID_EXCLUDE_CANDIDATES if c in raw_mimic.columns]
    predictions = raw_mimic[id_cols + [base.TIME_COL, base.EVENT_COL]].copy()
    result_rows = []

    for model_name in ("DeepSurv", "RSF"):
        config_id = selected_ids[model_name]
        config = grid.config_by_id(model_name, config_id)
        if model_name == "DeepSurv":
            params = grid.ds_runtime_params(config, len(x_eicu))
            final_epochs = (ds_final_epochs or {}).get(config_id)
            if final_epochs is not None and final_epochs > 0:
                params["max_epochs"] = int(final_epochs)
        else:
            params = {k: v for k, v in config.items() if k != "config_id"}
        _, risk, epochs = grid.fit_predict(
            model_name,
            x_eicu,
            t_eicu,
            e_eicu,
            x_mimic,
            params,
            seed + grid.stable_offset(model_name, config_id),
            valid=None,
        )
        if not base.np.isfinite(risk).all():
            raise ValueError(f"{model_name} produced non-finite MIMIC risk scores.")
        predictions[f"risk_{model_name}_{config_id}"] = risk
        result_rows.append(
            {
                "model": model_name,
                "config_id": config_id,
                "external_mimic_cindex": base.evaluate_cindex(t_mimic, e_mimic, risk),
                "final_training_epochs_if_applicable": epochs,
                "params": json.dumps(base.jsonable(config), ensure_ascii=False),
                "runtime_params": json.dumps(base.jsonable(params), ensure_ascii=False),
                "external_error": "",
            }
        )
    predictions.to_csv(endpoint_dir / "mimic_predictions_selected_only.csv", index=False)
    grid.save_json(
        endpoint_dir / "external_evaluation_scope.json",
        {
            "selection_source": "eICU four-fold validation only",
            "mimic_evaluated_configurations": selected_ids,
            "nonselected_configurations_evaluated_on_mimic": False,
        },
    )
    return base.pd.DataFrame(result_rows), predictions, feature_cols


def preflight(output_dir: Path) -> int:
    configure_runtime()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid.ensure_runtime(output_dir)
    rows = []
    for cfg in ENDPOINTS:
        eicu, mimic = base.load_endpoint_data(cfg)
        audit, warnings = base.make_data_audit(cfg, eicu, mimic)
        if warnings:
            raise RuntimeError(f"{cfg['name']} audit warnings: {warnings}")
        splits = four_endpoint_validation_splits(eicu, "cv4", 42)
        rows.append(
            {
                "endpoint": cfg["name"],
                "eicu_n": len(eicu),
                "eicu_events": int(eicu[base.EVENT_COL].sum()),
                "mimic_n": len(mimic),
                "mimic_events": int(mimic[base.EVENT_COL].sum()),
                "folds": len(splits),
                "splitter": "KFold" if eicu[base.EVENT_COL].nunique() == 1 else "StratifiedKFold",
                "audit": audit,
            }
        )
    (output_dir / "four_endpoint_preflight.json").write_text(
        json.dumps(base.jsonable(rows), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


def smoke(output_dir: Path) -> int:
    configure_runtime()
    grid.ensure_runtime(output_dir)
    cfg = ENDPOINTS[0]
    raw_eicu, raw_mimic = base.load_endpoint_data(cfg)
    tr_idx, va_idx = four_endpoint_validation_splits(raw_eicu, "cv4", 42)[0]
    tr_enc, prep = base.fit_preprocess(raw_eicu.iloc[tr_idx].reset_index(drop=True))
    va_enc = base.transform_preprocess(raw_eicu.iloc[va_idx].reset_index(drop=True), prep)
    x_tr, t_tr, e_tr, features = base.extract_xy(tr_enc)
    x_va, t_va, e_va, _ = base.extract_xy(va_enc)
    ds_cfg = dict(grid.DEEPSURV_CONFIGS[0])
    ds_cfg["max_epochs"] = 3
    ds_cfg["patience"] = 2
    ds_params = grid.ds_runtime_params(ds_cfg, len(x_tr))
    _, ds_risk, _ = grid.fit_predict(
        "DeepSurv", x_tr, t_tr, e_tr, x_va, ds_params, 42,
        valid=(x_va, t_va, e_va),
    )
    rsf_cfg = {k: v for k, v in grid.RSF_CONFIGS[0].items() if k != "config_id"}
    rsf_cfg["n_estimators"] = 10
    _, rsf_risk, _ = grid.fit_predict(
        "RSF", x_tr, t_tr, e_tr, x_va, rsf_cfg, 42
    )
    result = {
        "endpoint": cfg["name"],
        "features": len(features),
        "deepsurv_cindex": base.evaluate_cindex(t_va, e_va, ds_risk),
        "rsf_cindex": base.evaluate_cindex(t_va, e_va, rsf_risk),
        "deepsurv_finite": bool(base.np.isfinite(ds_risk).all()),
        "rsf_finite": bool(base.np.isfinite(rsf_risk).all()),
        "mimic_not_read_for_smoke": True,
    }
    (output_dir / "smoke_result.json").write_text(
        json.dumps(base.jsonable(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def run(output_dir: Path, endpoint: str | None, force: bool) -> int:
    configure_runtime()
    args = argparse.Namespace(
        validation="cv4",
        seed=42,
        force=force,
        endpoint=endpoint,
        my_algorithm_summary=None,
        my_algorithm_label="four-endpoint GP-Cox",
        output_dir=str(output_dir),
    )
    grid.ensure_runtime(output_dir)
    selected_rows = []
    all_rows = []
    selected = [cfg for cfg in ENDPOINTS if endpoint in (None, cfg["name"])]
    for cfg in selected:
        endpoint_selected, endpoint_all = grid.run_endpoint(cfg, args, output_dir)
        selected_rows.extend(endpoint_selected)
        all_rows.extend(endpoint_all)
    # Include cached successful endpoint results on resumed runs.
    for cfg in ENDPOINTS:
        selected_file = output_dir / cfg["name"] / "result_selected.xlsx"
        all_file = output_dir / cfg["name"] / "result_all_configs.xlsx"
        if selected_file.exists() and not any(r.get("endpoint") == cfg["name"] for r in selected_rows):
            selected_rows.extend(base.pd.read_excel(selected_file).to_dict("records"))
            all_rows.extend(base.pd.read_excel(all_file).to_dict("records"))
    grid.save_global_results(selected_rows, all_rows, output_dir)
    print(output_dir / "summary_selected_results.xlsx")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "smoke", "run"], required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--endpoint", choices=[cfg["name"] for cfg in ENDPOINTS])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if args.mode == "preflight":
        return preflight(out)
    if args.mode == "smoke":
        return smoke(out)
    return run(out, args.endpoint, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
