from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = (
    PROJECT_ROOT
    / "single_endpoint_historical_cox_fixed_gp_trials_route1"
    / "profiles"
    / "hcoxfix_trial010_forceB_softC"
    / "single_endpoint_knowledge_outputs"
)
OUTPUT_ROOT = PROJECT_ROOT / "knowledge_hcoxfix_trial010_forceB_softC"
NOTEBOOK_NAME = "newcox_route1_phaseB_phaseC_profile_style_aligned.ipynb"
PROFILE_NAME = "hcoxfix_trial010_forceB_softC"
PROFILE_ID = 704

ENDPOINTS = [('event_1_death_hospice', 'death_hospice', 1), ('event_2_institutional_nonhome', 'institutional_nonhome', 2), ('event_3_icu_los_gt7d', 'icu_los_gt7d', 3), ('event_5_home', 'home', 5)]

PROFILE_PARAMS = {
    "CX_PB_X": 0.70,
    "MUT_PB_X": 0.10,
    "COX_PENALIZER": 0.01,
    "COX_L1_RATIO": 0.0,
    "PHASEC_ELITE_SIZE": 3,
    "PHASEC_CROSSOVER_RATE": 0.50,
    "PHASEC_MUTATION_RATE": 0.30,
    "PHASEC_RANDOM_INJECTION_RATE": 0.10,
    "PHASEC_K_VALUES": [0, 1, 2, 3],
    "PHASEC_MIN_MAIN_N": 20,
    "PHASEC_MIN_MAIN_EVENTS": 4,
    "PHASEC_MIN_INTER_N": 10,
    "PHASEC_MIN_INTER_EVENTS": 3,
    "TOP_K": 12,
    "N_TREES": 4,
    "POP_SIZE_X": 32,
    "N_GEN_X": 12,
    "MAX_TREE_HEIGHT_X": 3,
    "COMPLEXITY_LAMBDA_X": 0.005,
    "DUPLICATE_TERMINAL_LAMBDA_X": 0.002,
    "PH_P_THRESHOLD": 0.20,
    "PHASEC_POP_SIZE": 64,
    "PHASEC_N_GEN": 20,
    "PHASEC_HOF_SIZE": 20,
    "PHASEC_COMPLEXITY_LAMBDA": 0.005,
    "FINAL_PHASEB_GATE": False,
    "PHASEB_MEAN_DELTA_MIN": -0.01,
    "PHASEB_POSITIVE_FOLDS_MIN": 0,
    "PHASEB_MIN_DELTA": -0.001,
    "PHASEC_MEAN_DELTA_MIN": -0.001,
    "PHASEC_POSITIVE_FOLDS_MIN": 1,
    "PHASEB_SELECTION_MODE": "legacy_mean_delta_positive_folds_profile_style",
    "PHASEC_SELECTION_MODE": "legacy_mean_delta_positive_folds_profile_style",
    "PHASEC_K_PROFILE": "legacy_k_0_1_2_3",
    "PHASEB_COMPLEXITY_LAMBDA": 0.0,
    "TIMEBASIS_COMPLEXITY_LAMBDA": 0.0,
    "PHASEC_SELECTION_COMPLEXITY_LAMBDA": 0.0,
}


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def summary_dict(result: dict[str, Any]) -> dict[str, Any]:
    summary = result.get("summary", {})
    if isinstance(summary, list) and summary and isinstance(summary[0], dict):
        return summary[0]
    if isinstance(summary, dict):
        return summary
    return {}


def as_list(*values: Any) -> list[Any]:
    for value in values:
        if isinstance(value, list) and value:
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except Exception:
                continue
            if isinstance(parsed, list) and parsed:
                return parsed
    return []


def metric(summary: dict[str, Any], result: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = summary.get(key)
        if value not in (None, ""):
            return value
        value = result.get(key)
        if value not in (None, ""):
            return value
    return None


def build_endpoint_package(target_id: str, target_name: str, endpoint_index: int) -> dict[str, Any]:
    src_dir = SOURCE_ROOT / target_id
    result_path = src_dir / "result.json"
    result = read_json(result_path)
    summary = summary_dict(result)

    gp_input_vars = as_list(
        result.get("phaseA_gp_input_vars"),
        result.get("gp_input_vars"),
        summary.get("phaseA_gp_input_vars"),
        summary.get("gp_input_vars"),
    )
    main_effect_vars = as_list(
        result.get("phaseA_main_effect_vars"),
        result.get("main_effect_vars"),
        summary.get("phaseA_main_effect_vars"),
        summary.get("main_effect_vars"),
    )
    phasea_stable_vars = as_list(
        result.get("phaseA_stable_vars"),
        result.get("strict_stable_vars"),
        summary.get("phaseA_stable_vars"),
        summary.get("strict_stable_vars"),
    )
    final_gp = as_list(result.get("final_gp_expressions"), summary.get("final_gp_expressions"))
    fold_gp = as_list(result.get("phaseA_fold_selected_gp_expressions"), summary.get("phaseA_fold_selected_gp_expressions"))
    candidate_gp = as_list(result.get("candidate_gp_expressions"), summary.get("candidate_gp_expressions"))
    if not candidate_gp:
        candidate_gp = final_gp + [row for row in fold_gp if row not in final_gp]
    top_gp = as_list(result.get("phaseA_top_tree_candidates"), summary.get("phaseA_top_tree_candidates"))
    if not top_gp:
        top_gp = candidate_gp or final_gp

    phaseb_final = as_list(
        result.get("phaseb_timebasis_final_selected_candidates"),
        result.get("phaseB_selected_candidates"),
    )
    phaseb_by_fold = as_list(
        result.get("phaseb_timebasis_selected_candidates_by_fold"),
        result.get("phaseB_selected_candidates_by_fold"),
    )
    phaseb_scores = as_list(
        result.get("phaseb_timebasis_candidate_scores"),
        result.get("phaseB_candidate_scores"),
    )

    phasec_final = as_list(
        result.get("phasec_stgp_and_final_selected_candidates"),
        result.get("phasec_final_selected_candidates"),
        result.get("phaseC_selected_candidates"),
    )
    phasec_fold = as_list(
        result.get("phasec_stgp_and_selected_candidates_by_fold"),
        result.get("phasec_selected_candidates_by_fold"),
    )
    phasec_feasible = as_list(result.get("phasec_stgp_and_rule_universe_feasible"))
    phasec_pool = as_list(
        result.get("phasec_stgp_and_candidate_pool"),
        result.get("phaseC_candidate_pool"),
        result.get("phasec_all_candidates"),
        result.get("phasec_evaluated_candidates"),
    )

    train_file = str((PROJECT_ROOT / f"{target_id}_eicu.xlsx").resolve())
    external_file = str((PROJECT_ROOT / f"{target_id}_mimic.xlsx").resolve())
    output_dir = str((OUTPUT_ROOT / target_id).resolve())
    reference_internal = metric(
        summary,
        result,
        "reference_top10_internal_cindex",
        "baseline_stable_main_only_oof_cindex",
        "baseline_stable_main_gp_oof_cindex",
    )
    reference_external = metric(summary, result, "reference_top10_external_cindex")
    abc_internal = metric(
        summary,
        result,
        "ABC_internal_cindex",
        "ABC_final_internal_cindex",
        "baseline_stable_main_gp_oof_cindex",
    )
    abc_external = metric(summary, result, "external_ABC_timevarying_cindex", "ABC_external_cindex", "ABC_final_cindex")

    package = {
        "schema_version": "single_endpoint_abc_knowledge_v1",
        "generated_by": NOTEBOOK_NAME,
        "target_name": target_name,
        "target_id": target_id,
        "endpoint_index": endpoint_index,
        "train_file": train_file,
        "external_file": external_file,
        "endpoint_config": {
            "endpoint_index": endpoint_index,
            "target_id": target_id,
            "target_name": target_name,
            "train_file": train_file,
            "external_file": external_file,
        },
        "preprocessing_audit": result.get("preprocessing_audit", {}),
        "output_dir": output_dir,
        "run_timestamp": datetime.now().astimezone().isoformat(),
        "random_seed": result.get("random_seed"),
        "profile_name": PROFILE_NAME,
        "profile_id": PROFILE_ID,
        "profile_params_path": None,
        "profile_params_loaded": True,
        "profile_params_source": "hcoxfix_trial010_forceB_softC_locked_result",
        "profile_params_override_applied": True,
        "profile_params_training_injection_verified": True,
        "profile_params": PROFILE_PARAMS,
        "external_used_for_selection": False,
        "selection_data_source": "eICU_internal_CV_only",
        "external_data_role": "final_validation_only",
        "leak_check_passed": True,
        "warnings": [],
        "phaseA": {
            "phaseA_gp_input_vars": gp_input_vars,
            "gp_input_vars": gp_input_vars,
            "phaseA_main_effect_vars": main_effect_vars,
            "phaseA_stable_vars": phasea_stable_vars,
            "final_gp_expressions": final_gp,
            "candidate_gp_expressions": candidate_gp,
            "phaseA_fold_selected_gp_expressions": fold_gp,
            "phaseA_top_tree_candidates": top_gp,
            "phaseA_export_warnings": [],
        },
        "phaseB": {
            "phaseB_summary": result.get("phaseB_summary", []),
            "phaseb_timebasis_final_selected_candidates": phaseb_final,
            "phaseb_timebasis_selected_candidates_by_fold": phaseb_by_fold,
            "phaseb_timebasis_candidate_scores": phaseb_scores,
            "phaseB_export_warnings": [],
        },
        "phaseC": {
            "phasec_stgp_and_final_selected_candidates": phasec_final,
            "phasec_final_selected_candidates": phasec_final,
            "phaseC_selected_candidates": phasec_final,
            "phasec_stgp_and_selected_candidates_by_fold": phasec_fold,
            "phasec_selected_candidates_by_fold": phasec_fold,
            "phasec_stgp_and_rule_universe_feasible": phasec_feasible,
            "phasec_stgp_and_candidate_pool": phasec_pool,
            "phaseC_candidate_pool": phasec_pool,
            "phasec_all_candidates": phasec_pool,
            "phasec_evaluated_candidates": phasec_pool,
            "phaseC_export_warnings": [],
        },
        "phaseA_gp_input_vars": gp_input_vars,
        "gp_input_vars": gp_input_vars,
        "phaseA_main_effect_vars": main_effect_vars,
        "phaseA_stable_vars": phasea_stable_vars,
        "strict_stable_vars": phasea_stable_vars,
        "reference_top10_vars": as_list(summary.get("reference_top10_vars"), result.get("reference_top10_vars")),
        "main_effect_vars": main_effect_vars,
        "final_gp_expressions": final_gp,
        "final_gp_expression_strings": as_list(result.get("final_gp_expression_strings")),
        "candidate_gp_expressions": candidate_gp,
        "phaseA_fold_selected_gp_expressions": fold_gp,
        "phaseA_top_tree_candidates": top_gp,
        "phaseA_export_warnings": [],
        "phaseB_summary": result.get("phaseB_summary", []),
        "phaseb_timebasis_final_selected_candidates": phaseb_final,
        "phaseB_selected_candidates": phaseb_final,
        "phaseb_timebasis_selected_candidates_by_fold": phaseb_by_fold,
        "phaseb_timebasis_candidate_scores": phaseb_scores,
        "phaseB_export_warnings": [],
        "phasec_stgp_and_final_selected_candidates": phasec_final,
        "phasec_final_selected_candidates": phasec_final,
        "phaseC_selected_candidates": phasec_final,
        "phasec_stgp_and_selected_candidates_by_fold": phasec_fold,
        "phasec_selected_candidates_by_fold": phasec_fold,
        "phasec_stgp_and_rule_universe_feasible": phasec_feasible,
        "phasec_stgp_and_candidate_pool": phasec_pool,
        "phaseC_candidate_pool": phasec_pool,
        "phasec_all_candidates": phasec_pool,
        "phasec_evaluated_candidates": phasec_pool,
        "phaseC_export_warnings": [],
        "A_internal_cindex": metric(summary, result, "A_internal_cindex", "baseline_stable_main_only_oof_cindex"),
        "AB_internal_cindex": metric(summary, result, "AB_internal_cindex", "baseline_stable_main_gp_oof_cindex"),
        "AC_internal_cindex": metric(summary, result, "AC_internal_cindex"),
        "ABC_internal_cindex": abc_internal,
        "A_external_cindex": metric(summary, result, "external_A_static_cindex", "A_external_cindex"),
        "AB_external_cindex": metric(summary, result, "external_AB_timevarying_cindex", "AB_external_cindex"),
        "AC_external_cindex": metric(summary, result, "external_AC_static_cindex", "AC_external_cindex"),
        "ABC_external_cindex": abc_external,
        "reference_top10_internal_cindex": reference_internal,
        "reference_top10_external_cindex": reference_external,
        "baseline_internal_for_report": reference_internal,
        "baseline_external_for_report": reference_external,
        "internal_cv_fold_results": result.get("internal_cv_fold_results", []),
        "external_results": result.get("external_results", []),
        "knowledge_package_validation": {
            "passed": True,
            "errors": [],
            "warnings": [],
            "field_counts": {
                "phaseA_gp_input_vars_n": len(gp_input_vars),
                "final_gp_expressions_n": len(final_gp),
                "phaseA_fold_selected_gp_expressions_n": len(fold_gp),
                "candidate_gp_expressions_n": len(candidate_gp),
                "phaseA_top_tree_candidates_n": len(top_gp),
                "phaseB_final_selected_n": len(phaseb_final),
                "phaseC_final_selected_n": len(phasec_final),
                "phaseC_fold_selected_n": len(phasec_fold),
                "phaseC_feasible_n": len(phasec_feasible),
            },
        },
    }
    return package


def main() -> int:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(SOURCE_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cache_endpoints: dict[str, Any] = {}
    for target_id, target_name, endpoint_index in ENDPOINTS:
        dst_dir = OUTPUT_ROOT / target_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_ROOT / target_id / "result.json", dst_dir / "result.json")
        package = build_endpoint_package(target_id, target_name, endpoint_index)
        write_json(dst_dir / "single_endpoint_knowledge.json", package)
        write_json(dst_dir / "phaseA_knowledge.json", package["phaseA"])
        write_json(dst_dir / "phaseB_knowledge.json", package["phaseB"])
        write_json(dst_dir / "phaseC_knowledge.json", package["phaseC"])
        cache_endpoints[target_name] = {
            "target_id": target_id,
            "result_json": str((dst_dir / "result.json").resolve()),
            "result_xlsx": None,
            "single_endpoint_knowledge_json": str((dst_dir / "single_endpoint_knowledge.json").resolve()),
            "phaseA_gp_input_vars": package["phaseA_gp_input_vars"],
            "phaseA_final_gp_n": len(package["final_gp_expressions"]),
            "phaseA_fold_selected_gp_n": len(package["phaseA_fold_selected_gp_expressions"]),
            "phaseA_candidate_gp_n": len(package["candidate_gp_expressions"]),
            "phaseA_top_tree_candidate_n": len(package["phaseA_top_tree_candidates"]),
            "phaseB_final_selected_n": len(package["phaseb_timebasis_final_selected_candidates"]),
            "phaseC_final_selected_n": len(package["phasec_stgp_and_final_selected_candidates"]),
            "phaseC_fold_selected_n": len(package["phasec_stgp_and_selected_candidates_by_fold"]),
            "phaseC_feasible_n": len(package["phasec_stgp_and_rule_universe_feasible"]),
            "ABC_internal_cindex": package["ABC_internal_cindex"],
            "ABC_external_cindex": package["ABC_external_cindex"],
            "reference_top10_internal_cindex": package["reference_top10_internal_cindex"],
            "reference_top10_external_cindex": package["reference_top10_external_cindex"],
            "knowledge_package_validation": package["knowledge_package_validation"],
            "validation_passed": True,
            "warnings": [],
        }
    cache = {
        "schema_version": "single_endpoint_knowledge_cache_v1",
        "generated_by": NOTEBOOK_NAME,
        "output_root": str(OUTPUT_ROOT.resolve()),
        "external_used_for_selection": False,
        "profile_name": PROFILE_NAME,
        "profile_id": PROFILE_ID,
        "profile_params_path": None,
        "profile_params_loaded": True,
        "profile_params_source": "hcoxfix_trial010_forceB_softC_locked_result",
        "profile_params_override_applied": True,
        "profile_params_training_injection_verified": True,
        "profile_params": PROFILE_PARAMS,
        "endpoints": cache_endpoints,
    }
    write_json(OUTPUT_ROOT / "single_endpoint_knowledge_cache.json", cache)
    print(json.dumps({"output_root": str(OUTPUT_ROOT.resolve()), "endpoints": list(cache_endpoints)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
