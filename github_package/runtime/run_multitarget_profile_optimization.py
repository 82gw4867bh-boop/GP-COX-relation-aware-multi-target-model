from __future__ import annotations


import argparse


import ast


import copy


import hashlib


import json


import math


import os


import random


import re


import subprocess


import sys


import time


import traceback


from datetime import datetime


from pathlib import Path


from typing import Any


import pandas as pd


TRAIN_SOURCE = "eicu"


EXTERNAL_SOURCE = "mimic"


OPTIMIZATION_MODE = "profile_competition"


TR_PROFILE_TRANSFER = 0.30


WORKDIR = Path(__file__).resolve().parent


PROJECT_ROOT = WORKDIR.parent


NOTEBOOK_FILE = WORKDIR / "newcox_route1_phaseB_phaseC_profile_style_aligned.ipynb"


OUTPUT_ROOT = WORKDIR / "outputs" / "profile_optimization_outputs"


PROFILES_DIR = OUTPUT_ROOT / "profiles"


FINAL_DIR = OUTPUT_ROOT / "selected_profile_final_external"


SUMMARY_DIR = OUTPUT_ROOT / "summaries"


DEFAULT_TIMEOUT_SECONDS = 24 * 60 * 60


SCORE_MODE = "original"


TARGETS = [('event_1_death_hospice', 'death_hospice'), ('event_2_institutional_nonhome', 'institutional_nonhome'), ('event_3_icu_los_gt7d', 'icu_los_gt7d'), ('event_5_home', 'home')]


PATH_VARIABLES = [
    "TRAIN_DATA_PATH",
    "EXTERNAL_VALID_DATA_PATH",
    "OUT_XLSX",
    "RESULT_JSON",
    "OUT_PNG_TRAJ",
    "OUT_PNG_PHASEA_4FOLD",
    "OUT_PNG",
    "OUTPUT_DIR",
    "FIGURE_DIR",
    "OUT_XLSX_BASELINE",
    "RESULT_JSON_BASELINE",
    "OUT_XLSX_ALGO",
    "RESULT_JSON_ALGO",
    "OUT_PNG_TRAJ_TEMPLATE",
    "SINGLE_ENDPOINT_KNOWLEDGE_OUTPUT_ROOT",
]


PATH_OBJECT_VARIABLES = {
    "SINGLE_ENDPOINT_KNOWLEDGE_OUTPUT_ROOT",
}


PROFILE_PATCH_KEYS = [
    "TOP_K",
    "N_TREES",
    "POP_SIZE_X",
    "N_GEN_X",
    "MAX_TREE_HEIGHT_X",
    "CX_PB_X",
    "MUT_PB_X",
    "COMPLEXITY_LAMBDA_X",
    "DUPLICATE_TERMINAL_LAMBDA_X",
    "COX_PENALIZER",
    "COX_L1_RATIO",
    "PH_P_THRESHOLD",
    "PHASEB_POSITIVE_FOLDS_MIN",
    "PHASEB_MEAN_DELTA_MIN",
    "PHASEB_COMPLEXITY_LAMBDA",
    "TIMEBASIS_COMPLEXITY_LAMBDA",
    "PHASEC_POP_SIZE",
    "PHASEC_N_GEN",
    "PHASEC_ELITE_SIZE",
    "PHASEC_HOF_SIZE",
    "PHASEC_CROSSOVER_RATE",
    "PHASEC_MUTATION_RATE",
    "PHASEC_RANDOM_INJECTION_RATE",
    "PHASEC_COMPLEXITY_LAMBDA",
    "PHASEC_K_VALUES",
    "PHASEC_MIN_MAIN_N",
    "PHASEC_MIN_MAIN_EVENTS",
    "PHASEC_MIN_INTER_N",
    "PHASEC_MIN_INTER_EVENTS",
    "MAIN_EFFECT_LOCK_MODE",
    "GP_INPUT_LOCK_MODE",
    "LOCK_MAIN_EFFECTS_FROM_SINGLE_PROFILE",
    "LOCK_GP_INPUTS_FROM_SINGLE_PROFILE",
    "LOCK_SINGLE_PROFILE_BACKBONE",
    "LOCKED_SINGLE_PROFILE_JSON",
    "LOCKED_MAIN_EFFECT_VARS",
    "LOCKED_GP_INPUT_VARS",
    "LOCKED_FINAL_FEATURE_SET_MODE",
    "LOCKED_FEATURE_SET_MODE",
    "PHASEA_EXTERNAL_TREES_JSON",
    "PHASEC_EXTERNAL_RULES_JSON",
]


LONG_FIELDS = [
    "profile_id", "profile_name", "target_id", "target_name", "status", "runtime_seconds",
    "A_internal_cindex", "AB_internal_cindex", "AC_internal_cindex", "ABC_internal_cindex",
    "delta_B", "delta_C_static", "delta_C_afterB", "delta_total",
    "A_external_cindex", "AB_external_cindex", "AC_external_cindex", "ABC_external_cindex",
    "phaseB_allowed", "phaseB_active_final", "selected_k_cat", "phaseC_active_final", "unified_abc_fallback_mode",
    "phaseB_selected_candidates", "phaseC_selected_candidates",
    "phaseB_candidate_n", "phaseC_candidate_n", "AC_internal_cindex_missing",
    "train_n", "train_events", "external_n", "external_events",
    "leak_check_passed", "used_full_train_stable_vars_in_cv", "external_used_for_selection",
    "warning", "error_message", "result_json_path", "result_xlsx_path", "log_path",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def path_str(path: Path) -> str:
    return str(path.resolve())


def py_literal(value: Any) -> str:
    if isinstance(value, Path):
        return "r" + repr(path_str(value))
    return repr(value)


def assignment_literal(key: str, value: Any) -> str:
    if isinstance(value, Path) and key in PATH_OBJECT_VARIABLES:
        return "Path(" + py_literal(value) + ")"
    return py_literal(value)


def read_notebook(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text))


def write_notebook(path: Path, notebook: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def source_text(source: Any) -> str:
    return "".join(source) if isinstance(source, list) else str(source or "")


def source_like(text: str, original: Any) -> Any:
    return text.splitlines(keepends=True) if isinstance(original, list) else text


def patch_notebook_for_profile(
    input_ipynb: Path,
    output_ipynb: Path,
    train_path: Path,
    external_path: Path,
    output_dir: Path,
    profile_dict: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    paths = {
        "TRAIN_DATA_PATH": train_path,
        "EXTERNAL_VALID_DATA_PATH": external_path,
        "OUT_XLSX": output_dir / "result.xlsx",
        "RESULT_JSON": output_dir / "result.json",
        "OUT_PNG_TRAJ": output_dir / "figure.png",
        "OUT_PNG_PHASEA_4FOLD": output_dir / "phaseA_8fold.png",
        "OUT_PNG": output_dir / "figure.png",
        "OUTPUT_DIR": output_dir,
        "FIGURE_DIR": output_dir,
        "OUT_XLSX_BASELINE": output_dir / "_unused_baseline.xlsx",
        "RESULT_JSON_BASELINE": output_dir / "_unused_baseline.json",
        "OUT_XLSX_ALGO": output_dir / "result.xlsx",
        "RESULT_JSON_ALGO": output_dir / "result.json",
        "OUT_PNG_TRAJ_TEMPLATE": output_dir / "outer{fold}_trajectory.png",
        "SINGLE_ENDPOINT_KNOWLEDGE_OUTPUT_ROOT": output_dir / "single_endpoint_knowledge_outputs",
    }
    replacements: dict[str, Any] = dict(paths)
    replacements.update(profile_dict)
    replacements["SEED"] = int(seed)

    nb = read_notebook(input_ipynb)
    counts = {k: 0 for k in replacements}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        original = cell.get("source", "")
        text = source_text(original)
        assignment_edits = []
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            key = node.targets[0].id
            if key not in replacements:
                continue
            value = replacements[key]
            assignment_edits.append((node.lineno-1, node.end_lineno,
                " " * node.col_offset + key + " = " + assignment_literal(key, value) + "\n"))
            counts[key] += 1
        source_lines = text.splitlines(keepends=True)
        for start_line, end_line, replacement in sorted(assignment_edits, reverse=True):
            source_lines[start_line:end_line] = [replacement]
        text = "".join(source_lines)
        feature_decision_anchor = 'globals()["FEATURE_SET_DECISION"] = feature_decision["feature_set_decision"]'
        if feature_decision_anchor in text and "_mt_gp_input_lock_enabled" not in text:
            guard = '''
    _mt_main_effect_lock_enabled = str(globals().get("MAIN_EFFECT_LOCK_MODE", "unlocked")) == "profile007_locked"
    _mt_gp_input_lock_enabled = str(globals().get("GP_INPUT_LOCK_MODE", "unlocked")) == "profile007_locked"
    globals()["_mt_main_effect_lock_enabled"] = _mt_main_effect_lock_enabled
    globals()["_mt_gp_input_lock_enabled"] = _mt_gp_input_lock_enabled
    globals()["LOCK_MAIN_EFFECTS_FROM_SINGLE_PROFILE"] = bool(globals().get("LOCK_MAIN_EFFECTS_FROM_SINGLE_PROFILE", False)) and _mt_main_effect_lock_enabled
    globals()["LOCK_SINGLE_PROFILE_BACKBONE"] = bool(globals().get("LOCK_SINGLE_PROFILE_BACKBONE", False)) and _mt_main_effect_lock_enabled
    globals()["LOCK_GP_INPUTS_FROM_SINGLE_PROFILE"] = bool(globals().get("LOCK_GP_INPUTS_FROM_SINGLE_PROFILE", False)) and _mt_gp_input_lock_enabled
    if _mt_main_effect_lock_enabled and globals()["LOCK_MAIN_EFFECTS_FROM_SINGLE_PROFILE"]:
        _mt_locked_main = [str(x) for x in (globals().get("LOCKED_MAIN_EFFECT_VARS") or [])]
        if _mt_locked_main:
            globals()["MAIN_EFFECT_VARS"] = list(_mt_locked_main)
            feature_decision["main_effect_vars"] = list(_mt_locked_main)
            feature_decision["use_main_effects_for_this_run"] = True
    if _mt_gp_input_lock_enabled and globals()["LOCK_GP_INPUTS_FROM_SINGLE_PROFILE"]:
        _mt_locked_gp = [str(x) for x in (globals().get("LOCKED_GP_INPUT_VARS") or [])]
        if not _mt_locked_gp:
            raise RuntimeError("GP_INPUT_LOCK_MODE=profile007_locked but LOCKED_GP_INPUT_VARS is empty")
        globals()["GP_INPUT_VARS"] = list(_mt_locked_gp)
        feature_decision["gp_input_vars"] = list(_mt_locked_gp)
        feature_decision["final_vars"] = list(_mt_locked_gp)
        feature_decision["gp_input_vars_source"] = "profile007_locked"
        feature_decision["gp_input_lock_mode"] = "profile007_locked"
        feature_decision["locked_gp_input_vars"] = list(_mt_locked_gp)
        # The GP parameter profile (3-tree main+GP versus 5-tree GP-only) must
        # follow the locked checkpoint inventory as well.  Previously only the
        # variable list was overwritten, so an endpoint with an empty locked
        # main-effect list could incorrectly retain the 3-tree profile chosen
        # by a fresh full-cohort feature screen and fail checkpoint recovery.
        _mt_locked_main = [str(x) for x in (globals().get("LOCKED_MAIN_EFFECT_VARS") or [])]
        # Keep the global feature state in sync with the checkpoint inventory.
        # In particular, an intentionally GP-only endpoint has an empty locked
        # main-effect list; leaving a freshly screened MAIN_EFFECT_VARS value in
        # place makes checkpoint validation compare [] with unrelated variables.
        globals()["MAIN_EFFECT_VARS"] = list(_mt_locked_main)
        feature_decision["main_effect_vars"] = list(_mt_locked_main)
        feature_decision["use_main_effects_for_this_run"] = bool(_mt_locked_main)
        feature_decision["final_feature_set_mode"] = (
            "baseline_stable_main_gp" if _mt_locked_main else "top10_gp_only"
        )
        feature_decision["feature_set_mode"] = feature_decision["final_feature_set_mode"]
        globals()["FEATURE_SET_DECISION"] = str(globals().get("FEATURE_SET_DECISION", "")) + "|gp_input_profile007_locked"
'''
            text = text.replace(feature_decision_anchor, feature_decision_anchor + guard)
            counts["_mt_independent_gp_input_lock_guard"] = counts.get("_mt_independent_gp_input_lock_guard", 0) + 1
        feature_selector_anchor = "def _select_ab_feature_sets(df_train_raw, seed_offset=0, inner_folds_for_varsel=INNER_FOLDS):\n"
        if feature_selector_anchor in text and "checkpoint_feature_inventory_locked" not in text:
            locked_selector = '''def _select_ab_feature_sets(df_train_raw, seed_offset=0, inner_folds_for_varsel=INNER_FOLDS):
    # checkpoint_feature_inventory_locked: checkpoint continuation must use the
    # exact feature inventory that defined the saved GP primitive set. This
    # also prevents an eICU placeholder external frame from changing the
    # candidate universe relative to the locked G run.
    _mt_gp_locked = str(globals().get("GP_INPUT_LOCK_MODE", "unlocked")) == "profile007_locked"
    if _mt_gp_locked:
        gp_input_vars = [str(x) for x in (globals().get("LOCKED_GP_INPUT_VARS") or [])]
        main_effect_vars = [str(x) for x in (globals().get("LOCKED_MAIN_EFFECT_VARS") or [])]
        if not gp_input_vars:
            raise RuntimeError("Checkpoint feature lock requested without LOCKED_GP_INPUT_VARS")
        missing = [name for name in gp_input_vars if name not in df_train_raw.columns]
        if missing:
            raise RuntimeError(f"Checkpoint feature lock variables missing from training data: {missing}")
        final_mode = "baseline_stable_main_gp" if main_effect_vars else "top10_gp_only"
        main_effect_indices = _main_effect_indices_from_final(gp_input_vars, main_effect_vars)
        return {
            "feature_set_mode": final_mode,
            "final_feature_set_mode": final_mode,
            "feature_set_decision": "checkpoint_feature_inventory_locked",
            "use_main_effects_for_this_run": bool(main_effect_vars),
            "final_vars": list(gp_input_vars),
            "gp_input_vars": list(gp_input_vars),
            "gp_input_vars_source": "locked G checkpoint inventory",
            "main_effect_vars": list(main_effect_vars),
            "main_effect_indices": main_effect_indices,
            "stable_vars": list(main_effect_vars),
            "stable_freq_df": pd.DataFrame(),
            "stable_min_freq_used": int(OUTER_FOLDS),
            "stable_fold_df": pd.DataFrame(),
            "freq_df": pd.DataFrame(),
            "inner_selected_df": pd.DataFrame(),
        }
'''
            text = text.replace(feature_selector_anchor, locked_selector, 1)
            counts["_mt_checkpoint_feature_selector_lock"] = counts.get("_mt_checkpoint_feature_selector_lock", 0) + 1
        cell["source"] = source_like(text, original)

    warnings = [f"{k} not found; injected override cell" for k, v in counts.items() if v == 0 and k in profile_dict]
    override_lines = [
        "# Auto-injected by run_multitarget_profile_optimization.py\n",
        "# This cell is a final guard; existing assignments were patched throughout the notebook first.\n",
    ]
    for key in PATH_VARIABLES:
        override_lines.append(f"{key} = {assignment_literal(key, paths[key])}\n")
    override_lines.append(f"SEED = {int(seed)}\n")
    injected_profile_keys = list(PROFILE_PATCH_KEYS)
    for key in profile_dict:
        if key not in injected_profile_keys:
            injected_profile_keys.append(key)
    for key in injected_profile_keys:
        if key in profile_dict:
            override_lines.append(f"{key} = {py_literal(profile_dict[key])}\n")
    override_lines.extend([
        "_mt_main_effect_lock_enabled = str(globals().get('MAIN_EFFECT_LOCK_MODE', 'unlocked')) == 'profile007_locked'\n",
        "_mt_gp_input_lock_enabled = str(globals().get('GP_INPUT_LOCK_MODE', 'unlocked')) == 'profile007_locked'\n",
        "LOCK_MAIN_EFFECTS_FROM_SINGLE_PROFILE = bool(globals().get('LOCK_MAIN_EFFECTS_FROM_SINGLE_PROFILE', False)) and _mt_main_effect_lock_enabled\n",
        "LOCK_SINGLE_PROFILE_BACKBONE = bool(globals().get('LOCK_SINGLE_PROFILE_BACKBONE', False)) and _mt_main_effect_lock_enabled\n",
        "LOCK_GP_INPUTS_FROM_SINGLE_PROFILE = bool(globals().get('LOCK_GP_INPUTS_FROM_SINGLE_PROFILE', False)) and _mt_gp_input_lock_enabled\n",
    ])
    nb.setdefault("cells", []).insert(
        1,
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "multitarget-profile-overrides",
            "metadata": {},
            "outputs": [],
            "source": override_lines,
        },
    )
    if str(profile_dict.get("EIGHTFOLD_RUN_ROLE", "")) == "ordinary_cox_baseline_outputs_from_original_notebook":
        nb["cells"] = nb.get("cells", [])[:3]
        write_notebook(output_ipynb, nb)
        return {"patch_counts": counts, "warnings": warnings, "ordinary_cox_baseline_only": True}
    nb.setdefault("cells", []).append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "multitarget-result-compat-fields",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Auto-injected compatibility output fields; does not alter model fitting or selection logic.\n",
                "try:\n",
                "    import json as _mt_json\n",
                "    from pathlib import Path as _MTPath\n",
                "    _mt_p = _MTPath(RESULT_JSON)\n",
                "    if _mt_p.exists():\n",
                "        _mt_d = _mt_json.loads(_mt_p.read_text(encoding='utf-8'))\n",
                "        _mt_summary = _mt_d.get('summary') if isinstance(_mt_d.get('summary'), list) else []\n",
                "        _mt_s0 = _mt_summary[0] if _mt_summary and isinstance(_mt_summary[0], dict) else {}\n",
                "        _mt_krows = _mt_d.get('internal_cv_abc_k_summary') or _mt_d.get('phasec_abc_unified_k_summary') or _mt_d.get('phasec_k_summary') or []\n",
                "        _mt_sel = next((r for r in _mt_krows if isinstance(r, dict) and int(r.get('selected', 0) or 0) == 1), _mt_krows[0] if _mt_krows else {})\n",
                "        _mt_ac = _mt_s0.get('AC_internal_cindex', _mt_s0.get('AC_static_internal_cindex', None))\n",
                "        if _mt_ac is None and isinstance(_mt_sel, dict):\n",
                "            _mt_ac = _mt_sel.get('phasec_AC_static_cindex', None)\n",
                "        _mt_d['AC_internal_cindex'] = _mt_ac\n",
                "        _mt_d['AC_internal_cindex_missing'] = _mt_ac is None\n",
                "        _mt_d['main_effect_lock_mode'] = globals().get('MAIN_EFFECT_LOCK_MODE', 'unlocked')\n",
                "        _mt_d['gp_input_lock_mode'] = globals().get('GP_INPUT_LOCK_MODE', 'unlocked')\n",
                "        _mt_d['single_knowledge_used_for_main_effect_lock'] = bool(globals().get('LOCK_MAIN_EFFECTS_FROM_SINGLE_PROFILE', False))\n",
                "        _mt_d['single_knowledge_used_for_gp_input_lock'] = bool(globals().get('LOCK_GP_INPUTS_FROM_SINGLE_PROFILE', False))\n",
                "        _mt_d['lock_main_effects_enabled'] = bool(globals().get('_mt_main_effect_lock_enabled', False))\n",
                "        _mt_d['lock_gp_inputs_enabled'] = bool(globals().get('_mt_gp_input_lock_enabled', False))\n",
                "        _mt_d['main_effect_vars'] = list(globals().get('MAIN_EFFECT_VARS', []))\n",
                "        _mt_d['gp_input_vars'] = list(globals().get('GP_INPUT_VARS', []))\n",
                "        _mt_d['locked_main_effect_vars_audit_only'] = list(globals().get('LOCKED_MAIN_EFFECT_VARS', []))\n",
                "        _mt_d['locked_gp_input_vars'] = list(globals().get('LOCKED_GP_INPUT_VARS', []))\n",
                "        if _mt_summary and isinstance(_mt_summary[0], dict):\n",
                "            _mt_summary[0]['AC_internal_cindex'] = _mt_ac\n",
                "            _mt_summary[0]['AC_internal_cindex_missing'] = _mt_ac is None\n",
                "            _mt_summary[0]['main_effect_lock_mode'] = _mt_d['main_effect_lock_mode']\n",
                "            _mt_summary[0]['gp_input_lock_mode'] = _mt_d['gp_input_lock_mode']\n",
                "            _mt_summary[0]['single_knowledge_used_for_main_effect_lock'] = _mt_d['single_knowledge_used_for_main_effect_lock']\n",
                "            _mt_summary[0]['single_knowledge_used_for_gp_input_lock'] = _mt_d['single_knowledge_used_for_gp_input_lock']\n",
                "            _mt_summary[0]['main_effect_vars'] = list(_mt_d['main_effect_vars'])\n",
                "            _mt_summary[0]['gp_input_vars'] = list(_mt_d['gp_input_vars'])\n",
                "        _mt_p.write_text(_mt_json.dumps(_mt_d, ensure_ascii=False, indent=2), encoding='utf-8')\n",
                "except Exception as _mt_exc:\n",
                "    print(f'[multitarget compat warning] failed to append AC_internal_cindex fields: {_mt_exc}')\n",
            ],
        }
    )
    syntax_errors = []
    for cell_index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        try:
            compile(source_text(cell.get("source", "")), f"notebook_cell_{cell_index}", "exec")
        except SyntaxError as exc:
            syntax_errors.append(f"cell={cell_index}: {exc}")
    if syntax_errors:
        write_notebook(output_ipynb, nb)
        raise RuntimeError("Patched notebook syntax check failed: " + "; ".join(syntax_errors))
    from portable_notebook import bind
    nb = bind(nb, train_path, external_path, output_dir, profile_dict)
    write_notebook(output_ipynb, nb)
    return {"patch_counts": counts, "warnings": warnings}


def profile_signature(profile_dict: dict[str, Any]) -> str:
    return json.dumps(profile_dict, sort_keys=True, ensure_ascii=False)


def run_notebook(run_dir: Path, target: dict[str, Any], profile: dict[str, Any], timeout: int, force: bool, seed: int) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    patched = run_dir / "patched_notebook.ipynb"
    executed = run_dir / "executed_notebook.ipynb"
    result_json = run_dir / "result.json"
    result_xlsx = run_dir / "result.xlsx"
    log_path = run_dir / "run.log"
    status_path = run_dir / "run_status.json"

    if not force and status_path.exists() and result_json.exists() and executed.exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("status") == "success":
            old["skipped_existing_success"] = True
            return old

    started = time.time()
    status = {
        "status": "running",
        "started_at": now_iso(),
        "finished_at": None,
        "runtime_seconds": None,
        "target_id": target["target_id"],
        "target_name": target["target_name"],
        "train_file": path_str(target["train_file"]),
        "external_file": path_str(target["external_file"]),
        "patched_notebook": path_str(patched),
        "executed_notebook": path_str(executed),
        "result_json": path_str(result_json),
        "result_xlsx": path_str(result_xlsx),
        "log_path": path_str(log_path),
        "seed": seed,
        "profile": profile,
        "profile_signature": profile_signature(profile),
        "error_message": "",
        "warnings": [],
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        patch_info = patch_notebook_for_profile(NOTEBOOK_FILE, patched, target["train_file"], target["external_file"], run_dir, profile, seed)
        status.update(patch_info)
        notebook_python = os.environ.get("NOTEBOOK_PYTHON") or sys.executable
        command = [notebook_python, str(WORKDIR / "execute_notebook.py"), str(patched), str(executed)]
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"Started at: {status['started_at']}\n")
            log.write(f"TRAIN_SOURCE={TRAIN_SOURCE}; EXTERNAL_SOURCE={EXTERNAL_SOURCE}\n")
            log.write(f"TRAIN_DATA_PATH={path_str(target['train_file'])}\n")
            log.write(f"EXTERNAL_VALID_DATA_PATH={path_str(target['external_file'])}\n")
            log.write("Profile: " + json.dumps(profile, ensure_ascii=False, sort_keys=True) + "\n")
            log.write("Command: " + " ".join(command) + "\n\n")
            proc = subprocess.run(command, cwd=WORKDIR, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=timeout + 300)
        status["returncode"] = proc.returncode
        status["status"] = "success" if proc.returncode == 0 and result_json.exists() else "failed"
        if proc.returncode != 0:
            status["error_message"] = f"nbconvert failed with return code {proc.returncode}"
        elif not result_json.exists():
            status["error_message"] = "Notebook executed but result.json was not created."
    except subprocess.TimeoutExpired:
        status["status"] = "failed"
        status["error_message"] = f"Notebook execution timed out after {timeout} seconds."
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(status["error_message"] + "\n")
    except Exception as exc:
        status["status"] = "failed"
        status["error_message"] = str(exc)
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(traceback.format_exc())
    finally:
        status["finished_at"] = now_iso()
        status["runtime_seconds"] = round(time.time() - started, 3)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return status


def first_row(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    if isinstance(value, dict):
        return value
    return {}


def to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def mean_finite(values: list[Any]) -> float:
    nums = [to_float(v) for v in values]
    nums = [v for v in nums if math.isfinite(v)]
    return float(sum(nums) / len(nums)) if nums else math.nan


def parse_maybe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return [] if value.strip() in {"", "[]"} else [value]
    return []


def candidate_count(value: Any) -> int:
    return len(parse_maybe_list(value))


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def selected_row(rows: Any) -> dict[str, Any]:
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, dict) and int(to_float(row.get("selected")) or 0) == 1:
            return row
    return first_row(rows)


def parse_result_json(path: Path) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = first_row(data.get("summary"))
    phaseb = first_row(data.get("phaseB_summary"))
    phasec = first_row(data.get("phaseC_summary"))
    krow = selected_row(data.get("internal_cv_abc_k_summary") or data.get("phasec_abc_unified_k_summary") or data.get("phasec_k_summary"))
    folds = data.get("fold_results") if isinstance(data.get("fold_results"), list) else []

    a_int = to_float(summary.get("A_internal_cindex", summary.get("baseline_stable_main_gp_oof_cindex", math.nan)))
    ab_int = to_float(summary.get("AB_internal_cindex", summary.get("baseline_stable_main_gp_phaseB_oof_cindex", phaseb.get("main_gp_phaseB_oof_cindex", math.nan))))
    abc_int = to_float(summary.get("ABC_internal_cindex", krow.get("mean_abc", math.nan)))
    ac_int = to_float(first_present(
        data.get("AC_internal_cindex"),
        summary.get("AC_internal_cindex"),
        summary.get("AC_static_internal_cindex"),
        krow.get("phasec_AC_static_cindex"),
    ))
    if not math.isfinite(a_int):
        a_int = mean_finite([r.get("main_gp_cindex") for r in folds if isinstance(r, dict)])
    if not math.isfinite(ab_int):
        ab_int = mean_finite([r.get("ab_final_cindex", r.get("main_gp_phaseB_cindex")) for r in folds if isinstance(r, dict)])
    if not math.isfinite(abc_int):
        abc_int = mean_finite([r.get("abc_final_cindex") for r in folds if isinstance(r, dict)])
    ac_missing = not math.isfinite(ac_int)
    if ac_missing:
        warnings.append("AC_internal_cindex_missing = True")

    ext = first_row(data.get("external_results"))
    out = {
        "A_internal_cindex": a_int,
        "AB_internal_cindex": ab_int,
        "AC_internal_cindex": ac_int,
        "AC_internal_cindex_missing": ac_missing,
        "ABC_internal_cindex": abc_int,
        "A_external_cindex": to_float(data.get("external_A_static_cindex", summary.get("external_A_static_cindex", summary.get("A_static_cindex", ext.get("A_static_cindex", math.nan))))),
        "AB_external_cindex": to_float(data.get("external_AB_timevarying_cindex", summary.get("external_AB_timevarying_cindex", summary.get("AB_timevarying_cindex", math.nan)))),
        "AC_external_cindex": to_float(data.get("external_AC_static_cindex", summary.get("external_AC_static_cindex", summary.get("AC_static_cindex", math.nan)))),
        "ABC_external_cindex": to_float(data.get("external_ABC_timevarying_cindex", summary.get("external_ABC_timevarying_cindex", summary.get("ABC_timevarying_cindex", summary.get("ABC_final_cindex", math.nan))))),
        "phaseB_allowed": data.get("phaseb_timebasis_allowed_by_gate", summary.get("phaseB_allowed", summary.get("phaseB_allowed_final", phaseb.get("phaseB_allowed_final")))),
        "phaseB_active_final": data.get("phaseB_active_final", summary.get("phaseB_active_final")),
        "selected_k_cat": data.get("selected_k_cat", data.get("phasec_selected_k", summary.get("selected_k_cat", summary.get("selected_k_phaseC", phasec.get("selected_k"))))),
        "phaseC_active_final": data.get("phaseC_active_final", summary.get("phaseC_active_final")),
        "unified_abc_fallback_mode": data.get("unified_abc_fallback_mode", summary.get("unified_abc_fallback_mode")),
        "phaseB_selected_candidates": data.get("phaseB_selected_candidates", summary.get("phaseB_selected_candidates", data.get("phaseb_timebasis_final_selected_candidates", phaseb.get("final_kept_phaseB_interactions")))),
        "phaseC_selected_candidates": data.get("phaseC_selected_candidates", summary.get("phaseC_selected_candidates", data.get("phasec_final_selected_candidates", data.get("phasec_stgp_and_final_selected_candidates")))),
        "train_n": summary.get("train_n"),
        "train_events": summary.get("train_events"),
        "external_n": summary.get("external_n"),
        "external_events": summary.get("external_events"),
        "leak_check_passed": data.get("leak_check_passed", summary.get("leak_check_passed")),
        "used_full_train_stable_vars_in_cv": data.get("used_full_train_stable_vars_in_cv", summary.get("used_full_train_stable_vars_in_cv")),
        "external_used_for_selection": data.get("external_used_for_selection", summary.get("external_used_for_selection")),
    }
    out["phaseB_candidate_n"] = candidate_count(first_present(
        out.get("phaseB_selected_candidates"),
        data.get("phaseb_timebasis_final_selected_candidates"),
        phaseb.get("final_kept_phaseB_interactions"),
    ))
    out["phaseC_candidate_n"] = candidate_count(first_present(
        out.get("phaseC_selected_candidates"),
        data.get("phasec_final_selected_candidates"),
        data.get("phasec_stgp_and_final_selected_candidates"),
    ))
    if out["phaseC_candidate_n"] == 0 and to_float(out.get("selected_k_cat")) > 0:
        out["phaseC_candidate_n"] = int(to_float(out.get("selected_k_cat")))
    for key in ["A_internal_cindex", "AB_internal_cindex", "ABC_internal_cindex"]:
        if not math.isfinite(to_float(out[key])):
            warnings.append(f"{key} could not be parsed.")
    return out, warnings


def parse_result_excel(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    xl = pd.ExcelFile(path)
    if "summary" in xl.sheet_names:
        df = pd.read_excel(path, sheet_name="summary")
        if len(df):
            out.update(df.iloc[0].to_dict())
    return out


def build_long_row(profile_meta: dict[str, Any], target: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    warnings = list(status.get("warnings") or [])
    error = status.get("error_message", "")
    result_json = Path(status["result_json"])
    result_xlsx = Path(status["result_xlsx"])
    if result_json.exists():
        try:
            metrics, parse_warnings = parse_result_json(result_json)
            warnings.extend(parse_warnings)
        except Exception as exc:
            error = (error + f"; JSON parse failed: {exc}").strip("; ")
    elif result_xlsx.exists():
        try:
            metrics = parse_result_excel(result_xlsx)
            warnings.append("Parsed Excel fallback; internal metric coverage may be incomplete.")
        except Exception as exc:
            error = (error + f"; Excel parse failed: {exc}").strip("; ")

    a = to_float(metrics.get("A_internal_cindex"))
    ab = to_float(metrics.get("AB_internal_cindex"))
    ac = to_float(metrics.get("AC_internal_cindex"))
    abc = to_float(metrics.get("ABC_internal_cindex"))
    row = {k: None for k in LONG_FIELDS}
    row.update({
        "profile_id": profile_meta["profile_id"],
        "profile_name": profile_meta["profile_name"],
        "target_id": target["target_id"],
        "target_name": target["target_name"],
        "status": status.get("status"),
        "runtime_seconds": status.get("runtime_seconds"),
        "A_internal_cindex": a,
        "AB_internal_cindex": ab,
        "AC_internal_cindex": ac,
        "ABC_internal_cindex": abc,
        "delta_B": ab - a if math.isfinite(ab) and math.isfinite(a) else math.nan,
        "delta_C_static": ac - a if math.isfinite(ac) and math.isfinite(a) else math.nan,
        "delta_C_afterB": abc - ab if math.isfinite(abc) and math.isfinite(ab) else math.nan,
        "delta_total": abc - a if math.isfinite(abc) and math.isfinite(a) else math.nan,
        "A_external_cindex": metrics.get("A_external_cindex"),
        "AB_external_cindex": metrics.get("AB_external_cindex"),
        "AC_external_cindex": metrics.get("AC_external_cindex"),
        "ABC_external_cindex": metrics.get("ABC_external_cindex"),
        "phaseB_allowed": metrics.get("phaseB_allowed"),
        "phaseB_active_final": metrics.get("phaseB_active_final"),
        "selected_k_cat": metrics.get("selected_k_cat"),
        "phaseC_active_final": metrics.get("phaseC_active_final"),
        "unified_abc_fallback_mode": metrics.get("unified_abc_fallback_mode"),
        "phaseB_selected_candidates": metrics.get("phaseB_selected_candidates"),
        "phaseC_selected_candidates": metrics.get("phaseC_selected_candidates"),
        "phaseB_candidate_n": metrics.get("phaseB_candidate_n"),
        "phaseC_candidate_n": metrics.get("phaseC_candidate_n"),
        "AC_internal_cindex_missing": metrics.get("AC_internal_cindex_missing", not math.isfinite(ac)),
        "train_n": metrics.get("train_n"),
        "train_events": metrics.get("train_events"),
        "external_n": metrics.get("external_n"),
        "external_events": metrics.get("external_events"),
        "leak_check_passed": metrics.get("leak_check_passed"),
        "used_full_train_stable_vars_in_cv": metrics.get("used_full_train_stable_vars_in_cv"),
        "external_used_for_selection": metrics.get("external_used_for_selection"),
        "warning": "; ".join(str(x) for x in warnings if x),
        "error_message": error,
        "result_json_path": status.get("result_json"),
        "result_xlsx_path": status.get("result_xlsx"),
        "log_path": status.get("log_path"),
    })
    if row["status"] == "success" and not metrics:
        row["status"] = "parsed_failed"
    return row


def build_profiles(profile_set="pilot"):
    params = json.loads((WORKDIR.parent / "config/single_target_G_defaults.json").read_text())
    return [{"profile_id": 704, "profile_name": "hcoxfix_trial010_forceB_softC", "params": params}]
