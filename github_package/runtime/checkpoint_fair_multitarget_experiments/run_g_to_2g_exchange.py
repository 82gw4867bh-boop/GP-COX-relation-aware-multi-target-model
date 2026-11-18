from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD_BRANCH = ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(OLD_BRANCH) not in sys.path:
    sys.path.insert(0, str(OLD_BRANCH))

from checkpoint_fair_multitarget_experiments import common
import multitarget_abc_runner as mt


BOUNDARY_NOTEBOOK = Path(
    os.environ.get(
        "MTGP_BOUNDARY_OUTPUT_NOTEBOOK",
        str(HERE / "boundary_notebook" / "newcox_route1_phaseB_phaseC_profile_style_aligned.ipynb"),
    )
)
ONLINE_NOTEBOOK = Path(
    os.environ.get(
        "MTGP_ONLINE_OUTPUT_NOTEBOOK",
        str(HERE / "online_notebook" / "newcox_route1_phaseB_phaseC_profile_style_aligned.ipynb"),
    )
)
DEFAULT_OUTPUTS = {
    "boundary": HERE / "outputs" / "g_boundary_exchange_then_g_evolution",
    "online": HERE / "outputs" / "g_online_exchange_during_g_evolution",
}


ALLOWED_EXCHANGE_PARAMS = {
    # Fixed internal-only terminal-outcome joint selection.  These values do
    # not alter the source G checkpoints and are never selected on MIMIC.
    "INTERNAL_EXCLUSIVE_SELECTION_WEIGHT",
    "INTERNAL_EXCLUSIVE_HIGH_QUANTILE",
    "INTERNAL_EXCLUSIVE_FORCE_K_BY_ENDPOINT",
    "INTERNAL_EXCLUSIVE_SELECTION_ENABLED",
    "INTERNAL_EXCLUSIVE_REPLAY",
    "PHASEA_MIN_TRANSFER_WEIGHT",
    "PHASEA_RELATION_WEIGHT_POWER",
    "MULTITARGET_PHASEA_MAX_TRANSFER_TREES_PER_TARGET",
    "MAX_PHASEA_SOURCE_TREES_PER_ENDPOINT",
    "PHASEC_MIN_TRANSFER_WEIGHT",
    "PHASEC_RELATION_WEIGHT_POWER",
    "PHASEC_MAX_RULES_PER_TARGET",
    "MAX_PHASEC_SOURCE_RULES_PER_ENDPOINT",
    "CROSS_ENDPOINT_CROSSOVER_PROB",
    "phaseA_relation_matrix_override",
    "phaseC_relation_matrix_override",
    "PHASEC_TRANSFER_MODE",
    "PHASEC_TRANSFER_SCORING_MODE",
    "PHASEC_NOVEL_EXTERNAL_RULE_QUOTA_PER_TARGET",
    "PHASEC_NOVEL_MIN_TRANSFER_SCORE",
    "PHASEC_NOVEL_MIN_RELATION_SCORE",
    "PHASEC_NOVEL_MIN_SOURCE_QUALITY_SCORE",
    "PHASEC_NOVEL_MIN_SOURCE_COUNT",
    "PHASEC_NOVEL_REQUIRE_SOURCE_FINAL_OR_FOLD_SELECTED",
    "PHASEC_NOVEL_MIN_SOURCE_SUPPORT_EVENTS",
    "PHASEC_NOVEL_TARGET_SUPPORT_GATE_REQUIRED",
    "PHASEC_NOVEL_DEDUP_BY_PAIR",
    "PHASEC_MATRIX_RELATION_SCORE_WEIGHT",
    "PHASEC_SOURCE_RULE_QUALITY_SCORE_WEIGHT",
    "PHASEC_SOURCE_SELECTION_SCORE_WEIGHT",
    "PHASEC_SOURCE_SUPPORT_SCORE_WEIGHT",
    "PHASEC_SOURCE_COUNT_SCORE_WEIGHT",
    "DEATH_PHASEA_TARGET_RELATION_MULTIPLIER",
    "NONHOME_PHASEA_TARGET_RELATION_MULTIPLIER",
    "EVENT3_PHASEA_TARGET_RELATION_MULTIPLIER",
    "EVENT4_PHASEA_TARGET_RELATION_MULTIPLIER",
    "HOME_PHASEA_TARGET_RELATION_MULTIPLIER",
    "HOSPLOS_PHASEA_TARGET_RELATION_MULTIPLIER",
    "DEATH_PHASEA_MAX_TREES_MULTIPLIER",
    "NONHOME_PHASEA_MAX_TREES_MULTIPLIER",
    "EVENT3_PHASEA_MAX_TREES_MULTIPLIER",
    "EVENT4_PHASEA_MAX_TREES_MULTIPLIER",
    "HOME_PHASEA_MAX_TREES_MULTIPLIER",
    "HOSPLOS_PHASEA_MAX_TREES_MULTIPLIER",
    "DEATH_PHASEC_TARGET_RELATION_MULTIPLIER",
    "NONHOME_PHASEC_TARGET_RELATION_MULTIPLIER",
    "EVENT3_PHASEC_TARGET_RELATION_MULTIPLIER",
    "EVENT4_PHASEC_TARGET_RELATION_MULTIPLIER",
    "HOME_PHASEC_TARGET_RELATION_MULTIPLIER",
    "HOSPLOS_PHASEC_TARGET_RELATION_MULTIPLIER",
    "DEATH_PHASEC_MAX_RULES_MULTIPLIER",
    "NONHOME_PHASEC_MAX_RULES_MULTIPLIER",
    "EVENT3_PHASEC_MAX_RULES_MULTIPLIER",
    "EVENT4_PHASEC_MAX_RULES_MULTIPLIER",
    "HOME_PHASEC_MAX_RULES_MULTIPLIER",
    "HOSPLOS_PHASEC_MAX_RULES_MULTIPLIER",
    "BOUNDARY_MAX_DONORS_PER_CALL",
    "BOUNDARY_RECIPIENT_POOL_SIZE",
    "BOUNDARY_MIN_PLACEMENT_GAIN",
    "BOUNDARY_MAX_MIGRANTS",
    "BOUNDARY_MIGRATION_GRACE_GENERATIONS",
    "BOUNDARY_MIGRANT_PARENT_SLOTS",
    "BOUNDARY_ALLOW_CROSS_FOLD_FALLBACK",
    "BOUNDARY_PHASEC_SEED_FRACTION",
}


def fair_profile(exchange_params=None):
    profile_path = Path(os.environ["MTGP_SINGLE_PROFILE_JSON"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["profile_name"] = common.PROFILE_NAME
    params = profile["params"]
    continuation = int(os.environ.get("MTGP_G_CONTINUATION_GENERATIONS", 12))
    params["N_GEN_X"] = continuation
    params["FINAL_FULL_TRAIN_N_GEN_X"] = continuation
    for key in ("MAIN_EFFECT_GP_PARAMS", "GP_ONLY_PARAMS"):
        params[key]["N_GEN_X"] = continuation
    params["PHASEC_N_GEN"] = int(os.environ.get("MTGP_PHASEC_TOTAL_GENERATIONS", 40))
    params["MAIN_EFFECT_LOCK_MODE"] = "unlocked"
    params["GP_INPUT_LOCK_MODE"] = "unlocked"
    params.update(exchange_params or {})
    profile["seed_profile_name"] = common.PROFILE_NAME
    return profile


def configure(mode: str, notebook: Path, exchange_params: dict[str, Any] | None = None) -> None:
    checkpoint_audit = common.validate_g_checkpoints()
    if checkpoint_audit["checkpoint_n"] != len(common.ENDPOINT_NAMES) * 5:
        raise RuntimeError("Approved G checkpoint inventory is incomplete")
    if not notebook.exists():
        raise FileNotFoundError(f"Build the {mode} notebook first: {notebook}")

    mt.SINGLE_ENDPOINT_KNOWLEDGE_CACHE = common.G_KNOWLEDGE_ROOT / "single_endpoint_knowledge_cache.json"
    mt.DEFAULT_SINGLE_ENDPOINT_REFERENCE_ROOT = common.G_KNOWLEDGE_ROOT
    mt.MAIN_EFFECT_LOCK_MODE = "unlocked"
    mt.GP_INPUT_LOCK_MODE = "unlocked"
    mt.SINGLE_NOTEBOOK_BACKEND = notebook
    mt.MULTITARGET_NOTEBOOK_BACKEND = notebook
    mt.legacy.NOTEBOOK_FILE = notebook
    mt.ABLATIONS["M3"] = {"relation_aware_phaseA": True, "relation_aware_phaseC": True}
    mt.build_phaseA_source_pool_from_single_endpoint_knowledge = (
        lambda _packages, max_per_endpoint=100: common.checkpoint_phasea_source_pool(mt, max_per_endpoint)
    )
    if mode == "boundary":
        mt.build_phaseA_transfer_candidates_from_source_pool = (
            lambda source_pool, target_gp_input_vars, transfer_matrix,
            min_transfer_weight=0.0, relation_weight_power=1.0,
            max_trees_per_target=100, max_trees_per_target_by_endpoint=None:
            common.call_aware_phasea_transfer_candidates(
                mt,
                source_pool,
                target_gp_input_vars,
                transfer_matrix,
                min_transfer_weight=min_transfer_weight,
                relation_weight_power=relation_weight_power,
                max_trees_per_target=max_trees_per_target,
                max_trees_per_target_by_endpoint=max_trees_per_target_by_endpoint,
            )
        )

    profile = fair_profile(exchange_params)

    # ``fair_profile`` is deliberately restricted to exchange parameters, but
    # controller-only internal selection fields must also survive into the
    # runner's profile payload.  Without this explicit merge the replay config
    # contains a forced K map that is silently absent from endpoint notebooks.
    for key in (
        "INTERNAL_EXCLUSIVE_SELECTION_ENABLED",
        "INTERNAL_EXCLUSIVE_SELECTION_WEIGHT",
        "INTERNAL_EXCLUSIVE_HIGH_QUANTILE",
        "INTERNAL_EXCLUSIVE_FORCE_K_BY_ENDPOINT",
        "INTERNAL_EXCLUSIVE_REPLAY",
    ):
        if key in (exchange_params or {}):
            profile["params"][key] = (exchange_params or {})[key]

    checkpoint_features: dict[str, dict[str, list[str]]] = {}
    for endpoint_name, endpoint_id in mt.ENDPOINT_IDS.items():
        metadata_files = sorted(
            (common.G_CHECKPOINT_ROOT / endpoint_id / "phaseA_checkpoints").glob(
                "phaseA_call_*_generation_*.json"
            )
        )
        if len(metadata_files) != 5:
            raise RuntimeError(
                f"Expected five checkpoint metadata files for {endpoint_id}, got {len(metadata_files)}"
            )
        rows = [json.loads(path.read_text(encoding="utf-8-sig")) for path in metadata_files]
        gp_sets = {tuple(row.get("gp_input_vars") or []) for row in rows}
        main_sets = {tuple(row.get("main_effect_vars") or []) for row in rows}
        if len(gp_sets) != 1 or len(main_sets) != 1:
            raise RuntimeError(
                f"Checkpoint feature inventory differs across calls for {endpoint_id}"
            )
        checkpoint_features[endpoint_name] = {
            "gp_input_vars": list(next(iter(gp_sets))),
            "main_effect_vars": list(next(iter(main_sets))),
        }

    def profiles(self: Any, _profile_set: str, only_profile: str | None = None) -> list[dict[str, Any]]:
        selected = json.loads(json.dumps(profile, ensure_ascii=False))
        # A nested internal-exclusivity replay constructs a new runner from a
        # replay config after ``configure`` has already captured ``profile``.
        # Propagate controller-only replay fields from that runner; otherwise
        # the endpoint notebook silently falls back to its locally selected K.
        for key in (
            "INTERNAL_EXCLUSIVE_SELECTION_ENABLED",
            "INTERNAL_EXCLUSIVE_SELECTION_WEIGHT",
            "INTERNAL_EXCLUSIVE_HIGH_QUANTILE",
            "INTERNAL_EXCLUSIVE_FORCE_K_BY_ENDPOINT",
            "INTERNAL_EXCLUSIVE_REPLAY",
        ):
            if key in self.multitarget_params:
                selected["params"][key] = self.multitarget_params[key]
        if only_profile:
            selected["profile_name"] = only_profile
        return [selected]

    seeds = common.original_endpoint_seeds()

    def exact_seed(self: Any, *parts: str) -> int:
        for part in parts:
            if part in seeds:
                return seeds[part]
        raise RuntimeError(f"Cannot resolve original endpoint seed from seed parts: {parts}")

    mt.MultiTargetABCCompetitionRunner.profiles = profiles
    mt.MultiTargetABCCompetitionRunner._seed = exact_seed
    original_profile_params = mt.MultiTargetABCCompetitionRunner._profile_params

    def checkpoint_locked_profile_params(
        self: Any,
        profile_value: dict[str, Any],
        ablation: str,
        cfg: dict[str, Any],
        target_name: str | None = None,
    ) -> dict[str, Any]:
        params = original_profile_params(
            self, profile_value, ablation, cfg, target_name
        )
        if target_name not in checkpoint_features:
            raise RuntimeError(f"Missing checkpoint feature inventory for {target_name}")
        inventory = checkpoint_features[target_name]
        params.update(
            {
                "MAIN_EFFECT_LOCK_MODE": "profile007_locked",
                "GP_INPUT_LOCK_MODE": "profile007_locked",
                "LOCK_MAIN_EFFECTS_FROM_SINGLE_PROFILE": True,
                "LOCK_GP_INPUTS_FROM_SINGLE_PROFILE": True,
                "LOCK_SINGLE_PROFILE_BACKBONE": True,
                "LOCKED_MAIN_EFFECT_VARS": list(inventory["main_effect_vars"]),
                "LOCKED_GP_INPUT_VARS": list(inventory["gp_input_vars"]),
            }
        )
        return params

    mt.MultiTargetABCCompetitionRunner._profile_params = checkpoint_locked_profile_params


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["boundary", "online"])
    parser.add_argument("--output-root")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-target")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--exchange-config")
    parser.add_argument("--external-only-from-locked", action="store_true")
    parser.add_argument("--run-external-after-internal", action="store_true")
    args = parser.parse_args()

    exchange_params: dict[str, Any] = {}
    if args.exchange_config:
        payload = json.loads(Path(args.exchange_config).read_text(encoding="utf-8"))
        exchange_params = dict(payload.get("params", payload))
        forbidden = sorted(set(exchange_params) - ALLOWED_EXCHANGE_PARAMS)
        if forbidden:
            raise ValueError(f"Non-exchange parameters are forbidden: {forbidden}")
        if args.mode == "boundary" and "CROSS_ENDPOINT_CROSSOVER_PROB" in exchange_params:
            raise ValueError("CROSS_ENDPOINT_CROSSOVER_PROB is only valid for online mode")
        if args.mode == "online":
            boundary_only = sorted(key for key in exchange_params if key.startswith("BOUNDARY_"))
            if boundary_only:
                raise ValueError(f"Boundary-only parameters are invalid for online mode: {boundary_only}")
    notebook = BOUNDARY_NOTEBOOK if args.mode == "boundary" else ONLINE_NOTEBOOK
    configure(args.mode, notebook, exchange_params)
    if args.preflight_only:
        profile = fair_profile(exchange_params)
        source = fair_profile({})
        actual = dict(profile["params"])
        actual.pop("MAIN_EFFECT_LOCK_MODE", None)
        actual.pop("GP_INPUT_LOCK_MODE", None)
        expected = dict(source["params"])
        expected.pop("MAIN_EFFECT_LOCK_MODE", None)
        expected.pop("GP_INPUT_LOCK_MODE", None)
        for key in exchange_params:
            actual.pop(key, None)
        if actual != expected:
            raise RuntimeError("Fair profile differs from hcoxfix_trial010_forceB_softC beyond PhaseC 20->40")
        print(json.dumps({
            "passed": True,
            "mode": args.mode,
            "profile": common.PROFILE_NAME,
            "checkpoint_audit": common.validate_g_checkpoints(),
            "endpoint_seeds": common.original_endpoint_seeds(),
            "notebook": str(notebook.resolve()),
            "old_trial025_parameter_override_used": False,
            "exchange_params": exchange_params,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.only_target:
        original_discover = mt.discover_targets
        mt.discover_targets = lambda: [
            target for target in original_discover() if target["target_id"] == args.only_target
        ]

    output = Path(args.output_root) if args.output_root else DEFAULT_OUTPUTS[args.mode]
    argv = [
        "--profile-set", "optuna_search",
        "--only-profile", common.PROFILE_NAME,
        "--only-ablation", "M3",
        "--output-root", str(output.resolve()),
        "--single-endpoint-reference-root", str(common.G_KNOWLEDGE_ROOT.resolve()),
        "--notebook-template", str(notebook.resolve()),
        "--trial-id", f"fair_{args.mode}_exchange_hcoxfix_trial010",
    ]
    if args.exchange_config:
        argv.extend(["--profile-override-json", str(Path(args.exchange_config).resolve())])
    if args.force:
        argv.append("--force")
    if args.external_only_from_locked:
        argv.append("--external-only-from-locked")
    rc = int(mt.main(argv) or 0)
    external_rc = None
    if rc == 0 and args.run_external_after_internal and not args.external_only_from_locked:
        external_argv = list(argv) + ["--external-only-from-locked"]
        external_rc = int(mt.main(external_argv) or 0)
        rc = external_rc
    output.mkdir(parents=True, exist_ok=True)
    (output / "fair_experiment_definition.json").write_text(
        json.dumps({
            "mode": args.mode,
            "single_target_parameter_profile": common.PROFILE_NAME,
            "source_G_checkpoint_root": str(common.G_CHECKPOINT_ROOT.resolve()),
            "source_G_knowledge_root": str(common.G_KNOWLEDGE_ROOT.resolve()),
            "phaseA_resume_budget": "12+12 for all five Phase-A calls",
            "phaseC_total_generations": 40,
            "boundary_exchange_mechanism": (
                "call_aware_paired_placement_with_temporary_lineage_opportunity"
                if args.mode == "boundary" else None
            ),
            "endpoint_seeds": common.original_endpoint_seeds(),
            "old_trial025_parameter_override_used": False,
            "runner_returncode": rc,
            "external_only_from_locked": bool(args.external_only_from_locked),
            "external_after_internal_returncode": external_rc,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
