from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checkpoint_fair_multitarget_experiments import common


SOURCE_BRANCH = ROOT / "multitarget_6endpoint_gp_exchange_policy_tree"
SOURCE_CONFIG = SOURCE_BRANCH / "branch_config.json"
DEFAULT_OUTPUT = HERE / "outputs" / "2g_end_only_gp_policy_exchange"


def validate_2g_knowledge() -> dict:
    cache_path = common.AUTO_2G_KNOWLEDGE_ROOT / "single_endpoint_knowledge_cache.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Build the frozen true-2G knowledge first: {cache_path}"
        )
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    expected_budget = {
        "main_effect_phaseA": 16,
        "gp_only_phaseA": 20,
        "final_full_train_phaseA": 60,
        "phaseC": 40,
    }
    if cache.get("generation_budget") != expected_budget:
        raise RuntimeError(f"2G knowledge budget mismatch: {cache.get('generation_budget')}")
    if cache.get("source_kind") != "completed_uninterrupted_from_zero_2G_results":
        raise RuntimeError("2G knowledge is not derived from the approved from-zero control")
    if len(cache.get("endpoints", {})) != 6:
        raise RuntimeError("2G knowledge does not contain six endpoints")
    equivalence_path = Path(cache["checkpoint_resume_equivalence_audit"])
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    if equivalence.get("passed") is not True:
        raise RuntimeError("The G->2G versus 0->2G equivalence audit did not pass")
    return cache


def build_config(output: Path) -> Path:
    validate_2g_knowledge()
    config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    config["branch_name"] = "fair_2G_end_only_gp_exchange_policy_tree"
    config["source_parameter_profile"] = common.PROFILE_NAME
    config["knowledge_root"] = str(common.AUTO_2G_KNOWLEDGE_ROOT.resolve())
    config["output_root"] = str(output.resolve())
    config["branch_root"] = str(SOURCE_BRANCH.resolve())
    config["run_output_root"] = str(output.resolve())
    config["frozen_source_generation"] = "2G"
    config["target_side_evolution_after_knowledge_load"] = False
    output.mkdir(parents=True, exist_ok=True)
    path = output / "resolved_branch_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_root)
    config_path = build_config(output)
    if args.prepare_only:
        print(config_path)
        return 0
    completed = subprocess.run(
        [sys.executable, str(SOURCE_BRANCH / "exchange_policy_tree.py"), "--config", str(config_path)],
        cwd=str(SOURCE_BRANCH),
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
