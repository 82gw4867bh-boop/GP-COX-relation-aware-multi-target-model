from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import run_multitarget_profile_optimization as legacy


class SingleEndpointABCPipeline:
    """
    Thin callable wrapper around the existing single-endpoint notebook pipeline.

    Cell 4 runs internal eICU selection only. Cell 5 uses the locked-structure
    helpers in ``multitarget_abc_runner.py`` for deterministic external validation.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def fit_internal_cv(
        self,
        train_file: Path,
        target: dict[str, Any],
        endpoint_config: dict[str, Any],
        run_dir: Path,
        phaseA_external_trees: list[dict[str, Any]] | None = None,
        phaseC_external_rules: list[dict[str, Any]] | None = None,
        return_oof_risk: bool = True,
        timeout: int = 24 * 60 * 60,
        force: bool = False,
        seed: int = 20260523,
    ) -> dict[str, Any]:
        """Run eICU internal CV for one endpoint; MIMIC is not used here."""
        target_local = dict(target)
        target_local["external_file"] = Path(train_file)
        profile = dict(endpoint_config)
        profile["CELL4_STRUCTURE_LOCK_ONLY"] = True
        profile["MIMIC_READ_DISABLED"] = True
        if phaseC_external_rules:
            rules_path = run_dir / "phaseC_external_rules.json"
            rules_path.parent.mkdir(parents=True, exist_ok=True)
            rules_path.write_text(json.dumps(phaseC_external_rules, ensure_ascii=False, indent=2), encoding="utf-8")
            profile["PHASEC_EXTERNAL_RULES_JSON"] = str(rules_path.resolve())
        if phaseA_external_trees:
            trees_path = run_dir / "phaseA_external_trees.json"
            trees_path.parent.mkdir(parents=True, exist_ok=True)
            trees_path.write_text(json.dumps(phaseA_external_trees, ensure_ascii=False, indent=2), encoding="utf-8")
            profile["PHASEA_EXTERNAL_TREES_JSON"] = str(trees_path.resolve())
        return legacy.run_notebook(run_dir, target_local, profile, timeout=timeout, force=force, seed=seed)

def load_result_json(path: Path) -> dict[str, Any]:
    """Load a notebook result JSON, returning an error dict if parsing fails."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "failed", "errors": [str(exc)]}


def read_endpoint_frame(path: Path) -> pd.DataFrame:
    """Read an endpoint Excel file for transfer-rule support checks."""
    return pd.read_excel(path)
