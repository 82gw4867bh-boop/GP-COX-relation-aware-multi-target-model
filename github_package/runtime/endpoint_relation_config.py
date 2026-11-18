from __future__ import annotations

import json
import os
from typing import Any, Sequence

import numpy as np


_DEFAULT_ENDPOINTS = ['death_hospice', 'institutional_nonhome', 'icu_los_gt7d', 'home']

_DEFAULT_ENDPOINT_IDS = {'death_hospice': 'event_1_death_hospice', 'institutional_nonhome': 'event_2_institutional_nonhome', 'icu_los_gt7d': 'event_3_icu_los_gt7d', 'home': 'event_5_home'}

_DEFAULT_ENDPOINT_GROUPS = {'terminal': ['death_hospice', 'institutional_nonhome', 'home'], 'resource': ['icu_los_gt7d']}

_DEFAULT_EXCLUSIVE_GROUPS = [["death_hospice", "institutional_nonhome", "home"]]


def _load_runtime_endpoint_config() -> dict[str, Any]:
    raw = os.environ.get("MTGP_ENDPOINT_CONFIG_JSON", "").strip()
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("MTGP_ENDPOINT_CONFIG_JSON must contain a JSON object")
    rows = payload.get("endpoints")
    if not isinstance(rows, list) or len(rows) < 2:
        raise RuntimeError("Runtime endpoint config requires at least two endpoints")
    names = [str(row["name"]) for row in rows]
    ids = [str(row["id"]) for row in rows]
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise RuntimeError("Runtime endpoint names and IDs must be unique")
    for key in ("phaseA_matrix", "phaseC_matrix"):
        matrix = payload.get(key)
        if not isinstance(matrix, list) or len(matrix) != len(rows) or any(
            not isinstance(row, list) or len(row) != len(rows) for row in matrix
        ):
            raise RuntimeError(f"Runtime {key} must be a square endpoint matrix")
    return payload


_RUNTIME_CONFIG = _load_runtime_endpoint_config()
if _RUNTIME_CONFIG:
    _RUNTIME_ROWS = list(_RUNTIME_CONFIG["endpoints"])
    ENDPOINTS = [str(row["name"]) for row in _RUNTIME_ROWS]
    ENDPOINT_IDS = {
        str(row["name"]): str(row["id"]) for row in _RUNTIME_ROWS
    }
    ENDPOINT_GROUPS: dict[str, list[str]] = {}
    for row in _RUNTIME_ROWS:
        group = str(row.get("group") or "ungrouped")
        ENDPOINT_GROUPS.setdefault(group, []).append(str(row["name"]))
    EXCLUSIVE_GROUPS = [
        [str(value) for value in group if str(value) in ENDPOINTS]
        for group in _RUNTIME_CONFIG.get("exclusive_groups", [])
    ]
    EXCLUSIVE_GROUPS = [group for group in EXCLUSIVE_GROUPS if len(group) >= 2]
else:
    ENDPOINTS = list(_DEFAULT_ENDPOINTS)
    ENDPOINT_IDS = dict(_DEFAULT_ENDPOINT_IDS)
    ENDPOINT_GROUPS = {
        key: list(values) for key, values in _DEFAULT_ENDPOINT_GROUPS.items()
    }
    EXCLUSIVE_GROUPS = [list(group) for group in _DEFAULT_EXCLUSIVE_GROUPS]


def endpoint_group(endpoint: str) -> str | None:
    """Return the clinical relation group for an endpoint name."""
    for group, members in ENDPOINT_GROUPS.items():
        if endpoint in members:
            return group
    return None


def build_exclusive_conflict_matrix(endpoint_names: Sequence[str]) -> np.ndarray:
    """
    Return a 0/1 symmetric matrix for mutually exclusive terminal disposition outcomes.

    This matrix is only used for prediction conflict penalty, not for Phase A or
    Phase C transfer.
    """
    names = list(endpoint_names)
    idx = {name: i for i, name in enumerate(names)}
    mat = np.zeros((len(names), len(names)), dtype=float)
    for group in EXCLUSIVE_GROUPS:
        present = [idx[name] for name in group if name in idx]
        for i in present:
            for j in present:
                if i != j:
                    mat[i, j] = 1.0
    return mat


def _matrix_from_fixed(endpoint_names: Sequence[str], fixed: Sequence[Sequence[float]]) -> np.ndarray:
    """Map an ENDPOINTS-ordered matrix onto a requested endpoint ordering."""
    names = list(endpoint_names)
    base_idx = {name: i for i, name in enumerate(ENDPOINTS)}
    mat = np.zeros((len(names), len(names)), dtype=float)
    for i, target in enumerate(names):
        for j, source in enumerate(names):
            if target in base_idx and source in base_idx:
                mat[i, j] = float(fixed[base_idx[target]][base_idx[source]])
    np.fill_diagonal(mat, 0.0)
    return mat


def build_phaseA_transfer_matrix(endpoint_names: Sequence[str]) -> np.ndarray:
    """
    Severity-aware transfer matrix for Phase A continuous primitive-tree transfer.

    Only feature/tree structures transfer. Cox coefficients are always
    re-estimated inside the target endpoint.
    """
    if _RUNTIME_CONFIG:
        return _matrix_from_fixed(
            endpoint_names,
            _RUNTIME_CONFIG["phaseA_matrix"],
        )
    phasea_transfer_matrix = [[0.0, 0.5, 0.6, 0.3], [0.5, 0.0, 0.4, 0.65], [0.6, 0.4, 0.0, 0.3], [0.3, 0.65, 0.3, 0.0]]
    return _matrix_from_fixed(endpoint_names, phasea_transfer_matrix)


def build_phaseC_transfer_matrix(endpoint_names: Sequence[str]) -> np.ndarray:
    """
    Rule-aware transfer matrix for Phase C AND-rule structure transfer.

    Only rule structures transfer. Cox coefficients are always re-estimated
    inside the target endpoint.
    """
    if _RUNTIME_CONFIG:
        return _matrix_from_fixed(
            endpoint_names,
            _RUNTIME_CONFIG["phaseC_matrix"],
        )
    phasec_transfer_matrix = [[0.0, 0.55, 0.65, 0.3], [0.55, 0.0, 0.45, 0.7], [0.65, 0.45, 0.0, 0.3], [0.3, 0.7, 0.3, 0.0]]
    return _matrix_from_fixed(endpoint_names, phasec_transfer_matrix)


def build_legacy_phaseC_transfer_matrix(endpoint_names: Sequence[str]) -> np.ndarray:
    """Build the legacy terminal/resource matrix used by current Phase C transfer."""
    return build_transfer_matrix(endpoint_names)


def build_transfer_matrix(endpoint_names: Sequence[str]) -> np.ndarray:
    """Deprecated: legacy terminal/resource transfer matrix. Do not use for Phase A."""
    names = list(endpoint_names)
    mat = np.zeros((len(names), len(names)), dtype=float)
    for i, target in enumerate(names):
        for j, source in enumerate(names):
            if i == j:
                continue
            gt = endpoint_group(target)
            gs = endpoint_group(source)
            mat[i, j] = 0.7 if gt and gt == gs else 0.3

    special = {}
    idx = {name: i for i, name in enumerate(names)}
    for (a, b), weight in special.items():
        if a in idx and b in idx:
            mat[idx[a], idx[b]] = weight
            mat[idx[b], idx[a]] = weight
    np.fill_diagonal(mat, 0.0)
    return mat


def sample_transfer_task(task_idx: int, transfer_matrix: np.ndarray, rng=None) -> int:
    """Sample a source endpoint index using relation weights, never returning task_idx."""
    rng = np.random.default_rng(rng)
    weights = np.asarray(transfer_matrix[task_idx], dtype=float).copy()
    if task_idx < 0 or task_idx >= len(weights):
        raise IndexError(f"task_idx out of range: {task_idx}")
    weights[task_idx] = 0.0
    candidates = np.array([i for i in range(len(weights)) if i != task_idx], dtype=int)
    if not len(candidates):
        raise ValueError("At least two endpoints are required for transfer sampling.")
    cand_weights = weights[candidates]
    if not np.isfinite(cand_weights).all() or cand_weights.sum() <= 0:
        return int(rng.choice(candidates))
    return int(rng.choice(candidates, p=cand_weights / cand_weights.sum()))
