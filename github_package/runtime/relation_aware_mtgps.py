from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from endpoint_relation_config import build_phaseA_transfer_matrix, sample_transfer_task
from locked_trial31_safe_repair_20260802.safe_repair_primitives import (
    soft_div as _audited_soft_div,
    soft_log as _audited_soft_log,
    soft_sqrt as _audited_soft_sqrt,
)


@dataclass
class TreeGroup:
    expressions: list[str]
    source_endpoint: str
    target_endpoint: str
    transfer_weight: float
    generation: int
    whether_transferred: bool
    transfer_parent_expression: str = ""
    target_parent_expression: str = ""


def _safe_div(a, b):
    return _audited_soft_div(a, b)


def _safe_log(x):
    return _audited_soft_log(x)


def _safe_sqrt(x):
    return _audited_soft_sqrt(x)


def _clip(x):
    return np.clip(x, -10.0, 10.0)


def _clean_feature_matrix(x: pd.DataFrame | np.ndarray) -> np.ndarray:
    arr = x.to_numpy(float) if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _tree_complexity(expr: str) -> int:
    return max(1, len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*\(", str(expr))) + len(re.findall(r"\bx\d+\b", str(expr))))


def _tree_height(expr: str) -> int:
    depth = 0
    max_depth = 0
    for ch in str(expr):
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    return max(1, max_depth)


def _eval_expr(expr: str, x: np.ndarray) -> np.ndarray:
    text = re.sub(r"\barray\(", "np.array(", str(expr))
    env = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "safe_div": _safe_div,
        "safe_log": _safe_log,
        "safe_sqrt": _safe_sqrt,
        "clip": _clip,
        "np": np,
        "array": np.array,
    }
    for i in range(x.shape[1]):
        env[f"x{i}"] = x[:, i]
    out = eval(text, {"__builtins__": {}}, env)
    out = np.asarray(out, dtype=float)
    if out.ndim == 0:
        out = np.repeat(float(out), x.shape[0])
    if out.shape[0] != x.shape[0]:
        raise ValueError(f"Expression returned length {out.shape[0]}, expected {x.shape[0]}")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_expr(expr: str) -> str:
    return re.sub(r"\s+", "", str(expr or ""))


def expression_used_features(expr: str, feature_names: Sequence[str]) -> list[str]:
    names = [str(x) for x in feature_names]
    used: list[str] = []
    for idx in sorted({int(x) for x in re.findall(r"\bx(\d+)\b", str(expr or ""))}):
        if idx < len(names):
            used.append(names[idx])
    return used


def remap_expression_by_feature_names(expr: str, source_features: Sequence[str], target_features: Sequence[str]) -> tuple[str | None, bool, str]:
    """Map source x-index expression into target x-index expression by feature name."""
    source = [str(x) for x in source_features]
    target_idx = {str(name): i for i, name in enumerate(target_features)}
    missing: list[str] = []

    def repl(match):
        old_i = int(match.group(1))
        if old_i >= len(source):
            missing.append(f"x{old_i}:source_index_out_of_range")
            return match.group(0)
        name = source[old_i]
        if name not in target_idx:
            missing.append(name)
            return match.group(0)
        return f"x{target_idx[name]}"

    mapped = re.sub(r"\bx(\d+)\b", repl, str(expr))
    if missing:
        return None, False, "missing_target_features_for_transfer:" + ",".join(sorted(set(missing)))
    return mapped, True, ""


def _cox_cv_folds(n: int, event: np.ndarray, n_splits: int = 4, seed: int = 20260523) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    pos = idx[event.astype(int) == 1]
    neg = idx[event.astype(int) != 1]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for k, arr in enumerate([pos, neg]):
        for i, value in enumerate(arr):
            folds[i % n_splits].append(int(value))
    out = []
    for val in folds:
        val_idx = np.array(sorted(val), dtype=int)
        train_idx = np.setdiff1d(idx, val_idx)
        if len(val_idx) and len(train_idx):
            out.append((train_idx, val_idx))
    return out


def _cox_cindex(time, event, risk) -> float:
    from lifelines.utils import concordance_index

    return float(concordance_index(np.asarray(time, dtype=float), -np.asarray(risk, dtype=float), np.asarray(event, dtype=int)))


def evaluate_tree_group_cox_cv(
    tree_group,
    endpoint: str,
    X,
    T,
    E,
    main_effect_matrix=None,
    folds=None,
    baseline_cindex=None,
    cox_penalizer: float = 0.05,
    complexity_weight: float = 0.01,
    instability_weight: float = 0.05,
    loglik_weight: float = 0.0,
) -> dict[str, Any]:
    """
    Evaluate a GP tree group with internal Cox CV.

    Fitness is minimized and is based on internal CV only. MIMIC external data is
    not accepted by this evaluator.
    """
    try:
        from lifelines import CoxPHFitter
    except Exception as exc:
        raise RuntimeError("Cox-aware Phase A requires lifelines.") from exc

    expressions = list(tree_group.expressions if isinstance(tree_group, TreeGroup) else tree_group)
    x = _clean_feature_matrix(X)
    t = np.asarray(T, dtype=float)
    e = np.asarray(E, dtype=int)
    m = _clean_feature_matrix(main_effect_matrix) if main_effect_matrix is not None else np.empty((x.shape[0], 0))
    if folds is None:
        folds = _cox_cv_folds(x.shape[0], e)

    fold_cindex: list[float] = []
    fold_loglik: list[float] = []
    coef_signs: list[np.ndarray] = []
    coef_values: list[np.ndarray] = []
    fold_success = 0
    failed = 0
    for train_idx, val_idx in folds:
        try:
            gp_cols = []
            for expr in expressions:
                values = _eval_expr(expr, x)
                train_vals = values[train_idx]
                sd = float(np.nanstd(train_vals))
                if not math.isfinite(sd) or sd < 1e-8:
                    raise ValueError("low_variance_gp_feature")
                mu = float(np.nanmean(train_vals))
                gp_cols.append((values - mu) / sd)
            gp_mat = np.column_stack(gp_cols) if gp_cols else np.empty((x.shape[0], 0))
            design = np.column_stack([m, gp_mat])
            cols = [f"main_{i}" for i in range(m.shape[1])] + [f"gp_{i}" for i in range(gp_mat.shape[1])]
            train_df = pd.DataFrame(design[train_idx], columns=cols)
            train_df["T"] = t[train_idx]
            train_df["E"] = e[train_idx]
            val_df = pd.DataFrame(design[val_idx], columns=cols)
            cph = CoxPHFitter(penalizer=cox_penalizer)
            cph.fit(train_df, duration_col="T", event_col="E", show_progress=False)
            risk = cph.predict_partial_hazard(val_df).to_numpy(float).reshape(-1)
            fold_cindex.append(_cox_cindex(t[val_idx], e[val_idx], risk))
            fold_loglik.append(float(getattr(cph, "log_likelihood_", np.nan)))
            gp_coefs = cph.params_.reindex([f"gp_{i}" for i in range(len(expressions))]).fillna(0.0).to_numpy(float)
            coef_signs.append(np.sign(gp_coefs))
            coef_values.append(gp_coefs)
            fold_success += 1
        except Exception:
            failed += 1

    if not fold_cindex:
        return {
            "fitness": 1e6,
            "mean_cindex": math.nan,
            "mean_delta_cindex": -1.0,
            "min_fold_cindex": math.nan,
            "mean_loglik": math.nan,
            "mean_delta_loglik": 0.0,
            "loglik_delta_active": False,
            "complexity_penalty": complexity_weight * sum(_tree_complexity(e) for e in expressions),
            "fold_instability_penalty": 1.0,
            "cox_converged": False,
            "fold_success_n": 0,
            "failed_cox_n": failed,
            "mean_coefficients": {},
            "mean_coefficient_sign": {},
            "coefficient_sign_stability": 0.0,
            "baseline_cindex_used": baseline_cindex,
            "baseline_missing": baseline_cindex is None or not math.isfinite(float(baseline_cindex)),
        }

    mean_c = float(np.mean(fold_cindex))
    min_c = float(np.min(fold_cindex))
    base_missing = baseline_cindex is None or not math.isfinite(float(baseline_cindex))
    delta_c = mean_c if base_missing else mean_c - float(baseline_cindex)
    mean_loglik = float(np.nanmean(fold_loglik)) if fold_loglik else math.nan
    delta_loglik = 0.0
    if coef_signs:
        signs = np.vstack(coef_signs)
        stability = float(np.nanmean(np.abs(np.nanmean(signs, axis=0)))) if signs.size else 0.0
    else:
        stability = 0.0
    instability = 1.0 - stability
    complexity = sum(_tree_complexity(expr) for expr in expressions)
    duplicate_penalty = len(expressions) - len({_normalize_expr(e) for e in expressions})
    convergence_failure_penalty = 0.25 * failed
    complexity_penalty = complexity_weight * (complexity + duplicate_penalty)
    instability_penalty = instability_weight * instability
    loglik_delta_active = False
    loglik_term = 0.0
    fitness = (
        -delta_c
        + loglik_term
        + complexity_penalty
        + instability_penalty
        + convergence_failure_penalty
    )
    mean_coef = np.nanmean(np.vstack(coef_values), axis=0).tolist() if coef_values else []
    mean_sign = np.nanmean(np.vstack(coef_signs), axis=0).tolist() if coef_signs else []
    return {
        "fitness": float(fitness),
        "mean_cindex": mean_c,
        "mean_delta_cindex": float(delta_c),
        "min_fold_cindex": min_c,
        "mean_loglik": mean_loglik,
        "mean_delta_loglik": float(delta_loglik),
        "loglik_delta_active": loglik_delta_active,
        "loglik_delta_note": "loglik_weight fixed at 0; baseline fold loglik delta is not active",
        "complexity_penalty": float(complexity_penalty),
        "fold_instability_penalty": float(instability_penalty),
        "cox_converged": failed == 0,
        "fold_success_n": int(fold_success),
        "failed_cox_n": int(failed),
        "mean_coefficients": {expr: mean_coef[i] if i < len(mean_coef) else math.nan for i, expr in enumerate(expressions)},
        "mean_coefficient_sign": {expr: mean_sign[i] if i < len(mean_sign) else math.nan for i, expr in enumerate(expressions)},
        "coefficient_sign_stability": stability,
        "baseline_cindex_used": baseline_cindex,
        "baseline_missing": base_missing,
    }


class RelationAwareMultiTargetGP:
    """Relation-aware Phase A sidecar using Cox internal-CV fitness."""

    def __init__(
        self,
        endpoint_names: Sequence[str],
        transfer_matrix: np.ndarray | None = None,
        fitness_mode: str = "cox_cv",
        n_generations: int = 8,
        pop_size: int = 24,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        transfer_prob: float = 0.3,
        tree_group_size: int = 3,
        max_tree_height: int = 5,
        random_state: int | None = None,
        warm_start_trees: dict[str, list[str]] | None = None,
        verbose: bool = True,
        cox_penalizer: float = 0.05,
        complexity_weight: float = 0.01,
        instability_weight: float = 0.05,
        loglik_weight: float = 0.0,
    ) -> None:
        if fitness_mode != "cox_cv":
            raise ValueError("RelationAwareMultiTargetGP default path only supports fitness_mode='cox_cv'. RMSE is deprecated.")
        self.endpoint_names = list(endpoint_names)
        self.transfer_matrix = np.asarray(transfer_matrix if transfer_matrix is not None else build_phaseA_transfer_matrix(self.endpoint_names), dtype=float)
        self.fitness_mode = fitness_mode
        self.n_generations = int(n_generations)
        self.pop_size = int(pop_size)
        self.crossover_prob = float(crossover_prob)
        self.mutation_prob = float(mutation_prob)
        self.transfer_prob = float(transfer_prob)
        self.tree_group_size = int(tree_group_size)
        self.max_tree_height = int(max_tree_height)
        self.random_state = random_state
        self.warm_start_trees = warm_start_trees or {}
        self.verbose = bool(verbose)
        self.cox_penalizer = float(cox_penalizer)
        self.complexity_weight = float(complexity_weight)
        self.instability_weight = float(instability_weight)
        self.loglik_weight = 0.0
        self.transfer_log: list[dict[str, Any]] = []
        self.generation_log_: list[dict[str, Any]] = []
        self.top_trees_: dict[str, list[dict[str, Any]]] = {}
        self.top_tree_groups_: dict[str, list[dict[str, Any]]] = {}
        self._eval_cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    def sample_transfer_task(self, task_idx: int) -> int:
        return sample_transfer_task(task_idx, self.transfer_matrix, self.rng)

    def _terminal(self, n_features: int) -> str:
        return f"x{int(self.rng.integers(0, max(1, n_features)))}"

    def _random_expr(self, n_features: int, depth: int | None = None) -> str:
        depth = self.max_tree_height if depth is None else depth
        if depth <= 1 or self.rng.random() < 0.45:
            term = self._terminal(n_features)
            if self.rng.random() < 0.35:
                return f"{self.rng.choice(['safe_log', 'safe_sqrt', 'clip'])}({term})"
            return term
        op = str(self.rng.choice(["add", "sub", "mul", "safe_div"]))
        return f"{op}({self._random_expr(n_features, depth - 1)}, {self._random_expr(n_features, depth - 1)})"

    def _new_group(self, endpoint: str, n_features: int) -> TreeGroup:
        warm = list(self.warm_start_trees.get(endpoint, []))
        exprs = []
        while len(exprs) < self.tree_group_size:
            if warm and self.rng.random() < 0.35:
                exprs.append(str(self.rng.choice(warm)))
            else:
                exprs.append(self._random_expr(n_features, min(3, self.max_tree_height)))
        return TreeGroup(exprs, endpoint, endpoint, 0.0, 0, False)

    def _mutate(self, group: TreeGroup, n_features: int, generation: int) -> TreeGroup:
        exprs = list(group.expressions)
        idx = int(self.rng.integers(0, len(exprs)))
        exprs[idx] = self._random_expr(n_features, min(3, self.max_tree_height))
        return TreeGroup(exprs, group.source_endpoint, group.target_endpoint, group.transfer_weight, generation, group.whether_transferred, group.transfer_parent_expression, group.target_parent_expression)

    def _crossover(
        self,
        target_group: TreeGroup,
        source_group: TreeGroup,
        target: str,
        source: str,
        weight: float,
        generation: int,
        source_features: Sequence[str],
        target_features: Sequence[str],
        n_target_features: int,
    ) -> tuple[TreeGroup, dict[str, Any]]:
        exprs = list(target_group.expressions)
        src_expr = str(self.rng.choice(source_group.expressions))
        idx = int(self.rng.integers(0, len(exprs)))
        old = exprs[idx]
        mapped_expr, ok, reason = remap_expression_by_feature_names(src_expr, source_features, target_features)
        if ok and mapped_expr is not None:
            exprs[idx] = mapped_expr
            return TreeGroup(exprs, source, target, weight, generation, True, mapped_expr, old), {
                "remap_success": True,
                "remap_failed_reason": "",
                "source_expression": src_expr,
                "mapped_expression": mapped_expr,
                "target_parent_expression": old,
            }
        return TreeGroup(exprs, target, target, 0.0, generation, False, "", old), {
            "remap_success": False,
            "remap_failed_reason": reason,
            "source_expression": src_expr,
            "mapped_expression": "",
            "target_parent_expression": old,
        }

    def _feature_names(self, endpoint: str, feature_names_by_endpoint: dict[str, list[str]] | None) -> list[str]:
        if feature_names_by_endpoint and endpoint in feature_names_by_endpoint:
            return [str(x) for x in feature_names_by_endpoint[endpoint]]
        return []

    def _eval(self, group: TreeGroup, endpoint: str, X, T, E, main, folds, baseline) -> dict[str, Any]:
        sig = (endpoint, tuple(_normalize_expr(e) for e in group.expressions))
        if sig not in self._eval_cache:
            self._eval_cache[sig] = evaluate_tree_group_cox_cv(
                group,
                endpoint,
                X,
                T,
                E,
                main,
                folds,
                baseline,
                cox_penalizer=self.cox_penalizer,
                complexity_weight=self.complexity_weight,
                instability_weight=self.instability_weight,
                loglik_weight=self.loglik_weight,
            )
        return dict(self._eval_cache[sig])

    def _group_log_row(self, group_id: str, group: TreeGroup, metric: dict[str, Any], feature_names: list[str], generation: int) -> dict[str, Any]:
        return {
            "tree_group_id": group_id,
            "tree_group_expressions": json_safe_list(group.expressions),
            "feature_names": feature_names,
            "source_endpoint": group.source_endpoint,
            "target_endpoint": group.target_endpoint,
            "transfer_weight": group.transfer_weight,
            "generation": generation,
            "fitness": metric["fitness"],
            "fitness_mode": self.fitness_mode,
            "whether_transferred": bool(group.whether_transferred),
            "generated_by_transfer": bool(group.whether_transferred),
            "transfer_parent_expression": group.transfer_parent_expression,
            "target_parent_expression": group.target_parent_expression,
            "cox_cv_mean_cindex": metric["mean_cindex"],
            "cox_cv_mean_delta_cindex": metric["mean_delta_cindex"],
            "cox_cv_min_fold_cindex": metric["min_fold_cindex"],
            "cox_cv_mean_delta_loglik": metric["mean_delta_loglik"],
            "loglik_delta_active": metric.get("loglik_delta_active", False),
            "cox_converged": metric["cox_converged"],
            "fold_success_n": metric["fold_success_n"],
            "mean_coefficients": metric.get("mean_coefficients", {}),
            "mean_coefficient_sign": metric.get("mean_coefficient_sign", {}),
            "coefficient_sign_stability": metric["coefficient_sign_stability"],
            "complexity_penalty": metric["complexity_penalty"],
            "instability_penalty": metric["fold_instability_penalty"],
            "baseline_cindex_used": metric["baseline_cindex_used"],
            "baseline_cindex_source": "ordinary_cox_internal_or_single_endpoint_phaseA",
            "injected_to_notebook": False,
            "parse_success": None,
            "final_selected_in_cox": False,
        }

    def fit(
        self,
        X_by_endpoint: dict[str, pd.DataFrame | np.ndarray],
        time_by_endpoint: dict[str, np.ndarray],
        event_by_endpoint: dict[str, np.ndarray],
        main_effects_by_endpoint: dict[str, pd.DataFrame | np.ndarray] | None = None,
        folds_by_endpoint: dict[str, list[tuple[np.ndarray, np.ndarray]]] | None = None,
        baseline_cindex_by_endpoint: dict[str, float] | None = None,
        warm_start_trees_by_endpoint: dict[str, list[str]] | None = None,
        feature_names_by_endpoint: dict[str, list[str]] | None = None,
    ) -> "RelationAwareMultiTargetGP":
        self.rng = np.random.default_rng(self.random_state)
        random.seed(self.random_state)
        if warm_start_trees_by_endpoint:
            self.warm_start_trees.update(warm_start_trees_by_endpoint)
        main_effects_by_endpoint = main_effects_by_endpoint or {}
        folds_by_endpoint = folds_by_endpoint or {}
        baseline_cindex_by_endpoint = baseline_cindex_by_endpoint or {}

        populations: dict[str, list[TreeGroup]] = {}
        metrics: dict[str, list[dict[str, Any]]] = {}
        for ep in self.endpoint_names:
            n_features = _clean_feature_matrix(X_by_endpoint[ep]).shape[1]
            populations[ep] = [self._new_group(ep, n_features) for _ in range(self.pop_size)]
            metrics[ep] = [
                self._eval(g, ep, X_by_endpoint[ep], time_by_endpoint[ep], event_by_endpoint[ep], main_effects_by_endpoint.get(ep), folds_by_endpoint.get(ep), baseline_cindex_by_endpoint.get(ep))
                for g in populations[ep]
            ]

        self.transfer_log = []
        self.generation_log_ = []
        for gen in range(1, self.n_generations + 1):
            for ti, target in enumerate(self.endpoint_names):
                n_features = _clean_feature_matrix(X_by_endpoint[target]).shape[1]
                target_feature_names = self._feature_names(target, feature_names_by_endpoint)
                order = np.argsort([m["fitness"] for m in metrics[target]]).tolist()
                elites = [populations[target][i] for i in order[: max(1, self.pop_size // 4)]]
                next_groups = list(elites)
                transfers = 0
                failed = 0
                while len(next_groups) < self.pop_size:
                    remap_info = None
                    if self.rng.random() < self.transfer_prob:
                        si = self.sample_transfer_task(ti)
                        source = self.endpoint_names[si]
                        s_order = np.argsort([m["fitness"] for m in metrics[source]]).tolist()
                        src = populations[source][int(self.rng.choice(s_order[: max(1, len(s_order) // 3)]))]
                        tgt = populations[target][int(self.rng.choice(order[: max(1, len(order) // 2)]))]
                        source_feature_names = self._feature_names(source, feature_names_by_endpoint)
                        child, remap_info = self._crossover(
                            tgt,
                            src,
                            target,
                            source,
                            float(self.transfer_matrix[ti, si]),
                            gen,
                            source_feature_names,
                            target_feature_names,
                            n_features,
                        )
                        transfers += 1
                    else:
                        parent = populations[target][int(self.rng.choice(order[: max(1, len(order) // 2)]))]
                        if self.rng.random() < self.mutation_prob:
                            child = self._mutate(parent, n_features, gen)
                        else:
                            child = TreeGroup(list(parent.expressions), parent.source_endpoint, target, parent.transfer_weight, gen, parent.whether_transferred, parent.transfer_parent_expression, parent.target_parent_expression)
                    metric = self._eval(child, target, X_by_endpoint[target], time_by_endpoint[target], event_by_endpoint[target], main_effects_by_endpoint.get(target), folds_by_endpoint.get(target), baseline_cindex_by_endpoint.get(target))
                    if not metric.get("cox_converged", False):
                        failed += 1
                    next_groups.append(child)
                    if remap_info is not None:
                        gid = f"{target}_g{gen:03d}_{len(self.transfer_log):05d}"
                        event_row = self._group_log_row(gid, child, metric, target_feature_names, gen)
                        event_row.update(remap_info)
                        event_row["expression"] = remap_info.get("mapped_expression") or remap_info.get("source_expression")
                        event_row["source_endpoint"] = source
                        event_row["target_endpoint"] = target
                        event_row["transfer_weight"] = float(self.transfer_matrix[ti, si])
                        event_row["generated_by_transfer"] = bool(remap_info.get("remap_success"))
                        event_row["whether_transferred"] = bool(remap_info.get("remap_success"))
                        self.transfer_log.append(event_row)
                scored = [(g, self._eval(g, target, X_by_endpoint[target], time_by_endpoint[target], event_by_endpoint[target], main_effects_by_endpoint.get(target), folds_by_endpoint.get(target), baseline_cindex_by_endpoint.get(target))) for g in next_groups]
                scored.sort(key=lambda x: x[1]["fitness"])
                populations[target] = [g for g, _ in scored[: self.pop_size]]
                metrics[target] = [m for _, m in scored[: self.pop_size]]
                best = metrics[target][0]
                mean_delta = float(np.nanmean([m["mean_delta_cindex"] for m in metrics[target]]))
                row = {
                    "generation": gen,
                    "endpoint": target,
                    "best_mean_delta_cindex": best["mean_delta_cindex"],
                    "best_mean_cindex": best["mean_cindex"],
                    "mean_population_delta_cindex": mean_delta,
                    "min_fold_cindex_of_best": best["min_fold_cindex"],
                    "n_transfers": transfers,
                    "n_converged": int(sum(bool(m.get("cox_converged")) for m in metrics[target])),
                    "n_failed_cox": failed,
                    "best_tree_group_expressions": json_safe_list(populations[target][0].expressions),
                    "fitness_mode": self.fitness_mode,
                }
                self.generation_log_.append(row)
                if self.verbose:
                    print(
                        f"[RelationAware PhaseA][Gen {gen:03d}/{self.n_generations:03d}][{target}] "
                        f"best_delta_cindex={row['best_mean_delta_cindex']:.4f} "
                        f"mean_delta_cindex={row['mean_population_delta_cindex']:.4f} "
                        f"min_fold_cindex={row['min_fold_cindex_of_best']:.4f} "
                        f"transfers={transfers} converged={row['n_converged']} failed_cox={failed}",
                        flush=True,
                    )

        self.top_trees_ = {}
        self.top_tree_groups_ = {}
        for ep in self.endpoint_names:
            top = []
            top_groups = []
            for rank, group in enumerate(populations[ep][: min(10, self.pop_size)], start=1):
                metric = self._eval(group, ep, X_by_endpoint[ep], time_by_endpoint[ep], event_by_endpoint[ep], main_effects_by_endpoint.get(ep), folds_by_endpoint.get(ep), baseline_cindex_by_endpoint.get(ep))
                gid = f"{ep}_top_{rank:02d}"
                feature_names = self._feature_names(ep, feature_names_by_endpoint)
                group_row = self._group_log_row(gid, group, metric, feature_names, group.generation or self.n_generations)
                group_row.update({
                    "top_rank": rank,
                    "remap_success": True,
                    "remap_failed_reason": "",
                    "expression_n": len(group.expressions),
                })
                top_groups.append(group_row)
                for expr in group.expressions:
                    tree_id = f"{gid}_tree_{len(top) + 1:03d}"
                    used_features = expression_used_features(expr, feature_names)
                    coef = metric.get("mean_coefficients", {}).get(expr) if isinstance(metric.get("mean_coefficients"), dict) else None
                    sign = metric.get("mean_coefficient_sign", {}).get(expr) if isinstance(metric.get("mean_coefficient_sign"), dict) else None
                    top.append({
                        "tree_group_id": gid,
                        "tree_id": tree_id,
                        "candidate_id": tree_id,
                        "tree_group_expressions": json_safe_list(group.expressions),
                        "feature_names": feature_names,
                        "source_feature_names": feature_names,
                        "used_features": used_features,
                        "source_endpoint": group.source_endpoint,
                        "target_endpoint": ep,
                        "transfer_weight": group.transfer_weight,
                        "generation": group.generation or self.n_generations,
                        "reestimated_in_target": True,
                        "expression": expr,
                        "tree_height": _tree_height(expr),
                        "tree_complexity": _tree_complexity(expr),
                        "fitness": metric["fitness"],
                        "fitness_mode": self.fitness_mode,
                        "whether_transferred": bool(group.whether_transferred),
                        "generated_by_transfer": bool(group.whether_transferred),
                        "transfer_parent_expression": group.transfer_parent_expression,
                        "target_parent_expression": group.target_parent_expression,
                        "survived_to_top_trees": True,
                        "cox_cv_mean_cindex": metric["mean_cindex"],
                        "cox_cv_mean_delta_cindex": metric["mean_delta_cindex"],
                        "cox_cv_min_fold_cindex": metric["min_fold_cindex"],
                        "cox_cv_mean_delta_loglik": metric["mean_delta_loglik"],
                        "loglik_delta_active": metric.get("loglik_delta_active", False),
                        "cox_converged": metric["cox_converged"],
                        "fold_success_n": metric["fold_success_n"],
                        "mean_coefficient": coef,
                        "mean_coefficient_sign": sign,
                        "coefficient_sign_stability": metric["coefficient_sign_stability"],
                        "complexity_penalty": metric["complexity_penalty"],
                        "instability_penalty": metric["fold_instability_penalty"],
                        "baseline_cindex_used": metric["baseline_cindex_used"],
                        "baseline_cindex_source": "ordinary_cox_internal_or_single_endpoint_phaseA",
                        "injected_to_notebook": False,
                        "parse_success": None,
                        "final_selected_in_cox": False,
                    })
            self.top_tree_groups_[ep] = top_groups
            self.top_trees_[ep] = top
        return self


def json_safe_list(values: Sequence[Any]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"
