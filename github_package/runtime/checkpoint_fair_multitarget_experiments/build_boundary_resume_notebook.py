from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NO_EXCHANGE_BUILDER = ROOT / "checkpoint_no_exchange_G_continuation" / "build_resume_notebook.py"
SOURCE_NOTEBOOK = Path(
    os.environ.get(
        "MTGP_BOUNDARY_SOURCE_NOTEBOOK",
        str(
            ROOT
            / "single_endpoint_phaseA_G_checkpoint_hcoxfix_trial010"
            / "newcox_route1_phaseB_phaseC_profile_style_aligned_checkpoint.ipynb"
        ),
    )
)
OUTPUT_NOTEBOOK = Path(
    os.environ.get(
        "MTGP_BOUNDARY_OUTPUT_NOTEBOOK",
        str(
            HERE
            / "boundary_notebook"
            / "newcox_route1_phaseB_phaseC_profile_style_aligned.ipynb"
        ),
    )
)


def load_no_exchange_builder():
    spec = importlib.util.spec_from_file_location("fair_no_exchange_resume_builder", NO_EXCHANGE_BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one {label}, got {text.count(old)}")
    return text.replace(old, new, 1)


EXCHANGE_BLOCK = r'''
        # relation_aware_phaseA_tree_injection: one checkpoint-boundary migration.
        # Donors are routed by checkpoint call, tested against several recipients
        # and every target tree slot, and admitted by paired target-fitness gain.
        for individual in pop + saved_hof:
            if individual.fitness.valid:
                del individual.fitness.values
            individual.fitness.values = toolbox.evaluate(individual)
        hof = tools.HallOfFame(ELITE_K)
        hof.update(saved_hof)
        hof.update(pop)

        transfer_attempted_n = 0
        transfer_valid_n = 0
        transfer_accepted_n = 0
        placement_evaluated_n = 0
        transfer_items = []
        transfer_rejected_items = []
        boundary_lineage_by_generation = []
        boundary_accepted_expression_by_id = {}
        boundary_route_mode = "none"
        _boundary_original_select = toolbox.select
        _boundary_original_mate = toolbox.mate
        _boundary_original_mutate = toolbox.mutate
        _boundary_wrapped = False

        def _boundary_lineage_ids(individual):
            return tuple(sorted(set(str(value) for value in getattr(individual, "boundary_lineage_ids", ()) if str(value))))

        def _boundary_lineage_snapshot(generation, population, hall_of_fame):
            population_with_lineage = [ind for ind in population if _boundary_lineage_ids(ind)]
            hof_with_lineage = [ind for ind in hall_of_fame if _boundary_lineage_ids(ind)]
            boundary_lineage_by_generation.append({
                "generation": int(generation),
                "population_lineage_individual_n": int(len(population_with_lineage)),
                "population_lineage_id_n": int(len({value for ind in population_with_lineage for value in _boundary_lineage_ids(ind)})),
                "hof_lineage_individual_n": int(len(hof_with_lineage)),
                "hof_lineage_id_n": int(len({value for ind in hof_with_lineage for value in _boundary_lineage_ids(ind)})),
            })

        phasea_transfer_path = str(globals().get("PHASEA_EXTERNAL_TREES_JSON", "") or "")
        if bool(globals().get("RELATION_AWARE_PHASEA_ENABLED", False)) and phasea_transfer_path:
            all_transfer_records = [
                record
                for record in json.loads(Path(phasea_transfer_path).read_text(encoding="utf-8"))
                if str(record.get("source_endpoint", ""))
                and str(record.get("source_endpoint", "")) != str(record.get("target_endpoint", ""))
                and str(record.get("mapped_expression") or record.get("expression") or "").strip()
            ]
            same_call_records = [
                record for record in all_transfer_records
                if int(record.get("checkpoint_call_index", -1) or -1) == int(phasea_checkpoint_index)
            ]
            if same_call_records:
                transfer_records = same_call_records
                boundary_route_mode = "same_call"
            elif int(phasea_checkpoint_index) <= 4 and bool(globals().get("BOUNDARY_ALLOW_CROSS_FOLD_FALLBACK", True)):
                transfer_records = [
                    record for record in all_transfer_records
                    if 1 <= int(record.get("checkpoint_call_index", -1) or -1) <= 4
                ]
                boundary_route_mode = "cross_fold_fallback" if transfer_records else "no_eligible_fold_donor"
            else:
                transfer_records = []
                boundary_route_mode = "no_same_call_donor"

            transfer_records = sorted(
                transfer_records,
                key=lambda record: (
                    bool(record.get("selected_in_final_model", False)),
                    bool(record.get("selected_by_fold", False)),
                    float(record.get("source_fitness_percentile", 0.0) or 0.0),
                    float(record.get("transfer_weight", 0.0) or 0.0),
                    str(record.get("source", "")) == "checkpoint_hof",
                ),
                reverse=True,
            )[:max(1, int(globals().get("BOUNDARY_MAX_DONORS_PER_CALL", 6)))]

            original_pop = list(pop)
            ranked_original = sorted(original_pop, key=lambda ind: ind.fitness.values[0], reverse=True)
            recipient_pool_n = max(1, min(int(globals().get("BOUNDARY_RECIPIENT_POOL_SIZE", 3)), len(ranked_original)))
            if recipient_pool_n == 1:
                recipient_positions = [0]
            else:
                recipient_positions = sorted(set(
                    int(round(index * (len(ranked_original) - 1) / float(recipient_pool_n - 1)))
                    for index in range(recipient_pool_n)
                ))
            recipient_entries = [(position, ranked_original[position]) for position in recipient_positions]
            best_transfer_candidates = []
            seen_candidate_structures = set()

            for record in transfer_records:
                transfer_attempted_n += 1
                mapped_expression = str(record.get("mapped_expression") or record.get("expression") or "")
                candidate_id = str(record.get("candidate_id", record.get("tree_id", f"boundary_candidate_{transfer_attempted_n}")))
                base_item = {
                    "phaseA_call_index": int(phasea_checkpoint_index),
                    "source_checkpoint_call_index": int(record.get("checkpoint_call_index", -1) or -1),
                    "routing_mode": boundary_route_mode,
                    "source_endpoint": record.get("source_endpoint"),
                    "target_endpoint": record.get("target_endpoint"),
                    "expression": record.get("expression_original", record.get("expression")),
                    "mapped_expression": mapped_expression,
                    "transfer_weight": record.get("transfer_weight"),
                    "source_fitness_percentile": record.get("source_fitness_percentile"),
                    "source_individual_fitness": record.get("source_individual_fitness"),
                    "source_tree_slot": record.get("source_tree_slot"),
                    "candidate_id": candidate_id,
                    "generated_by_transfer": True,
                }
                try:
                    transfer_tree = tree_from_string(mapped_expression)
                    if int(transfer_tree.height) > int(MAX_TREE_HEIGHT_X):
                        raise ValueError("tree_height_exceeds_MAX_TREE_HEIGHT_X")
                    placement_rows = []
                    for recipient_rank, original in recipient_entries:
                        base_fitness = float(original.fitness.values[0])
                        for tree_slot in range(len(original)):
                            if str(original[tree_slot]) == mapped_expression:
                                continue
                            candidate = toolbox.clone(original)
                            candidate[tree_slot] = gp.PrimitiveTree(transfer_tree)
                            if candidate.fitness.valid:
                                del candidate.fitness.values
                            candidate.fitness.values = toolbox.evaluate(candidate)
                            placement_evaluated_n += 1
                            target_fitness = float(candidate.fitness.values[0])
                            if not np.isfinite(target_fitness):
                                continue
                            structure_key = tuple(str(tree) for tree in candidate)
                            placement_rows.append((
                                target_fitness - base_fitness,
                                target_fitness,
                                -int(recipient_rank),
                                candidate,
                                int(recipient_rank),
                                int(tree_slot),
                                base_fitness,
                                structure_key,
                            ))
                    if not placement_rows:
                        raise ValueError("no_height_valid_expression_changing_placement")
                    best = max(placement_rows, key=lambda row: (row[0], row[1], row[2]))
                    placement_gain, target_fitness, _, candidate, recipient_rank, tree_slot, base_fitness, structure_key = best
                    candidate.boundary_lineage_ids = (candidate_id,)
                    candidate.boundary_birth_generation = 0
                    item = dict(base_item)
                    item.update({
                        "valid": True,
                        "recipient_fitness_rank_zero_based": int(recipient_rank),
                        "tree_slot": int(tree_slot),
                        "base_fitness": float(base_fitness),
                        "target_fitness": float(target_fitness),
                        "placement_gain": float(placement_gain),
                        "placement_tested_n": int(len(placement_rows)),
                    })
                    transfer_valid_n += 1
                    if structure_key not in seen_candidate_structures:
                        seen_candidate_structures.add(structure_key)
                        best_transfer_candidates.append((candidate, item))
                except Exception as exc:
                    item = dict(base_item)
                    item.update({"valid": False, "reason": str(exc)})
                    transfer_rejected_items.append(item)

            minimum_gain = float(globals().get("BOUNDARY_MIN_PLACEMENT_GAIN", -0.005))
            max_migrants = max(1, min(int(globals().get("BOUNDARY_MAX_MIGRANTS", 4)), len(original_pop)))
            eligible = [
                (candidate, item) for candidate, item in best_transfer_candidates
                if float(item.get("placement_gain", -np.inf)) >= minimum_gain
            ]
            eligible.sort(
                key=lambda pair: (
                    float(pair[1].get("placement_gain", -np.inf)),
                    float(pair[1].get("target_fitness", -np.inf)),
                    float(pair[1].get("source_fitness_percentile", 0.0) or 0.0),
                ),
                reverse=True,
            )
            admitted = eligible[:max_migrants]
            elite_n = min(int(ELITE_K), max(0, len(original_pop) - len(admitted)))
            protected = ranked_original[:elite_n]
            remaining_slots = max(0, len(original_pop) - len(protected) - len(admitted))
            pop = protected + [candidate for candidate, _ in admitted] + ranked_original[elite_n:elite_n + remaining_slots]
            admitted_ids = {str(item["candidate_id"]) for _, item in admitted}
            for candidate, item in best_transfer_candidates:
                item["minimum_placement_gain"] = minimum_gain
                item["accepted_into_resumed_population"] = str(item["candidate_id"]) in admitted_ids
                if item["accepted_into_resumed_population"]:
                    transfer_accepted_n += 1
                    boundary_accepted_expression_by_id[str(item["candidate_id"])] = str(item["mapped_expression"])
                transfer_items.append(item)
            hof.update(pop)

            if admitted:
                grace_generations = max(0, int(globals().get("BOUNDARY_MIGRATION_GRACE_GENERATIONS", 2)))
                parent_slots = max(1, min(int(globals().get("BOUNDARY_MIGRANT_PARENT_SLOTS", len(admitted))), len(pop)))
                selection_call_n = 0

                def boundary_select(population, k):
                    nonlocal selection_call_n
                    selected = list(_boundary_original_select(population, k))
                    if selection_call_n < grace_generations:
                        lineage_population = sorted(
                            [individual for individual in population if _boundary_lineage_ids(individual)],
                            key=lambda individual: individual.fitness.values[0],
                            reverse=True,
                        )[:min(parent_slots, k)]
                        for offset, migrant in enumerate(lineage_population, 1):
                            selected[-offset] = migrant
                    selection_call_n += 1
                    return selected

                def boundary_mate(individual1, individual2):
                    lineage = tuple(sorted(set(_boundary_lineage_ids(individual1)) | set(_boundary_lineage_ids(individual2))))
                    result = _boundary_original_mate(individual1, individual2)
                    if lineage:
                        for child in result:
                            child.boundary_lineage_ids = lineage
                            child.boundary_recombined = True
                    return result

                def boundary_mutate(individual):
                    lineage = _boundary_lineage_ids(individual)
                    before = tuple(str(tree) for tree in individual)
                    result = _boundary_original_mutate(individual)
                    if lineage:
                        for child in result:
                            child.boundary_lineage_ids = lineage
                            if tuple(str(tree) for tree in child) != before:
                                child.boundary_recombined = True
                    return result

                toolbox.select = boundary_select
                toolbox.mate = boundary_mate
                toolbox.mutate = boundary_mutate
                _boundary_wrapped = True

        _boundary_lineage_snapshot(0, pop, hof)
        globals()["RELATION_AWARE_PHASEA_ATTEMPTED_N"] = int(transfer_attempted_n)
        globals()["RELATION_AWARE_PHASEA_INJECTED_N"] = int(transfer_valid_n)
        globals()["RELATION_AWARE_PHASEA_PARSE_FAILED_N"] = int(len(transfer_rejected_items))
        globals()["RELATION_AWARE_PHASEA_INJECTED_ITEMS"] = list(transfer_items)
        globals()["RELATION_AWARE_PHASEA_PARSE_FAILED_ITEMS"] = list(transfer_rejected_items)
        globals()["RELATION_AWARE_PHASEA_CURRENT_GP_INPUT_VARS"] = list(GP_INPUT_VARS)
        globals().setdefault("RELATION_AWARE_PHASEA_BY_FOLD", []).append({
            "fold": int(phasea_checkpoint_index),
            "endpoint": endpoint_id,
            "source_n": int(len(all_transfer_records)) if 'all_transfer_records' in locals() else 0,
            "attempted_n": int(transfer_attempted_n),
            "injected_n": int(transfer_valid_n),
            "placement_evaluated_n": int(placement_evaluated_n),
            "accepted_into_resumed_population_n": int(transfer_accepted_n),
            "routing_mode": boundary_route_mode,
            "checkpoint_generation": int(resume_payload["generation"]),
            "additional_generation": int(n_gen),
            "checkpoint_population_fitness_recomputed": True,
        })
        print(
            f"  [{label}] CHECKPOINT-BOUNDARY route={boundary_route_mode} "
            f"donors={transfer_attempted_n} placements={placement_evaluated_n} "
            f"valid={transfer_valid_n} admitted={transfer_accepted_n}"
        )
'''


def build_resume_function(base) -> str:
    function = str(base.RESUME_FUNCTION)
    needle = '        np.random.set_state(resume_payload["numpy_random_state"])\n'
    function = replace_once(function, needle, needle + EXCHANGE_BLOCK, "RNG restore insertion point")
    function = replace_once(
        function,
        "        pop[:] = offspring\n        hof.update(pop)\n",
        "        pop[:] = offspring\n        hof.update(pop)\n"
        "        if is_phasea:\n"
        "            _boundary_lineage_snapshot(gen, pop, hof)\n",
        "boundary lineage generation hook",
    )
    function = replace_once(
        function,
        '            "cross_endpoint_exchange_enabled": False,\n'
        '            "injected_n": 0,\n'
        '            "transferred_n": 0,\n'
        '            "donor_endpoint_n": 0,\n',
        '            "cross_endpoint_exchange_enabled": True,\n'
        '            "transfer_attempted_n": int(transfer_attempted_n),\n'
        '            "injected_n": int(transfer_valid_n),\n'
        '            "transferred_n": int(transfer_accepted_n),\n'
        '            "placement_evaluated_n": int(placement_evaluated_n),\n'
        '            "routing_mode": boundary_route_mode,\n'
        '            "donor_endpoint_n": len({str(x.get("source_endpoint")) for x in transfer_items if x.get("accepted_into_resumed_population")}),\n'
        '            "checkpoint_population_fitness_recomputed": True,\n'
        '            "transfer_items": transfer_items,\n'
        '            "transfer_rejected_items": transfer_rejected_items,\n'
        '            "lineage_by_generation": boundary_lineage_by_generation,\n'
        '            "lineage_survived_grace_individual_n": int(next((row["population_lineage_individual_n"] for row in boundary_lineage_by_generation if row["generation"] == min(int(n_gen), max(0, int(globals().get("BOUNDARY_MIGRATION_GRACE_GENERATIONS", 2))))), 0)),\n'
        '            "lineage_final_population_individual_n": int(boundary_lineage_by_generation[-1]["population_lineage_individual_n"] if boundary_lineage_by_generation else 0),\n'
        '            "lineage_final_hof_individual_n": int(boundary_lineage_by_generation[-1]["hof_lineage_individual_n"] if boundary_lineage_by_generation else 0),\n'
        '            "lineage_final_population_id_n": int(boundary_lineage_by_generation[-1]["population_lineage_id_n"] if boundary_lineage_by_generation else 0),\n'
        '            "lineage_final_hof_id_n": int(boundary_lineage_by_generation[-1]["hof_lineage_id_n"] if boundary_lineage_by_generation else 0),\n',
        "resume audit exchange fields",
    )
    function = function.replace(
        '"schema_version": "phaseA_no_exchange_resume_v1"',
        '"schema_version": "phaseA_call_aware_paired_boundary_exchange_resume_v2"',
    )
    function = function.replace("additional_gen={n_gen} exchange=disabled", "additional_gen={n_gen} exchange=call_aware_boundary")
    function = replace_once(
        function,
        "    return hof\n",
        "    if is_phasea and _boundary_wrapped:\n"
        "        toolbox.select = _boundary_original_select\n"
        "        toolbox.mate = _boundary_original_mate\n"
        "        toolbox.mutate = _boundary_original_mutate\n"
        "    return hof\n",
        "toolbox restoration",
    )
    return function


def patch_phasec_boundary_seed(source: str) -> str:
    old_init = '''    population, population_seen = [], set()
    shuffled = list(feasible)
    random.shuffle(shuffled)
    for cand in shuffled:
'''
    new_init = '''    population, population_seen = [], set()
    boundary_rule_by_pair = {}
    boundary_rules_path = str(globals().get("PHASEC_EXTERNAL_RULES_JSON", "") or "")
    if boundary_rules_path:
        boundary_rule_rows = json.loads(Path(boundary_rules_path).read_text(encoding="utf-8"))
        for rule in boundary_rule_rows:
            if not isinstance(rule, dict) or rule.get("phaseC_origin") == "single_profile007_backbone":
                continue
            if rule.get("passed_support_gate") is False:
                continue
            pair = rule.get("interaction_pair")
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            pair_key = tuple(sorted((str(pair[0]), str(pair[1]))))
            source_endpoints = list(rule.get("source_endpoints", []) or [])
            if rule.get("source_endpoint"):
                source_endpoints.append(str(rule.get("source_endpoint")))
            target_endpoint = str(rule.get("target_endpoint", "") or "")
            if not any(str(source) and str(source) != target_endpoint for source in source_endpoints):
                continue
            score = (
                float(rule.get("phaseC_transfer_score", rule.get("relation_score", 0.0)) or 0.0),
                float(rule.get("source_rule_quality_score", 0.0) or 0.0),
            )
            previous = boundary_rule_by_pair.get(pair_key)
            if previous is None or score > previous[0]:
                boundary_rule_by_pair[pair_key] = (score, dict(rule))
    boundary_external = []
    for cand in feasible:
        pair = cand.get("interaction_pair")
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        pair_key = tuple(sorted((str(pair[0]), str(pair[1]))))
        matched = boundary_rule_by_pair.get(pair_key)
        if matched is None and not bool(cand.get("relation_aware_transfer", False)):
            continue
        promoted = dict(cand)
        if matched is not None:
            rule = matched[1]
            for key in (
                "candidate_id", "source_endpoint", "source_endpoints", "target_endpoint",
                "phaseC_transfer_score", "relation_score", "source_rule_quality_score",
                "source_selection_score", "source_support_score", "source_count_score",
                "phaseC_transfer_mode", "phaseC_transfer_scoring_mode",
            ):
                if rule.get(key) is not None:
                    promoted[key] = rule.get(key)
        promoted["relation_aware_transfer"] = True
        promoted["boundary_external_pair_key"] = list(pair_key)
        boundary_external.append(promoted)
    boundary_external.sort(
        key=lambda c: (
            float(c.get("phaseC_transfer_score", c.get("relation_score", 0.0)) or 0.0),
            float(c.get("source_rule_quality_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    boundary_seed_fraction = min(0.50, max(0.0, float(globals().get("BOUNDARY_PHASEC_SEED_FRACTION", 0.20))))
    boundary_seed_n = min(len(boundary_external), int(round(int(PHASEC_POP_SIZE) * boundary_seed_fraction)))
    for seed_index, cand in enumerate(boundary_external[:boundary_seed_n]):
        sig = phasec_candidate_signature(cand)
        if sig in population_seen:
            continue
        population_seen.add(sig)
        candidate_id = str(cand.get("candidate_id") or f"{cand.get('source_endpoint', 'source')}::{cand.get('name', sig)}")
        population.append(dict(
            cand,
            init_source="boundary_external_seed",
            fold=fold_label,
            boundary_phasec_lineage_ids=[candidate_id],
            boundary_phasec_direct=True,
        ))
    globals().setdefault("BOUNDARY_PHASEC_SEED_AUDIT", []).append({
        "fold": str(fold_label),
        "external_feasible_n": int(len(boundary_external)),
        "requested_seed_n": int(boundary_seed_n),
        "actual_seed_n": int(len(population)),
        "seed_fraction": float(boundary_seed_fraction),
    })
    shuffled = [c for c in feasible if phasec_candidate_signature(c) not in population_seen]
    random.shuffle(shuffled)
    for cand in shuffled:
'''
    source = replace_once(source, old_init, new_init, "PhaseC external seed initialization")
    source = replace_once(
        source,
        '    init_rows = [dict(c, fold=fold_label, initialization="random_legal", base_fit_once=True, support_precomputed=True, fitness_cache_enabled=True) for c in population]\n',
        '    init_rows = [dict(c, fold=fold_label, initialization=c.get("init_source", "random_legal"), base_fit_once=True, support_precomputed=True, fitness_cache_enabled=True) for c in population]\n',
        "PhaseC initialization audit",
    )
    source = replace_once(
        source,
        '    print(f"[PhaseC-STGP-AND][Fold {fold_label}] init_population={len(population)} random_legal={len(population)}")\n',
        '    print(f"[PhaseC-STGP-AND][Fold {fold_label}] init_population={len(population)} boundary_external_seed={sum(c.get(\'init_source\') == \'boundary_external_seed\' for c in population)}")\n',
        "PhaseC seed log",
    )

    crossover_start = source.index("def phasec_stgp_and_crossover_candidates(")
    crossover_end = source.index("\n\ndef ", crossover_start)
    crossover = source[crossover_start:crossover_end]
    crossover = replace_once(
        crossover,
        "        out.append(kept if kept is not None else old)\n",
        "        child = dict(kept if kept is not None else old)\n"
        "        lineage = sorted(set(p1.get(\"boundary_phasec_lineage_ids\", []) or []) | set(p2.get(\"boundary_phasec_lineage_ids\", []) or []))\n"
        "        if lineage:\n"
        "            child[\"boundary_phasec_lineage_ids\"] = lineage\n"
        "            child[\"boundary_phasec_descendant\"] = True\n"
        "        out.append(child)\n",
        "PhaseC crossover lineage propagation",
    )
    source = source[:crossover_start] + crossover + source[crossover_end:]

    mutate_start = source.index("def phasec_stgp_and_mutate_candidate(")
    mutate_end = source.index("\n\ndef ", mutate_start)
    mutate = source[mutate_start:mutate_end]
    mutate = replace_once(
        mutate,
        "            if kept is not None:\n                return kept\n",
        "            if kept is not None:\n"
        "                lineage = list(candidate.get(\"boundary_phasec_lineage_ids\", []) or [])\n"
        "                if lineage:\n"
        "                    kept = dict(kept)\n"
        "                    kept[\"boundary_phasec_lineage_ids\"] = lineage\n"
        "                    kept[\"boundary_phasec_descendant\"] = True\n"
        "                return kept\n",
        "PhaseC mutation lineage propagation",
    )
    source = source[:mutate_start] + mutate + source[mutate_end:]
    return source


def main() -> None:
    base = load_no_exchange_builder()
    resume_function = build_resume_function(base)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    evolution_replaced = 0
    phasec_patched = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def run_evolution(" in source:
            start = source.index("def run_evolution(")
            end = source.find("\n\ndef ", start + 1)
            if end < 0:
                raise RuntimeError("Could not find run_evolution function boundary")
            constant = f'PHASEA_RESUME_CHECKPOINT_ROOT = r"{Path(base.CHECKPOINT_ROOT).resolve()}"\n'
            source = source[:start] + constant + resume_function + source[end:]
            evolution_replaced += 1
        if "def phasec_stgp_and_run_evolution(" in source:
            source = patch_phasec_boundary_seed(source)
            phasec_patched += 1
        cell["source"] = source.splitlines(keepends=True)
    if evolution_replaced != 1:
        raise RuntimeError(f"Expected one run_evolution replacement, got {evolution_replaced}")
    if phasec_patched != 1:
        raise RuntimeError(f"Expected one PhaseC evolution patch, got {phasec_patched}")
    OUTPUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUTPUT_NOTEBOOK)


if __name__ == "__main__":
    main()
