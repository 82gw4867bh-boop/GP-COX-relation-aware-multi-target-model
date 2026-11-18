from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = Path(
    os.environ.get(
        "HCA_RESUME_SOURCE_NOTEBOOK",
        str(
            ROOT.parent
            / "single_endpoint_phaseA_G_checkpoint_hcoxfix_trial010"
            / "newcox_route1_phaseB_phaseC_profile_style_aligned_checkpoint.ipynb"
        ),
    )
)
OUTPUT_NOTEBOOK = Path(
    os.environ.get(
        "HCA_RESUME_OUTPUT_NOTEBOOK",
        str(ROOT / "newcox_route1_phaseB_phaseC_profile_style_aligned_resume_no_exchange.ipynb"),
    )
)
CHECKPOINT_ROOT = Path(
    os.environ.get(
        "MTGP_G_CHECKPOINT_ROOT",
        str(
            ROOT.parent
            / "single_endpoint_phaseA_G_checkpoint_hcoxfix_trial010"
            / "outputs"
            / "profiles"
            / "hcoxfix_trial010_forceB_softC"
            / "single_endpoint_knowledge_outputs"
        ),
    )
)


RESUME_FUNCTION = r'''def run_evolution(toolbox, pop_size, n_gen, cx_pb, mut_pb, label="", on_generation_end=None):
    global PHASEA_CHECKPOINT_COUNTER
    phasea_checkpoint_index = None
    resume_payload = None
    is_phasea = "Phase-A" in str(label) or "PhaseA" in str(label)
    if is_phasea:
        import ast
        import pickle

        PHASEA_CHECKPOINT_COUNTER = int(globals().get("PHASEA_CHECKPOINT_COUNTER", 0)) + 1
        phasea_checkpoint_index = PHASEA_CHECKPOINT_COUNTER
        endpoint_id = str(
            globals().get("CURRENT_TARGET_ID")
            or globals().get("TARGET_ID")
            or ""
        )
        if not endpoint_id:
            endpoint_id = Path(TRAIN_DATA_PATH).stem
            if endpoint_id.endswith("_eicu"):
                endpoint_id = endpoint_id[:-5]
        checkpoint_dir = Path(PHASEA_RESUME_CHECKPOINT_ROOT) / endpoint_id / "phaseA_checkpoints"
        matches = sorted(checkpoint_dir.glob(f"phaseA_call_{phasea_checkpoint_index:02d}_generation_*.pkl"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one PhaseA checkpoint for endpoint={endpoint_id} "
                f"call={phasea_checkpoint_index}; found {len(matches)} under {checkpoint_dir}"
            )
        with matches[0].open("rb") as f:
            resume_payload = pickle.load(f)

        expected = {
            "phaseA_call_index": int(phasea_checkpoint_index),
            "generation": int(n_gen),
            "population_size": int(pop_size),
            "n_trees": int(N_TREES),
            "gp_input_vars": list(GP_INPUT_VARS),
            "main_effect_vars": list(MAIN_EFFECT_VARS),
        }
        mismatches = {
            key: {"checkpoint": resume_payload.get(key), "current": value}
            for key, value in expected.items()
            if resume_payload.get(key) != value
        }
        if abs(float(resume_payload.get("cx_probability")) - float(cx_pb)) > 1e-12:
            mismatches["cx_probability"] = {
                "checkpoint": resume_payload.get("cx_probability"), "current": float(cx_pb)
            }
        if abs(float(resume_payload.get("mutation_probability")) - float(mut_pb)) > 1e-12:
            mismatches["mutation_probability"] = {
                "checkpoint": resume_payload.get("mutation_probability"), "current": float(mut_pb)
            }
        if mismatches:
            raise RuntimeError(f"Checkpoint/current configuration mismatch: {mismatches}")

        pset = toolbox.compile.keywords["pset"]

        def tree_from_string(expression):
            nodes = []

            def visit(node):
                if isinstance(node, ast.Call):
                    name = node.func.id
                    if name == "array":
                        value = ast.literal_eval(node.args[0])
                        nodes.append(gp.Terminal(np.array(value, dtype=float), False, np.ndarray))
                        return
                    primitive = pset.mapping[name]
                    nodes.append(primitive)
                    for arg in node.args:
                        visit(arg)
                    return
                if isinstance(node, ast.Name):
                    nodes.append(pset.mapping[node.id])
                    return
                if isinstance(node, (ast.Constant, ast.UnaryOp)):
                    value = ast.literal_eval(node)
                    nodes.append(gp.Terminal(np.array(value, dtype=float), False, np.ndarray))
                    return
                raise TypeError(f"Unsupported checkpoint expression node: {ast.dump(node)}")

            visit(ast.parse(expression, mode="eval").body)
            return gp.PrimitiveTree(nodes)

        def restore_individual(expressions, fitness):
            individual = creator.IndividualX([tree_from_string(expr) for expr in expressions])
            individual.fitness.values = tuple(float(v) for v in fitness)
            return individual

        pop = [
            restore_individual(expressions, fitness)
            for expressions, fitness in zip(
                resume_payload["population_expressions"], resume_payload["population_fitness"]
            )
        ]
        saved_hof = [
            restore_individual(expressions, fitness)
            for expressions, fitness in zip(
                resume_payload["hall_of_fame_expressions"], resume_payload["hall_of_fame_fitness"]
            )
        ]
        # Restore the Hall of Fame in its serialized order.  Rebuilding it via
        # ``HallOfFame.update`` can reverse equally fit individuals because
        # DEAP inserts ties on the right.  That changes hof[0] after a resume
        # even though the saved expressions and fitness values are identical.
        hof = tools.HallOfFame(ELITE_K)
        hof.items = [toolbox.clone(ind) for ind in saved_hof]
        hof.keys = [ind.fitness for ind in reversed(hof.items)]
        restored_hof_expressions = [[str(tree) for tree in ind] for ind in hof]
        if restored_hof_expressions != resume_payload["hall_of_fame_expressions"]:
            raise RuntimeError("Hall-of-Fame order changed while restoring checkpoint")
        hof.update(pop)
        if [[str(tree) for tree in ind] for ind in hof] != restored_hof_expressions:
            raise RuntimeError("Checkpoint population changed the restored Hall-of-Fame")
        random.setstate(resume_payload["python_random_state"])
        np.random.set_state(resume_payload["numpy_random_state"])
        print(
            f"  [{label}] RESUMED endpoint={endpoint_id} call={phasea_checkpoint_index} "
            f"from_gen={resume_payload['generation']} population={len(pop)} hof={len(hof)} "
            f"additional_gen={n_gen} exchange=disabled"
        )
    else:
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(ELITE_K)

    stat = tools.Statistics(lambda ind: ind.fitness.values[0])
    stat.register("avg", np.mean)
    stat.register("max", np.max)

    for ind in [i for i in pop if not i.fitness.valid]:
        ind.fitness.values = toolbox.evaluate(ind)
    hof.update(pop)

    if on_generation_end is not None:
        on_generation_end(gen=0, pop=pop, hof=hof)

    for gen in range(1, n_gen + 1):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for mut in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mut)
                del mut.fitness.values

        for ind in [i for i in offspring if not i.fitness.valid]:
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring
        hof.update(pop)

        rec = stat.compile(pop)
        if PRINT_GEN_LOG:
            print(f"  [{label}] Gen {gen:02d} | avg={rec['avg']:.4f} max={rec['max']:.4f} hofFit={hof[0].fitness.values[0]:.4f}")

        if on_generation_end is not None:
            on_generation_end(gen=gen, pop=pop, hof=hof)

    if is_phasea:
        audit_dir = Path(OUT_XLSX).parent / "phaseA_resume_audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit = {
            "schema_version": "phaseA_no_exchange_resume_v1",
            "checkpoint_path": str(matches[0].resolve()),
            "phaseA_call_index": int(phasea_checkpoint_index),
            "checkpoint_generation": int(resume_payload["generation"]),
            "additional_generation": int(n_gen),
            "effective_total_generation": int(resume_payload["generation"]) + int(n_gen),
            "population_n": len(pop),
            "hall_of_fame_n": len(hof),
            "checkpoint_population_fitness_valid_n": len(resume_payload["population_fitness"]),
            "cross_endpoint_exchange_enabled": False,
            "injected_n": 0,
            "transferred_n": 0,
            "donor_endpoint_n": 0,
            "python_rng_restored": True,
            "numpy_rng_restored": True,
            "hall_of_fame_order_restored_exactly": True,
            "hall_of_fame_population_update_unchanged": True,
        }
        audit_path = audit_dir / f"phaseA_call_{phasea_checkpoint_index:02d}_resume.json"
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    return hof
'''


def main() -> None:
    if not SOURCE_NOTEBOOK.exists():
        raise FileNotFoundError(SOURCE_NOTEBOOK)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    replaced = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def run_evolution(" not in source:
            continue
        start = source.index("def run_evolution(")
        end_marker = "\n\ndef "
        end = source.find(end_marker, start + 1)
        if end < 0:
            raise RuntimeError("Could not find the function boundary after run_evolution")
        constant = f'PHASEA_RESUME_CHECKPOINT_ROOT = r"{CHECKPOINT_ROOT.resolve()}"\n'
        source = source[:start] + constant + RESUME_FUNCTION + source[end:]
        cell["source"] = source.splitlines(keepends=True)
        replaced += 1
    if replaced != 1:
        raise RuntimeError(f"Expected one run_evolution replacement, got {replaced}")
    OUTPUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUTPUT_NOTEBOOK)


if __name__ == "__main__":
    main()
