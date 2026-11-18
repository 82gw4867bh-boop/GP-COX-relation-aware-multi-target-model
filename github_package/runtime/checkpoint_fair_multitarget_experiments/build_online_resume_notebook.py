from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE_NOTEBOOK = Path(
    os.environ.get(
        "MTGP_ONLINE_SOURCE_NOTEBOOK",
        str(
            ROOT
            / "single_endpoint_phaseA_G_checkpoint_hcoxfix_trial010"
            / "newcox_route1_phaseB_phaseC_profile_style_aligned_checkpoint.ipynb"
        ),
    )
)
OUTPUT_NOTEBOOK = Path(
    os.environ.get(
        "MTGP_ONLINE_OUTPUT_NOTEBOOK",
        str(HERE / "online_notebook" / "newcox_route1_phaseB_phaseC_profile_style_aligned.ipynb"),
    )
)


def load_resume_builder():
    path = ROOT / "checkpoint_no_exchange_G_continuation" / "build_resume_notebook.py"
    spec = importlib.util.spec_from_file_location("fair_resume_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ONLINE_SETUP = r'''
        # relation_aware_phaseA_tree_injection: online crossover owns PhaseA
        # transfer; this marker disables the runner's legacy post-evolution
        # best_ind_x.append(external_tree) injection.
        online_transfer_records = []
        online_exchange_rows = []
        online_exchange_attempted_n = 0
        online_exchange_completed_n = 0
        phasea_transfer_path = str(globals().get("PHASEA_EXTERNAL_TREES_JSON", "") or "")
        if bool(globals().get("RELATION_AWARE_PHASEA_ENABLED", False)) and phasea_transfer_path:
            online_transfer_records = [
                record
                for record in json.loads(Path(phasea_transfer_path).read_text(encoding="utf-8"))
                if str(record.get("source_endpoint", ""))
                and str(record.get("source_endpoint", "")) != str(record.get("target_endpoint", ""))
                and str(record.get("mapped_expression") or record.get("expression") or "").strip()
            ]
'''


ONLINE_GENERATION_BLOCK = r'''
        # Cross-endpoint GP crossover happens inside every resumed generation.
        online_probability = float(globals().get("CROSS_ENDPOINT_CROSSOVER_PROB", 0.25))
        if online_transfer_records and online_probability > 0.0:
            for child_index, child in enumerate(offspring):
                if random.random() >= online_probability:
                    continue
                record = random.choice(online_transfer_records)
                online_exchange_attempted_n += 1
                row = {
                    "generation": int(gen),
                    "child_index": int(child_index),
                    "source_endpoint": record.get("source_endpoint"),
                    "target_endpoint": record.get("target_endpoint"),
                    "candidate_id": record.get("candidate_id", record.get("tree_id")),
                    "operator": "gp.cxOnePoint_target_slot_with_mapped_checkpoint_donor",
                    "completed": False,
                }
                try:
                    original_child = toolbox.clone(child)
                    donor_slot = random.randrange(len(child))
                    donor_template = tree_from_string(
                        str(record.get("mapped_expression") or record.get("expression"))
                    )
                    if int(donor_template.height) > int(MAX_TREE_HEIGHT_X):
                        raise ValueError("mapped donor exceeds MAX_TREE_HEIGHT_X")
                    before_expression = str(child[donor_slot])
                    effective = False
                    after_expression = before_expression
                    # Directly cross the selected recipient slot with the mapped
                    # external tree. The old toolbox.mate() path applied another
                    # hidden 0.5 gate and counted no-op calls as completed.
                    for crossover_try in range(1, 4):
                        child[:] = toolbox.clone(original_child)
                        donor_tree = gp.PrimitiveTree(donor_template)
                        gp.cxOnePoint(child[donor_slot], donor_tree)
                        after_expression = str(child[donor_slot])
                        heights_ok = all(int(tree.height) <= int(MAX_TREE_HEIGHT_X) for tree in child)
                        if heights_ok and after_expression != before_expression:
                            effective = True
                            break
                    if not effective:
                        child[:] = toolbox.clone(original_child)
                        raise ValueError("no effective height-valid external crossover after 3 attempts")
                    if child.fitness.valid:
                        del child.fitness.values
                    online_exchange_completed_n += 1
                    row.update({
                        "completed": True,
                        "effective_crossover": True,
                        "donor_tree_slot": int(donor_slot),
                        "recipient_tree_slot": int(donor_slot),
                        "crossover_try_n": int(crossover_try),
                        "recipient_expression_before": before_expression,
                        "recipient_expression_after": after_expression,
                    })
                except Exception as exc:
                    row["effective_crossover"] = False
                    row["reason"] = str(exc)
                online_exchange_rows.append(row)
'''


def main() -> None:
    base = load_resume_builder()
    function = str(base.RESUME_FUNCTION)
    rng_needle = '        np.random.set_state(resume_payload["numpy_random_state"])\n'
    if function.count(rng_needle) != 1:
        raise RuntimeError("Could not locate checkpoint RNG restore")
    function = function.replace(rng_needle, rng_needle + ONLINE_SETUP, 1)

    generation_needle = '        for ind in [i for i in offspring if not i.fitness.valid]:\n'
    if function.count(generation_needle) != 1:
        raise RuntimeError("Could not locate resumed generation evaluation")
    function = function.replace(generation_needle, ONLINE_GENERATION_BLOCK + generation_needle, 1)
    function = function.replace(
        '            "cross_endpoint_exchange_enabled": False,\n'
        '            "injected_n": 0,\n'
        '            "transferred_n": 0,\n'
        '            "donor_endpoint_n": 0,\n',
        '            "cross_endpoint_exchange_enabled": True,\n'
        '            "exchange_timing": "during_each_resumed_generation",\n'
        '            "injected_n": 0,\n'
        '            "exchange_attempted_n": int(online_exchange_attempted_n),\n'
        '            "transferred_n": int(online_exchange_completed_n),\n'
        '            "donor_endpoint_n": len({str(row.get("source_endpoint")) for row in online_exchange_rows if row.get("completed")}),\n'
        '            "online_exchange_rows": online_exchange_rows,\n',
    )
    function = function.replace(
        '"schema_version": "phaseA_no_exchange_resume_v1"',
        '"schema_version": "phaseA_fair_G_online_exchange_resume_v1"',
    )
    function = function.replace(
        "additional_gen={n_gen} exchange=disabled",
        "additional_gen={n_gen} exchange=online",
    )

    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    replaced = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "def run_evolution(" not in source:
            continue
        start = source.index("def run_evolution(")
        end = source.find("\n\ndef ", start + 1)
        if end < 0:
            raise RuntimeError("Could not find run_evolution boundary")
        constant = f'PHASEA_RESUME_CHECKPOINT_ROOT = r"{Path(base.CHECKPOINT_ROOT).resolve()}"\n'
        source = source[:start] + constant + function + source[end:]
        cell["source"] = source.splitlines(keepends=True)
        replaced += 1
    if replaced != 1:
        raise RuntimeError(f"Expected one run_evolution replacement, got {replaced}")
    OUTPUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUTPUT_NOTEBOOK)


if __name__ == "__main__":
    main()
