# Relation-aware GP-Cox for four-outcome survival prediction

The project models four outcomes after cervical spinal cord injury using staged genetic programming and Cox survival models. eICU-CRD supplies the development cohort and MIMIC-IV supplies the external evaluation cohort. The four tasks are death/hospice, institutional discharge, favourable disposition, and ICU exit.

## Environment

Use Python 3.11 in an isolated environment. Run the commands below from this `github_package` directory.

```sh
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install -r requirements.txt
python selfcheck.py
```

The runtime executes notebook cells using the invoking Python interpreter. A separately installed Jupyter kernel is unnecessary. `selfcheck.py` checks module imports, four-task configurations, notebook builders and parameter binding without patient data.

## Data

Prepare the eight harmonised endpoint workbooks listed in [data/README.md](data/README.md). Store them in `data/` or specify another directory with `--data-dir`. Column names and reference types are in `config/input_schema.json`.

The original cohorts contain 313 eICU stays and 340 MIMIC-IV stays. Each endpoint workbook contains positive follow-up `time`, binary `event`, identifiers and harmonised predictors. Total and ionised calcium remain separate variables. Controlled-access patient data are not included in the repository.

```sh
python run_pipeline.py preflight --data-dir data
```

The analytic entry point expects the harmonised tables. Cohort SQL identifies eligible stays; it does not automatically extract every predictor needed for those tables.

## Primary GP analysis

```sh
python run_pipeline.py all --data-dir data --output-dir outputs/primary
```

This runs G, checkpoint-resumed 2G without exchange, Boundary, Live-online and End-only. It then reconstructs the published fixed final models and computes the comparisons from the supplied selected seed records. The default training seed is 20260523 and the End-only policy seed is 2026080903; use `--seed` and `--policy-seed` to set alternatives.

The stages can also be run separately:

```sh
python run_pipeline.py train --budget G --data-dir data --output-dir outputs/primary
python run_pipeline.py train --budget resume2G --data-dir data --output-dir outputs/primary
python run_pipeline.py train --budget 2G --data-dir data --output-dir outputs/direct2g
python run_pipeline.py exchange --strategy boundary --data-dir data --output-dir outputs/primary
python run_pipeline.py exchange --strategy live_online --data-dir data --output-dir outputs/primary
python run_pipeline.py exchange --strategy end_only --data-dir data --output-dir outputs/primary
```

G uses 12 Phase-A and 20 Phase-C generations. Direct 2G uses 24 and 40 respectively. `resume2G` restores G checkpoints and continues Phase A by another 12 generations, with Phase C=40. Run G first in the same output tree before continuation and early-exchange stages. End-only requires the resumed-2G knowledge library. `--upstream-dir` permits an exchange-only run to use G/resume2G artifacts from another output tree.

Endpoint results, logs, patched notebooks, checkpoints and knowledge libraries are written to the output tree. Use separate output directories for different seeds, profiles or smoke budgets.

## Reduced-budget training

```sh
python run_pipeline.py train --budget G --endpoint death_hospice --smoke --data-dir data --output-dir outputs/smoke
```

This executes the training pipeline with a reduced search budget for an initial environment/data check. Smoke results are separate from the study settings. Formal exchange requires formal G/resume2G artifacts.

## Fixed final models and statistical comparisons

```sh
python run_pipeline.py locked --data-dir data --output-dir outputs/locked
python run_pipeline.py statistics --output-dir outputs/statistics
```

`locked` uses `config/locked_structures/` to fit the prescribed final models and apply the safe transformations without structure reselection. It saves coefficient summaries and reconstruction diagnostics. This fixed-model route is separate from a fresh stochastic search.

`statistics` recalculates mean differences, confidence intervals and P values from `reproducibility/selected_10seed_G_2G_results.csv` and `selected_10seed_multitarget_results.csv`. It needs no patient data and does not train models.

## Seed schedules

```sh
python run_seeds.py --method g2g --data-dir data --output-dir outputs/seeds
python run_seeds.py --method boundary --data-dir data --upstream-dir outputs/primary --output-dir outputs/seeds
python run_seeds.py --method live_online --data-dir data --upstream-dir outputs/primary --output-dir outputs/seeds
python run_seeds.py --method end_only --data-dir data --upstream-dir outputs/primary --output-dir outputs/seeds
```

The schedules are in `config/selected_seed_manifest.json`. Add `--seed` to run one seed. G/2G schedules train the corresponding upstream models. Exchange schedules use the supplied upstream knowledge/checkpoints while varying the relevant endpoint/exchange seed. Newly calculated summaries are saved as `*_recomputed.csv`, separately from the supplied selected records.

## DeepSurv and random survival forest

```sh
python -m pip install -r requirements-baselines.txt
python run_pipeline.py baselines --data-dir data --output-dir outputs/primary
```

These commands run the four-endpoint nonlinear baseline grids and evaluation modules. The primary `all` command covers the GP analysis; baseline grids and multi-seed schedules have separate commands.

## Source organization

[PROJECT_CONSTRUCTION.md](PROJECT_CONSTRUCTION.md) explains the data/model flow. [CODE_MAP.md](CODE_MAP.md) maps the components to `runtime/`. Configurations specify the four endpoint identities, G settings, relation matrices and exchange parameters. The package retains the selected transfer thresholds and protected-operator definitions.

Patient-level workbooks, generated aliases, predictions, checkpoints and fitted objects belong in local ignored directories. Database credentials are not stored in the source package.
