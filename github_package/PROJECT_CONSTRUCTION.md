# Project construction

`run_pipeline.py` connects the four-endpoint model components through explicit configurations and portable input/output bindings.

1. Cohort queries identify eligible eICU/MIMIC-IV stays. Harmonised endpoint/predictor tables follow the input contract in `data/README.md` and `config/input_schema.json`.
2. Preflight validates survival fields and cohort sizes. The runner creates endpoint-ID aliases under the local output directory; original input tables are not edited.
3. The checkpoint-enabled single-target notebook fits preprocessing and builds continuous GP expressions (Phase A), time interactions (Phase B), and binary/categorical rules (Phase C). The G profile uses Phase A=12 and Phase C=20 generations. Five Phase-A calls per endpoint supply outer-fold and full-training checkpoints.
4. No-exchange continuation restores populations, fitness values and random states and runs another 12 Phase-A generations. Phase C uses 40 generations. Direct 2G training is an alternative route.
5. Boundary and Live-online builders add the corresponding donor-transfer hooks to the continuation source. The controller supplies four-task relation matrices and selected parameters, fits the models and evaluates their predictions.
6. Knowledge export packages endpoint structures for End-only. Its backend builds a donor pool, evolves a ranking policy, evaluates candidate transfers in recipients and applies the selected acceptance/capacity rules. Negative-delta tolerance is retained in `config/end_only_defaults.json`.
7. The fixed-model route reads the published structural definitions, refits their coefficients and applies protected primitives with smoothing constant 0.25, without structural reselection.
8. Seed schedules, selected seed records and statistical comparisons are handled separately. Optional DeepSurv/RSF modules have their own dependency file.

Configuration order is death/hospice, institutional discharge, ICU exit and favourable disposition. Historical ID `event_5_home` refers to the fourth configured task. Relation-matrix rows are recipients and columns are donors; strategy-specific overrides supersede reference weights.

The runtime uses the four-endpoint backend directly. Notebook parameter binding replaces complete Python assignments, and cell execution uses the invoking environment. The source retains model functions, selected strategy settings, acceptance thresholds and safe-operator formulas.
