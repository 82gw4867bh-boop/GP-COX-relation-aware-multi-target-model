# Source map

| Component | Source |
|---|---|
| Main CLI | `run_pipeline.py` |
| Seed schedule execution | `run_seeds.py` |
| Environment/configuration checks | `selfcheck.py` |
| Study settings and task specification | `config/` |
| Staged GP-Cox and checkpoint export | `runtime/single_target_checkpoint.ipynb` |
| Notebook execution/status helpers | `runtime/run_multitarget_profile_optimization.py` |
| Complete-assignment parameter binding | `runtime/portable_notebook.py` |
| Same-interpreter cell execution | `runtime/execute_notebook.py` |
| Exchange-specific checkpoint template | `runtime/exchange_checkpoint.ipynb` |
| No-exchange G continuation | `runtime/checkpoint_no_exchange_G_continuation/` |
| Boundary/Live-online builders and controller | `runtime/checkpoint_fair_multitarget_experiments/` |
| Multi-target controller and locked evaluation | `runtime/multitarget_abc_runner.py` |
| Relation matrices | `runtime/endpoint_relation_config.py` |
| Rule transfer and expression mapping | `runtime/phaseC_relation_transfer.py`, `runtime/relation_aware_mtgps.py` |
| Single-endpoint wrapper | `runtime/single_endpoint_abc_pipeline.py` |
| Knowledge export | `runtime/build_knowledge_hcoxfix_trial010_from_single_gp.py` |
| End-only policy and recipient evaluation | `runtime/end_only_backend/` |
| Published model structures | `config/locked_structures/` |
| Final reconstruction | `runtime/four_endpoint_results_core_20260814.py`, `runtime/four_endpoint_safe_endonly_core_20260814.py` |
| Shared protected primitives | `runtime/locked_trial31_safe_repair_20260802/` |
| Four-endpoint baseline driver | `runtime/run_four_endpoint_deepsurv_rsf_20260812.py` |
| Baseline/grid helpers | `runtime/run_deepsurv_rsf_baselines.py`, `runtime/run_deepsurv_rsf_limited_grid.py` |
| Cohort queries | `data_extraction/` |

Some module names are retained for import compatibility. Public entry points configure four outcomes. Structural JSON files contain feature/formula definitions, without patient-level prediction arrays.
