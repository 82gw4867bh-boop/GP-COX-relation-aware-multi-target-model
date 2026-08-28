# Result-to-code verification

This package was checked against `result_FINAL.pdf` and `RESULTS_MAIN_TEXT_FOUR_ENDPOINT_20260815.docx`.

| Reported item | Released material | Verification result |
|---|---|---|
| Four endpoints | `code/single_target_cox_gp.ipynb`; `code/final_safe_repair/` | Matches death or hospice, high-dependency institutional discharge, favourable disposition after index ICU stay, and remaining in ICU at day 7. |
| Development and external cohorts | `data_extraction/` | Matches eICU development (313 ICU stays) and MIMIC-IV external validation (340 ICU stays). |
| Boundary exchange | `code/multitarget_gp/boundary_exchange.py`; `config/boundary_defaults.json` | Matches the Boundary strategy reported in the results. |
| Live-online exchange | `code/multitarget_gp/live_online_exchange.py`; `config/live_online_defaults.json` | Matches the Live-online strategy reported in the results. |
| End-only exchange | `code/multitarget_gp/end_only_exchange.py`; `code/multitarget_gp/end_only_backend/`; `config/end_only_defaults.json` | Matches the End-only strategy carried forward as the final multi-target model. |
| Final-model figures | `code/final_safe_repair/` | Matches the fixed-structure safe-repaired End-only model used for the final figures. |
| Stability ranges | `reproducibility/selected_10seed_multitarget_results.csv` | Contains 10 independently selected non-formal seeds for each of Boundary, Live-online, and End-only. |

The SHA-256 values in the stability file correspond to the fixed default configuration files in `config/`. The repository intentionally excludes raw data, intermediate model objects, and checkpoints.
