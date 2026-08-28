**# Reproducibility materials for the Cox-GP study

This repository contains the analysis code and cohort-definition queries supporting a four-endpoint study of ICU patients with cervical spinal cord injury. eICU-CRD was used for model development and MIMIC-IV for independent external validation.

The released materials include the single-target Cox-GP implementation, the three fixed multi-target exchange implementations (Boundary, Live-online, and End-only), the fixed-structure safe-repair code used for the final model figures, and the MIMIC-IV/eICU cohort queries. No participant-level data, database credentials, endpoint workbooks, fitted model objects, or intermediate checkpoints are included.

## Repository structure

- `code/single_target_cox_gp.ipynb`: single-target Cox-GP implementation.
- `code/multitarget_gp/`: code for the three multi-target exchange strategies.
- `code/final_safe_repair/`: fixed-structure End-only reconstruction used for the final-model figures.
- `config/`: fixed default configurations for the three multi-target strategies.
- `data_extraction/`: MIMIC-IV and eICU cohort queries and the eICU cohort definition.
- `reproducibility/`: compact records from 10 randomly selected non-formal seeds per multi-target strategy.

## Data access

MIMIC-IV (v3.1) and eICU Collaborative Research Database (v2.0) are de-identified, controlled-access datasets available through PhysioNet to credentialed users who complete the applicable data-use requirements. The data themselves are not redistributed here. The supplied queries define the study cohort; the analytic tables must retain the required `subject_id`, `hadm_id`, `stay_id`, `time`, `event`, and predictor columns.

## Reproducibility note

The three multi-target strategies use the fixed default configurations stored in `config/`. The compact stability file records 10 independently seeded runs for each strategy and was used to assess stability.

## Software

Create a Python environment using `requirements.txt`. The source analysis was executed with Python 3.11. The code expects spreadsheet input in `.xlsx` format.

## Data and code availability text

Suggested manuscript wording is provided in `CODE_AVAILABILITY.md` and `DATA_AVAILABILITY.md`.**
