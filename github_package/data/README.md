# Analytic data requirements

Supply the following workbooks here or pass another directory with `--data-dir`:

| Endpoint | eICU file | MIMIC-IV file |
|---|---|---|
| Death/hospice | `event_death_hospice_corrected_time_eicu.xlsx` | `event_death_hospice_corrected_time_mimic.xlsx` |
| Institutional discharge | `event_care_dependent_survival_after_index_icu_eicu.xlsx` | `event_care_dependent_survival_after_index_icu_mimic.xlsx` |
| ICU exit | `event_3_icu_time_to_exit_eicu.xlsx` | `event_3_icu_time_to_exit_mimic.xlsx` |
| Favourable disposition | `event_favourable_outcome_after_index_icu_eicu.xlsx` | `event_favourable_outcome_after_index_icu_mimic.xlsx` |

Each table has one row per ICU stay, positive follow-up `time` in days, binary `event`, identifiers and harmonised clinical predictors. Original cohort sizes are 313 eICU stays and 340 MIMIC-IV stays. Column names and reference dtypes are in `../config/input_schema.json`, without patient row values.

ICU exit uses observed index-ICU exit as its event. Discharge outcomes follow the study's corrected endpoint definitions. Match predictor definitions, units and measurement windows to the manuscript and supplementary mapping. `Total.Ca` (mg/dL) and `Ionized.Ca` (mmol/L) are distinct fields and are not filled from one another.

The SQL in `../data_extraction/` identifies eligible stays. It does not construct every predictor and endpoint workbook automatically. Credentialed users prepare the harmonised tables before invoking the modelling pipeline. Missing-value transformations are fitted by the code; changing predictor definitions cannot be handled by merely renaming columns.

Store generated inputs, predictions, checkpoints and fitted patient objects locally. Patient workbooks are excluded by the ignore rules.
