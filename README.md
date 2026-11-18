# Four-outcome relation-aware GP-Cox

Relation-aware genetic programming and Cox survival modelling for four outcomes after cervical spinal cord injury: death or hospice, high-dependency institutional discharge, favourable disposition after the index ICU stay, and ICU exit.

The project combines staged symbolic feature construction with Boundary, Live-online and End-only knowledge exchange. It includes G/2G training, checkpoint continuation, model evaluation, fixed-structure interpretation, seed schedules and optional DeepSurv/RSF baselines.

- [Installation and analysis commands](github_package/README.md)
- [Project construction](github_package/PROJECT_CONSTRUCTION.md)
- [Source map](github_package/CODE_MAP.md)
- [Analytic data requirements](github_package/data/README.md)

The analytic pipeline uses harmonised eICU-CRD and MIMIC-IV endpoint tables. Cohort queries are included; controlled-access patient data are obtained separately.
