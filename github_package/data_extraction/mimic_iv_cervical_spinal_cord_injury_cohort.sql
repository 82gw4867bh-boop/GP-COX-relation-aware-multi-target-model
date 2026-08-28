-- MIMIC-IV v3.1 cohort definition: cervical spinal cord injury with an ICU stay.
-- Unit of analysis: ICU stay.

WITH cervical_sci_admissions AS (
    SELECT DISTINCT
        diagnoses.subject_id,
        diagnoses.hadm_id
    FROM mimiciv_hosp.diagnoses_icd AS diagnoses
    WHERE
        (diagnoses.icd_version = 10 AND regexp_replace(upper(diagnoses.icd_code), '[^A-Z0-9]', '', 'g') LIKE 'S140%')
        OR (diagnoses.icd_version = 10 AND regexp_replace(upper(diagnoses.icd_code), '[^A-Z0-9]', '', 'g') LIKE 'S141%')
        OR (diagnoses.icd_version = 9 AND regexp_replace(upper(diagnoses.icd_code), '[^A-Z0-9]', '', 'g') LIKE '9520%')
        OR (diagnoses.icd_version = 9 AND regexp_replace(upper(diagnoses.icd_code), '[^A-Z0-9]', '', 'g') LIKE '8060%')
        OR (diagnoses.icd_version = 9 AND regexp_replace(upper(diagnoses.icd_code), '[^A-Z0-9]', '', 'g') LIKE '8061%')
)
SELECT DISTINCT
    icu.subject_id,
    icu.hadm_id,
    icu.stay_id
FROM cervical_sci_admissions AS cohort
JOIN mimiciv_icu.icustays AS icu
    ON cohort.subject_id = icu.subject_id
   AND cohort.hadm_id = icu.hadm_id;
