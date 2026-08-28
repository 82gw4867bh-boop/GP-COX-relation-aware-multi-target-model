-- eICU-CRD v2.0 cohort definition: cervical spinal cord injury with an ICU stay.
-- Unit of analysis: ICU stay (patientunitstayid).

WITH strict_diagnosis AS (
    SELECT DISTINCT diagnosis.patientunitstayid
    FROM public.diagnosis AS diagnosis
    WHERE diagnosis.diagnosisstring ~* 'cervical'
      AND diagnosis.diagnosisstring ~* '(spinal\\s*cord\\s*injury|cord\\s*injury|quadriplegia|tetraplegia|cervical\\s*myelopathy)'
      AND NOT (
          diagnosis.diagnosisstring ~* '(brachial\\s*plexus|plexopathy|sympathetic\\s*(nerve|injury|chain)|horner|nerve\\s*root|radiculopathy|peripheral\\s*nerve|neuropathy|ulnar\\s*nerve|median\\s*nerve|radial\\s*nerve)'
          AND diagnosis.diagnosisstring !~* '(spinal\\s*cord\\s*injury|cord\\s*injury|quadriplegia|tetraplegia|myelopathy)'
      )
)
SELECT
    patient.uniquepid AS subject_id,
    patient.patienthealthsystemstayid AS hadm_id,
    patient.patientunitstayid AS stay_id,
    patient.gender,
    patient.age,
    patient.ethnicity,
    patient.hospitaladmitsource,
    patient.unitdischargeoffset,
    patient.hospitaldischargelocation,
    patient.hospitaldischargestatus
FROM public.patient AS patient
JOIN strict_diagnosis AS cohort
    ON cohort.patientunitstayid = patient.patientunitstayid;
