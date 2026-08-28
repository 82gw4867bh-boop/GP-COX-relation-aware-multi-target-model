# eICU cohort definition

The eICU-CRD v2.0 cohort was identified at the ICU-stay level from `public.diagnosis` and `public.patient`.

Inclusion required a diagnosis text containing `cervical` and at least one of: `spinal cord injury`, `cord injury`, `quadriplegia`, `tetraplegia`, or `cervical myelopathy`. Text indicating an isolated brachial plexus, nerve root, peripheral nerve, sympathetic nerve, or Horner-related condition was excluded unless an explicit spinal-cord-related term was also present.

`patientunitstayid` was the ICU-stay identifier. The released query standardizes the patient, hospital-stay, and ICU-stay fields to `subject_id`, `hadm_id`, and `stay_id`, respectively. The query yielded 313 ICU stays from 270 patients and 271 hospital admissions, matching the eICU analytic cohort used in the reported results.
