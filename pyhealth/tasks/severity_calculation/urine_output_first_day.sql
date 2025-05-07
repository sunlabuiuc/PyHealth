-- THIS SCRIPT IS AUTOMATICALLY GENERATED. DO NOT EDIT IT DIRECTLY.
DROP TABLE IF EXISTS urine_output_first_day;
CREATE TABLE urine_output_first_day AS
-- ------------------------------------------------------------------
-- Purpose: Create a table of the urine output for each ICUSTAY_ID over the first 24 hours.
-- ------------------------------------------------------------------

SELECT
  -- patient identifiers
  ie.subject_id, ie.hadm_id, ie.icustay_id

  -- volumes associated with urine output ITEMIDs
  , SUM(
      -- we consider input of GU irrigant as a negative volume
      CASE
        WHEN oe.itemid = 227488 AND oe.value > 0 THEN -1 * oe.value
        ELSE oe.value
      END
    ) AS urineoutput -- Note: This will be NULL if no matching outputevents are found for an icustay_id. Use COALESCE(SUM(...), 0) if 0 is preferred.

FROM icustays ie -- Assuming icustays table exists
-- Join to the outputevents table to get urine output
LEFT JOIN outputevents oe -- Assuming outputevents table exists
  -- join on all patient identifiers
  ON ie.subject_id = oe.subject_id AND ie.hadm_id = oe.hadm_id AND ie.icustay_id = oe.icustay_id
  -- and ensure the data occurs during the first day
  -- Replace DATETIME_ADD with PostgreSQL interval addition
  AND oe.charttime BETWEEN ie.intime AND (ie.intime + INTERVAL '1 day') -- first ICU day (inclusive)
WHERE oe.itemid IN -- Filter for relevant item IDs in outputevents
(
-- these are the most frequently occurring urine output observations in CareVue
40055, -- "Urine Out Foley"
43175, -- "Urine ."
40069, -- "Urine Out Void"
40094, -- "Urine Out Condom Cath"
40715, -- "Urine Out Suprapubic"
40473, -- "Urine Out IleoConduit"
40085, -- "Urine Out Incontinent"
40057, -- "Urine Out Rt Nephrostomy"
40056, -- "Urine Out Lt Nephrostomy"
40405, -- "Urine Out Other"
40428, -- "Urine Out Straight Cath"
40086, --  Urine Out Incontinent
40096, -- "Urine Out Ureteral Stent #1"
40651, -- "Urine Out Ureteral Stent #2"

-- these are the most frequently occurring urine output observations in MetaVision
226559, -- "Foley"
226560, -- "Void"
226561, -- "Condom Cath"
226584, -- "Ileoconduit"
226563, -- "Suprapubic"
226564, -- "R Nephrostomy"
226565, -- "L Nephrostomy"
226567, --  Straight Cath
226557, -- R Ureteral Stent
226558, -- L Ureteral Stent
227488, -- GU Irrigant Volume In --> This is subtracted in the SUM() calculation
227489  -- GU Irrigant/Urine Volume Out --> This is added in the SUM() calculation
)
GROUP BY ie.subject_id, ie.hadm_id, ie.icustay_id
ORDER BY ie.subject_id, ie.hadm_id, ie.icustay_id;