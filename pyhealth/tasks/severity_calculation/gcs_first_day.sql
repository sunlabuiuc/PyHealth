-- THIS SCRIPT IS AUTOMATICALLY GENERATED. DO NOT EDIT IT DIRECTLY.
DROP TABLE IF EXISTS gcs_first_day;
CREATE TABLE gcs_first_day AS
-- ITEMIDs used:

-- CAREVUE
--     723 as GCSVerbal
--     454 as GCSMotor
--     184 as GCSEyes

-- METAVISION
--     223900 GCS - Verbal Response
--     223901 GCS - Motor Response
--     220739 GCS - Eye Opening

-- The code combines the ITEMIDs into the carevue itemids, then pivots those
-- So 223900 is changed to 723, then the ITEMID 723 is pivoted to form GCSVerbal

-- Note:
--  The GCS for sedated patients is defaulted to 15 in this code.
--  This is in line with how the data is meant to be collected.
--  e.g., from the SAPS II publication:
--    For sedated patients, the Glasgow Coma Score before sedation was used.
--    This was ascertained either from interviewing the physician who ordered the sedation,
--    or by reviewing the patient's medical record.

WITH base AS
(
  SELECT pvt.ICUSTAY_ID
  , pvt.charttime

  -- Easier names - note we coalesced Metavision and CareVue IDs below
  , MAX(CASE WHEN pvt.itemid = 454 THEN pvt.valuenum ELSE NULL END) AS GCSMotor
  , MAX(CASE WHEN pvt.itemid = 723 THEN pvt.valuenum ELSE NULL END) AS GCSVerbal
  , MAX(CASE WHEN pvt.itemid = 184 THEN pvt.valuenum ELSE NULL END) AS GCSEyes

  -- If verbal was set to 0 in the below select, then this is an intubated patient
  , CASE
      WHEN MAX(CASE WHEN pvt.itemid = 723 THEN pvt.valuenum ELSE NULL END) = 0
      THEN 1
      ELSE 0
    END AS EndoTrachFlag

  , ROW_NUMBER() OVER (PARTITION BY pvt.ICUSTAY_ID ORDER BY pvt.charttime ASC) AS rn

  FROM  (
    SELECT l.ICUSTAY_ID
    -- merge the ITEMIDs so that the pivot applies to both metavision/carevue data
    , CASE
        WHEN l.ITEMID IN (723, 223900) THEN 723
        WHEN l.ITEMID IN (454, 223901) THEN 454
        WHEN l.ITEMID IN (184, 220739) THEN 184
        ELSE l.ITEMID END
      AS ITEMID

    -- convert the data into a number, reserving a value of 0 for ET/Trach
    -- Assuming VALUENUM is already a numeric type (e.g., double precision, numeric, integer)
    -- If VALUE is text and needs conversion, use CAST(l.VALUE AS numeric) or l.VALUE::numeric
    , CASE
        -- endotrach/vent is assigned a value of 0, later parsed specially
        WHEN l.ITEMID = 723 AND l.VALUE = '1.0 ET/Trach' THEN 0 -- carevue
        WHEN l.ITEMID = 223900 AND l.VALUE = 'No Response-ETT' THEN 0 -- metavision
        ELSE l.VALUENUM -- Ensure this column is numeric in the chartevents table
      END AS VALUENUM
    , l.CHARTTIME
    FROM chartevents l -- Assuming chartevents table exists in PostgreSQL

    -- get intime for charttime subselection
    INNER JOIN icustays b -- Assuming icustays table exists
      ON l.icustay_id = b.icustay_id

    -- Isolate the desired GCS variables
    WHERE l.ITEMID IN
    (
      -- 198 -- GCS (commented out in original)
      -- GCS components, CareVue
      184, 454, 723
      -- GCS components, Metavision
      , 223900, 223901, 220739
    )
    -- Only get data for the first 24 hours
    -- Replace DATETIME_ADD with PostgreSQL interval addition
    AND l.charttime BETWEEN b.intime AND (b.intime + INTERVAL '1 day')
    -- exclude rows marked as error
    AND (l.error IS NULL OR l.error = 0) -- Ensure error column type allows for = 0 comparison
  ) pvt
  GROUP BY pvt.ICUSTAY_ID, pvt.charttime
)
, gcs AS (
  SELECT b.*
  , b2.GCSVerbal AS GCSVerbalPrev
  , b2.GCSMotor AS GCSMotorPrev
  , b2.GCSEyes AS GCSEyesPrev
  -- Calculate GCS, factoring in special case when they are intubated and prev vals
  -- note that the coalesce are used to implement the following if:
  --  if current value exists, use it
  --  if previous value exists, use it
  --  otherwise, default to normal
  , CASE
      -- replace GCS during sedation with 15
      WHEN b.GCSVerbal = 0
        THEN 15
      WHEN b.GCSVerbal IS NULL AND b2.GCSVerbal = 0
        THEN 15
      -- if previously they were intub, but they aren't now, do not use previous GCS values
      WHEN b2.GCSVerbal = 0
        THEN
            COALESCE(b.GCSMotor, 6)
          + COALESCE(b.GCSVerbal, 5)
          + COALESCE(b.GCSEyes, 4)
      -- otherwise, add up score normally, imputing previous value if none available at current time
      ELSE
            COALESCE(b.GCSMotor, COALESCE(b2.GCSMotor, 6))
          + COALESCE(b.GCSVerbal, COALESCE(b2.GCSVerbal, 5))
          + COALESCE(b.GCSEyes, COALESCE(b2.GCSEyes, 4))
    END AS GCS

  FROM base b
  -- join to itself within 6 hours to get previous value
  LEFT JOIN base b2
    ON b.ICUSTAY_ID = b2.ICUSTAY_ID AND b.rn = b2.rn + 1
    -- Replace DATETIME_SUB with PostgreSQL interval subtraction
    AND b2.charttime > (b.charttime - INTERVAL '6 hour')
)
, gcs_final AS (
  SELECT gcs.*
  -- This sorts the data by GCS, so rn=1 is the the lowest GCS values to keep
  , ROW_NUMBER() OVER (PARTITION BY gcs.ICUSTAY_ID ORDER BY gcs.GCS ASC) AS IsMinGCS
  FROM gcs
)
SELECT ie.subject_id, ie.hadm_id, ie.icustay_id
-- The minimum GCS is determined by the above row partition, we only join if IsMinGCS=1
, gs.GCS AS mingcs
, COALESCE(gs.GCSMotor, gs.GCSMotorPrev) AS gcsmotor
, COALESCE(gs.GCSVerbal, gs.GCSVerbalPrev) AS gcsverbal
, COALESCE(gs.GCSEyes, gs.GCSEyesPrev) AS gcseyes
, gs.EndoTrachFlag AS endotrachflag

-- subselect down to the cohort of eligible patients
FROM icustays ie -- Assuming icustays table exists
LEFT JOIN gcs_final gs
  ON ie.icustay_id = gs.icustay_id AND gs.IsMinGCS = 1
ORDER BY ie.icustay_id;