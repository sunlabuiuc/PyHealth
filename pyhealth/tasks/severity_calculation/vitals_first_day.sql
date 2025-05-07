-- THIS SCRIPT IS AUTOMATICALLY GENERATED. DO NOT EDIT IT DIRECTLY.
DROP TABLE IF EXISTS vitals_first_day;
CREATE TABLE vitals_first_day AS
-- This query pivots the vital signs for the first 24 hours of a patient's stay
-- Vital signs include heart rate, blood pressure, respiration rate, temperature, spo2, and glucose

SELECT pvt.subject_id, pvt.hadm_id, pvt.icustay_id

-- Easier names using standard conditional aggregation
, MIN(CASE WHEN VitalID = 1 THEN valuenum ELSE NULL END) AS heartrate_min
, MAX(CASE WHEN VitalID = 1 THEN valuenum ELSE NULL END) AS heartrate_max
, AVG(CASE WHEN VitalID = 1 THEN valuenum ELSE NULL END) AS heartrate_mean
, MIN(CASE WHEN VitalID = 2 THEN valuenum ELSE NULL END) AS sysbp_min
, MAX(CASE WHEN VitalID = 2 THEN valuenum ELSE NULL END) AS sysbp_max
, AVG(CASE WHEN VitalID = 2 THEN valuenum ELSE NULL END) AS sysbp_mean
, MIN(CASE WHEN VitalID = 3 THEN valuenum ELSE NULL END) AS diasbp_min
, MAX(CASE WHEN VitalID = 3 THEN valuenum ELSE NULL END) AS diasbp_max
, AVG(CASE WHEN VitalID = 3 THEN valuenum ELSE NULL END) AS diasbp_mean
, MIN(CASE WHEN VitalID = 4 THEN valuenum ELSE NULL END) AS meanbp_min
, MAX(CASE WHEN VitalID = 4 THEN valuenum ELSE NULL END) AS meanbp_max
, AVG(CASE WHEN VitalID = 4 THEN valuenum ELSE NULL END) AS meanbp_mean
, MIN(CASE WHEN VitalID = 5 THEN valuenum ELSE NULL END) AS resprate_min
, MAX(CASE WHEN VitalID = 5 THEN valuenum ELSE NULL END) AS resprate_max
, AVG(CASE WHEN VitalID = 5 THEN valuenum ELSE NULL END) AS resprate_mean
, MIN(CASE WHEN VitalID = 6 THEN valuenum ELSE NULL END) AS tempc_min
, MAX(CASE WHEN VitalID = 6 THEN valuenum ELSE NULL END) AS tempc_max
, AVG(CASE WHEN VitalID = 6 THEN valuenum ELSE NULL END) AS tempc_mean
, MIN(CASE WHEN VitalID = 7 THEN valuenum ELSE NULL END) AS spo2_min
, MAX(CASE WHEN VitalID = 7 THEN valuenum ELSE NULL END) AS spo2_max
, AVG(CASE WHEN VitalID = 7 THEN valuenum ELSE NULL END) AS spo2_mean
, MIN(CASE WHEN VitalID = 8 THEN valuenum ELSE NULL END) AS glucose_min
, MAX(CASE WHEN VitalID = 8 THEN valuenum ELSE NULL END) AS glucose_max
, AVG(CASE WHEN VitalID = 8 THEN valuenum ELSE NULL END) AS glucose_mean

FROM ( -- Start of the subquery that will be aliased as pvt
    -- Select the necessary columns from the inner subquery
    SELECT
        sub.subject_id, sub.hadm_id, sub.icustay_id, sub.vitalid, sub.valuenum
    FROM ( -- Start of the inner subquery to calculate vitalid and valuenum first
        SELECT
            ie.subject_id, ie.hadm_id, ie.icustay_id,
            -- Define vitalid based on itemid and value ranges
            CASE
                WHEN ce.itemid IN (211, 220045) AND ce.valuenum > 0 AND ce.valuenum < 300 THEN 1 -- HeartRate
                WHEN ce.itemid IN (51, 442, 455, 6701, 220179, 220050) AND ce.valuenum > 0 AND ce.valuenum < 400 THEN 2 -- SysBP
                WHEN ce.itemid IN (8368, 8440, 8441, 8555, 220180, 220051) AND ce.valuenum > 0 AND ce.valuenum < 300 THEN 3 -- DiasBP
                WHEN ce.itemid IN (456, 52, 6702, 443, 220052, 220181, 225312) AND ce.valuenum > 0 AND ce.valuenum < 300 THEN 4 -- MeanBP
                WHEN ce.itemid IN (615, 618, 220210, 224690) AND ce.valuenum > 0 AND ce.valuenum < 70 THEN 5 -- RespRate
                WHEN ce.itemid IN (223761, 678) AND ce.valuenum > 70 AND ce.valuenum < 120 THEN 6 -- TempF
                WHEN ce.itemid IN (223762, 676) AND ce.valuenum > 10 AND ce.valuenum < 50 THEN 6 -- TempC
                WHEN ce.itemid IN (646, 220277) AND ce.valuenum > 0 AND ce.valuenum <= 100 THEN 7 -- SpO2
                WHEN ce.itemid IN (807, 811, 1529, 3745, 3744, 225664, 220621, 226537) AND ce.valuenum > 0 THEN 8 -- Glucose
                ELSE NULL
            END AS vitalid,

            -- Convert F to C, ensuring floating point division
            CASE
                WHEN ce.itemid IN (223761, 678) THEN (ce.valuenum - 32.0) / 1.8
                ELSE ce.valuenum
            END AS valuenum

        FROM icustays ie -- Assuming icustays table exists
        LEFT JOIN chartevents ce -- Assuming chartevents table exists
            ON ie.icustay_id = ce.icustay_id
            -- Filter events to the first 24 hours of the ICU stay
            -- Replace BigQuery's DATETIME_ADD and DATETIME_DIFF with PostgreSQL equivalents
            AND ce.charttime > ie.intime -- Ensures event is after ICU admission
            AND ce.charttime <= (ie.intime + INTERVAL '1 day') -- Ensures event is within the first 24 hours
            -- exclude rows marked as error
            AND (ce.error IS NULL OR ce.error = 0) -- Ensure error column type allows for = 0 comparison
        WHERE ce.itemid IN (
            -- List all itemids needed for CASE statements above
            -- HEART RATE
            211, 220045,
            -- Systolic/diastolic BP
            51, 442, 455, 6701, 220179, 220050,
            8368, 8440, 8441, 8555, 220180, 220051,
            -- MEAN ARTERIAL PRESSURE
            456, 52, 6702, 443, 220052, 220181, 225312,
            -- RESPIRATORY RATE
            618, 615, 220210, 224690,
            -- SPO2, peripheral
            646, 220277,
            -- GLUCOSE, both lab and fingerstick
            807, 811, 1529, 3745, 3744, 225664, 220621, 226537,
            -- TEMPERATURE
            223762, 676, 223761, 678
        )
    ) sub -- End of inner subquery, aliased as sub
    -- Now filter using the vitalid calculated in the inner subquery 'sub'
    WHERE sub.vitalid IS NOT NULL
) pvt -- End of the subquery aliased as pvt
GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id
ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id;