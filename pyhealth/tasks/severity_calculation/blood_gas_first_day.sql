-- THIS SCRIPT IS AUTOMATICALLY GENERATED. DO NOT EDIT IT DIRECTLY.
-- Converted for PostgreSQL compatibility

DROP TABLE IF EXISTS blood_gas_first_day;
CREATE TABLE blood_gas_first_day AS

-- The aim of this query is to pivot entries related to blood gases and
-- chemistry values which were found in LABEVENTS

-- things to check:
--  when a mixed venous/arterial blood sample are taken at the same time, is the store time different?

WITH pvt AS (
    -- begin query that extracts the data
    SELECT
        ie.subject_id,
        ie.hadm_id,
        ie.icustay_id,
        -- here we assign labels to ITEMIDs
        -- this also fuses together multiple ITEMIDs containing the same data
        CASE
            WHEN itemid = 50800 THEN 'SPECIMEN'
            WHEN itemid = 50801 THEN 'AADO2'
            WHEN itemid = 50802 THEN 'BASEEXCESS'
            WHEN itemid = 50803 THEN 'BICARBONATE'
            WHEN itemid = 50804 THEN 'TOTALCO2'
            WHEN itemid = 50805 THEN 'CARBOXYHEMOGLOBIN'
            WHEN itemid = 50806 THEN 'CHLORIDE'
            WHEN itemid = 50808 THEN 'CALCIUM' -- Note: ITEMID 50807 Calcium measured? Omitted in original BQ query.
            WHEN itemid = 50809 THEN 'GLUCOSE'
            WHEN itemid = 50810 THEN 'HEMATOCRIT'
            WHEN itemid = 50811 THEN 'HEMOGLOBIN'
            WHEN itemid = 50812 THEN 'INTUBATED'
            WHEN itemid = 50813 THEN 'LACTATE'
            WHEN itemid = 50814 THEN 'METHEMOGLOBIN'
            WHEN itemid = 50815 THEN 'O2FLOW'
            WHEN itemid = 50816 THEN 'FIO2'
            WHEN itemid = 50817 THEN 'SO2' -- OXYGENSATURATION
            WHEN itemid = 50818 THEN 'PCO2'
            WHEN itemid = 50819 THEN 'PEEP'
            WHEN itemid = 50820 THEN 'PH'
            WHEN itemid = 50821 THEN 'PO2'
            WHEN itemid = 50822 THEN 'POTASSIUM'
            WHEN itemid = 50823 THEN 'REQUIREDO2'
            WHEN itemid = 50824 THEN 'SODIUM'
            WHEN itemid = 50825 THEN 'TEMPERATURE'
            WHEN itemid = 50826 THEN 'TIDALVOLUME'
            WHEN itemid = 50827 THEN 'VENTILATIONRATE'
            WHEN itemid = 50828 THEN 'VENTILATOR'
            ELSE NULL
        END AS label,
        le.charttime, -- Use charttime from labevents
        le.value,     -- Original textual value
        -- add in some sanity checks on the values
        CASE
            WHEN le.valuenum <= 0 AND le.itemid != 50802 THEN NULL -- allow negative baseexcess
            WHEN le.itemid = 50810 AND le.valuenum > 100 THEN NULL -- hematocrit
            -- ensure FiO2 is a valid number between ~21-100
            -- mistakes are rare (<100 obs out of ~100,000)
            -- there are 862 obs of valuenum == 20 - some people round down!
            -- rather than risk imputing garbage data for FiO2, we simply NULL invalid values
            WHEN le.itemid = 50816 AND le.valuenum < 20 THEN NULL -- FiO2 % can be 20, maybe keep? Original query nulls < 20.
            WHEN le.itemid = 50816 AND le.valuenum > 100 THEN NULL -- FiO2 %
            WHEN le.itemid = 50817 AND le.valuenum > 100 THEN NULL -- O2 sat %
            WHEN le.itemid = 50815 AND le.valuenum > 70 THEN NULL -- O2 flow L/min
            WHEN le.itemid = 50821 AND le.valuenum > 800 THEN NULL -- PO2 mmHg
            -- conservative upper limit for PO2
            ELSE le.valuenum
        END AS valuenum

    FROM icustays ie -- This table needs to exist in your PostgreSQL database
    LEFT JOIN labevents le -- This table also needs to exist
        ON le.subject_id = ie.subject_id AND le.hadm_id = ie.hadm_id
        -- Select events within the first 24 hours of ICU admission (+/- 6 hours)
        -- Use standard PostgreSQL interval arithmetic:
        AND le.charttime BETWEEN (ie.intime - INTERVAL '6 HOUR') AND (ie.intime + INTERVAL '1 DAY')
        AND le.itemid IN (
            -- blood gases / chemistry / ventilation settings specified in labevents
            50800, 50801, 50802, 50803, 50804, 50805, 50806, 50808, -- 50807 Calcium Lvl missing in original query
            50809, 50810, 50811, 50812, 50813, 50814, 50815, 50816, 50817, 50818, 50819,
            50820, 50821, 50822, 50823, 50824, 50825, 50826, 50827, 50828
            -- 51545 is base excess (art). Included ITEMID 50802 instead which is Base Excess.
            -- If 51545 is also needed, add it to the list and potentially adjust the CASE statement for label='BASEEXCESS'
        )
    WHERE le.valuenum IS NOT NULL OR le.itemid = 50800 -- Keep specimen type even if valuenum is null
)
-- Pivot the data
SELECT
    pvt.subject_id,
    pvt.hadm_id,
    pvt.icustay_id,
    pvt.charttime,
    MAX(CASE WHEN label = 'SPECIMEN' THEN value ELSE NULL END) AS specimen, -- Text value
    MAX(CASE WHEN label = 'AADO2' THEN valuenum ELSE NULL END) AS aado2,
    MAX(CASE WHEN label = 'BASEEXCESS' THEN valuenum ELSE NULL END) AS baseexcess,
    MAX(CASE WHEN label = 'BICARBONATE' THEN valuenum ELSE NULL END) AS bicarbonate,
    MAX(CASE WHEN label = 'TOTALCO2' THEN valuenum ELSE NULL END) AS totalco2,
    MAX(CASE WHEN label = 'CARBOXYHEMOGLOBIN' THEN valuenum ELSE NULL END) AS carboxyhemoglobin,
    MAX(CASE WHEN label = 'CHLORIDE' THEN valuenum ELSE NULL END) AS chloride,
    MAX(CASE WHEN label = 'CALCIUM' THEN valuenum ELSE NULL END) AS calcium,
    MAX(CASE WHEN label = 'GLUCOSE' THEN valuenum ELSE NULL END) AS glucose,
    MAX(CASE WHEN label = 'HEMATOCRIT' THEN valuenum ELSE NULL END) AS hematocrit,
    MAX(CASE WHEN label = 'HEMOGLOBIN' THEN valuenum ELSE NULL END) AS hemoglobin,
    MAX(CASE WHEN label = 'INTUBATED' THEN valuenum ELSE NULL END) AS intubated,
    MAX(CASE WHEN label = 'LACTATE' THEN valuenum ELSE NULL END) AS lactate,
    MAX(CASE WHEN label = 'METHEMOGLOBIN' THEN valuenum ELSE NULL END) AS methemoglobin,
    MAX(CASE WHEN label = 'O2FLOW' THEN valuenum ELSE NULL END) AS o2flow,
    MAX(CASE WHEN label = 'FIO2' THEN valuenum ELSE NULL END) AS fio2,
    MAX(CASE WHEN label = 'SO2' THEN valuenum ELSE NULL END) AS so2, -- OXYGENSATURATION
    MAX(CASE WHEN label = 'PCO2' THEN valuenum ELSE NULL END) AS pco2,
    MAX(CASE WHEN label = 'PEEP' THEN valuenum ELSE NULL END) AS peep,
    MAX(CASE WHEN label = 'PH' THEN valuenum ELSE NULL END) AS ph,
    MAX(CASE WHEN label = 'PO2' THEN valuenum ELSE NULL END) AS po2,
    MAX(CASE WHEN label = 'POTASSIUM' THEN valuenum ELSE NULL END) AS potassium,
    MAX(CASE WHEN label = 'REQUIREDO2' THEN valuenum ELSE NULL END) AS requiredo2,
    MAX(CASE WHEN label = 'SODIUM' THEN valuenum ELSE NULL END) AS sodium,
    MAX(CASE WHEN label = 'TEMPERATURE' THEN valuenum ELSE NULL END) AS temperature,
    MAX(CASE WHEN label = 'TIDALVOLUME' THEN valuenum ELSE NULL END) AS tidalvolume,
    MAX(CASE WHEN label = 'VENTILATIONRATE' THEN valuenum ELSE NULL END) AS ventilationrate,
    MAX(CASE WHEN label = 'VENTILATOR' THEN valuenum ELSE NULL END) AS ventilator
FROM pvt
GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime
ORDER BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id, pvt.charttime;