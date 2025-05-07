-- THIS SCRIPT IS CONVERTED FOR POSTGRESQL.
DROP TABLE IF EXISTS ventilation_classification;
CREATE TABLE ventilation_classification AS
-- This query identifies events related to mechanical ventilation, oxygen therapy, and extubation.
-- It classifies events from chartevents and procedureevents_mv based on itemid and value.

-- Part 1: Process data from chartevents
SELECT
  icustay_id,
  charttime,
  -- case statement determining whether it is an instance of mech vent
  MAX(
    CASE
      WHEN itemid IS NULL OR value IS NULL THEN 0 -- can't have null values
      WHEN itemid = 720 AND value <> 'Other/Remarks' THEN 1 -- VentTypeRecorded (using <> for not equal)
      WHEN itemid = 223848 AND value <> 'Other' THEN 1
      WHEN itemid = 223849 THEN 1 -- ventilator mode
      WHEN itemid = 467 AND value = 'Ventilator' THEN 1 -- O2 delivery device == ventilator
      WHEN itemid IN (
        -- List of itemids indicating mechanical ventilation settings
        445, 448, 449, 450, 1340, 1486, 1600, 224687, -- minute volume
        639, 654, 681, 682, 683, 684, 224685, 224684, 224686, -- tidal volume
        218, 436, 535, 444, 459, 224697, 224695, 224696, 224746, 224747, -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
        221, 1, 1211, 1655, 2000, 226873, 224738, 224419, 224750, 227187, -- Insp pressure
        543, -- PlateauPressure
        5865, 5866, 224707, 224709, 224705, 224706, -- APRV pressure
        60, 437, 505, 506, 686, 220339, 224700, -- PEEP
        3459, -- high pressure relief
        501, 502, 503, 224702, -- PCV
        223, 667, 668, 669, 670, 671, 672, -- TCPCV
        224701 -- PSVlevel
      ) THEN 1
      ELSE 0
    END
  ) AS MechVent,
  MAX(
    CASE
      -- initiation of oxygen therapy indicates the ventilation has ended
      WHEN itemid = 226732 AND value IN (
        'Nasal cannula', 'Face tent', 'Aerosol-cool', 'Trach mask ', 'High flow neb',
        'Non-rebreather', 'Venti mask ', 'Medium conc mask ', 'T-piece',
        'High flow nasal cannula', 'Ultrasonic neb', 'Vapomist'
      ) THEN 1
      WHEN itemid = 467 AND value IN (
        'Cannula', 'Nasal Cannula', 'Face Tent', 'Aerosol-Cool', 'Trach Mask',
        'Hi Flow Neb', 'Non-Rebreather', 'Venti Mask', 'Medium Conc Mask',
        'Vapotherm', 'T-Piece', 'Hood', 'Hut', 'TranstrachealCat', 'Heated Neb',
        'Ultrasonic Neb'
        -- Note: 'None' might be relevant depending on interpretation, excluded here as per original script
      ) THEN 1
      ELSE 0
    END
  ) AS OxygenTherapy,
  MAX(
    CASE
      WHEN itemid IS NULL OR value IS NULL THEN 0
      -- extubated indicates ventilation event has ended
      WHEN itemid = 640 AND value = 'Extubated' THEN 1
      WHEN itemid = 640 AND value = 'Self Extubation' THEN 1
      ELSE 0
    END
  ) AS Extubated,
  MAX(
    CASE
      WHEN itemid IS NULL OR value IS NULL THEN 0
      WHEN itemid = 640 AND value = 'Self Extubation' THEN 1
      ELSE 0
    END
  ) AS SelfExtubated
FROM chartevents ce -- Ensure this table exists in your PostgreSQL database
WHERE ce.value IS NOT NULL
  -- exclude rows marked as error (PostgreSQL uses standard NULL comparison)
  AND (ce.error IS NULL OR ce.error <> 1) -- Check if error is NULL or not equal to 1
  AND itemid IN (
    -- Combined list of relevant itemids for ventilation, extubation, and oxygen therapy
    -- Ventilation settings:
    720, 223849, 223848, 445, 448, 449, 450, 1340, 1486, 1600, 224687,
    639, 654, 681, 682, 683, 684, 224685, 224684, 224686, 218, 436, 535,
    444, 224697, 224695, 224696, 224746, 224747, 221, 1, 1211, 1655, 2000,
    226873, 224738, 224419, 224750, 227187, 543, 5865, 5866, 224707, 224709,
    224705, 224706, 60, 437, 505, 506, 686, 220339, 224700, 3459, 501, 502,
    503, 224702, 223, 667, 668, 669, 670, 671, 672, 224701,
    -- Extubation setting:
    640,
    -- Oxygen/NIV settings:
    468, 469, 470, 471, 227287, 226732, 223834,
    -- Setting used in both O2 and Vent:
    467
  )
GROUP BY icustay_id, charttime

UNION -- UNION DISTINCT is the default in PostgreSQL, but UNION is sufficient and standard

-- Part 2: Add extubation flags from procedureevents_mv
SELECT
  icustay_id,
  starttime AS charttime,
  0 AS MechVent,
  0 AS OxygenTherapy,
  1 AS Extubated,
  CASE WHEN itemid = 225468 THEN 1 ELSE 0 END AS SelfExtubated
FROM procedureevents_mv -- Ensure this table exists in your PostgreSQL database
WHERE itemid IN (
  227194, -- "Extubation"
  225468, -- "Unplanned Extubation (patient-initiated)"
  225477 -- "Unplanned Extubation (non-patient initiated)"
);

-- Optional: Add an index for faster lookups later
-- CREATE INDEX ON ventilation_classification (icustay_id, charttime);
