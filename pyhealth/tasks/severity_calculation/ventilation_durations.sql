-- THIS SCRIPT IS CONVERTED FOR POSTGRESQL.
DROP TABLE IF EXISTS ventilation_durations;
CREATE TABLE ventilation_durations AS
-- This query extracts the duration of mechanical ventilation
-- The main goal of the query is to aggregate sequential ventilator settings
-- into single mechanical ventilation "events". The start and end time of these
-- events can then be used for various purposes: calculating the total duration
-- of mechanical ventilation, cross-checking values (e.g. PaO2:FiO2 on vent), etc

-- The query's logic is roughly:
--    1) The presence of a mechanical ventilation setting starts a new ventilation event
--    2) Any instance of a setting in the next 8 hours continues the event
--    3) Certain elements end the current ventilation event
--       a) documented extubation ends the current ventilation
--       b) initiation of non-invasive vent and/or oxygen ends the current vent

-- See the ventilation_classification.sql query for step 1 of the above.
-- This query has the logic for converting events into durations.

-- Step 1: Lag charttime for mechanical ventilation events
WITH vd0 AS (
  SELECT
    icustay_id,
    -- this carries over the previous charttime which had a mechanical ventilation event
    CASE
      WHEN MechVent = 1 THEN
        LAG(charttime, 1) OVER (PARTITION BY icustay_id, MechVent ORDER BY charttime)
      ELSE
        NULL
    END AS charttime_lag,
    charttime,
    MechVent,
    OxygenTherapy,
    Extubated,
    SelfExtubated
  FROM ventilation_classification -- Ensure this table exists in your PostgreSQL database
),

-- Step 2: Calculate time differences and identify new ventilation events
vd1 AS (
  SELECT
    icustay_id,
    charttime_lag,
    charttime,
    MechVent,
    OxygenTherapy,
    Extubated,
    SelfExtubated,

    -- if this is a mechanical ventilation event, we calculate the time since the last event
    CASE
      -- if the current observation indicates mechanical ventilation is present
      -- calculate the time since the last vent event in hours
      WHEN MechVent = 1 THEN
        -- PostgreSQL uses EXTRACT(EPOCH FROM ...) to get seconds, then divide by 3600 for hours
        EXTRACT(EPOCH FROM (charttime - charttime_lag)) / 3600.0
      ELSE
        NULL
    END AS ventduration,

    LAG(Extubated, 1) OVER (
      PARTITION BY icustay_id, CASE WHEN MechVent = 1 OR Extubated = 1 THEN 1 ELSE 0 END
      ORDER BY charttime
    ) AS ExtubatedLag,

    -- now we determine if the current mech vent event is a "new", i.e. they've just been intubated
    CASE
      -- if the previous row was an extubation, this is a new ventilation event
      WHEN LAG(Extubated, 1) OVER (
            PARTITION BY icustay_id, CASE WHEN MechVent = 1 OR Extubated = 1 THEN 1 ELSE 0 END
            ORDER BY charttime
           ) = 1 THEN 1
      -- if patient has initiated oxygen therapy, and is not currently vented, start a newvent
      WHEN MechVent = 0 AND OxygenTherapy = 1 THEN 1
      -- if there is less than 8 hours between vent settings, we do not treat this as a new ventilation event
      -- PostgreSQL uses timestamp + INTERVAL 'X unit' syntax
      WHEN charttime > (charttime_lag + INTERVAL '8 hours') THEN 1
      ELSE 0
    END AS newvent
  FROM vd0
),

-- Step 3: Assign a unique number to each ventilation period
vd2 AS (
  SELECT
    vd1.*,
    -- create a cumulative sum of the instances of new ventilation
    -- this results in a monotonic integer assigned to each instance of ventilation
    CASE
      WHEN MechVent = 1 OR Extubated = 1 THEN
        SUM(newvent) OVER (PARTITION BY icustay_id ORDER BY charttime)
      ELSE
        NULL
    END AS ventnum
  FROM vd1
)

-- Step 4: Create the final durations for each mechanical ventilation instance
SELECT
  icustay_id,
  -- regenerate ventnum so it's sequential within the icustay_id
  ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY ventnum) AS ventnum,
  MIN(charttime) AS starttime,
  MAX(charttime) AS endtime,
  -- Calculate duration in hours using PostgreSQL syntax
  EXTRACT(EPOCH FROM (MAX(charttime) - MIN(charttime))) / 3600.0 AS duration_hours
FROM vd2
WHERE ventnum IS NOT NULL -- Filter out rows that are not part of a ventilation period
GROUP BY icustay_id, vd2.ventnum
HAVING
  -- Ensure the start and end times are different
  MIN(charttime) != MAX(charttime)
  -- Patient had to be mechanically ventilated at least once during the period
  -- This excludes situations like NIV/oxygen before intubation
  AND MAX(MechVent) = 1
ORDER BY icustay_id, ventnum;
