-- THIS SCRIPT HAS BEEN CONVERTED FOR POSTGRESQL.
DROP TABLE IF EXISTS echo_data;
CREATE TABLE echo_data AS
-- This code extracts structured data from echocardiographies
-- You can join it to the text notes using ROW_ID
-- Just note that ROW_ID might differ across versions/imports of MIMIC-III.

SELECT
    ne.ROW_ID,
    ne.subject_id,
    ne.hadm_id,
    ne.chartdate,

    -- charttime is often null for echoes in the original table.
    -- However, the time is usually available in the echo text, e.g., 'Date/Time: [**YYYY-MM-DD**] at HH:MM'
    -- We can attempt to impute it and re-create charttime using TO_TIMESTAMP.
    -- If the pattern doesn't match, the result will be NULL.
    TO_TIMESTAMP(
        TO_CHAR(ne.chartdate, 'YYYY-MM-DD') || -- Get the date part
        SUBSTRING(ne.text FROM 'Date/Time: .+? at ([0-9]{1,2}:[0-9]{2})') || -- Extract HH:MI
        ':00', -- Add seconds
        'YYYY-MM-DDHH24:MI:SS' -- Specify the combined format
    ) AS charttime,

    -- Explanation of substring:
    --  'Indication: ' - matched verbatim
    --  '(.*?)' - capture any character non-greedily
    --  '\n' - until the end of the line
    -- substring only returns the item in parentheses ()s
    SUBSTRING(ne.text FROM 'Indication: (.*?)\n') AS Indication,

    -- Extract numeric values, casting them to NUMERIC.
    -- Handle potential non-matches which will result in NULL.
    CAST(SUBSTRING(ne.text FROM 'Height: \(in\) ([0-9]+)') AS NUMERIC) AS Height,
    CAST(SUBSTRING(ne.text FROM 'Weight \(lb\): ([0-9]+)\n') AS NUMERIC) AS Weight,
    -- Note: Original regex captured only integer BSA. Consider '([0-9.]+)' if decimals are expected.
    CAST(SUBSTRING(ne.text FROM 'BSA \(m2\): ([0-9.]+) m2\n') AS NUMERIC) AS BSA,
    SUBSTRING(ne.text FROM 'BP \(mm Hg\): (.+)\n') AS BP, -- Full BP string like '120/80'
    CAST(SUBSTRING(ne.text FROM 'BP \(mm Hg\): ([0-9]+)/[0-9]+\n') AS NUMERIC) AS BPSys, -- Systolic part
    CAST(SUBSTRING(ne.text FROM 'BP \(mm Hg\): [0-9]+/([0-9]+)\n') AS NUMERIC) AS BPDias, -- Diastolic part
    CAST(SUBSTRING(ne.text FROM 'HR \(bpm\): ([0-9]+)\n') AS NUMERIC) AS HR,

    SUBSTRING(ne.text FROM 'Status: (.*?)\n') AS Status,
    SUBSTRING(ne.text FROM 'Test: (.*?)\n') AS Test,
    SUBSTRING(ne.text FROM 'Doppler: (.*?)\n') AS Doppler,
    SUBSTRING(ne.text FROM 'Contrast: (.*?)\n') AS Contrast,
    SUBSTRING(ne.text FROM 'Technical Quality: (.*?)\n') AS TechnicalQuality

FROM
    noteevents ne
WHERE
    ne.category = 'Echo';

-- Optional: Add indexes for faster lookups
-- CREATE INDEX idx_echo_data_row_id ON echo_data (ROW_ID);
-- CREATE INDEX idx_echo_data_subject_id ON echo_data (subject_id);
-- CREATE INDEX idx_echo_data_hadm_id ON echo_data (hadm_id);