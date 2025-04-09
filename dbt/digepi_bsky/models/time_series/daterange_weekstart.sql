WITH iso_weeks AS (
    -- Truncate each date to the start of its ISO week
    SELECT 
        DISTINCT DATE_TRUNC(date, ISOWEEK) AS iso_week_start_date
    FROM 
        {{ ref('daterange_days') }}
    ),

dates AS (
    -- Return the unique ISO week start dates
    SELECT 
    iso_week_start_date as iso_weekstartdate
    FROM 
    iso_weeks
    ORDER BY 
    iso_week_start_date
)

SELECT
 *
FROM dates