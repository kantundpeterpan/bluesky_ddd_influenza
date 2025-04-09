{% set start = var('start')%}
{% set today = var('today')%}
-- Generate all dates within the given year range
    SELECT 
        day as date
    FROM 
        UNNEST(GENERATE_DATE_ARRAY(DATE('{{start}}'), DATE('{{today}}'))) AS day
