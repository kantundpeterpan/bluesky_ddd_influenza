{% set ili_kws_sql = var('ili_kws_de') %}
{% set control_kws_sql = var('control_kws_de') %}
{% set lang = 'de'%}
{% set country_code = 'DEU'%}
{% set today = var('today')%}
WITH grippe_fr AS
    (
        SELECT DATE_TRUNC(period_start, ISOWEEK) as period_start, SUM(post_count) as post_count
        FROM {{ source('bsky_housekeeping', 'housekeeping')}}
        WHERE query IN ({{",".join(ili_kws_sql)}}) and langs LIKE '%{{lang}}%'
        GROUP BY 1 
    ),
    rest_fr AS
    (
        SELECT DATE_TRUNC(period_start, ISOWEEK) as period_start, SUM(post_count) as post_count
        FROM {{ source('bsky_housekeeping', 'housekeeping')}}
        WHERE query IN ({{",".join(control_kws_sql)}}) and langs LIKE '%{{lang}}%'
        GROUP BY period_start 
    ),

    year_dates AS (
    -- Generate all dates within the given year range
    SELECT 
        day
    FROM 
        UNNEST(GENERATE_DATE_ARRAY(DATE('2023-01-01'), DATE('{{today}}'))) AS day
    ),
    iso_weeks AS (
    -- Truncate each date to the start of its ISO week
    SELECT 
        DISTINCT DATE_TRUNC(day, ISOWEEK) AS iso_week_start_date
    FROM 
        year_dates
    ),

    dates AS (
    -- Return the unique ISO week start dates
    SELECT 
    iso_week_start_date as period_start
    FROM 
    iso_weeks
    ORDER BY 
    iso_week_start_date
    ),

    total as (
    SELECT 
        DATE_TRUNC(d.period_start, ISOWEEK) as iso_weekstartdate,
        SUM(COALESCE(g.post_count, 0)) as grippe_posts,
        SUM(COALESCE(r.post_count, 0)) as rest_posts,
    FROM dates d
    LEFT JOIN grippe_fr g
        on d.period_start = g.period_start
    LEFT JOIN rest_fr r
        on d.period_start = r.period_start
    GROUP BY 1
    ),

    ili_fr AS (
    SELECT 
        iso_weekstartdate,
        ili_case, ili_pop_cov,
        ari_case, ari_pop_cov
    FROM `digepizcde.case_data.who_fluid`
    WHERE country_code = '{{country_code}}' AND
          agegroup_code IN ('ALL', 'All')
    ),

    final AS (
        SELECT
        t.iso_weekstartdate as date,
        COALESCE(i.ili_case, 0) as ili_case,
        COALESCE(i.ili_pop_cov, 0) as ili_pop_cov,
        COALESCE(i.ari_case, 0) as ari_case,
        COALESCE(i.ari_pop_cov, 0) as ari_pop_cov,
        t.grippe_posts,
        t.rest_posts,
        COALESCE(t.grippe_posts / (t.grippe_posts + t.rest_posts), 0) as norm_post_count
        FROM total t
        LEFT JOIN ili_fr i
        ON t.iso_weekstartdate = i.iso_weekstartdate
        WHERE t.iso_weekstartdate > '2023-08-01'
        ORDER BY 1
    )

    SELECT
        *,
        ari_case / (ari_pop_cov + 1) as ari_incidence,
        ili_case / (ili_pop_cov + 1) as ili_incidence
    FROM final