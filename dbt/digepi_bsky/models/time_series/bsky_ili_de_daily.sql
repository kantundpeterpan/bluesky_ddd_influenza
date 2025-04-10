{% set ili_kws_sql = var('ili_kws_de') %}
{% set control_kws_sql = var('control_kws_de') %}
{% set lang = 'de'%}
{% set country_code = 'DEU'%}
{% set today = var('today')%}
WITH grippe_fr AS
    (
        SELECT period_start, SUM(post_count) as post_count
        FROM {{ source('bsky_housekeeping', 'housekeeping')}}
        WHERE query IN ({{",".join(ili_kws_sql)}}) and langs LIKE '%{{lang}}%'
        GROUP BY 1 
    ),
    rest_fr AS
    (
        SELECT period_start, SUM(post_count) as post_count
        FROM {{ source('bsky_housekeeping', 'housekeeping')}}
        WHERE query IN ({{",".join(control_kws_sql)}}) and langs LIKE '%{{lang}}%'
        GROUP BY period_start 
    ),

    total as (
    SELECT 
        date,
        SUM(COALESCE(g.post_count, 0)) as grippe_posts,
        SUM(COALESCE(r.post_count, 0)) as rest_posts,
    FROM {{ ref('daterange_days') }} d
    LEFT JOIN grippe_fr g
        on d.date = g.period_start
    LEFT JOIN rest_fr r
        on d.date = r.period_start
    GROUP BY 1
    ),

    final AS (
        SELECT
        i.date,
--        COALESCE(i.ili_case, 0) as ili_case,
--        COALESCE(i.ili_pop_cov, 0) as ili_pop_cov,
--        COALESCE(i.ari_case, 0) as ari_case,
--        COALESCE(i.ari_pop_cov, 0) as ari_pop_cov,
        COALESCE(i.ari_incidence, 0) as ari_incidence,
--        i.ari_incidence,
        t.grippe_posts,
        t.rest_posts,
        COALESCE(t.grippe_posts / (t.grippe_posts + t.rest_posts + 1), 0) as norm_post_count
        FROM total t
        RIGHT JOIN {{ ref('de_incidence_daily_upsampled_fluid') }} i
        ON t.date = i.date
        WHERE t.date > '2023-08-01'
        ORDER BY 1
    )

    SELECT
        *
    FROM final