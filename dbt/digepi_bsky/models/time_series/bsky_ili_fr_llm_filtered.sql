{{
  config(
    materialized = 'table',
    )
}}
{%  set lang = 'fr' %}
WITH llm_filtered_posts AS (
    {# filter llm tagged posts #}
            SELECT
            *,
            EXTRACT(DATE FROM DATE_TRUNC(p.record__created_at, ISOWEEK)) as iso_weekstartdate
            FROM {{ ref('bsky_all_posts_fr') }} p
            WHERE p.uri in (
                SELECT uri
                FROM `digepizcde.bsky_posts.llm_hints`
                WHERE ili_related=true
            ) {# already filtered for language#}
        ),
    {# count per week #}

    llm_filt_per_week AS (
        SELECT
            iso_weekstartdate,
            COUNT(DISTINCT uri) as llm_post_count_na
        FROM llm_filtered_posts
        GROUP BY 1
    )

    {# join with time post count time seriess #}

    SELECT
        *,
        COALESCE(l.llm_post_count_na, 0) as llm_post_count
    FROM llm_filt_per_week l
    RIGHT JOIN {{ ref('bsky_ili_fr') }} ts
    ON l.iso_weekstartdate = ts.date