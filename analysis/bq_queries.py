
def get_post_count_ili_sql(ili_kws_sql, control_kws_sql, lang, country_code, dataset_id):    

    post_count_ili_sql = f"""WITH grippe_fr AS
    (
        SELECT DATE_TRUNC(period_start, ISOWEEK) as period_start, SUM(post_count) as post_count
        FROM `digepizcde.bsky_housekeeping.housekeeping` 
        WHERE query IN ({",".join(ili_kws_sql)}) and langs LIKE '%{lang}%'
        GROUP BY 1 
    ),
    rest_fr AS
    (
        SELECT DATE_TRUNC(period_start, ISOWEEK) as period_start, SUM(post_count) as post_count
        FROM `digepizcde.bsky_housekeeping.housekeeping` 
        WHERE query IN ({",".join(control_kws_sql)}) and langs LIKE '%{lang}%'
        GROUP BY period_start 
    ),

    year_dates AS (
    -- Generate all dates within the given year range
    SELECT 
        day
    FROM 
        UNNEST(GENERATE_DATE_ARRAY(DATE('2023-01-01'), DATE('2025-04-05'))) AS day
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
    FROM `digepizcde.case_data.who_{dataset_id}`
    WHERE country_code = '{country_code}'
    )

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
    """
    
    return post_count_ili_sql

def get_llm_ili_sql(ili_kws, lang, country_code):
    
    llm_union_snippet = "UNION ALL\n".join(
        [
            f"""SELECT 
                   uri,
                   record__created_at,
                   record__langs as langs
                FROM `digepizcde.bsky_posts.{kw}`\n
              """ for kw in ili_kws
        ]
        )
    
    llm_ili_sql=f"""WITH posts as(
        {llm_union_snippet}
        ),

        llm_filtered_posts AS (
            SELECT
            *,
            EXTRACT(DATE FROM DATE_TRUNC(p.record__created_at, ISOWEEK)) as iso_weekstartdate
            FROM posts p
            WHERE p.uri in (
                SELECT uri
                FROM `digepizcde.bsky_posts.llm_hints`
                WHERE ili_related=true
            ) AND langs LIKE '%{lang}%'
        ),


        year_dates AS (
        -- Generate all dates within the given year range
        SELECT 
            day
        FROM 
            UNNEST(GENERATE_DATE_ARRAY(DATE('2023-01-01'), DATE('2025-04-05'))) AS day
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
        
        llm_counts AS (
            SELECT
                iso_weekstartdate,
            COUNT(DISTINCT uri) as ili_rel_posts
            FROM llm_filtered_posts
            GROUP BY iso_weekstartdate
        ),
        
        total as (
            SELECT 
                DATE_TRUNC(d.period_start, ISOWEEK) as iso_weekstartdate,
                COALESCE(l.ili_rel_posts) as ili_rel_posts
            FROM dates d
            LEFT JOIN llm_counts l
                on d.period_start = l.iso_weekstartdate
        ),


        ili_fr AS (
        SELECT 
            iso_weekstartdate,
            ili_case, ili_pop_cov,
            ari_case, ari_pop_cov
        FROM `digepizcde.case_data.who_fluid`
        WHERE country_code = 'FRA' AND
            agegroup_code IN ('ALL', 'All')
        ),

        final AS (
            SELECT
            t.iso_weekstartdate as date,
            COALESCE(i.ili_case, 0) as ili_case,
            COALESCE(i.ili_pop_cov, 0) as ili_pop_cov,
            COALESCE(i.ari_case, 0) as ari_case,
            COALESCE(i.ari_pop_cov, 0) as ari_pop_cov,
            COALESCE(t.ili_rel_posts, 0) as ili_rel_posts,
            COALESCE(t.ili_rel_posts / (t.ili_rel_posts + fr.rest_posts), 0) as norm_post_count,
            fr.rest_posts
            FROM total t
            LEFT JOIN ili_fr i
            ON t.iso_weekstartdate = i.iso_weekstartdate
            LEFT JOIN `digepizcde.bsky_ili.bsky_ili_fr` fr
            ON t.iso_weekstartdate = fr.date
            WHERE t.iso_weekstartdate > '2023-08-01'
            ORDER BY 1
        )
        
        SELECT
            *,
            ari_case / (ari_pop_cov + 1) as ari_incidence,
            ili_case / (ili_pop_cov + 1) as ili_incidence
        FROM final
        """
        
    return llm_ili_sql
        
# def get_