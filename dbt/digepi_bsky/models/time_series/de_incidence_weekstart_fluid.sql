/*
SELECT
  dr.iso_weekstartdate,
  COALESCE(ili_case,0) as ili_case, COALESCE(ili_pop_cov, 0) as ili_pop_cov,
  COALESCE(ari_case, 0) as ari_case, ari_pop_cov,
  COALESCE(ili_case,0) / (COALESCE(ili_pop_cov,0)+1) as ili_incidence,
  COALESCE(ari_case,0) / (COALESCE(ari_pop_cov,0)+1) as ari_incidence
FROM {{ ref('daterange_weekstart') }} dr
LEFT JOIN (
  SELECT *
  FROM {{ source('who_ili', 'who_fluid') }}
  WHERE country_code = "DEU" AND
        agegroup_code IN ("All", "ALL")
) i
ON dr.iso_weekstartdate = i.iso_weekstartdate
*/

WITH weekly_null AS (
SELECT
  dr.iso_weekstartdate,
  CASE WHEN i.ili_case IS NULL THEN NULL ELSE i.ili_case END as ili_case,
  CASE WHEN i.ili_pop_cov IS NULL THEN NULL ELSE i.ili_pop_cov END as ili_pop_cov,
  CASE WHEN i.ari_case IS NULL THEN NULL ELSE i.ari_case END as ari_case,
  i.ari_pop_cov,
  CASE
    WHEN i.ili_case IS NULL OR i.ili_pop_cov IS NULL THEN NULL
    ELSE i.ili_case / (i.ili_pop_cov + 1)
  END as ili_incidence,
  CASE
    WHEN i.ari_case IS NULL OR i.ari_pop_cov IS NULL THEN NULL
    ELSE i.ari_case / (i.ari_pop_cov + 1)
  END as ari_incidence
FROM {{ ref('daterange_weekstart') }} dr
LEFT JOIN (
  SELECT *
  FROM {{ source('who_ili', 'who_fluid') }}
  WHERE country_code = "DEU" AND
        agegroup_code IN ("All", "ALL")
) i
ON dr.iso_weekstartdate = i.iso_weekstartdate
)

SELECT w.iso_weekstartdate, gf.ili_incidence, ili_case, ili_pop_cov, ari_case, ari_pop_cov, ari_incidence
FROM GAP_FILL(
  TABLE weekly_null,
  ts_column => 'iso_weekstartdate',
  bucket_width => INTERVAL 1 DAY,
  value_columns => [
    ('ili_incidence', 'linear')
  ]
) gf
RIGHT JOIN weekly_null w
ON w.iso_weekstartdate = gf.iso_weekstartdate
ORDER BY iso_weekstartdate