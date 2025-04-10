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