WITH ili_incidence_weekstart AS (
    SELECT
        iso_weekstartdate,
        ili_incidence
    FROM {{ ref('fr_incidence_weekstart_fluid') }}
),

ili_incidence_days_null AS (
    SELECT
        dr.date,
        i.ili_incidence
    FROM {{ ref('daterange_days') }} dr
    LEFT JOIN ili_incidence_weekstart i
    ON dr.date = i.iso_weekstartdate
)


SELECT *
FROM GAP_FILL(
  TABLE ili_incidence_days_null,
  ts_column => 'date',
  bucket_width => INTERVAL 1 DAY,
  value_columns => [
    ('ili_incidence', 'linear')
  ]
)
ORDER BY date