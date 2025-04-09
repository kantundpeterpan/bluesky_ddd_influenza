{{
  config(
    materialized = 'view',
    )
}}

SELECT
  DISTINCT *
FROM {{ ref('bsky_all_posts_fr') }}
WHERE uri IN (
    SELECT uri
    FROM  {{ source('bsky_posts_fr', 'llm_hints') }}
)