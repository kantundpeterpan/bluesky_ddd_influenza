/*
  Union all of the bsky_post_fr tables 
*/

{{
  config(
    materialized = 'table',
    )
}}

{% set sources = [] -%}
{% for node in graph.sources.values() -%}
  {%- if  node.source_name == 'bsky_posts_fr' -%}
    {%- do sources.append(source(node.source_name, node.name)) -%}
  {%- endif -%}
{%- endfor %}

select uri, record__created_at, record__text from (
  {%- for source in sources %}
    select uri, record__created_at, record__text 
    from {{ source }}
    where record__langs LIKE '%fr%' 
    {% if not loop.last %} union all {% endif %}
  {% endfor %}
)