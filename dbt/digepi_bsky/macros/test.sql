{% macro create_replica_views() %}

{% set sources = [] %}

{% for node in graph['sources'].values() %}
    {%- do sources.append(node.name) %}
{% endfor %}

{%- for source_name in sources %}
    {%- set query %}
    CREATE OR REPLACE VIEW {{ target.schema }}.{{ source_name }} AS
    SELECT *
    FROM {{ source('your_source_name', source_name) }}
    {% endset -%}
    
    {% do run_query(query) %}
{% endfor %}

{% endmacro %}
