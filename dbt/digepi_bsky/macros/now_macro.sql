-- macros/now_macro.sql
{% macro now() %}
  {{ return(modules.datetime.datetime.now()) }}
{% endmacro %}
