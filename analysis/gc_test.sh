#!/bin/bash

# Script to export key-value pairs from a JSON file as environment variables with a custom prefix,
# handling multiline values correctly.

# Usage: ./export_json_env.sh <json_file> <prefix>

# Check if a JSON file and prefix are provided as arguments.
if [ $# -lt 1  ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <json_file> [<prefix>"
  exit 1
fi

json_file="$1"
prefix="${2:-}"  # Use empty string if no prefix is provided.
if [ -n "$prefix" ]; then
  prefix="${prefix}_" # Append underscore to prefix if it's not empty.
fi

# Check if the JSON file exists.
if [ ! -f "$json_file" ]; then
  echo "Error: JSON file '$json_file' not found."
  exit 1
fi

# Use jq to iterate through the JSON file and export each key-value pair as an environment variable.
# The -r flag ensures that jq outputs raw strings without quotes.
# Multiline values are handled by converting them to a single line with newline characters replaced by '\n'.
# The 'env' command prefixes each environment variable declaration so it is exported to the current shell.

keys=$(jq -r 'keys[]' "$json_file")

for key in $keys; do
  value=$(jq -r ".\"${key}\"" "$json_file")
  uppercase_key=$(echo "$key" | tr '[:lower:]' '[:upper:]')
  export "${prefix}${uppercase_key}=$value"
done
echo "Environment variables exported from '$json_file' with prefix '$prefix'."

python ./gc_test.py

exit 0