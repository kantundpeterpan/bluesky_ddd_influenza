import json
import base64
import argparse
import yaml
import os

def encode_secrets(mapping_file_path, output_env_file):
    """
    Encodes values from multiple JSON files into base64 and writes them to an .env file
    with the "SECRET_" prefix for use with Kestra, using a multi-level mapping from a YAML configuration.

    The YAML configuration specifies file mappings and key mappings within those files.
    Values defined under `from_env` are read from environment variables.

    Args:
        mapping_file_path (str): The path to the YAML file containing the file and key mappings.
        output_env_file (str): The path to the output .env file.
    """

    try:
        with open(mapping_file_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Mapping file not found: {mapping_file_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in mapping file: {mapping_file_path}\n{e}")
        return

    if 'files' not in config and 'from_env' not in config:
        print("Error: Neither 'files' nor 'from_env' section not found in the YAML configuration.")
        return

    if 'files' in config:
        file_data = {}
        for file_header, file_path in config['files'].items():
            try:
                with open(file_path, 'r') as f:
                    file_data[file_header] = json.load(f)
            except FileNotFoundError:
                print(f"Error: File not found: {file_path}")
                return
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in file: {file_path}")
                return

    with open(output_env_file, 'w') as env_file:
        for section_header, mappings in config.items():
            if section_header == 'files':
                continue

            if section_header == 'from_env':
                for env_var, env_key in mappings.items():
                    value = os.environ.get(env_var)
                    if value is None:
                        print(f"Warning: Environment variable '{env_var}' not found, skipping.")
                        continue
                encoded_value = base64.b64encode(str(value).encode('utf-8')).decode('utf-8')
                env_file.write(f"SECRET_{env_key}={encoded_value}\n")
                continue

            if section_header not in file_data:
                print(f"Warning: File header '{section_header}' not found in file data, skipping mappings for this file.")
                continue
            data = file_data[section_header]

            for json_key, env_key in mappings.items():
                if json_key not in data:
                    print(f"Warning: Key '{json_key}' not found in JSON data for file header '{section_header}', skipping.")
                    continue
                value = data[json_key]
                encoded_value = base64.b64encode(str(value).encode('utf-8')).decode('utf-8')
                env_file.write(f"SECRET_{env_key}={encoded_value}\n")

    print(f"Successfully encoded secrets using mapping from {mapping_file_path} and saved to {output_env_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode secrets from JSON files to an .env file for Kestra using a YAML mapping.")
    parser.add_argument("mapping_file", help="Path to the YAML mapping file")
    parser.add_argument("-o", "--output", dest="output_file", default=".env_encoded", help="Path to the output .env file (default: .env_encoded)")

    args = parser.parse_args()
    encode_secrets(args.mapping_file, args.output_file)
