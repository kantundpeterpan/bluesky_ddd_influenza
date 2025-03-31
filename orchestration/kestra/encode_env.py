import json
import base64
import argparse
import yaml

def encode_secrets(json_file_path, mapping_file_path, output_env_file):
    """
    Encodes values from a JSON file into base64 and writes them to an .env file
    with the "SECRET_" prefix for use with Kestra, using a mapping from a YAML file.

    Args:
        json_file_path (str): The path to the JSON file containing the data.
        mapping_file_path (str): The path to the YAML file containing the key mappings.
        output_env_file (str): The path to the output .env file.
    """

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {json_file_path}")
        return

    try:
        with open(mapping_file_path, 'r') as f:
            mappings = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Mapping file not found: {mapping_file_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in mapping file: {mapping_file_path}\n{e}")
        return

    with open(output_env_file, 'w') as env_file:
        for json_key, env_key in mappings.items():
            if json_key not in data:
                print(f"Warning: Key '{json_key}' not found in JSON data, skipping.")
                continue
            value = data[json_key]
            encoded_value = base64.b64encode(str(value).encode('utf-8')).decode('utf-8')
            env_file.write(f"SECRET_{env_key}={encoded_value}\n")
    print(f"Successfully encoded secrets from {json_file_path} using mapping from {mapping_file_path} and saved to {output_env_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode secrets from a JSON file to an .env file for Kestra using a YAML mapping.")
    parser.add_argument("json_file", help="Path to the JSON file")
    parser.add_argument("mapping_file", help="Path to the YAML mapping file")
    parser.add_argument("-o", "--output", dest="output_file", default=".env_encoded", help="Path to the output .env file (default: .env_encoded)")

    args = parser.parse_args()
    encode_secrets(args.json_file, args.mapping_file, args.output_file)
