import os
from google.oauth2 import service_account

def create_service_account_credentials():
    """Creates Google service account credentials using environment variables."""

    try:
        credentials_info = {
            "type": "service_account",
            "project_id": os.environ["BIGQUERY_PROJECT_ID"],
            "private_key_id": os.environ["BIGQUERY_PRIVATE_KEY_ID"],
            "private_key": os.environ["BIGQUERY_PRIVATE_KEY"].replace('\\n', '\n'), # Handle escaped newline characters
            "client_email": os.environ["BIGQUERY_CLIENT_EMAIL"],
            "client_id": os.environ["BIGQUERY_CLIENT_ID"],
            "auth_uri": os.environ["BIGQUERY_AUTH_URI"],
            "token_uri": os.environ["BIGQUERY_TOKEN_URI"],
            "auth_provider_x509_cert_url": os.environ["BIGQUERY_AUTH_PROVIDER_X509_CERT_URL"],
            "client_x509_cert_url": os.environ["BIGQUERY_CLIENT_X509_CERT_URL"],
            "universe_domain": os.environ["BIGQUERY_UNIVERSE_DOMAIN"]
        }

        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        print('credentials from env variables ... OK')
        return credentials

    except KeyError as e:
        print("falling back to keyfile")
        credentials = service_account.Credentials.from_service_account_file(
            '../.gc_creds/digepizcde-71333237bf40.json'
        )
        return credentials
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    credentials = create_service_account_credentials()

    if credentials:
        print("Service account credentials created successfully.")
        # You can now use these credentials to authenticate with Google services.
        # For example, to create a BigQuery client:
        # from google.cloud import bigquery
        # client = bigquery.Client(credentials=credentials)
    else:
        print("Failed to create service account credentials.")