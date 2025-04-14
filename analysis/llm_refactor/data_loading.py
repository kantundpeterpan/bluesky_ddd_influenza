import pandas as pd
from google.oauth2 import service_account
import pandas_gbq
import logging
from pathlib import Path
from typing import Optional, List
import sys
import os
sys.path.append(os.path.abspath("../"))
from analysis.feature_eng import extact_time_features

_here_dir = Path(__file__).parent

# Load credentials from file
credentials_local = service_account.Credentials.from_service_account_file(
    '../.gc_creds/digepizcde-71333237bf40.json')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_post_count_ili(lang: str = 'fr', credentials: Optional[service_account.Credentials] = None) -> pd.DataFrame:
    """Loads post count and ILI data from BigQuery.

    Args:
        lang (str): Language code (e.g., 'fr'). Defaults to 'fr'.
        credentials (Optional[service_account.Credentials]): Google Cloud credentials.
            If None, uses local credentials.

    Returns:
        pd.DataFrame: DataFrame with post count and ILI data, indexed by date.
    """
    if not credentials:
        credentials = credentials_local

    post_count_ili_sql = f"SELECT * FROM `digepizcde.bsky_ili.bsky_ili_{lang}` ORDER BY date"
    try:
        post_count_ili_df = pandas_gbq.read_gbq(
            post_count_ili_sql, credentials=credentials
        ).set_index('date')
        post_count_ili_df.index = pd.to_datetime(post_count_ili_df.index)
        post_count_ili_df = extact_time_features(post_count_ili_df) #Adding time series features

        logging.info(f"Loaded post count ILI data for language '{lang}' from BigQuery. Shape: {post_count_ili_df.shape}")
        return post_count_ili_df.iloc[:-1,:]

    except Exception as e:
        logging.error(f"Error loading post count ILI data: {e}")
        raise

def load_post_count_ili_upsampled(lang: str = 'fr', credentials: Optional[service_account.Credentials] = None) -> pd.DataFrame:
    """Loads upsampled (daily) post count and ILI data from BigQuery.

    Args:
        lang (str): Language code (e.g., 'fr'). Defaults to 'fr'.
        credentials (Optional[service_account.Credentials]): Google Cloud credentials.
            If None, uses local credentials.

    Returns:
        pd.DataFrame: DataFrame with upsampled post count and ILI data, indexed by date.
    """
    if not credentials:
        credentials = credentials_local
    
    post_count_ili_sql = f"SELECT * FROM `digepizcde.bsky_ili.bsky_ili_{lang}_daily` ORDER BY date"
    try:
        post_count_ili_daily_df = pandas_gbq.read_gbq(
            post_count_ili_sql, credentials=credentials
        ).set_index('date')
        post_count_ili_daily_df.index = pd.to_datetime(post_count_ili_daily_df.index)
        post_count_ili_daily_df = extact_time_features(post_count_ili_daily_df) #Adding time series features
        post_count_ili_daily_df['day'] = post_count_ili_daily_df.index.day.astype("category")
        
        logging.info(f"Loaded upsampled post count ILI data for language '{lang}' from BigQuery. Shape: {post_count_ili_daily_df.shape}")
        return post_count_ili_daily_df

    except Exception as e:
        logging.error(f"Error loading upsampled post count ILI data: {e}")
        raise

def load_weekly_words() -> pd.DataFrame:
    """Loads weekly token counts from a CSV file.

    Returns:
        pd.DataFrame: DataFrame with weekly token counts, indexed by date.
    """
    try:
        weekly_words = pd.read_csv(
            _here_dir / "weekly_token_counts.csv", parse_dates=['iso_weekstartdate']
        ).set_index("iso_weekstartdate")
        logging.info(f"Loaded weekly token counts from CSV. Shape: {weekly_words.shape}")
        return weekly_words
    except Exception as e:
        logging.error(f"Error loading weekly words data: {e}")
        raise

def load_merged_posts_ww(min_weeks: int = 12, min_mentions: int = 10) -> pd.DataFrame:
    """Loads and merges post count data with weekly token counts.

    Args:
        min_weeks (int): Minimum number of weeks a token must appear. Defaults to 12.
        min_mentions (int): Minimum number of mentions a token must have. Defaults to 10.

    Returns:
        pd.DataFrame: Merged DataFrame with post counts and weekly token counts.
    """
    try:
        post_count_ili_df = load_post_count_ili()
        weekly_words = load_weekly_words()

        from analysis.data_proc_tools import filter_vocab  # Import here to avoid circular dependency
        weekly_words = weekly_words.loc[:, filter_vocab(weekly_words, min_weeks=min_weeks, min_mentions=min_mentions)]
        post_count_ili_df = post_count_ili_df.merge(weekly_words, left_index=True, right_index=True)

        logging.info(f"Merged post count data with weekly words. Shape: {post_count_ili_df.shape}")
        return post_count_ili_df
    except Exception as e:
        logging.error(f"Error loading and merging data: {e}")
        raise

def load_llm_filtered_post_count(credentials: Optional[service_account.Credentials] = None) -> pd.DataFrame:
    """Loads LLM-filtered post count data from BigQuery.

    Args:
        credentials (Optional[service_account.Credentials]): Google Cloud credentials.
            If None, uses local credentials.

    Returns:
        pd.DataFrame: DataFrame with LLM-filtered post counts, indexed by date.
    """
    if not credentials:
        credentials = credentials_local

    llm_filtered_post_count_sql = "SELECT date, llm_post_count, ili_incidence, rest_posts FROM `digepizcde.bsky_ili.bsky_ili_fr_llm_filtered` ORDER BY date"
    try:
        llm_filtered_post_count = pandas_gbq.read_gbq(
            llm_filtered_post_count_sql, credentials=credentials
        ).set_index('date')
        llm_filtered_post_count.index = pd.to_datetime(llm_filtered_post_count.index)
        llm_filtered_post_count = extact_time_features(llm_filtered_post_count) #Adding time series features
        logging.info(f"Loaded LLM-filtered post count data from BigQuery. Shape: {llm_filtered_post_count.shape}")
        return llm_filtered_post_count
    except Exception as e:
        logging.error(f"Error loading LLM-filtered post count data: {e}")
        raise

def load_data(data_path: str, dataset: str, credentials = None) -> pd.DataFrame:
    """
    Loads data based on the specified dataset.

    Args:
        data_path (str): Path to the data (either a csv or instructs to load data).
        dataset (str): Name of the dataset to load ('grippe_posts' or 'llm_filtered').
        credentials: Google Cloud credentials.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        ValueError: If an unsupported dataset is specified.
        FileNotFoundError: If the specified CSV file does not exist.
        Exception: For any other errors during data loading.
    """
    logging.info(f"Loading data for dataset: {dataset}")
    try:
        if dataset.startswith("grippe_posts"):
            lang = dataset.split("_")[-1]
            df = load_post_count_ili(lang, credentials=credentials)
            df = df.rename(columns={'rest_posts':'control_posts', 'grippe_posts':'ili_posts'})
        
        elif dataset == 'llm_filtered':
            df = load_llm_filtered_post_count(credentials=credentials)
            
        elif 'upsampled' in dataset:
            df = load_post_count_ili_upsampled(dataset.split("_")[-1], credentials=credentials).loc['2023-08-06':'2025-03-30']
        
        else:
            raise ValueError("Unsupported dataset")
            
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
