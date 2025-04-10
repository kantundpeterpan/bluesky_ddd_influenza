from google.oauth2 import service_account
import pandas as pd
import matplotlib.pyplot as plt
import pandas_gbq
import sys
import os
sys.path.append(os.path.abspath("../"))
from analysis.bq_queries import get_post_count_ili_sql, get_llm_ili_sql
from analysis.feature_eng import *
from analysis.model_evaluation import *
from analysis.data_proc_tools import filter_vocab
from pathlib import Path
from typing import Optional, List

_here_dir = Path(__file__).parent

credentials = service_account.Credentials.from_service_account_file(
    '../.gc_creds/digepizcde-71333237bf40.json')

def load_post_count_ili(lang: str = 'fr'):

    post_count_ili_sql = f"SELECT * FROM `digepizcde.bsky_ili.bsky_ili_{lang}` ORDER BY date"
    post_count_ili_df = pandas_gbq.read_gbq(
        post_count_ili_sql, credentials=credentials
    ).set_index('date')
    post_count_ili_df.index = pd.to_datetime(post_count_ili_df.index)
    
    post_count_ili_df = extact_time_features(post_count_ili_df)
    
    return post_count_ili_df.iloc[:-1,:]

def load_post_count_ili_upsampled(lang: str = 'fr'):
    
    post_count_ili_sql = f"SELECT * FROM `digepizcde.bsky_ili.bsky_ili_{lang}_daily` ORDER BY date"
    post_count_ili_daily_df = pandas_gbq.read_gbq(
        post_count_ili_sql, credentials=credentials
    ).set_index('date')
    post_count_ili_daily_df.index = pd.to_datetime(post_count_ili_daily_df.index)
    
    post_count_ili_daily_df = extact_time_features(post_count_ili_daily_df)
    post_count_ili_daily_df['day'] = post_count_ili_daily_df.index.day.astype("category")
    
    return post_count_ili_daily_df

def load_weekly_words():
    
    weekly_words = pd.read_csv(
        _here_dir / "weekly_token_counts.csv", parse_dates=['iso_weekstartdate']
        ) \
        .set_index("iso_weekstartdate")
        
    return weekly_words

def load_merged_posts_ww(min_weeks = 12, min_mentions = 10):
    
    post_count_ili_df = load_post_count_ili()
    weekly_words = load_weekly_words()
    
    weekly_words = weekly_words.loc[:,filter_vocab(weekly_words, min_weeks =  min_weeks, min_mentions=min_mentions)]
    
    post_count_ili_df = post_count_ili_df.merge(weekly_words, left_index=True, right_index=True)
    
    return post_count_ili_df

def load_llm_filtered_post_count():

    llm_filtered_post_count_sql ="SELECT date, llm_post_count, ili_incidence, rest_posts FROM `digepizcde.bsky_ili.bsky_ili_fr_llm_filtered` ORDER BY date"
    llm_filtered_post_count = pandas_gbq.read_gbq(
        llm_filtered_post_count_sql, credentials=credentials
    ).set_index('date')
    llm_filtered_post_count.index = pd.to_datetime(llm_filtered_post_count.index)
    
    llm_filtered_post_count = extact_time_features(llm_filtered_post_count)
    
    return llm_filtered_post_count

def prepare_data_cv(
    df,lags: int = 2, weeks_ahead: int = 1,
    target_col: str = 'ili_incidence', normalize_y: bool = True, 
    cols_to_drop: Optional[List[str]] = None
    ):
    
    y = df[target_col].iloc[lags + weeks_ahead:]
    
    if 'incidence' in target_col:
        y = y.multiply(100_000)
    
    if normalize_y:
        y = y.divide(y.max())
        
    if cols_to_drop:
        X = df.drop(cols_to_drop, axis = 1)
    else:
        X = df.copy()
        
    lagdfs = []

    for l in range(1, lags+1):
        lagdf = X.shift(l)
        lagdf.columns = [f"{c}_lag{l}" for c in lagdf.columns]
        lagdfs.append(lagdf)

    X = pd.concat([X, *lagdfs], axis = 1).dropna().iloc[:-weeks_ahead,:]
    print(X)
    print(y)
    return X, y

def make_train_test(
    df, split_date: str = '2024-08-01', lags: int = 2, weeks_ahead: int = 1,
    target_col: str = 'ili_incidence', normalize_y: bool = True, 
    cols_to_trop: Optional[List[str]] = None
    ):
    
    ytrain = df[target_col].loc[:split_date].iloc[lags+weeks_ahead:]
    ytest = df[target_col].loc[split_date:].iloc[weeks_ahead:]
    
    if 'incidence' in target_col:
        ytrain = ytrain.multiply(100_000)
        ytest = ytest.multiply(100_000)

    
    if normalize_y:
        ytrain = ytrain.divide(ytrain.max())
        ytest = ytest.divide(ytest.max())
        
    if cols_to_trop:
        X = df.drop(cols_to_trop, axis = 1)
    else:
        X = df.copy()
        
    lagdfs = []

    for l in range(1, lags+1):
        lagdf = X.shift(l)
        lagdf.columns = [f"{c}_lag{l}" for c in lagdf.columns]
        lagdfs.append(lagdf)
        
    X = pd.concat([X, *lagdfs], axis = 1).dropna()
    
    Xtrain = X.loc[:split_date].iloc[:-weeks_ahead]
    Xtest = X.loc[split_date:].iloc[:-weeks_ahead]
    
    return Xtrain, ytrain, Xtest, ytest