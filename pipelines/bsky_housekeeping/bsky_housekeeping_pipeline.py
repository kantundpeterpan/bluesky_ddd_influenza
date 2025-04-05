# use pandas 
import pandas as pd
from datetime import timedelta
from multiprocessing import Pool, Queue, Process, Manager
import os
from typing import Callable, List, Optional
import pandas as pd
from datetime import timedelta
import threading
from tqdm import tqdm  # For the progress bar
import time # for debugging

if __name__ != '__main__':
    from ...bsky_tools.dlt_helpers import *
else:
    import sys
    sys.path.append(
        os.path.join(
            os.path.dirname(__file__), '../../../'
                     )
    )
    from project.bsky_tools.dlt_helpers.parsing import *
    from project.bsky_tools.dlt_helpers.multicore_runners import pool_func_post_count, _run_query_pool
# from .dlt_bluesky import get_posts_count_adaptive_sliding_window_reverse  # Keep the import

def by_day_aggregation(
    df: pd.DataFrame,
    date_col: str = 'period_start',
    agg_func: str = 'sum',
    agg_col: str = 'post_count'
) -> pd.DataFrame:
    """
    Aggregates the provided DataFrame by day.
    Args:
        df (pd.DataFrame): The input DataFrame containing a date column and a column to aggregate.
        date_col (str): The name of the column containing the date values. Defaults to 'period_start'.
        agg_func (str): The aggregation function to use (e.g., 'sum', 'mean'). Defaults to 'sum'.
        agg_col (str): The name of the column to aggregate. Defaults to 'post_count'.
    Returns:
        pd.DataFrame: A DataFrame with the aggregated values by day.
    """
    # print(df)
    # first convert the date column to datetime objects
    df[date_col] = pd.to_datetime(df[date_col])
    return df.groupby([df[date_col].dt.date,'langs'])[agg_col].agg(agg_func)


def _parse_query_results(
    results: List[List[dict]],
    agg_func: Callable = by_day_aggregation,
    **agg_kwargs
) -> pd.DataFrame:
    """
    Parses the query results from a list of dictionaries into a Pandas DataFrame
    and optionally aggregates the results.
    Args:
        results (List[dict]): A list of dictionaries representing the query results.
        agg_func (Callable, optional): A function used to aggregate the results (ex. by_day_aggregation). Defaults to None.
        **agg_kwargs: Keyword arguments passed to the aggregation function.
    Returns:
        pd.DataFrame: A DataFrame containing the parsed query results, optionally aggregated.
    """
    
    res = pd.concat([pd.DataFrame.from_dict(rr) for r in results for rr in r])
    
    if agg_func:
        res = agg_func(res, **agg_kwargs)
        if res.empty:
            res = pd.DataFrame(columns=['period_start', 'langs', 'post_count'])
            
    return res

# @dlt.resource(primary_key=("period_start", "query"), write_disposition="merge")
def bsky_housekeeping_query(
    query: str,
    start_date: str,
    end_date: Optional[str] = None,
    out_file: str = None,
    n_jobs: int = os.cpu_count()
) -> pd.DataFrame:
    """
    Queries the Bluesky API for a given query term within a specified date range,
    aggregates the results by day, and returns a Pandas DataFrame.

    Args:
        query (str): The search query term.
        start_date (str): The start date for the query (YYYY-MM-DD).
        end_date (str): The end date for the query (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A DataFrame with the aggregated results by day.
    """
    
    if end_date:
        query_params = create_query_params(query, start_date, end_date)

    else:
        query_params = create_query_params_from_date(query, start_date, n_chunks=n_jobs)
        
    # Unzip the query_params to get separate lists for query, start_date, and end_date
    unzipped_params = list(zip(*query_params))

    # Run the query pool with the unzipped parameters
    query_results = _run_query_pool(
        zip(*unzipped_params),
        pool_func=pool_func_post_count,
        yield_flag=False,
        n_jobs=n_jobs
    )
    
    # No messages found for query
    if all([not r for r in query_results]):
        return None
    # print(query_results)

    parsed_results = _parse_query_results(query_results).to_frame()

    if parsed_results.empty:
        parsed_results = pd.DataFrame([0], index = [start_date], columns = ['post_count'])
        parsed_results.index.name = 'period_start'
        # parsed_results.name = "post_count"

    # return parsed_results

    parsed_results['query'] = query

    if not out_file:
        out_file = f'baseline_{query}.csv'

    # return parsed_results
    
    # parsed_results.to_csv(out_file)
    # print(parsed_results)
    
    parsed_results = parsed_results.reset_index().astype({
        'period_start': "datetime64[ns]",
        'post_count': 'int',
        'query': 'string'
    })
    
    parsed_results['period_start'] = parsed_results.period_start.dt.date
    
    # print(parsed_results)
    
    return parsed_results
    
    
def run(
    query: str,
    start_date: str,
    end_date: Optional[str] = None,
    out_file: str = None,
    n_jobs: int = 50,
    verbose: bool = True
):
    # Initialize dlt pipeline
    pipeline = dlt.pipeline(
        pipeline_name="bsky_housekeeping",
        destination="bigquery",
        dataset_name="bsky_housekeeping"
    )
    
    results_df = bsky_housekeeping_query(
            query=query, start_date=start_date,
            end_date = end_date, n_jobs=n_jobs
        )
    
    if results_df is not None:
    
        load_info = pipeline.run(
            results_df,
            table_name='housekeeping',
            write_disposition='merge',
            primary_key=("period_start", "langs", "query")
        )

        if verbose:
            print(load_info)
            
    else:
        print("No results for query")

if __name__ == '__main__':
    import fire
    fire.Fire(run)