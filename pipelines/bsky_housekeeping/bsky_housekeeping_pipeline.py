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
    from ..bsky_tools import *
else:
    import sys
    sys.path.append(
        os.path.join(
            os.path.dirname(__file__), '../../../'
                     )
    )
    from project.bsky_tools import *
# from .dlt_bluesky import get_posts_count_adaptive_sliding_window_reverse  # Keep the import

def create_query_params(
    query:str,
    start_date: str,
    end_date: str
) -> tuple[str, str, str]:
    """
    Generates query parameters (query, start_date, end_date) tuples for each day
    within the specified date range.

    Args:
        query (str): The search query.
        start_date (str): The start date for the range (YYYY-MM-DD).
        end_date (str): The end date for the range (YYYY-MM-DD).

    Returns:
        zip: A zip object containing tuples of (query, start_date, end_date) for each day.
    """
    start_dates = pd.date_range(
    start=start_date, end = end_date, tz = 'utc'
    )

    end_dates = start_dates + timedelta(
        hours = 23, minutes = 59, seconds = 59
    )

    # Format start and end dates to strings and add the 'T' separator
    start_dates = start_dates.astype(str).str.replace(" " , "T").to_list()
    end_dates = end_dates.astype(str).str.replace(" " , "T").to_list()

    return zip([query]*len(start_dates), start_dates, end_dates)

def create_query_params_from_date(
    query: str,
    date: str,
    n_chunks: int
) -> zip:
    """
    Generates query parameters (query, start_date, end_date) tuples for chunks of a single day.

    Args:
        query (str): The search query.
        date (str): The date to chunk (YYYY-MM-DD).
        n_chunks (int): The number of chunks to divide the day into.

    Returns:
        zip: A zip object containing tuples of (query, start_date, end_date) for each chunk.
    """
    start_date = pd.to_datetime(date)
    total_seconds = 24 * 60 * 60  # Total seconds in a day
    chunk_seconds = total_seconds / n_chunks

    start_dates = [start_date + timedelta(seconds=i * chunk_seconds) for i in range(n_chunks)]
    end_dates = [start_date + timedelta(seconds=(i + 1) * chunk_seconds - 1) for i in range(n_chunks)]

    # Format start and end dates to strings and add the 'T' separator
    start_dates = [dt.strftime('%Y-%m-%dT%H:%M:%S+00:00') for dt in start_dates]
    end_dates = [dt.strftime('%Y-%m-%dT%H:%M:%S+00:00') for dt in end_dates]

    return zip([query] * len(start_dates), start_dates, end_dates)

def pool_func(task_queue, progress_queue, result_queue):
    """Worker function for the process pool."""
    while True:
        task = task_queue.get()
        if task is None:
            break

        query, start_date, end_date = task
        try:
            result = [*get_posts_count_adaptive_sliding_window_reverse(query, start_date, end_date)]
            progress_queue.put(1)  # Signal completion
            result_queue.put(result)
            
        except Exception as e:
            print(f"Error processing task: {task}, error: {e}")
            result = []  # Or handle the error as needed
            progress_queue.put(1) # Signal to move to next task (or end threads)
        # task_queue.task_done()  # Signal that the task is done here -> IMPORTANT

def _run_query_pool(
    param_tuple: tuple,
    pool_func: Callable = pool_func,
    n_cpus: int = os.cpu_count()
) -> List[List[dict]]:
    """
    Executes the given function in parallel using a multiprocessing pool,
    along with a progress bar updated via a queue.
    Args:
        param_tuple (tuple): A tuple of parameters to pass to the `pool_func`.
        pool_func (Callable): The function to execute in parallel.
                          Defaults to `get_posts_count_adaptive_sliding_window_reverse`
        n_cpus (int): The number of CPUs to use for the pool. Defaults to `os.cpu_count()`.
    Returns:
        List[List[dict]]: A list of results from the `pool_func` for each input parameter set.
    """

    results = []
    manager = Manager()
    task_queue = manager.Queue()
    progress_queue = manager.Queue()
    result_queue = manager.Queue()
    tasks = [*param_tuple]
    total_tasks = len(tasks)


    def progress_updater(queue, total):
        with tqdm(total=total) as progress_bar:
            count = 0
            while count < total:
                try:
                    queue.get(timeout=1)  # Set a timeout
                    progress_bar.update(1)
                    count += 1
                except Exception as e: # queue.Empty:
                    # Handle timeout or empty queue as needed
                    #print("Progress queue empty, checking for completion...")
                    #time.sleep(0.1)
                    pass

    # Start the progress bar thread
    progress_thread = threading.Thread(
        target=progress_updater, args=(progress_queue, total_tasks)
    )
    progress_thread.daemon = True  # Thread will exit when the main program exits
    progress_thread.start()

    # Create and start worker processes
    processes = []
    for _ in range(n_cpus):
        p = Process(target=pool_func, args=(task_queue, progress_queue, result_queue))
        p.start()
        processes.append(p)
    # Queue them up to the work queue
    [task_queue.put(task) for task in tasks]

    # Signal the workers to terminate
    for _ in range(n_cpus):
       task_queue.put(None)

    # Wait for all tasks to be completed by each of the child processes
    for p in processes:
        p.join()
    # It is important to join all processes before proceeding

    progress_queue.put(None)
    # Signal all jobs processed
    # progress_queue.put("DONE")
    progress_thread.join()


    # Collect results from the result queue
    while not result_queue.empty():
        results.append(result_queue.get())
        # print(f"Remaining: {result_queue.qsize()}")

    #result_queue.put(None) # Not needed

    return results

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
    return df.groupby(df[date_col].dt.date)[agg_col].agg(agg_func).to_frame()

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

    # flatten the result list and create a DataFrame
    res = pd.DataFrame.from_records(
        [e for entry in results for e in entry],
        columns=['period_start', 'post_count']
        )

    if agg_func:
        res = agg_func(res, **agg_kwargs)
        if res.empty:
            res = pd.DataFrame(columns=['period_start', 'post_count'])
            
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
    query_results = _run_query_pool(zip(*unzipped_params))

    parsed_results = _parse_query_results(query_results)

    
    if parsed_results.empty:
        parsed_results = pd.DataFrame([0], index = [start_date], columns = ['post_count'])
        parsed_results.index.name = 'period_start'
        # parsed_results.name = "post_count"

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
    
    print(parsed_results)
    
    return parsed_results
    
    
def run(
    query: str,
    start_date: str,
    end_date: Optional[str] = None,
    out_file: str = None,
    n_jobs: int = os.cpu_count(),
    verbose: bool = True
):
    # Initialize dlt pipeline
    pipeline = dlt.pipeline(
        pipeline_name="bsky_housekeeping",
        destination="bigquery",
        dataset_name="bsky_housekeeping"
    )
    
    load_info = pipeline.run(
        bsky_housekeeping_query(
            query=query, start_date=start_date,
            end_date = end_date, n_jobs=n_jobs
        ),
        table_name='housekeeping',
        write_disposition='merge',
        primary_key=("period_start", "query")
    )

    if verbose:
        print(load_info)

if __name__ == '__main__':
    import fire
    fire.Fire(run)