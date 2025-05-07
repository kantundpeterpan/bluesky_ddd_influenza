import duckdb
import dlt
from dlt.sources.helpers.rest_client import RESTClient
from dlt.sources.helpers.rest_client.paginators import JSONResponseCursorPaginator
from datetime import datetime, timezone
from datetime import timedelta
from urllib.parse import quote_plus
from typing import Optional
import fire
import os
import logging

# Create a logger
logger = logging.getLogger('dlt')
# Set the log level
logger.setLevel(logging.INFO)
    

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
    from project.bsky_tools.dlt_helpers.multicore_runners import pool_func_posts, _run_query_pool

import pandas as pd
import multiprocessing
from multiprocessing import Manager

@dlt.resource(primary_key=("uri"), write_disposition="merge")
def fetch_posts(
    query: str,
    start_date: str,
    end_date: Optional[str] = None,
    out_file: str = None,
    n_jobs: int = 1,
    verbose: bool = True
):
    
    if end_date:
        query_params = create_query_params(query, start_date, end_date)

    else:
        query_params = create_query_params_from_date(query, start_date, n_chunks=n_jobs)
    
    # Unzip the query_params to get separate lists for query, start_date, and end_date
    unzipped_params = list(zip(*query_params))

    # Run the query pool with the unzipped parameters
    query_results = _run_query_pool(
        zip(*unzipped_params),
        pool_func=pool_func_posts,
        yield_flag=True,
        n_jobs=n_jobs
    )
    
    yield from query_results

def run(
    query: str,
    start_date: str,
    end_date: Optional[str] = None,
    n_jobs: int = 1,
    verbose: bool = True
):
    
    # Initialize dlt pipeline
    pipeline = dlt.pipeline(
        pipeline_name="bsky_posts",
        destination="bigquery",
        dataset_name="bsky_posts"
    )
    
    data = fetch_posts(
            query = query, start_date = start_date,
            end_date = end_date, n_jobs=n_jobs
        )
    
    print("Start data pipeline ... ")
    
    load_info = pipeline.run(
        data,
        table_name=query
    )
    
    # load_info = pipeline.run(
    #     fetch_posts_multiproc(
    #         query=query, date_str=date, n_chunks=1
    #     ),
    #     table_name=query
    # )

    if verbose:
        print(load_info)
            
if __name__ == "__main__":
    
    fire.Fire(run)
