import duckdb
import dlt
from dlt.sources.helpers.rest_client import RESTClient
from dlt.sources.helpers.rest_client.paginators import JSONResponseCursorPaginator
from datetime import datetime, timezone
from datetime import timedelta
from urllib.parse import quote_plus
from collections import Counter
import pandas as pd
from requests.exceptions import HTTPError

# Bluesky API Client
bluesky_client = RESTClient(
    # base_url="https://public.api.bsky.app/xrpc/",
    base_url="https://api.bsky.app/xrpc/",
    paginator=JSONResponseCursorPaginator(cursor_path="cursor", cursor_param="cursor"),
)


def sanitize_string_with_emojis(input_string):
    """
    Sanitizes a string for safe insertion into a database while preserving emojis.
    - Replaces newlines with spaces.
    - Escapes single quotes by doubling them.
    """
    if not isinstance(input_string, str):
        return input_string  # Return as-is if not a string

    # Replace newlines with spaces
    # sanitized = input_string.replace("\n", " ").replace("\r", " ")

    # Escape single quotes by doubling them
    # sanitized = sanitized.replace("'", " ").replace("’", "")
    sanitized = input_string.replace("\x00", "")

    # logging.info(f"Input: {input_string}")
    # logging.info(f"Sanitized string: {sanitized}") # Logging only a snippet

    return sanitized

def sanitize_strings_in_record(record, sanitize_function):
    """
    Recursively traverses a record (dict or list) and applies the sanitize_function to all string fields.
    """
    if isinstance(record, dict):
        # Traverse dictionary
        return {key: sanitize_strings_in_record(value, sanitize_function) for key, value in record.items()}
    elif isinstance(record, list):
        # Traverse list
        return [sanitize_strings_in_record(item, sanitize_function) for item in record]
    elif isinstance(record, str):
        # Apply sanitization to strings
        return sanitize_function(record)
    else:
        # Return other data types as-is
        return record
    
def get_posts_count_adaptive_sliding_window_reverse(
    query: str, start_date: str, end_date: str,
    limit: int = 100):
    """
    Fetches posts using an adaptive sliding window, starting each window from the earliest
    post time of the previous window, given that the API returns recent posts first.
    Ensures all datetime objects are timezone-aware and in UTC.
    """
    next_date = datetime.fromisoformat(end_date.replace("Z", "+00:00")).astimezone(
        timezone.utc)  # Ensure UTC
    start_date_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00")).astimezone(
        timezone.utc)  # Ensure UTC

    # posts_count = 0

    while next_date > start_date_dt:
        
        # Format dates for the API
        until_str = next_date.isoformat().replace("+00:00", "Z")
        encoded_query = quote_plus(query)

        params = {
            "q": encoded_query,
            "until": until_str,
            "since": start_date,  # Fetch until the overall start date, for each query the start date is a fixed value
            "limit": limit,
        }

        response = bluesky_client.get(  # Explicitly using GET
            "app.bsky.feed.searchPosts",
            params=params
        )
        
        if response.status_code != 200:
            raise HTTPError(f"Error code: {response.status_code}")

        posts = response.json()
        
        # print(posts)
        try:
            posts = posts['posts']
            
            #extract post languages
            langs: list[str] = [','.join(p['record'].get('langs','')) for p in posts]
            langs_count = Counter(langs)
            langs = langs_count.keys()
            no_langs: int = len(langs)
            post_counts: list = [langs_count[l] for l in langs]
            
        except Exception as e:
            print(e)
            print(posts[:1])
            break
        # No results found in this window
        # that means that there are no posts between current next_date
        # and start_date, which means there no posts left to retrieve
        if not posts:
            # print('no posts retrieved')
            break
        
        #count no of posts
        posts_count = len(posts)
        
        earliest_post_time = None
        # Update next_date based on the earliest post in the current window
        # post are returned in chronological order
        # most recent first
        # earliest_post_time = datetime.fromisoformat(
        #     posts[-1]['record']['createdAt'].replace("Z", "+00:00")
        #     ).astimezone(
        #             timezone.utc
        #             )
        
        for post in posts:
            record = post['record']
            indexed_at = post.get('indexedAt')
            created_at = record.get('createdAt')
            created_at = created_at if indexed_at is None else min(created_at, indexed_at)

            if created_at:
                post_datetime = datetime.fromisoformat(created_at.replace("Z", "+00:00")).astimezone(
                    timezone.utc)  # Ensure UTC

                if earliest_post_time is None or post_datetime < earliest_post_time:
                    earliest_post_time = post_datetime
            
        yield {
            "period_start":[earliest_post_time]*no_langs,
            "period_end": [next_date]*no_langs,
            "langs": langs,
            "post_count": post_counts
            }
        
        next_date = earliest_post_time
        next_date -= timedelta(seconds=5)


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
    start_dates = [dt.strftime('%Y-%m-%dT%H:%M:%SZ') for dt in start_dates]
    end_dates = [dt.strftime('%Y-%m-%dT%H:%M:%SZ') for dt in end_dates]

    return zip([query] * len(start_dates), start_dates, end_dates)

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

# # Fetch Posts with adaptive sliding window
# @dlt.resource(
#     # columns={"record": {"langs": "array<string>"}}
#     write_disposition='append',
#     parallelized=True
# )
def get_posts_adaptive_sliding_window_reverse(query: str, start_date: str, end_date: str, limit: int = 100):
    """
    Fetches posts using an adaptive sliding window, starting each window from the earliest
    post time of the previous window, given that the API returns recent posts first.
    Ensures all datetime objects are timezone-aware and in UTC.
    """
    next_date = datetime.fromisoformat(end_date.replace("Z", "+00:00")).astimezone(
        timezone.utc)  # Ensure UTC
    start_date_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00")).astimezone(
        timezone.utc)  # Ensure UTC

    while next_date > start_date_dt:
        # Format dates for the API
        until_str = next_date.isoformat().replace("+00:00", "Z")
        encoded_query = quote_plus(query)

        params = {
            "q": encoded_query,
            "until": until_str,
            "since": start_date,  # Fetch until the overall start date, for each query the start date is a fixed value
            "limit": limit,
        }


        response = bluesky_client.get(  # Explicitly using GET
            "app.bsky.feed.searchPosts",
            params=params
        )
        
        if response.status_code != 200:
            raise HTTPError(f"Error code: {response.status_code}")


        posts = response.json()['posts']

        # No results found in this window
        # that means that there are no posts between current next_date
        # and start_date, which means there no posts left to retrieve
        if not posts:
            print('no posts retrieved')
            break
            # next_date -= timedelta(days=1)
            # if next_date < start_date_dt:
            #     break
            # continue  # Skip yielding and go to the next iteration

        # Update next_date based on the earliest post in the current window
        # do some string sanitization
        earliest_post_time = None
        for post in posts:
            record = post.get("record", {})
            embed = post.get("embed", {})
            # process langs fields
            if 'langs' in record.keys():
                record['langs'] = ','.join(record['langs']) if record['langs'] else ''
            else:
                record['langs'] = ''
            # record['text'] = sanitize_string_with_emojis(record['text'])
            if record:
                post['record'] = sanitize_strings_in_record(
                    record, sanitize_string_with_emojis
                )
            if embed:
                post['embed'] = sanitize_strings_in_record(
                    embed, sanitize_string_with_emojis
                )

            # handle inconsistency between indexedAt and createdAt
            # to avoid infinite loops in the closing window approach
            indexed_at = post.get('indexedAt')
            created_at = record.get('createdAt')
            created_at = created_at if indexed_at is None else min(created_at, indexed_at)

            # created_at = record.get("createdAt")
            if created_at:
                post_datetime = datetime.fromisoformat(created_at.replace("Z", "+00:00")).astimezone(
                    timezone.utc)  # Ensure UTC

                if earliest_post_time is None or post_datetime < earliest_post_time:
                    earliest_post_time = post_datetime

            yield post #from posts

        if earliest_post_time:
            next_date = earliest_post_time
        else:
            print("no earliest post date determined")
            next_date -= timedelta(days=1)
            if next_date < start_date_dt:
                break

        # Add a small buffer to avoid duplicate entries -- subtracting to move it back
        next_date -= timedelta(microseconds=1)

# Initialize dlt pipeline with DuckDB
# pipeline = dlt.pipeline(
#     pipeline_name="proc_test",
#     destination="duckdb",
#     dataset_name="posts"
# )

# query = 'influenza'
# start_date = "2025-03-14T00:00:00Z"
# end_date = "2025-03-14T23:59:59Z"

# Run and store data in DuckDB

# pipeline.run(get_posts_adaptive_sliding_window_reverse(query, start_date, end_date), table_name = 'posts',
#              write_disposition = "append")
