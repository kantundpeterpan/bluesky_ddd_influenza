import duckdb
import dlt
from dlt.sources.helpers.rest_client import RESTClient
from dlt.sources.helpers.rest_client.paginators import JSONResponseCursorPaginator
from datetime import datetime, timezone
from datetime import timedelta
from urllib.parse import quote_plus

# Bluesky API Client
bluesky_client = RESTClient(
    base_url="https://public.api.bsky.app/xrpc/",
    paginator=JSONResponseCursorPaginator(cursor_path="cursor", cursor_param="cursor"),
)

# Fetch Followers
@dlt.resource
def get_followers(actor: str):
    for page in bluesky_client.paginate(
        "app.bsky.graph.getFollowers",
        params={"actor": actor, "limit": 100},
    ):
        yield page

# Fetch Follows
@dlt.resource
def get_follows(actor: str):
    for page in bluesky_client.paginate(
        "app.bsky.graph.getFollows",
        params={"actor": actor, "limit": 100},
    ):
        yield page

# # Fetch Posts with sliding window
# @dlt.resource
# def get_posts_sliding_window(query: str, start_date: str, end_date: str, window_size_days: int = 1, limit: int = 100):
#     """
#     Fetches posts using a sliding window approach to respect API limits and retrieve all posts within a date range.
#     """
#     current_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))  # handle Z correctly
#     end_date_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

#     while current_date < end_date_dt:
#         window_end_date = current_date + timedelta(days=window_size_days)
#         if window_end_date > end_date_dt:
#             window_end_date = end_date_dt

#         since_str = current_date.isoformat().replace('+00:00', 'Z')  # back to zulu time
#         until_str = window_end_date.isoformat().replace('+00:00', 'Z')

#         # Properly encode the query string
#         encoded_query = quote_plus(query)

#         params = {
#             "q": encoded_query,
#             "since": since_str,
#             "until": until_str,
#             "limit": limit
#         }

#         yield from bluesky_client.get(  # Explicitly using GET
#             "app.bsky.feed.searchPosts",
#             params=params
#         )

#         current_date = window_end_date

import logging

# Configure logging
logging.basicConfig(filename='string_sanitizer.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


        posts = bluesky_client.get(  # Explicitly using GET
            "app.bsky.feed.searchPosts",
            params=params
        ).json()
        
        # print(posts)
        
        posts = posts['posts']
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
            created_at = record.get("createdAt")
            if created_at:
                post_datetime = datetime.fromisoformat(created_at.replace("Z", "+00:00")).astimezone(
                    timezone.utc)  # Ensure UTC

                if earliest_post_time is None or post_datetime < earliest_post_time:
                    earliest_post_time = post_datetime
            
        yield {
            "period_start":earliest_post_time,
            "period_end": next_date,
            "post_count": posts_count
            }
        
        next_date = earliest_post_time
        next_date -= timedelta(microseconds=1)

        


# Fetch Posts with adaptive sliding window
@dlt.resource(
    # columns={"record": {"langs": "array<string>"}}
    write_disposition='append',
)
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

        posts = bluesky_client.get(  # Explicitly using GET
            "app.bsky.feed.searchPosts",
            params=params
        ).json()['posts']

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
            created_at = record.get("createdAt")
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


@dlt.resource
def test_res():
    record = {'embed':"E'Cette image est tirée de l''article de Slate cité par Laure Dasinière.\nCovid, les virus de la grippe se transmettent par voie aérosol, par ces trs fi"}
    posts = [record]
    for post in posts:
        record = post.get("record", {})
        embed = post.get("embed", {})
        # process langs fields
        if 'langs' in record.keys():
            record['langs'] = ','.join(record['langs']) if record['langs'] else ''
        else:
            record['langs'] = ''
        # record['text'] = sanitize_string_with_emojis(record['text'])
        record = sanitize_strings_in_record(
            record, sanitize_string_with_emojis
        )
        embed = sanitize_strings_in_record(
            embed, sanitize_string_with_emojis
        )
        
        post['embed'] = embed

        print(post)
        
    yield from posts

# Initialize dlt pipeline with DuckDB
pipeline = dlt.pipeline(
    pipeline_name="bluesky",
    destination="duckdb",
    dataset_name="influenza"
)

# query = 'influenza'
# start_date = "2025-03-14T00:00:00Z" 
# end_date = "2025-03-14T23:59:59Z"

# Run and store data in DuckDB

# pipeline.run(get_posts_adaptive_sliding_window_reverse(query, start_date, end_date), table_name = 'posts',
#              write_disposition = "append") 