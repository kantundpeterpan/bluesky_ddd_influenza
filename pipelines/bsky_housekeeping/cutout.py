
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