import dlt
import pandas as pd
import os
import sys

# Add the project directory to the Python path
# to enable importing local modules.
sys.path.append(
    os.path.join(
        os.path.dirname(__file__), '../../../'
                    )
)

from project.llm_tools import SymptomExtractor


def run(
    query_kw: str,
    language: str = 'fr',
    is_test: bool = False,
    max_test_records: int = 10,
    chunk_size: int = 500,
    n_jobs: int = os.cpu_count()
):
    """
    Extracts symptoms from text data fetched from a BigQuery table,
    uses a SymptomExtractor, and saves the results back to BigQuery.
    
    Args:
        query_kw (str): The name of the BigQuery table to query.
        language (str, optional): The language of the text to extract symptoms from. Defaults to 'fr'.
        is_test (bool, optional): If True, limits the query to a small number of rows for testing. Defaults to False.
        max_test_records (int, optional): The maximum number of records to use when `is_test` is True. Defaults to 10.
        chunk_size (int, optional): The number of records to process in each chunk to avoid memory issues. Defaults to 500.
        n_jobs (int, optional): The number of parallel processes to use for symptom extraction. Defaults to the number of CPU cores.
    """
    # Initialize the SymptomExtractor.
    se = SymptomExtractor()
    
    # Configure the dlt pipeline for BigQuery.
    pipeline = dlt.pipeline(
        pipeline_name="bsky_posts",
        destination="bigquery",
        dataset_name="bsky_posts"
    )

    # Construct the SQL query to fetch records from BigQuery.
    sql_query = f"""SELECT uri, record__text 
    FROM {query_kw.lower()} 
    WHERE record__langs LIKE '{language}' AND
    uri NOT IN (
        SELECT uri FROM llm_hints
        )
    """

    # Limit the query if it's a test run.
    if is_test:
        sql_query += f" LIMIT {max_test_records}"
    
    # Execute the query and fetch the data into a Pandas DataFrame.
    with pipeline.sql_client() as client:
        with client.execute_query(
            sql_query,
        ) as cursor:
            # get all data from the cursor as a list of tuples and create a dataframe
            df = pd.DataFrame.from_records(
                    cursor.fetchall(),
                    columns = ['uri', 'record__text']
                    )

    # Print the shape of the DataFrame for debugging.
    print(df.shape)

    # If there's no data, exit early.
    if df.empty:
        print("No messages to label")
        return

    # Process the DataFrame in chunks.
    for i in range(0, len(df), chunk_size):
        df_chunk = df.iloc[i:i + chunk_size,:]

        # Extract the symptoms using the SymptomExtractor in parallel.
        se.multi_extract(
                [*zip(df_chunk['uri'], df_chunk['record__text'])],
            n_jobs = n_jobs
            )

        # During test, print out individual tweets and extracted symptoms for verification.
        if is_test:
            print(se.resdf)
            
            for i, (ind, sl, uri) in enumerate(zip(se.resdf.ili_related, se.resdf.symptoms, se.resdf.uri)):
                if ind or sl:
                    print(f"############# Tweet {i}, ili: {ind}, symptoms: {sl}")
                    print(df_chunk.set_index("uri").loc[uri].record__text)  # Corrected to use df_chunk
                    print("#############")
                    print()

        # Convert list of symptoms into a single string for easier storage in BigQuery.
        se.resdf['symptoms'] = se.resdf.symptoms.apply(lambda x: ",".join(x))

        # Run the dlt pipeline to write the results to BigQuery, merging on the 'uri' column.
        pipeline.run(
            se.resdf,
            write_disposition="merge", primary_key="uri",
            table_name="llm_hints"
        )

if __name__ == "__main__":
    import fire
    fire.Fire(run)
