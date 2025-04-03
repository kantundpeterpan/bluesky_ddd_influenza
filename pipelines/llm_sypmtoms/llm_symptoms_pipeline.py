import dlt
import pandas as pd
import os
import sys

# Add the project directory to the Python path to enable importing local modules.
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
    max_test_records: int = 10
):
    """
    Extracts symptoms from text data fetched from a BigQuery table,
    using a SymptomExtractor, and saves the results back to BigQuery.
    
    Args:
        query_kw (str): The name of the BigQuery table to query.
        language (str, optional): The language of the text to extract symptoms from. Defaults to 'fr'.
        is_test (bool, optional): If True, limits the query to 10 rows and prints additional debugging information. Defaults to False.
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
    FROM {query_kw} 
    WHERE record__langs LIKE '{language}' AND
    uri NOT IN (
        SELECT uri FROM llm_hints
        )"""

    # Limit the query if it's a test run.
    if is_test:
        sql_query += " LIMIT 100"
    
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

    # Print the shape of the DataFrame.
    print(df.shape)

    if df.empty:
        print("No messages to label")
        return

    # Extract the symptoms using the SymptomExtractor.
    se.multi_extract([*zip(df['uri'], df['record__text'])])

    # Print out individual tweets during test
    if is_test:
        print(se.resdf)
        
        for i, (ind, sl, uri) in enumerate(zip(se.resdf.ili_related, se.resdf.symptoms, se.resdf.uri)):
            if ind or sl:
                print(f"############# Tweet {i}, ili: {ind}, symptoms: {sl}")
                print(df.set_index("uri").loc[uri].record__text)
                print("#############")
                print()

    # Converts list of symptoms into a single string
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
