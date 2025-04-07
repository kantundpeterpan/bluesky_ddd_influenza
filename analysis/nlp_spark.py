import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, to_date, date_trunc
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import col as pyspcol
from pyspark.sql.functions import desc, asc
from pyspark.ml.functions import vector_to_array
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem.snowball import SnowballStemmer
import fire
from typing import Optional
from pyspark.ml.feature import CountVectorizer
from nltk.stem.snowball import SnowballStemmer

stopwords_list = set(stopwords.words('french'))
stemmer = SnowballStemmer("french")
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text):
    """Cleans the text by removing URLs, mentions, hashtags, and non-alphanumeric characters."""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text)
    text = text.lower()
    return text

def tokenize_and_remove_stopwords(text):
    """Tokenizes the text, stems, and removes French stopwords."""
    word_tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(w) for w in word_tokens]
    filtered_tokens = [w for w in stemmed_tokens if not w in stopwords_list]
    return filtered_tokens

def prepare_french_tweets_for_count_vectorizer(spark, df, vectorizer_params=None, output_model_path: str = "count_vectorizer_model"):
    """
    Prepares French tweets for Count Vectorization and aggregates token counts per week using Spark.
    Args:
        spark (SparkSession): SparkSession object.
        df (pd.DataFrame): DataFrame with columns 'uri' (key), 'record__text',
                           and 'record__created_at' (publication date)
                           containing French tweet messages.
        vectorizer_params (dict, optional): Parameters to pass to CountVectorizer. Defaults to None.
        output_model_path (str, optional): Path to save the CountVectorizerModel. Defaults to "count_vectorizer_model".

    Returns:
        pyspark.sql.DataFrame: DataFrame with preprocessed text.
    """
    logging.info("Starting prepare_french_tweets_for_count_vectorizer")

    spark_df = spark.createDataFrame(df)
    logging.info("Created Spark DataFrame from Pandas DataFrame")

    # 1. Cleaning text (UDF)
    logging.info("Defining clean_text UDF")
    clean_text_udf = udf(clean_text, StringType())
    spark_df = spark_df.withColumn("cleaned_text", clean_text_udf(spark_df["record__text"]))
    logging.info("Applied clean_text UDF to create 'cleaned_text' column")

    # 2. Tokenize and remove stopwords (UDF)
    logging.info("Defining tokenize_and_remove_stopwords UDF")
    tokenize_and_remove_stopwords_udf = udf(tokenize_and_remove_stopwords, ArrayType(StringType()))
    spark_df = spark_df.withColumn("tokenized_text", tokenize_and_remove_stopwords_udf(spark_df["cleaned_text"]))
    logging.info("Applied tokenize_and_remove_stopwords UDF to create 'tokenized_text' column")

    # 3. Preprocessed Text
    logging.info("Creating preprocessed text column")
    spark_df = spark_df.withColumn("preprocessed_text", udf(lambda x: " ".join(x), StringType())(spark_df["tokenized_text"]))
    logging.info("Created the column `preprocessed_text`")

    # 4. Count Vectorization
    logging.info("Initializing CountVectorizer")
    vectorizer = CountVectorizer(**vectorizer_params, inputCol="tokenized_text", outputCol="count_vector")
    model = vectorizer.fit(spark_df)
    logging.info("Fitted CountVectorizer model")

    # Save the CountVectorizerModel
    # model.save(output_model_path)
    logging.info(f"CountVectorizerModel saved to {output_model_path}")
    # 5. Prepare for aggregation
    vectorized_df = model.transform(spark_df)
    logging.info("Transformed DataFrame with CountVectorizer model")
    vectorized_df = vectorized_df.withColumn("record__created_at", to_date(vectorized_df["record__created_at"]))
    vectorized_df = vectorized_df.withColumn("iso_weekstartdate", date_trunc("week", vectorized_df["record__created_at"]))
    logging.info("Created 'record__created_at' and 'iso_weekstartdate' columns")

    #Prepare a new table to efficiently store the dense vectors for weekly aggregation
    count_matrix_df = vectorized_df.select("iso_weekstartdate", "count_vector")
    logging.info("Created `count_matrix_df`")

    #Create Struct of Values (ISO Week, dense matrix)

    #Create a UDF to convert sparse vector to dense vector for easier aggregation
    def to_dense(vector):
        return vector.toArray().tolist()

    #UDF to convert the vectors to a dense representation
    to_dense_udf = udf(to_dense, ArrayType(StringType()))

    #Apply the function to the table
    count_matrix_df = count_matrix_df.withColumn('dense_vector', vector_to_array('count_vector'))
    logging.info("Created `dense_vector` column")

    #Drop sparse vector as no longer needed
    count_matrix_df = count_matrix_df.drop('count_vector')
    logging.info("Dropped `count_vector` column")


    #Rename the "count_vector_dense" to the vocabulary words (this is required to make the group by operation workable)
    vocabulary = model.vocabulary
    logging.info(f"Total vocabulary size: {len(vocabulary)}")
    # for i, word in enumerate(vocabulary):
    #     count_matrix_df = count_matrix_df.withColumn(word, count_matrix_df["dense_vector"].getItem(i).cast("integer"))
    
    count_matrix_df_renamed = count_matrix_df.select(
        [pyspcol("iso_weekstartdate")] + \
            [pyspcol("dense_vector")[i].alias(word).cast('integer') for i, word in zip(range(len(vocabulary)), vocabulary)]
    )
    
    logging.info("Added vocabulary columns")

    #Remove the intermediate column "count_vector_dense"
    # count_matrix_df = count_matrix_df.drop("dense_vector")
    logging.info("Dropped `dense_vector` column")

    #Group by iso_weekstartdate and sum the words from vocabulary
    weekly_token_counts = count_matrix_df_renamed.groupBy("iso_weekstartdate").sum()
    weekly_token_counts = weekly_token_counts.withColumn("iso_weekstartdate", to_date("iso_weekstartdate"))
    logging.info("Performed weekly aggregation")
    
    #Convert columns names to be correct (remove sum(.) prefix)
    agg_cols = weekly_token_counts.columns
    aliases = [col.replace("sum(", "").replace(")", "") for col in agg_cols]
    
    weekly_token_counts_renamed  = weekly_token_counts.select(
        [pyspcol(col).alias(a) for col,a in zip(agg_cols, aliases)]
    )
    
    logging.info("Renamed aggregated columns")

    logging.info("Finished prepare_french_tweets_for_count_vectorizer")
    return weekly_token_counts_renamed.orderBy(asc("iso_weekstartdate"))

def main(
        query: Optional = None,
        output_weekly_counts: str = "weekly_token_counts.csv",
        credentials: str = '../.gc_creds/digepizcde-71333237bf40.json',
        input_file: Optional = None,
        vectorizer_params: dict = None,
        output_model_path: str = "count_vectorizer_model"):
    """
    Main function to process French tweets from a file and save the results using Spark.

    Args:
        query: str = SQL-style query
        output_weekly_counts (str): Path to save the processed Parquet file.
        input_file (str): Path to the input CSV file.
        vectorizer_params (dict, optional): Parameters to pass to CountVectorizer. Defaults to None.
        output_model_path (str, optional): Path to save the CountVectorizerModel. Defaults to "count_vectorizer_model".
    """
    import os
    logging.info("Starting main function")
    # Create SparkSession
    spark = SparkSession.builder.master("local[*]").appName("FrenchTweetsAnalysis").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    logging.info("SparkSession created")

    try:
        if input_file:
            logging.info(f"Reading input from CSV file: {input_file}")
            df = pd.read_csv(input_file)
            logging.info(f"Shape of DataFrame read from CSV: {df.shape}")
        elif query:
            from google.oauth2 import service_account
            import pandas_gbq as pdbq

            credentials = service_account.Credentials.from_service_account_file(credentials)
            logging.info(f"Reading data from GBQ with query: {query}")
            df = pdbq.read_gbq(
                query, credentials=credentials
            )
            logging.info(f"Shape of DataFrame read from GBQ: {df.shape}")
        else:
            error_message = "Must provide .csv file or GBQ query"
            logging.error(error_message)
            raise ValueError(error_message)
    
        weekly_counts_df = prepare_french_tweets_for_count_vectorizer(spark, df, vectorizer_params, output_model_path)

        # Write to a temporary directory
        temp_dir = "temp_weekly_counts"
        logging.info(f"Writing weekly counts to temporary directory: {temp_dir}")
        weekly_counts_df.coalesce(1).write.option("header", "true").csv(temp_dir, mode = 'overwrite')
        
        # Find the output file (there should be only one)
        part_file = [f for f in os.listdir(temp_dir) if f.startswith("part-")][0]
        logging.info(f"Found part file: {part_file}")
        
        # Rename the file to the desired output name
        output_file_path = os.path.join(temp_dir, part_file)
        final_output_path = output_weekly_counts
        os.rename(output_file_path, final_output_path)
        logging.info(f"Renamed part file to: {final_output_path}")
        # Remove the temporary directory
        import shutil
        shutil.rmtree(temp_dir)
        logging.info(f"Removed temporary directory: {temp_dir}")

        logging.info(f"Weekly counts data saved to {output_weekly_counts}")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        raise

    finally:
        spark.stop()
        logging.info("SparkSession stopped")
        logging.info("Finished main function")
        

if __name__ == '__main__':
    fire.Fire(main)
