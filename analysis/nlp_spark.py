import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, to_date, date_trunc
from pyspark.sql.types import ArrayType, StringType
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem.snowball import SnowballStemmer
import fire
from typing import Optional
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
stopwords_list = set(stopwords.words('french'))
stemmer = SnowballStemmer("french")

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
        pd.DataFrame: DataFrame with preprocessed text.
    """
    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(df)

    # 1. Cleaning text (UDF)
    clean_text_udf = udf(clean_text, StringType())
    spark_df = spark_df.withColumn("cleaned_text", clean_text_udf(spark_df["record__text"]))

    # 2. Tokenize and remove stopwords (UDF)
    tokenize_and_remove_stopwords_udf = udf(tokenize_and_remove_stopwords, ArrayType(StringType()))
    spark_df = spark_df.withColumn("tokenized_text", tokenize_and_remove_stopwords_udf(spark_df["cleaned_text"]))

    # 3. Preprocessed Text
    spark_df = spark_df.withColumn("preprocessed_text", udf(lambda x: " ".join(x), StringType())(spark_df["tokenized_text"]))

    # 4. Count Vectorization
    vectorizer = CountVectorizer(**vectorizer_params, inputCol="tokenized_text", outputCol="count_vector")
    model = vectorizer.fit(spark_df)

    # Save the CountVectorizerModel
    # model.save(output_model_path)
    print(f"CountVectorizerModel saved to {output_model_path}")
    # 5. Prepare for aggregation
    vectorized_df = model.transform(spark_df)
    vectorized_df = vectorized_df.withColumn("record__created_at", to_date(vectorized_df["record__created_at"]))
    vectorized_df = vectorized_df.withColumn("iso_weekstartdate", date_trunc("week", vectorized_df["record__created_at"]))

    #Prepare a new table to efficiently store the dense vectors for weekly aggregation
    count_matrix_df = vectorized_df.select("iso_weekstartdate", "count_vector")

    #Create Struct of Values (ISO Week, dense matrix)

    #Create a UDF to convert sparse vector to dense vector for easier aggregation
    def to_dense(vector):
        return vector.toArray().tolist()

    #UDF to convert the vectors to a dense representation
    to_dense_udf = udf(to_dense, ArrayType(StringType()))

    #Apply the function to the table
    count_matrix_df = count_matrix_df.withColumn('dense_vector', to_dense_udf(count_matrix_df['count_vector']))

    #Drop sparse vector as no longer needed
    count_matrix_df = count_matrix_df.drop('count_vector')

    #Rename the "count_vector_dense" to the vocabulary words (this is required to make the group by operation workable)
    vocabulary = model.vocabulary
    for i, word in enumerate(vocabulary):
        count_matrix_df = count_matrix_df.withColumn(word, count_matrix_df["dense_vector"].getItem(i).cast("double"))

    #Remove the intermediate column "count_vector_dense"
    count_matrix_df = count_matrix_df.drop("dense_vector")

    #Group by iso_weekstartdate and sum the words from vocabulary
    weekly_token_counts = count_matrix_df.groupBy("iso_weekstartdate").sum()

    #Convert columns names to be correct (remove sum(.) prefix)
    for col in weekly_token_counts.columns:
        if col != "iso_weekstartdate":
            weekly_token_counts = weekly_token_counts.withColumnRenamed(col, col.replace("sum(", "").replace(")", ""))

    weekly_token_counts_pd = weekly_token_counts.toPandas()

    return weekly_token_counts_pd

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
        output_weekly_counts (str): Path to save the processed CSV file.
        input_file (str): Path to the input CSV file.
        vectorizer_params (dict, optional): Parameters to pass to CountVectorizer. Defaults to None.
        output_model_path (str, optional): Path to save the CountVectorizerModel. Defaults to "count_vectorizer_model".
    """

    # Create SparkSession
    spark = SparkSession.builder.master("local[*]").appName("FrenchTweetsAnalysis").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        if input_file:
            df = pd.read_csv(input_file)
        elif query:
            from google.oauth2 import service_account
            import pandas_gbq as pdbq

            credentials = service_account.Credentials.from_service_account_file(credentials)
            
            df = pdbq.read_gbq(
                query, credentials=credentials
            )
        else:
            raise ValueError("Must provide .csv file or GBQ query")
    
        print(df.shape)
    
        weekly_counts_df = prepare_french_tweets_for_count_vectorizer(spark, df, vectorizer_params, output_model_path)  # IMPORTANT: pass a copy to avoid modifying the original DataFrame when using fire
        weekly_counts_df.to_csv(output_weekly_counts)
    
        print(f"Weekly counts data saved to {output_weekly_counts}")

    finally:
        spark.stop()

if __name__ == '__main__':
    fire.Fire(main)
