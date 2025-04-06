import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem.snowball import SnowballStemmer
import fire
from typing import Optional
import pandas_gbq as pdbq

nltk.download('stopwords')
nltk.download('punkt')

def prepare_french_tweets_for_count_vectorizer(df, **vectorizer_params):
    """
    Prepares French tweets for Count Vectorization and aggregates token counts per week.

    Args:
        df (pd.DataFrame): DataFrame with columns 'uri' (key), 'record__text',
                           and 'record__created_at' (publication date)
                           containing French tweet messages.
        vectorizer_params (dict, optional): Parameters to pass to CountVectorizer. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with preprocessed text.
        pd.DataFrame: DataFrame with weekly token counts.
    """

    def clean_text(text):
        """Cleans the text by removing URLs, mentions, hashtags, and non-alphanumeric characters."""
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\S+', '', text)  # Remove mentions
        text = re.sub(r'#\S+', '', text)  # Remove hashtags
        text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', text)  # Remove non-alphanumeric characters
        text = text.lower()  # Convert to lowercase
        return text

    def tokenize_and_remove_stopwords(text):
        """Tokenizes the text, stems, and removes French stopwords."""
        stop_words = set(stopwords.words('french'))
        word_tokens = word_tokenize(text)
        stemmer = SnowballStemmer("french")
        stemmed_tokens = [stemmer.stem(w) for w in word_tokens] # Stemming
        filtered_tokens = [w for w in stemmed_tokens if not w in stop_words]
        return filtered_tokens

    # 1. Clean the text
    df['cleaned_text'] = df['record__text'].apply(clean_text)

    # 2. Tokenize and remove stopwords
    df['tokenized_text'] = df['cleaned_text'].apply(tokenize_and_remove_stopwords)
    df['preprocessed_text'] = df['tokenized_text'].apply(lambda x: " ".join(x))

    # 3. Count Vectorization
    vectorizer = CountVectorizer(**vectorizer_params)

    count_matrix = vectorizer.fit_transform(df['preprocessed_text'])
    df['count_vector'] = list(count_matrix.toarray())

    # 4. Aggregate token counts per week
    df['record__created_at'] = pd.to_datetime(df['record__created_at'])
    df['iso_weekstartdate'] = df['record__created_at'].dt.to_period('W-MON').dt.start_time  # Weeks start on Monday

    print(count_matrix.shape)
    return

    token_counts_dense = pd.DataFrame(
        count_matrix.todense(),
        index = df.week_start, columns = [",".join([k for k in vectorizer.vocabulary_.keys()])] 
        )
    
    weekly_token_counts = token_counts_dense.groupby('iso_weekstartdate').sum()

    return weekly_token_counts

def main(
    query: Optional = None,
    output_weekly_counts: str = "weekly_token_counts.csv",
    credentials: str = '../.gc_creds/digepizcde-71333237bf40.json',
    input_file : Optional = None,
    vectorizer_params=None):
    """
    Main function to process French tweets from a file and save the results.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the processed CSV file.
        vectorizer_params (str, optional): JSON string of parameters to pass to CountVectorizer. Defaults to None.
    """
    if input_file:
        df = pd.read_csv(input_file)
    elif query:
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(credentials)
        
        df = pdbq.read_gbq(
            query, credentials=credentials
        )
    else:
        raise ValueError("Must provide .csv file or GBQ query")
    
    # if vectorizer_params:
    #     import json
    #     vectorizer_params = json.loads(vectorizer_params)
    
    weekly_counts_df = prepare_french_tweets_for_count_vectorizer(df, **vectorizer_params) # IMPORTANT: pass a copy to avoid modifying the original DataFrame when using fire
    weekly_counts_df.to_csv(output_weekly_counts)
    
    print(f"Weekly counts data saved to {output_weekly_counts}")

if __name__ == '__main__':
    # Example Usage (within the script):
    # data = {'uri': ['1', '2', '3'],
    #         'record__text': ["Bonjour le monde ! Ceci est un tweet en français.",
    #                           "J'aime le café et les croissants.",
    #                           "Le ciel est bleu aujourd'hui."],
    #         'record__created_at': ['2024-01-08', '2024-01-10', '2024-01-15']} # added created_at
    # example_df = pd.DataFrame(data)

    # processed_df, weekly_tokens = prepare_french_tweets_for_count_vectorizer(example_df)
    # print(processed_df.head())
    # print(weekly_tokens.head())

    #run from command line
    fire.Fire(main)
    #Example: python your_script.py --input_file input.csv --output_file output.csv --output_weekly_counts weekly_counts.csv --vectorizer_params '{"ngram_range": [1, 2]}'
