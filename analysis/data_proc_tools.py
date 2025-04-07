def filter_vocab(df, min_weeks: int = 10, min_mention: int = 10):

    col_idx = df.ge(min_mention).sum(axis = 1).ge(min_weeks).values
    
    return col_idx