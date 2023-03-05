def encode_categories(df, cols):
    """Preprocesses the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to preprocess.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    for col in cols:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
    return df

def encode_ranks(df, cols, conversions: dict[str, dict[str, int]]):
    """Preprocesses the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to preprocess.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    for col in cols:
        df[col] = df[col].map(conversions[col])
    return df
