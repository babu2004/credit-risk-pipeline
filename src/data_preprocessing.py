import pandas as pd


def load_data(path):
    """
    Load credit risk dataset.
    """
    return pd.read_csv(path)


def preprocess_data(df):
    """
    Preprocessing exactly as in the notebook:
    - Drop missing values
    - No encoding (data already numeric)
    """
    df = df.dropna()
    return df
