import pandas as pd
import os

def load_insurance_data(file_path):
    """
    Loads the insurance dataset from a CSV file.
    Checks if the file exists before loading.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NO File Found at:{file_path}")
    df = pd.read_csv(file_path)
    return df
