import pandas as pd

def load_csv(filepath: str):
    """
    Load a CSV file and return a DataFrame.
    If loading fails, print an error and return None.
    """
    try:
        df = pd.read_csv(filepath)
        print("CSV loaded successfully.")
        print("Columns:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
