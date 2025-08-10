import pandas as pd

def opensimTableToDataFrame(file_path):
    """
    Reads an OpenSim .mot or .sto file into a clean Pandas DataFrame.
    Skips the first 6 lines of metadata.
    """
    try:
        df = pd.read_csv(file_path, sep='\t', skiprows=6)
        df = df.dropna(axis=1, how='all')  # Drop columns that are fully NaN
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load OpenSim file: {file_path}\n{e}")
