# STFGenerator/modules/loader.py

import os
import pandas as pd
from config.settings import log

def load_csv(file_path: str, delimiter: str = ",") -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        delimiter (str): Field delimiter, default is ','.

    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        log(f"CSV file loaded successfully: {file_path}")
        return df
    except Exception as e:
        log(f"Error loading CSV file {file_path}: {e}", level="ERROR")
        return pd.DataFrame()

def load_excel(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """
    Load an Excel file into a pandas DataFrame.

    Args:
        file_path (str): Path to the Excel file.
        sheet_name (str): Sheet to load. If None, loads the first sheet.

    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        sheet_info = sheet_name if sheet_name else "first sheet"
        log(f"Excel file loaded successfully: {file_path} ({sheet_info})")
        return df
    except Exception as e:
        log(f"Error loading Excel file {file_path}: {e}", level="ERROR")
        return pd.DataFrame()

def list_input_files(input_dir: str) -> list:
    """
    List all files in a given input directory.

    Args:
        input_dir (str): Path to the directory.

    Returns:
        list: List of file paths.
    """
    try:
        files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        log(f"Found {len(files)} files in {input_dir}")
        return files
    except Exception as e:
        log(f"Error listing files in {input_dir}: {e}", level="ERROR")
        return []
