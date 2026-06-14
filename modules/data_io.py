# modules/data_io.py
import functools
import pandas as pd
from pathlib import Path


# --- Internal cached loader ---

@functools.lru_cache(maxsize=16)
def _load_csv_cached(path: str) -> pd.DataFrame:
    """Internal cached loader for CSV files."""
    return pd.read_csv(path, index_col=0)


# --- Validation helpers ---

def validate_columns(df: pd.DataFrame, required: list) -> None:
    """
    Raise a ValueError if any required columns are missing from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    required : list of str
        Column names that must be present.

    Raises
    ------
    ValueError
        If any required columns are absent, with a message listing them.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_dtypes(
    df: pd.DataFrame,
    expected: dict
) -> None:
    """
    Raise a TypeError if any columns do not match their expected dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    expected : dict
        Mapping of column name to expected dtype string,
        e.g. {"TV": "float64", "Sales": "float64"}.

    Raises
    ------
    TypeError
        If any column dtype does not match the expected dtype.
    """
    mismatches = {
        col: (str(df[col].dtype), exp_dtype)
        for col, exp_dtype in expected.items()
        if col in df.columns and str(df[col].dtype) != exp_dtype
    }
    if mismatches:
        details = ", ".join(
            f"'{col}': expected {exp}, got {got}"
            for col, (got, exp) in mismatches.items()
        )
        raise TypeError(f"Column dtype mismatches: {details}")


# --- Public interface ---

def load_dataset(
    path: str,
    use_cache: bool = True,
    required_columns: list = None,
    expected_dtypes: dict = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load a dataset from a local path or remote URL with optional validation.

    Parameters
    ----------
    path : str
        File path or URL. Supported formats: .csv, .parquet, .json,
        .xlsx, .xls.
    use_cache : bool, optional
        Use in-memory cache for CSV files. Default is True.
    required_columns : list of str, optional
        Columns that must be present in the loaded data.
    expected_dtypes : dict, optional
        Mapping of column name to expected dtype string.
    **kwargs
        Additional keyword arguments passed to the underlying reader.

    Returns
    -------
    pd.DataFrame
        A validated DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing or the file format is unsupported.
    TypeError
        If column dtypes do not match expected types.

    Examples
    --------
    >>> url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
    >>> df = load_dataset(url, index_col=0)
    >>> df.shape
    (200, 4)
    """
    suffix = Path(path).suffix.lower()

    readers = {
        ".csv":     pd.read_csv,
        ".parquet": pd.read_parquet,
        ".json":    pd.read_json,
        ".xlsx":    pd.read_excel,
        ".xls":     pd.read_excel,
    }

    if suffix not in readers:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported formats: {list(readers.keys())}"
        )

    # Cached CSV loading
    if suffix == ".csv" and use_cache and not kwargs:
        df = _load_csv_cached(path)
    else:
        df = readers[suffix](path, **kwargs)

    # Validate required columns
    if required_columns:
        validate_columns(df, required_columns)

    # Validate dtypes
    if expected_dtypes:
        validate_dtypes(df, expected_dtypes)

    return df


def save_dataset(df: pd.DataFrame, path: str, **kwargs) -> None:
    """
    Save a DataFrame to a local file.

    Supported formats: .csv, .parquet, .json, .xlsx.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str
        Destination file path. Format is inferred from the extension.
    **kwargs
        Additional keyword arguments passed to the underlying writer.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    # Create parent directories if they do not exist
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    writers = {
        ".csv":     lambda: df.to_csv(path_obj, **kwargs),
        ".parquet": lambda: df.to_parquet(path_obj, **kwargs),
        ".json":    lambda: df.to_json(path_obj, **kwargs),
        ".xlsx":    lambda: df.to_excel(path_obj, **kwargs),
    }

    if suffix not in writers:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported formats: {list(writers.keys())}"
        )

    writers[suffix]()
    print(f"Saved: {path_obj}")
