from pathlib import Path
import warnings
import pandas as pd
from pandas.api.types import is_numeric_dtype

def data_validation(name: str) -> pd.DataFrame:
    """
    Ingest and validate a Yahoo Finance CSV.
    Guarantees on return:
      - Index: DatetimeIndex (date-only), tz-naive, unique, ascending, no NaT.
      - Columns: normalized to ['open','high','low','close','adj_close','volume'] (plus any extras kept),
                 and reordered so required columns come first.
      - Dtypes: price columns numeric; volume numeric (int/nullable-int/float acceptable here).
    Fails fast with clear errors if the contract is violated.
    """
    
    data_dir = Path("data/raw")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")
    
    if not name or not isinstance(name, str):
        raise FileNotFoundError("You must provide a non-empty file base name (without .csv).")

    file_path = data_dir / f"{name}"
    if not file_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {file_path.resolve()}")
    
    # Read CSV; dates in 'Date' column; set as index.
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    
    # --- Index sanity ---
    try:
        df.index = df.index.tz_localize(None)
    except (TypeError, AttributeError):
        # Already tz-naive or index not timezone-aware; ignore
        pass

    # strip HH:MM:SS
    if getattr(df.index, "hour", None) is not True:
        df.index = df.index.normalize()

    # No NaT
    if getattr(df.index, "hasnans", False):
        bad_count = df.index.isna().sum()
        raise ValueError(f"Unparseable dates (NaT) detected: {bad_count} rows in {file_path.name}.")

    # Uniqueness
    if not df.index.is_unique:
        duplicated = df.index[df.index.duplicated()].unique()
        example = duplicated[0] if len(duplicated) else None
        raise ValueError(f"Duplicate dates in index (e.g., {example.date() if example is not None else 'unknown'}) in {file_path.name}.")

    # Ascending order
    if not df.index.is_monotonic_increasing:
        diffs = df.index.to_series().diff().dropna()
        first_bad = diffs[diffs < pd.Timedelta(0)].index[0]
        prev = df.index[df.index.get_loc(first_bad) - 1]
        raise ValueError(f"Dates not sorted ascending around {prev.date()} -> {first_bad.date()} in {file_path.name}.")

    # --- Column sanity ---
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    required = ["open", "high", "low", "close", "adj_close", "volume"]
    
    # Missing columns
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {file_path.name}. "
                         f"Found columns: {list(df.columns)}")
    
    # Extra columns
    extras = [c for c in df.columns if c not in required]
    if extras:
        warnings.warn(f"Extra columns retained for {file_path.name}: {extras}", UserWarning)

    ordered_cols = required + extras
    df = df[ordered_cols]

    # --- Dtype checks ---
    price_cols = ["open", "high", "low", "close", "adj_close"]
    bad_price = [c for c in price_cols if not is_numeric_dtype(df[c])]
    if bad_price:
        raise TypeError(f"Non-numeric price columns {bad_price} in {file_path.name}. "
                        f"Check thousand separators/locale or bad cells.")

    if not is_numeric_dtype(df["volume"]):
        raise TypeError(f"'volume' must be numeric in {file_path.name}.")
    
    return df

