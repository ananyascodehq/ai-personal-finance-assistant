import pandas as pd

REQUIRED_COLUMNS = ["Date", "Category", "Amount", "Description"]


class InvalidExpenseFile(ValueError):
    """Raised when an uploaded file cannot be used."""


def load_csv(file):
    """Read a CSV (file path or uploaded file) into a DataFrame."""
    try:
        return pd.read_csv(file)
    except pd.errors.EmptyDataError:
        raise InvalidExpenseFile("The file is empty.")
    except pd.errors.ParserError as error:
        raise InvalidExpenseFile(f"Could not read the file as a CSV: {error}")


def check_columns(df):
    """Make sure the 4 required columns exist (ignoring case and spaces).
    Returns the DataFrame with standard column names, in standard order."""
    found = {str(col).strip().lower(): col for col in df.columns}
    missing = [col for col in REQUIRED_COLUMNS if col.lower() not in found]
    if missing:
        raise InvalidExpenseFile("Missing columns: " + ", ".join(missing))
    renamed = df.rename(columns={found[col.lower()]: col for col in REQUIRED_COLUMNS})
    return renamed[REQUIRED_COLUMNS]