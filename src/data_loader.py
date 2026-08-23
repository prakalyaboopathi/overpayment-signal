from pathlib import Path
import pandas as pd

def load_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads cases.csv and payments.csv from the specified directory.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (cases_df, payments_df)
    """
    data_path = Path(data_dir)
    cases_file = data_path / "cases.csv"
    payments_file = data_path / "payments.csv"

    if not cases_file.exists():
        raise FileNotFoundError(f"Cases file not found at: {cases_file.resolve()}")
    if not payments_file.exists():
        raise FileNotFoundError(f"Payments file not found at: {payments_file.resolve()}")

    cases_df = pd.read_csv(cases_file, dtype=str)
    payments_df = pd.read_csv(payments_file, dtype=str)

    return cases_df, payments_df
