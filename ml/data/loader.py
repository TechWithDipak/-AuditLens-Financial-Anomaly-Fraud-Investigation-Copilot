"""Repeatable data loading module with schema verification."""

from pathlib import Path

import pandas as pd

from ml.data.validator import ValidationResult, validate_dataframe


def load_dataset(
    filepath: str | Path,
    is_training: bool = False,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, ValidationResult]:
    """Load a dataset from CSV and execute validation checks.

    Args:
        filepath: Path to the CSV dataset file.
        is_training: If True, requires the target 'Class' column to be present.
        nrows: Optional limit on rows to read for quick sampling/profiling.

    Returns:
        tuple[pd.DataFrame, ValidationResult]: Loaded DataFrame and validation report.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file format is unsupported (non-CSV).
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported file format '{path.suffix}'. Only '.csv' files are supported."
        )

    # Read CSV safely
    df = pd.read_csv(path, nrows=nrows)

    # Execute schema validation
    validation = validate_dataframe(df, is_training=is_training)

    return df, validation
