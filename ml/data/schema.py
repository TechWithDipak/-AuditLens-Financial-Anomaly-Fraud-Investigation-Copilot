"""Data schema definitions, column constraints, and bounds."""

# 28 PCA dimensions
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]

# Required features for scoring and inference (30 features total)
REQUIRED_FEATURE_COLUMNS = ["Time", *PCA_COLUMNS, "Amount"]

# Target label for supervised training
TARGET_COLUMN = "Class"

# Full dataset schema including target
ALL_COLUMNS = [*REQUIRED_FEATURE_COLUMNS, TARGET_COLUMN]

# Maximum allowed file upload size: 50 Megabytes
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024

# Expected data types
EXPECTED_DTYPES = {
    "Time": "float64",
    **{col: "float64" for col in PCA_COLUMNS},
    "Amount": "float64",
    "Class": "int64",
}
