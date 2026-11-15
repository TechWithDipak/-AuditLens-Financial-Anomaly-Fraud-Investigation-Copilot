"""Configuration module for AuditLens."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DOCS_DIR = BASE_DIR / "docs"

TRAIN_RAW_FILE = RAW_DATA_DIR / "fraudTrain.csv"
TEST_RAW_FILE = RAW_DATA_DIR / "fraudTest.csv"

# Global Random Seed
RANDOM_SEED = 42

# Currency conversion (display only)
USD_TO_INR = 83.0

# Split settings
TRAIN_SPLIT_RATIO = 0.8  # Earlier 80% of fraudTrain for training, last 20% for validation

# Feature definitions
EXCLUDED_COLUMNS = [
    "Unnamed: 0",
    "cc_num",
    "trans_num",
    "first",
    "last",
    "street",
    "unix_time",
    "job",
    "merchant",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "dob",
]

BASE_NUMERICAL_FEATURES = [
    "amt",
    "log_amt",
    "hour",
    "day_of_week",
    "is_night",
    "distance_km",
    "age",
    "gender",
    "city_pop",
    "secs_since_last_txn",
    "amt_zscore_vs_customer_history",
    "txn_count_last_24h",
]

UNSUPERVISED_FEATURE_COLUMNS = [
    "gmm_score",
    "hmm_score",
    "kmeans_cluster",
    "kmeans_distance",
]

# Risk Level Threshold defaults (overridden by validation tuning)
DEFAULT_THRESHOLD_SUSPICIOUS = 0.20
DEFAULT_THRESHOLD_FRAUD = 0.50
