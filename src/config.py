"""Configuration module for AuditLens."""

from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

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

# ==============================================================================
# UI COLOR PALETTE (Defined once here and in .streamlit/config.toml only)
# ==============================================================================
COLOR_BACKGROUND = "#FFFFFF"
COLOR_PANELS = "#F5F7FA"
COLOR_BORDERS = "#D9DEE5"
COLOR_TEXT = "#1A1D21"
COLOR_TEXT_SECONDARY = "#5C6670"

# Accent: blue (buttons, selected tab, chart base series)
COLOR_ACCENT = "#1F4E8C"

# Risk colors (used ONLY for risk labels, risk bars and flagged rows)
COLOR_NORMAL_TEXT = "#2E6B3F"
COLOR_NORMAL_BG = "#EAF3EC"

COLOR_SUSPICIOUS_TEXT = "#8A5A00"
COLOR_SUSPICIOUS_BG = "#FBF1DC"

COLOR_FRAUD_TEXT = "#9B1C1C"
COLOR_FRAUD_BG = "#FBE9E9"

# Semantic mappings for visualization
COLOR_DOT_NORMAL = COLOR_BORDERS  # light grey-blue
COLOR_DOT_FRAUD = COLOR_FRAUD_TEXT  # red
COLOR_CHART_THRESHOLD = COLOR_FRAUD_TEXT  # dashed threshold red
COLOR_CHART_OTHER_MODEL = COLOR_TEXT_SECONDARY  # grey
COLOR_SHAP_INCREASE = COLOR_FRAUD_TEXT  # increases risk red
COLOR_SHAP_DECREASE = COLOR_ACCENT  # decreases risk blue


def create_auditlens_plotly_template() -> go.layout.Template:
    """Create a custom Plotly template ensuring no default Plotly colors appear."""
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=COLOR_BACKGROUND,
        plot_bgcolor=COLOR_BACKGROUND,
        font=dict(
            color=COLOR_TEXT,
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            size=12,
        ),
        colorway=[
            COLOR_ACCENT,
            COLOR_TEXT_SECONDARY,
            COLOR_FRAUD_TEXT,
            COLOR_NORMAL_TEXT,
            COLOR_SUSPICIOUS_TEXT,
        ],
        xaxis=dict(
            linecolor=COLOR_BORDERS,
            gridcolor=COLOR_PANELS,
            tickcolor=COLOR_BORDERS,
            tickfont=dict(color=COLOR_TEXT_SECONDARY, size=11),
            title=dict(font=dict(color=COLOR_TEXT_SECONDARY, size=12)),
            zerolinecolor=COLOR_BORDERS,
        ),
        yaxis=dict(
            linecolor=COLOR_BORDERS,
            gridcolor=COLOR_PANELS,
            tickcolor=COLOR_BORDERS,
            tickfont=dict(color=COLOR_TEXT_SECONDARY, size=11),
            title=dict(font=dict(color=COLOR_TEXT_SECONDARY, size=12)),
            zerolinecolor=COLOR_BORDERS,
        ),
        legend=dict(
            font=dict(color=COLOR_TEXT, size=11),
            bordercolor=COLOR_BORDERS,
            bgcolor=COLOR_BACKGROUND,
        ),
    )
    return template


CUSTOM_PLOTLY_TEMPLATE = create_auditlens_plotly_template()
pio.templates["auditlens"] = CUSTOM_PLOTLY_TEMPLATE
pio.templates.default = "auditlens"
