"""Dataset schemas for upload, schema validation, and data quality profiling."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None


class ClassImbalanceReport(BaseModel):
    label_column: str
    class_counts: Dict[str, int]
    positive_class_ratio: float
    is_imbalanced: bool


class DataQualityReport(BaseModel):
    row_count: int
    column_count: int
    duplicate_rows: int
    missing_cells_total: int
    columns: List[ColumnProfile]
    class_imbalance: Optional[ClassImbalanceReport] = None
    warnings: List[str] = Field(default_factory=list)
    is_valid_for_ml: bool = True


class DatasetValidationResult(BaseModel):
    status: str  # "valid", "warning", "error"
    filename: str
    file_size_bytes: int
    quality: DataQualityReport
    preview_rows: List[Dict[str, Any]] = Field(default_factory=list)
