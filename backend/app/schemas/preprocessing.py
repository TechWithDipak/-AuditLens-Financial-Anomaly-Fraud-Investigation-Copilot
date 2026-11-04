"""Schemas for Exploratory Data Analysis and Leakage-Safe Preprocessing."""

from typing import List, Optional

from pydantic import BaseModel, Field


class FeatureDistribution(BaseModel):
    feature: str
    mean: float
    std: float
    median: float
    min: float
    max: float
    skewness: float


class TimeAnalysis(BaseModel):
    min_time: float
    max_time: float
    duration_hours: float
    transaction_count: int
    hourly_rate_avg: float


class EDAResponse(BaseModel):
    total_records: int
    total_features: int
    fraud_count: int
    fraud_rate_pct: float
    time_analysis: Optional[TimeAnalysis] = None
    feature_distributions: List[FeatureDistribution] = Field(default_factory=list)


class SplitRequest(BaseModel):
    dataset_name: str = "creditcard_sample_fixture.csv"
    strategy: str = "stratified"  # "stratified" or "chronological"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


class SplitPartitionInfo(BaseModel):
    sample_count: int
    fraud_count: int
    fraud_rate_pct: float
    min_time: Optional[float] = None
    max_time: Optional[float] = None


class SplitResponse(BaseModel):
    strategy: str
    random_state: int
    train: SplitPartitionInfo
    val: SplitPartitionInfo
    test: SplitPartitionInfo
    feature_names: List[str]
