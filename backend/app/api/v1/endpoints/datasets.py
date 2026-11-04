"""Datasets API endpoints for upload, validation, and sample profiling."""

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.security import sanitize_filename
from app.schemas.dataset import DatasetValidationResult
from app.services.dataset_service import MAX_UPLOAD_BYTES, DatasetService, DatasetValidationError

router = APIRouter()


@router.get("/sample", response_model=DatasetValidationResult)
def get_sample_dataset() -> DatasetValidationResult:
    """Load and return data quality profile for the standard credit card fraud sample fixture."""
    try:
        return DatasetService.load_fixture_dataset("creditcard_sample_fixture.csv")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to load sample fixture: {str(e)}",
        ) from e


@router.get("/sequence-sample", response_model=DatasetValidationResult)
def get_sequence_sample_dataset() -> DatasetValidationResult:
    """Load and return data quality profile for the sequential account fixture."""
    try:
        return DatasetService.load_fixture_dataset("sequence_account_fixture.csv")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to load sequential fixture: {str(e)}",
        ) from e


@router.post("/validate-upload", response_model=DatasetValidationResult)
async def validate_csv_upload(file: UploadFile = File(...)) -> DatasetValidationResult:
    """Upload a CSV dataset, validate structure, check quality, and return profile."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    safe_name = sanitize_filename(file.filename)
    if not safe_name.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only CSV files (.csv) are accepted.",
        )

    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1024)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File exceeds maximum upload limit of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )
        return DatasetService.validate_csv_bytes(content, filename=safe_name)
    except HTTPException:
        raise
    except DatasetValidationError as dve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(dve)) from dve
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing dataset: {str(e)}",
        ) from e
