"""Data upload and validation API router."""

from typing import Any

from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import JSONResponse

from backend.app.core.errors import APIError, create_error_response
from ml.data.schema import MAX_UPLOAD_SIZE_BYTES
from ml.data.validator import validate_csv_bytes

router = APIRouter(prefix="/data", tags=["Data"])


@router.post("/upload", status_code=status.HTTP_200_OK)
async def upload_csv_dataset(
    file: UploadFile = File(..., description="CSV file containing transactions"),  # noqa: B008
    is_training: bool = Form(
        False,
        description="Whether this dataset is for supervised training (requires 'Class' column)",
    ),
) -> JSONResponse:
    """Upload and validate a transaction dataset in CSV format.

    Enforces file format, maximum size limits (50MB), schema completeness,
    data types, and domain bounds. Rejects malformed uploads with actionable errors.
    """
    # 1. Validate file extension
    filename = file.filename or "unknown.csv"
    if not filename.lower().endswith(".csv"):
        raise APIError(
            message=f"Unsupported file type for '{filename}'. Only .csv files are supported.",
            code="UNSUPPORTED_MEDIA_TYPE",
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )

    # 2. Read file bytes with size bounding
    content = await file.read()
    size_bytes = len(content)

    if size_bytes > MAX_UPLOAD_SIZE_BYTES:
        raise APIError(
            message=(
                f"File '{filename}' exceeds maximum allowed size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB "
                f"(received {size_bytes / (1024 * 1024):.2f}MB)."
            ),
            code="PAYLOAD_TOO_LARGE",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )

    # 3. Validate CSV bytes and parse
    _df, validation = validate_csv_bytes(
        content=content,
        filename=filename,
        is_training=is_training,
        max_size_bytes=MAX_UPLOAD_SIZE_BYTES,
    )

    if not validation.is_valid:
        return create_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="INVALID_DATASET",
            message=f"Validation failed for '{filename}'.",
            details={
                "errors": validation.errors,
                "missing_columns": validation.missing_columns,
                "warnings": validation.warnings,
            },
        )

    # 4. Return success report
    payload: dict[str, Any] = {
        "success": True,
        "data": {
            "filename": filename,
            "size_bytes": size_bytes,
            "row_count": validation.row_count,
            "column_count": validation.column_count,
            "columns_present": validation.columns_present,
            "duplicate_count": validation.duplicate_count,
            "summary": validation.summary,
            "warnings": validation.warnings,
        },
    }
    return JSONResponse(status_code=status.HTTP_200_OK, content=payload)
