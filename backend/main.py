"""
FastAPI application entrypoint for GENESIS-AI Dataset Intelligence Profile (DIP) v1 engine.
"""

import logging
from typing import Dict, Any
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from backend.dataset.loader import CSVLoaderError
from backend.dataset.validator import DatasetValidationError
from backend.dataset.dip import generate_dip, DIP_VERSION


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("genesis.api")

app = FastAPI(
    title="GENESIS-AI DIP Engine",
    description="Dataset Intelligence Profile (DIP) v1 REST API for tabular AutoML pre-search profiling.",
    version=DIP_VERSION,
)


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint returning system status and version."""
    return {
        "status": "ok",
        "version": DIP_VERSION,
        "service": "GENESIS-AI Dataset Intelligence Profile Engine",
    }


@app.post("/dip", tags=["DIP"])
async def create_dataset_profile(
    file: UploadFile = File(..., description="CSV dataset file"),
    target_column: str = Form(..., description="Target column name")
) -> JSONResponse:
    """
    Generate Dataset Intelligence Profile (DIP) v1 for an uploaded CSV dataset and target column.
    """
    filename = file.filename or "dataset.csv"
    logger.info(f"Received DIP request for file: '{filename}', target_column: '{target_column}'")

    if not filename.lower().endswith(".csv"):
        logger.warning(f"Rejected non-CSV file: '{filename}'")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only CSV files are supported. Received filename: '{filename}'"
        )

    try:
        content = await file.read()
        if not content or len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset is empty (0 bytes)."
            )

        dip_result = generate_dip(content, target_column=target_column, dataset_name=filename)
        logger.info(
            f"Successfully generated DIP v1 for '{filename}'. Complexity Score: {dip_result['complexity_score']}"
        )
        return JSONResponse(content=dip_result, status_code=status.HTTP_200_OK)

    except (CSVLoaderError, DatasetValidationError) as e:
        logger.warning(f"Validation error for '{filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal error processing DIP for '{filename}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating dataset profile."
        )
