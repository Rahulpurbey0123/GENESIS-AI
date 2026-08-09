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
from backend.recommendation.engine import RecommendationEngine, RecommendationEngineError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("genesis.api")

app = FastAPI(
    title="GENESIS-AI Engine",
    description="Dataset Intelligence Profile (DIP) v1.1 & Recommendation Engine REST API.",
    version=DIP_VERSION,
)


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint returning system status and version."""
    return {
        "status": "ok",
        "version": DIP_VERSION,
        "service": "GENESIS-AI Dataset Intelligence & Recommendation Engine",
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


@app.post("/recommend", tags=["Recommendation"])
async def create_recommendation(
    file: UploadFile = File(..., description="CSV dataset file"),
    target_column: str = Form(..., description="Target column name"),
    top_k: int = Form(5, description="Number of top pipeline recommendations to return")
) -> JSONResponse:
    """
    Generate deterministic pipeline recommendations derived from Dataset Intelligence Profile (DIP) v1.1.
    """
    filename = file.filename or "dataset.csv"
    logger.info(f"Received Recommendation request for file: '{filename}', target: '{target_column}', top_k: {top_k}")

    if not filename.lower().endswith(".csv"):
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

        engine = RecommendationEngine()
        report = engine.recommend(content, target_column=target_column, dataset_name=filename, top_k=top_k)
        logger.info(
            f"Successfully generated Recommendations for '{filename}'. Task: {report.task_type}, Top-K: {len(report.recommendations)}"
        )
        return JSONResponse(content=report.model_dump(), status_code=status.HTTP_200_OK)

    except (CSVLoaderError, DatasetValidationError, RecommendationEngineError) as e:
        logger.warning(f"Validation/Recommendation error for '{filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal error generating recommendations for '{filename}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating pipeline recommendations."
        )

