import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, status, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.dataset.loader import load_csv, CSVLoaderError
from backend.dataset.validator import validate_dataset, DatasetValidationError
from backend.dataset.dip import generate_dip, DIP_VERSION
from backend.recommendation.engine import RecommendationEngine, RecommendationEngineError
from backend.optimization.schemas import OptimizationConfig
from backend.optimization.optimizer import EvolutionaryOptimizer, EvolutionaryOptimizerError
from backend.llm.service import LLMService
from backend.llm.schemas import LLMExplanationRequest, LLMExplanationOutput
from backend.database import DatabaseService
from backend.jobs import JobManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("genesis.api")

app = FastAPI(
    title="GENESIS-AI Engine",
    description="Dataset Intelligence Profile (DIP) v1.1, Recommendation, Evolutionary Optimization, Explainability & LLM API.",
    version=DIP_VERSION,
)

# Enable CORS for frontend application with configurable origins
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB limit


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to prevent directory traversal or script execution."""
    filename = Path(filename).name
    filename = re.sub(r"[^\w\.-]", "_", filename)
    return filename or "dataset.csv"


# ============================================================
# LEGACY / BACKWARD COMPATIBLE ENDPOINTS
# ============================================================

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint returning system status and version."""
    return {
        "status": "ok",
        "version": DIP_VERSION,
        "service": "GENESIS-AI Dataset Intelligence, Recommendation, Optimization & LLM Engine",
    }


@app.post("/dip", tags=["DIP"])
async def create_dataset_profile(
    file: UploadFile = File(..., description="CSV dataset file"),
    target_column: str = Form(..., description="Target column name")
) -> JSONResponse:
    """Generate Dataset Intelligence Profile (DIP) v1 for an uploaded CSV dataset and target column."""
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
        logger.info(f"Successfully generated DIP v1 for '{filename}'. Complexity Score: {dip_result['complexity_score']}")
        return JSONResponse(content=dip_result, status_code=status.HTTP_200_OK)

    except (CSVLoaderError, DatasetValidationError) as e:
        logger.warning(f"Validation error for '{filename}': {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
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
    """Generate deterministic pipeline recommendations derived from Dataset Intelligence Profile (DIP) v1.1."""
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
        logger.info(f"Successfully generated Recommendations for '{filename}'. Task: {report.task_type}, Top-K: {len(report.recommendations)}")
        return JSONResponse(content=report.model_dump(), status_code=status.HTTP_200_OK)

    except (CSVLoaderError, DatasetValidationError, RecommendationEngineError) as e:
        logger.warning(f"Validation/Recommendation error for '{filename}': {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal error generating recommendations for '{filename}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating pipeline recommendations."
        )


@app.post("/optimize", tags=["Optimization"])
async def create_optimization(
    file: UploadFile = File(..., description="CSV dataset file"),
    target_column: str = Form(..., description="Target column name"),
    mode: str = Form("genesis", description="Optimization mode: 'genesis' or 'baseline'"),
    top_k: int = Form(2, description="Number of top candidate pipelines for GENESIS mode"),
    population_size: int = Form(20, description="Population size for Genetic Algorithm"),
    generations: int = Form(10, description="Number of GA generations"),
    max_evaluations: int = Form(200, description="Maximum evaluations budget"),
    mutation_rate: float = Form(0.10, description="Hyperparameter mutation rate"),
    pipeline_mutation_rate: float = Form(0.10, description="Model-family pipeline mutation rate"),
    random_state: int = Form(42, description="Random seed")
) -> JSONResponse:
    """Run Evolutionary Pipeline Optimization in GENESIS or BASELINE mode."""
    filename = file.filename or "dataset.csv"
    logger.info(f"Received Optimization request for file: '{filename}', mode: '{mode}', target: '{target_column}'")

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

        config = OptimizationConfig(
            mode=mode,
            top_k=top_k,
            population_size=population_size,
            generations=generations,
            max_evaluations=max_evaluations,
            mutation_rate=mutation_rate,
            pipeline_mutation_rate=pipeline_mutation_rate,
            random_state=random_state
        )
        optimizer = EvolutionaryOptimizer(config=config)
        result = optimizer.optimize(content, target_column=target_column, dataset_name=filename)

        logger.info(f"Successfully executed Optimization for '{filename}'. Mode: {result.mode}, Best Fitness: {result.best_fitness}")
        return JSONResponse(content=result.model_dump(), status_code=status.HTTP_200_OK)

    except (CSVLoaderError, DatasetValidationError, EvolutionaryOptimizerError, ValueError) as e:
        logger.warning(f"Validation/Optimization error for '{filename}': {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal error executing optimization for '{filename}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while running evolutionary optimization."
        )


@app.get("/config/llm", tags=["LLM Explanation"])
@app.get("/api/config/llm", tags=["LLM Explanation"])
@app.get("/llm/status", tags=["LLM Explanation"])
@app.get("/api/llm/status", tags=["LLM Explanation"])
async def get_llm_status_api() -> JSONResponse:
    """Retrieve safe metadata for current LLM provider configuration status."""
    raw_provider = os.getenv("LLM_PROVIDER", "mock").lower()
    model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
    has_key = bool(os.getenv("OPENROUTER_API_KEY"))

    if raw_provider in ("mock", "test", "offline"):
        content = {
            "provider": "mock",
            "mode": "mock",
            "configured": True,
            "model": None,
            "has_api_key": has_key
        }
    elif raw_provider == "openrouter":
        if has_key:
            content = {
                "provider": "openrouter",
                "mode": "real",
                "configured": True,
                "model": model,
                "has_api_key": True
            }
        else:
            content = {
                "provider": "openrouter",
                "mode": "real",
                "configured": False,
                "model": model,
                "has_api_key": False,
                "error": "LLM provider is not configured"
            }
    else:
        content = {
            "provider": raw_provider,
            "mode": "invalid",
            "configured": False,
            "model": None,
            "has_api_key": False,
            "error": "Unsupported LLM provider configuration."
        }

    return JSONResponse(content=content, status_code=status.HTTP_200_OK)


@app.post("/explain/llm", tags=["LLM Explanation"])
async def generate_llm_explanation_endpoint(
    request: LLMExplanationRequest = Body(..., description="LLM Explanation Request payload with evidence and mode")
) -> JSONResponse:
    """Generate an evidence-grounded LLM interpretation for Week 5 structured output."""
    logger.info(f"Received LLM Explanation request for mode: '{request.mode}'")
    try:
        service = LLMService()
        output = service.explain(
            raw_evidence=request.evidence,
            mode=request.mode,
            provider_override=request.provider_override,
            model_override=request.model_override
        )
        return JSONResponse(content=output.model_dump(), status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Internal error generating LLM explanation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred generating LLM explanation: {str(e)}"
        )


# ============================================================
# WEEK 8 DASHBOARD & END-TO-END APPLICATION ENDPOINTS
# ============================================================

@app.post("/datasets/upload", tags=["Dashboard Datasets"])
@app.post("/api/datasets/upload", tags=["Dashboard Datasets"])
async def upload_dataset_api(
    file: UploadFile = File(..., description="CSV dataset file")
) -> JSONResponse:
    """Upload and validate a CSV dataset file. Returns dataset metadata and suggested target column."""
    filename = sanitize_filename(file.filename or "dataset.csv")
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only .csv files are supported."
        )

    try:
        content = await file.read()
        if not content or len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size exceeds maximum allowed limit of 50 MB."
            )

        # Parse CSV to validate content
        df = load_csv(content)
        if df.empty or len(df) == 0:
            raise DatasetValidationError("Dataset is empty.")
        if len(df.columns) < 2:
            raise DatasetValidationError(f"Dataset must have at least 2 columns. Found {len(df.columns)}.")

        content_hash = hashlib.sha256(content).hexdigest()
        dataset_id = f"ds_{content_hash[:12]}"
        saved_filepath = UPLOAD_DIR / f"{dataset_id}_{filename}"

        with open(saved_filepath, "wb") as f:
            f.write(content)
        columns = list(df.columns)

        rows, cols_count = df.shape

        # Intelligent Generic Target Suggestion Heuristic (P1 Target Suggestion Audit)
        from backend.dataset.contract import detect_identifier_columns
        identifier_cols = detect_identifier_columns(df)
        candidate_cols = [c for c in columns if c not in identifier_cols]

        suggested_target = None
        lower_cols = [c.lower() for c in candidate_cols]
        target_keywords = ["target", "survived", "label", "class", "outcome", "y", "churn", "price", "status", "dependent"]

        for kw in target_keywords:
            if kw in lower_cols:
                suggested_target = candidate_cols[lower_cols.index(kw)]
                break

        if not suggested_target and candidate_cols:
            suggested_target = candidate_cols[-1]
        elif not suggested_target and len(columns) > 0:
            suggested_target = columns[-1]


        record = DatabaseService.save_dataset(
            dataset_id=dataset_id,
            name=filename,
            content_hash=content_hash,
            filepath=str(saved_filepath),
            rows=rows,
            columns=cols_count,
            features=columns,
            suggested_target=suggested_target
        )
        return JSONResponse(content=record, status_code=status.HTTP_200_OK)

    except (CSVLoaderError, DatasetValidationError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process uploaded dataset: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to upload dataset. Please check that the file is a valid CSV."
        )


@app.get("/datasets/{id}", tags=["Dashboard Datasets"])
@app.get("/api/datasets/{id}", tags=["Dashboard Datasets"])
async def get_dataset_api(id: str) -> JSONResponse:
    """Retrieve metadata for a previously uploaded dataset."""
    record = DatabaseService.get_dataset(id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")
    return JSONResponse(content=record, status_code=status.HTTP_200_OK)


@app.post("/datasets/{id}/profile", tags=["Dashboard DIP"])
@app.post("/api/datasets/{id}/profile", tags=["Dashboard DIP"])
async def profile_dataset_api(
    id: str,
    target_column: str = Form(..., description="Confirmed target column name")
) -> JSONResponse:
    """Generate Dataset Intelligence Profile (DIP) v1.1 for dataset and confirmed target column."""
    record = DatabaseService.get_dataset(id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")

    if target_column not in record["features"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{target_column}' is not present in dataset features."
        )

    try:
        with open(record["filepath"], "rb") as f:
            csv_bytes = f.read()

        profile = generate_dip(csv_bytes, target_column=target_column, dataset_name=record["name"])
        DatabaseService.save_dip_profile(id, target_column, profile)
        return JSONResponse(content=profile, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"DIP generation failed for dataset '{id}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset profiling failed: {str(e)}"
        )


@app.get("/datasets/{id}/profile", tags=["Dashboard DIP"])
@app.get("/api/datasets/{id}/profile", tags=["Dashboard DIP"])
async def get_dataset_profile_api(
    id: str,
    target_column: Optional[str] = None
) -> JSONResponse:
    """Retrieve stored Dataset Intelligence Profile (DIP) for dataset."""
    try:
        profile = DatabaseService.get_dip_profile(id, target_column)
        if not profile:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="DIP profile not found for dataset.")
        return JSONResponse(content=profile, status_code=status.HTTP_200_OK)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))


@app.get("/datasets/{id}/recommendations", tags=["Dashboard Recommendations"])
@app.get("/api/datasets/{id}/recommendations", tags=["Dashboard Recommendations"])
@app.post("/datasets/{id}/recommendations", tags=["Dashboard Recommendations"])
@app.post("/api/datasets/{id}/recommendations", tags=["Dashboard Recommendations"])
async def get_dataset_recommendations_api(
    id: str,
    target_column: Optional[str] = Form(None, description="Optional target column name")
) -> JSONResponse:
    """Retrieve model recommendations and search-space reduction for dataset WITHOUT launching an experiment (Read-Only)."""
    record = DatabaseService.get_dataset(id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")

    target = target_column or record.get("suggested_target")
    if not target or target not in record["features"]:
        target = record["features"][-1]

    try:
        with open(record["filepath"], "rb") as f:
            csv_bytes = f.read()

        engine = RecommendationEngine()
        report = engine.recommend(csv_bytes, target_column=target, dataset_name=record["name"])
        return JSONResponse(content=report.model_dump(), status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error fetching dataset recommendations for '{id}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations."
        )


@app.post("/experiments", tags=["Dashboard Experiments"])
@app.post("/api/experiments", tags=["Dashboard Experiments"])
async def create_experiment_api(
    dataset_id: str = Form(..., description="Uploaded dataset ID"),
    target_column: str = Form(..., description="Target column name"),
    mode: str = Form("genesis", description="Optimization mode: 'genesis' or 'baseline'"),
    top_k: int = Form(2, description="Top K candidate pipelines for GENESIS mode"),
    population_size: int = Form(20, description="GA population size"),
    generations: int = Form(10, description="GA generations"),
    max_evaluations: int = Form(200, description="Max evaluation budget"),
    mutation_rate: float = Form(0.10, description="Mutation rate"),
    random_state: int = Form(42, description="Random state seed")
) -> JSONResponse:
    """Submit a non-blocking background optimization experiment."""
    record = DatabaseService.get_dataset(dataset_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found.")

    if target_column not in record["features"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{target_column}' is not present in dataset features."
        )

    config_dict = {
        "top_k": top_k,
        "population_size": population_size,
        "generations": generations,
        "max_evaluations": max_evaluations,
        "mutation_rate": mutation_rate,
        "random_state": random_state
    }

    exp_record = JobManager.submit_experiment(
        dataset_id=dataset_id,
        target_column=target_column,
        mode=mode,
        config_dict=config_dict
    )
    return JSONResponse(content=exp_record, status_code=status.HTTP_200_OK)


@app.get("/experiments", tags=["Dashboard Experiments"])
@app.get("/api/experiments", tags=["Dashboard Experiments"])
async def list_experiments_api() -> JSONResponse:
    """Retrieve history of all executed experiments."""
    experiments = DatabaseService.list_experiments()
    return JSONResponse(content=experiments, status_code=status.HTTP_200_OK)


@app.get("/experiments/{id}", tags=["Dashboard Experiments"])
@app.get("/api/experiments/{id}", tags=["Dashboard Experiments"])
async def get_experiment_api(id: str) -> JSONResponse:
    """Retrieve current status and progress of an experiment."""
    experiment = DatabaseService.get_experiment(id)
    if not experiment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found.")
    return JSONResponse(content=experiment, status_code=status.HTTP_200_OK)


@app.get("/experiments/{id}/recommendations", tags=["Dashboard Recommendations"])
@app.get("/api/experiments/{id}/recommendations", tags=["Dashboard Recommendations"])
async def get_experiment_recommendations_api(id: str) -> JSONResponse:
    """Retrieve model recommendations and search-space reduction details for experiment using stored config."""
    experiment = DatabaseService.get_experiment(id)
    if not experiment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found.")

    dataset = DatabaseService.get_dataset(experiment["dataset_id"])
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Associated dataset not found.")

    try:
        with open(dataset["filepath"], "rb") as f:
            csv_bytes = f.read()

        exp_config = experiment.get("config", {})
        exp_mode = str(experiment.get("mode", "genesis")).lower()
        engine = RecommendationEngine()

        if exp_mode == "genesis":
            top_k = exp_config.get("top_k", 2)
            report = engine.recommend(csv_bytes, target_column=experiment["target_column"], dataset_name=dataset["name"], top_k=top_k)
            report_dict = report.model_dump()
            report_dict["mode"] = "genesis"
            report_dict["recommendation_mode_context"] = f"Mode 'GENESIS' restricts search space to Top-K={top_k} recommended candidates."
            report_dict["top_k_applied"] = True
            report_dict["top_k"] = top_k
        else:
            report = engine.recommend(csv_bytes, target_column=experiment["target_column"], dataset_name=dataset["name"], top_k=100)
            report_dict = report.model_dump()
            report_dict["mode"] = exp_mode
            report_dict["recommendation_mode_context"] = f"Mode '{exp_mode.upper()}' uses all compatible candidate pipelines ({report.candidate_count_after_filtering}) without GENESIS Top-K restriction."
            report_dict["top_k_applied"] = False
            report_dict["top_k"] = report.candidate_count_after_filtering

        return JSONResponse(content=report_dict, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error fetching recommendations for experiment '{id}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations."
        )


@app.get("/experiments/{id}/results", tags=["Dashboard Results"])
@app.get("/api/experiments/{id}/results", tags=["Dashboard Results"])
async def get_experiment_results_api(id: str) -> JSONResponse:
    """Retrieve evaluation results and best pipeline metrics for completed experiment."""
    experiment = DatabaseService.get_experiment(id)
    if not experiment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found.")

    results = DatabaseService.get_experiment_results(id)
    if not results:
        if experiment["status"] == "RUNNING":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Experiment is still running.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Results not available for this experiment.")

    return JSONResponse(content=results, status_code=status.HTTP_200_OK)


@app.get("/experiments/{id}/explanations", tags=["Dashboard Explainability"])
@app.get("/api/experiments/{id}/explanations", tags=["Dashboard Explainability"])
async def get_experiment_explanations_api(id: str) -> JSONResponse:
    """Retrieve SHAP, global/local feature importances, and diagnostic plots for experiment."""
    experiment = DatabaseService.get_experiment(id)
    if not experiment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found.")

    explanations = DatabaseService.get_experiment_explanations(id)
    if not explanations:
        if experiment["status"] == "RUNNING":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Experiment is still running.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Explanations not available for this experiment.")

    return JSONResponse(content=explanations, status_code=status.HTTP_200_OK)


@app.post("/experiments/{id}/chat", tags=["Dashboard AI Assistant"])
@app.post("/api/experiments/{id}/chat", tags=["Dashboard AI Assistant"])
async def chat_experiment_api(
    id: str,
    prompt: str = Form(..., description="User question for AI Assistant")
) -> JSONResponse:
    """Evidence-grounded AI Assistant providing scientific explanations based exclusively on stored experiment facts."""
    experiment = DatabaseService.get_experiment(id)
    if not experiment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found.")

    results = DatabaseService.get_experiment_results(id) or {}
    explanations = DatabaseService.get_experiment_explanations(id) or {}
    dip = DatabaseService.get_dip_profile(experiment["dataset_id"], experiment.get("target_column")) or {}

    # Safely fetch recommendation evidence if available for this dataset
    recommendations_summary = {}
    try:
        from backend.recommendation.engine import RecommendationEngine
        ds_rec = DatabaseService.get_dataset(experiment.get("dataset_id"))
        if ds_rec and os.path.exists(ds_rec.get("filepath", "")):
            with open(ds_rec["filepath"], "rb") as f:
                c_bytes = f.read()
            rec_report = RecommendationEngine().recommend(
                c_bytes,
                target_column=experiment.get("target_column", "target"),
                dataset_name=ds_rec.get("name", "dataset.csv")
            )
            recommendations_summary = {
                "top_recommendations": [
                    {"name": r.model_name, "score": r.score}
                    for r in rec_report.recommendations
                ],
                "search_space_reduction": round(rec_report.search_space_reduction, 4)
            }
    except Exception:
        pass

    global_imp_raw = explanations.get("global_importance", {})
    global_imp_list = []
    if isinstance(global_imp_raw, dict):
        feats = global_imp_raw.get("features", [])
        scores = global_imp_raw.get("scores", [])
        for idx, f in enumerate(feats[:5]):
            s = scores[idx] if idx < len(scores) else None
            global_imp_list.append({"feature": f, "importance": s, "rank": idx + 1})

    best_pipe = results.get("best_pipeline", {})
    m_dict = results.get("metrics", {})
    m_primary = "f1" if "f1" in m_dict else ("rmse" if "rmse" in m_dict else "score")
    m_score = m_dict.get(m_primary)

    evidence = {
        "experiment_id": experiment.get("id"),
        "dataset_id": experiment.get("dataset_id"),
        "dataset_name": experiment.get("dataset_name"),
        "target_column": experiment.get("target_column"),
        "mode": experiment.get("mode"),
        "pipeline_id": best_pipe.get("pipeline_id", "custom_pipeline"),
        "model_name": best_pipe.get("model_name", "Unknown Estimator"),
        "task_type": "regression" if "rmse" in m_dict or "mae" in m_dict else "classification",
        "metric": m_primary,
        "model_score": m_score,
        "method": explanations.get("method", "permutation_importance"),
        "global_importance": global_imp_list,
        "status": experiment.get("status", "COMPLETED"),
        "dip_summary": {
            "rows": dip.get("dataset", {}).get("rows", 0),
            "columns": dip.get("dataset", {}).get("columns", 0),
            "complexity_score": dip.get("complexity_score")
        },
        "recommendation_summary": recommendations_summary,
        "efficiency": results.get("efficiency", {}),
        "metrics": m_dict
    }

    chat_id = f"chat_{uuid.uuid4().hex[:8]}"

    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    api_key = os.getenv("OPENROUTER_API_KEY")

    if provider not in ("mock", "test", "offline", "openrouter"):
        fallback_data = {
            "explanation": "The AI Assistant is temporarily unavailable. Unsupported LLM provider configuration. Supported providers are 'mock' and 'openrouter'. Please try again.",
            "evidence_used": evidence,
            "llm_provider": provider,
            "llm_model": None,
            "question_intent": "GENERAL_EXPERIMENT",
            "mode": "evidence_grounded",
            "is_fallback": True,
            "validation_status": "FALLBACK",
            "warnings": ["Unsupported LLM provider configuration."]
        }
        DatabaseService.save_chat(experiment_id=id, chat_id=chat_id, prompt=prompt, response=fallback_data)
        return JSONResponse(content=fallback_data, status_code=status.HTTP_200_OK)

    if provider == "openrouter" and not api_key:
        fallback_data = {
            "explanation": "The AI Assistant is temporarily unavailable. The configured LLM provider could not process this request. Please try again.",
            "evidence_used": evidence,
            "llm_provider": "openrouter",
            "llm_model": os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001"),
            "question_intent": "GENERAL_EXPERIMENT",
            "mode": "evidence_grounded",
            "is_fallback": True,
            "validation_status": "FALLBACK",
            "warnings": ["OPENROUTER_API_KEY is missing."]
        }
        DatabaseService.save_chat(experiment_id=id, chat_id=chat_id, prompt=prompt, response=fallback_data)
        return JSONResponse(content=fallback_data, status_code=status.HTTP_200_OK)

    try:
        llm_service = LLMService()
        output: LLMExplanationOutput = llm_service.explain(
            raw_evidence=evidence,
            mode="technical",
            user_prompt=prompt
        )
        response_data = output.model_dump()
        intent_val = getattr(output.structured_explanation, "question_intent", "GENERAL_EXPERIMENT")
        response_data["explanation"] = f"{output.structured_explanation.summary}\n\n{output.structured_explanation.model_explanation}"
        response_data["evidence_used"] = evidence
        response_data["llm_provider"] = output.llm_provider
        response_data["llm_model"] = output.llm_model
        response_data["question_intent"] = intent_val
        response_data["is_fallback"] = output.metadata.get("is_fallback", False) if isinstance(output.metadata, dict) else False
        response_data["validation_status"] = output.validation_status
        DatabaseService.save_chat(experiment_id=id, chat_id=chat_id, prompt=prompt, response=response_data)
        return JSONResponse(content=response_data, status_code=status.HTTP_200_OK)
    except Exception as e:
        from backend.llm.client import _sanitize_text
        clean_err = _sanitize_text(str(e), api_key)
        logger.error(
            f"LLM explanation endpoint error. Provider: '{provider}', Model: '{os.getenv('OPENROUTER_MODEL')}', "
            f"Experiment: '{id}', Error: {clean_err}",
            exc_info=True
        )
        from backend.llm.client import MockLLMClient
        intent = MockLLMClient()._detect_intent(prompt)
        m = results.get("metrics", {}) or {}
        metric_str = "the evaluation metric is unavailable."
        for metric_key in ["f1", "accuracy", "precision", "recall", "roc_auc", "balanced_accuracy", "mae", "rmse", "r2"]:
            val = m.get(metric_key)
            if val is not None:
                metric_str = f"the best pipeline achieved a {metric_key.upper()} score of {val}."
                break

        fallback_data = {
            "explanation": f"The AI Assistant is temporarily unavailable. Based on verified experiment evidence for '{experiment.get('dataset_name')}', target column '{experiment.get('target_column')}' was evaluated using '{results.get('best_pipeline', {}).get('model_name', 'model')}'. {metric_str}",
            "evidence_used": evidence,
            "llm_provider": provider,
            "llm_model": os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001"),
            "question_intent": intent,
            "mode": "evidence_grounded",
            "is_fallback": True,
            "validation_status": "FALLBACK",
            "warnings": ["LLM service endpoint unavailable. Grounded fallback active."]
        }

        DatabaseService.save_chat(experiment_id=id, chat_id=chat_id, prompt=prompt, response=fallback_data)
        return JSONResponse(content=fallback_data, status_code=status.HTTP_200_OK)
