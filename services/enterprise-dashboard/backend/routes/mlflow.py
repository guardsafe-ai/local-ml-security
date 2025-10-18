"""
Enterprise Dashboard Backend - MLflow Routes
MLflow experiment and model registry endpoints
"""

import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from models.responses import MLflowExperiment, MLflowRun
from services.main_api_client import MainAPIClient

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize API client
api_client = MainAPIClient()


@router.get("/experiments")
async def get_mlflow_experiments():
    """Get MLflow experiments"""
    try:
        experiments = await api_client.get_mlflow_experiments()
        return {"experiments": experiments, "count": len(experiments)}
    except Exception as e:
        logger.error(f"Failed to get MLflow experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/detailed")
async def get_mlflow_experiments_detailed():
    """Get detailed MLflow experiments"""
    try:
        experiments = await api_client.get_mlflow_experiments()
        # Add more detailed information
        detailed_experiments = []
        for exp in experiments:
            detailed_exp = {
                **exp,
                "run_count": exp.get("run_count", 0),
                "last_run_time": exp.get("last_update_time"),
                "tags": exp.get("tags", {}),
                "description": exp.get("description", "")
            }
            detailed_experiments.append(detailed_exp)
        
        return {"experiments": detailed_experiments, "count": len(detailed_experiments)}
    except Exception as e:
        logger.error(f"Failed to get detailed MLflow experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/runs")
async def get_mlflow_runs(experiment_id: str):
    """Get runs for a specific experiment"""
    try:
        # This would typically get runs from MLflow API
        return {
            "experiment_id": experiment_id,
            "runs": [],
            "count": 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get MLflow runs for experiment {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_mlflow_models():
    """Get MLflow registered models"""
    try:
        # This would typically get models from MLflow model registry
        return {"models": [], "count": 0}
    except Exception as e:
        logger.error(f"Failed to get MLflow models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
