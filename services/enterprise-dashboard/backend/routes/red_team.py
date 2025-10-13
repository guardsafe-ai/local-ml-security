"""
Enterprise Dashboard Backend - Red Team Routes
Red team testing and security analysis endpoints
"""

import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from models.requests import RedTeamTestRequest, RedTeamResultsRequest
from models.responses import AttackResult, SuccessResponse
from services.api_client import APIClient

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize API client
api_client = APIClient()


@router.get("/results")
async def get_red_team_results():
    """Get red team test results"""
    try:
        results = await api_client.get_red_team_results()
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Failed to get red team results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_red_team_status():
    """Get red team service status"""
    try:
        # This would typically get status from red team service
        return {
            "status": "running",
            "active_tests": 0,
            "last_test": None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get red team status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=SuccessResponse)
async def start_red_team_test(request: RedTeamTestRequest):
    """Start a red team test"""
    try:
        request_data = {
            "model_name": request.model_name,
            "attack_categories": request.attack_categories,
            "num_attacks": request.num_attacks,
            "config": request.config
        }
        
        result = await api_client.start_red_team_test(request_data)
        
        return SuccessResponse(
            status="success",
            message="Red team test started successfully",
            timestamp=datetime.now(),
            data=result
        )
    except Exception as e:
        logger.error(f"Failed to start red team test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=SuccessResponse)
async def stop_red_team_test():
    """Stop all red team tests"""
    try:
        # This would typically call stop endpoint on red team service
        return SuccessResponse(
            status="success",
            message="Red team tests stopped",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Failed to stop red team tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test", response_model=SuccessResponse)
async def run_red_team_test(request: RedTeamTestRequest):
    """Run a single red team test"""
    try:
        request_data = {
            "model_name": request.model_name,
            "attack_categories": request.attack_categories,
            "num_attacks": request.num_attacks,
            "config": request.config
        }
        
        result = await api_client.start_red_team_test(request_data)
        
        return SuccessResponse(
            status="success",
            message="Red team test completed",
            timestamp=datetime.now(),
            data=result
        )
    except Exception as e:
        logger.error(f"Failed to run red team test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_red_team_models():
    """Get models available for red team testing"""
    try:
        models_data = await api_client.get_available_models()
        return {
            "models": models_data.get("models", {}),
            "count": len(models_data.get("models", {}))
        }
    except Exception as e:
        logger.error(f"Failed to get red team models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_red_team_metrics():
    """Get red team testing metrics"""
    try:
        # This would typically get metrics from analytics service
        return {
            "total_tests": 0,
            "detection_rate": 0.0,
            "vulnerability_rate": 0.0,
            "last_test": None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get red team metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/results/store", response_model=SuccessResponse)
async def store_red_team_results(request: RedTeamResultsRequest):
    """Store red team test results"""
    try:
        # This would typically store results in analytics service
        return SuccessResponse(
            status="success",
            message=f"Red team results stored for test {request.test_id}",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Failed to store red team results: {e}")
        raise HTTPException(status_code=500, detail=str(e))
