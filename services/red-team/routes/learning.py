"""
Red Team Service - Learning Routes
Continuous learning and pattern management endpoints
"""

import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from models.requests import LearningConfigRequest, CustomAttackRequest
from models.responses import LearningStatus, SuccessResponse
from services.red_team_service import RedTeamService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize service
red_team_service = RedTeamService()


@router.get("/status", response_model=LearningStatus)
async def get_learning_status():
    """Get continuous learning status"""
    try:
        status = await red_team_service.get_learning_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get learning status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enable")
async def enable_learning(request: LearningConfigRequest = None):
    """Enable continuous learning"""
    try:
        enabled = request.enabled if request else True
        await red_team_service.enable_learning(enabled)
        
        return SuccessResponse(
            status="success",
            message=f"Continuous learning {'enabled' if enabled else 'disabled'}",
            timestamp="2025-09-26T17:40:00.000000"
        )
    except Exception as e:
        logger.error(f"Failed to enable learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disable")
async def disable_learning():
    """Disable continuous learning"""
    try:
        await red_team_service.enable_learning(False)
        
        return SuccessResponse(
            status="success",
            message="Continuous learning disabled",
            timestamp="2025-09-26T17:40:00.000000"
        )
    except Exception as e:
        logger.error(f"Failed to disable learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/configure")
async def configure_learning(request: LearningConfigRequest):
    """Configure continuous learning parameters"""
    try:
        # Update learning configuration
        await red_team_service.enable_learning(request.enabled)
        
        return SuccessResponse(
            status="success",
            message="Learning configuration updated",
            timestamp="2025-09-26T17:40:00.000000",
            data={
                "enabled": request.enabled,
                "update_frequency_hours": request.update_frequency_hours,
                "learning_rate": request.learning_rate,
                "min_confidence_threshold": request.min_confidence_threshold,
                "max_patterns_per_category": request.max_patterns_per_category
            }
        )
    except Exception as e:
        logger.error(f"Failed to configure learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-custom-patterns")
async def add_custom_patterns(request: CustomAttackRequest):
    """Add custom attack patterns for learning"""
    try:
        # This would integrate with the learning system
        # For now, just return success
        
        return SuccessResponse(
            status="success",
            message=f"Added {len(request.patterns)} custom patterns to {request.category}",
            timestamp="2025-09-26T17:40:00.000000",
            data={
                "category": request.category,
                "pattern_count": len(request.patterns),
                "persist": request.persist
            }
        )
    except Exception as e:
        logger.error(f"Failed to add custom patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_learned_patterns(category: str = None):
    """Get learned attack patterns"""
    try:
        # This would return patterns from the learning system
        # For now, return empty list
        
        return {
            "patterns": [],
            "category": category,
            "count": 0,
            "timestamp": "2025-09-26T17:40:00.000000"
        }
    except Exception as e:
        logger.error(f"Failed to get learned patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/patterns")
async def clear_learned_patterns(category: str = None):
    """Clear learned attack patterns"""
    try:
        # This would clear patterns from the learning system
        # For now, just return success
        
        return SuccessResponse(
            status="success",
            message=f"Cleared learned patterns{' for ' + category if category else ''}",
            timestamp="2025-09-26T17:40:00.000000"
        )
    except Exception as e:
        logger.error(f"Failed to clear learned patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger-update")
async def trigger_learning_update():
    """Manually trigger learning update"""
    try:
        # This would trigger the learning system to update
        # For now, just return success
        
        return SuccessResponse(
            status="success",
            message="Learning update triggered",
            timestamp="2025-09-26T17:40:00.000000"
        )
    except Exception as e:
        logger.error(f"Failed to trigger learning update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_learning_performance():
    """Get learning performance metrics"""
    try:
        # This would return learning performance metrics
        # For now, return mock data
        
        return {
            "patterns_learned": 0,
            "accuracy_improvement": 0.0,
            "detection_rate_improvement": 0.0,
            "last_update": None,
            "next_update": None,
            "learning_efficiency": 0.0,
            "timestamp": "2025-09-26T17:40:00.000000"
        }
    except Exception as e:
        logger.error(f"Failed to get learning performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
