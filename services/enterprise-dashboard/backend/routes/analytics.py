"""
Enterprise Dashboard Backend - Analytics Routes
Analytics and reporting endpoints
"""

import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from models.requests import ModelPerformanceRequest
from models.responses import SuccessResponse
from services.main_api_client import MainAPIClient

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize API client
api_client = MainAPIClient()


@router.get("/summary")
async def get_analytics_summary():
    """Get analytics summary"""
    try:
        summary = await api_client.get_analytics_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get analytics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/overview")
async def get_analytics_overview():
    """Get analytics overview"""
    try:
        # This would typically get overview data from analytics service
        return {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_processing_time": 0.0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get analytics overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_analytics_trends():
    """Get analytics trends"""
    try:
        # This would typically get trends from analytics service
        return {
            "trends": [],
            "time_range": "24h",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get analytics trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison/{model_name}")
async def get_model_comparison(model_name: str):
    """Get model comparison data"""
    try:
        # This would typically get comparison data from analytics service
        return {
            "model_name": model_name,
            "comparison_data": {},
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get model comparison for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance/store", response_model=SuccessResponse)
async def store_model_performance(request: ModelPerformanceRequest):
    """Store model performance data"""
    try:
        # This would typically store performance data in analytics service
        return SuccessResponse(
            status="success",
            message=f"Model performance stored for {request.model_name}",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Failed to store model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== CRITICAL MISSING ANALYTICS ENDPOINTS FOR FRONTEND USERS =====

@router.get("/drift")
async def get_drift_analysis(model_name: str = None, time_range: str = "7d"):
    """
    Get data and model drift analysis
    
    Frontend Usage: Drift Detection Dashboard showing:
    - Data drift over time
    - Model performance degradation
    - Drift alerts and recommendations
    - Historical drift patterns
    """
    try:
        drift_data = await api_client.analytics.get_drift_analysis(model_name, time_range)
        
        return {
            "drift_analysis": drift_data,
            "model_name": model_name,
            "time_range": time_range,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get drift analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift/check")
async def trigger_drift_check(model_name: str):
    """
    Manually trigger drift check for a model
    
    Frontend Usage: Manual Drift Check:
    - User suspects model performance issues
    - Clicks "Check for Drift"
    - System analyzes recent data vs training data
    - Returns drift report
    """
    try:
        result = await api_client.analytics.trigger_drift_check(model_name)
        
        return {
            "status": "success",
            "message": f"Drift check triggered for {model_name}",
            "model_name": model_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to trigger drift check for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/drift/alerts")
async def get_drift_alerts(model_name: str = None, severity: str = None):
    """
    Get drift alerts and notifications
    
    Frontend Usage: Drift Alerts Panel:
    - Shows active drift alerts
    - Alert severity levels
    - Recommended actions
    - Alert history
    """
    try:
        alerts = await api_client.analytics.get_drift_alerts(model_name, severity)
        
        return {
            "drift_alerts": alerts,
            "model_name": model_name,
            "severity_filter": severity,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get drift alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auto-retrain/config")
async def get_auto_retrain_config():
    """
    Get auto-retrain configuration settings
    
    Frontend Usage: Auto-Retrain Settings Page:
    - Configure drift thresholds
    - Set retrain triggers
    - Schedule retraining
    - Enable/disable auto-retrain
    """
    try:
        config = await api_client.analytics.get_auto_retrain_config()
        
        return {
            "auto_retrain_config": config,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get auto-retrain config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/auto-retrain/config")
async def update_auto_retrain_config(config: Dict[str, Any]):
    """
    Update auto-retrain configuration
    
    Frontend Usage: Auto-Retrain Configuration:
    - User adjusts drift thresholds
    - Sets retrain schedules
    - Configures notification settings
    - Saves configuration
    """
    try:
        result = await api_client.analytics.update_auto_retrain_config(config)
        
        return {
            "status": "success",
            "message": "Auto-retrain configuration updated",
            "config": config,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to update auto-retrain config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-retrain/trigger/{model_name}")
async def trigger_manual_retrain(model_name: str):
    """
    Manually trigger model retraining
    
    Frontend Usage: Manual Retrain Button:
    - User notices performance issues
    - Clicks "Retrain Model Now"
    - System starts retraining process
    - User gets progress updates
    """
    try:
        result = await api_client.analytics.trigger_retrain(model_name)
        
        return {
            "status": "success",
            "message": f"Retraining triggered for {model_name}",
            "model_name": model_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to trigger retrain for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/attack-patterns")
async def get_attack_patterns(time_range: str = "7d"):
    """
    Get attack pattern analysis
    
    Frontend Usage: Security Analytics Dashboard:
    - Shows attack patterns over time
    - Identifies new threat types
    - Attack success rates
    - Geographic attack distribution
    """
    try:
        patterns = await api_client.analytics.get_attack_patterns(time_range)
        
        return {
            "attack_patterns": patterns,
            "time_range": time_range,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get attack patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vulnerability-analysis")
async def get_vulnerability_analysis(severity: str = None, time_range: str = "7d"):
    """
    Get vulnerability analysis and security insights
    
    Frontend Usage: Security Vulnerability Dashboard:
    - Shows security vulnerabilities
    - Severity distribution
    - Vulnerability trends
    - Remediation recommendations
    """
    try:
        analysis = await api_client.analytics.get_vulnerability_analysis(severity, time_range)
        
        return {
            "vulnerability_analysis": analysis,
            "severity_filter": severity,
            "time_range": time_range,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get vulnerability analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/trends")
async def get_performance_trends(model_name: str = None, days: int = 30):
    """
    Get model performance trends over time
    
    Frontend Usage: Performance Trends Chart:
    - Shows accuracy, precision, recall trends
    - Performance degradation detection
    - Model comparison over time
    - Performance predictions
    """
    try:
        trends = await api_client.analytics.get_performance_trends(days, model_name)
        
        return {
            "performance_trends": trends,
            "model_name": model_name,
            "days": days,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get performance trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-comparison")
async def get_model_comparison_analytics(model_name: str, days: int = 30):
    """
    Get detailed model comparison analytics
    
    Frontend Usage: Model Comparison Page:
    - Side-by-side model performance
    - Feature importance comparison
    - Training time and resource usage
    - Recommendation engine
    """
    try:
        comparison = await api_client.analytics.get_model_comparison(model_name, days)
        
        return {
            "model_comparison": comparison,
            "model_name": model_name,
            "days": days,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get model comparison analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/production/inference-data")
async def get_production_inference_data(hours: int = 24, model_name: str = None):
    """
    Get production inference data for analysis
    
    Frontend Usage: Production Analytics Dashboard:
    - Real-time inference statistics
    - Request patterns and volumes
    - Performance metrics
    - Error rates and patterns
    """
    try:
        data = await api_client.analytics.get_production_inference_data(hours, model_name)
        
        return {
            "production_data": data,
            "hours": hours,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get production inference data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/production/drift-check")
async def check_production_drift(model_name: str, hours: int = 24):
    """
    Check for drift in production data
    
    Frontend Usage: Production Drift Monitor:
    - Real-time drift detection
    - Production vs training data comparison
    - Immediate drift alerts
    - Automated response triggers
    """
    try:
        result = await api_client.analytics.detect_production_drift(model_name, hours)
        
        return {
            "drift_check_result": result,
            "model_name": model_name,
            "hours": hours,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to check production drift for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
