"""
Business Metrics Service - KPI Routes
Business KPI calculation and retrieval endpoints
"""

import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from models.requests import AttackSuccessRateRequest, CostAnalysisRequest, SystemEffectivenessRequest, KPICalculationRequest
from models.responses import AttackSuccessRate, ModelDriftMetrics, CostMetrics, SystemEffectiveness, BusinessKPI, SuccessResponse
from services.metrics_calculator import MetricsCalculator
from services.drift_detector import DriftDetector

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
metrics_calculator = MetricsCalculator()
drift_detector = DriftDetector()


@router.get("/kpis", response_model=BusinessKPI)
async def get_business_kpis(
    time_range_days: int = Query(30, description="Time range in days"),
    include_recommendations: bool = Query(True, description="Include recommendations")
):
    """Get comprehensive business KPIs"""
    try:
        # Mock data - in real implementation, fetch from database
        attacks_data = []
        usage_data = []
        performance_data = []
        
        # Calculate attack success rate
        attack_metrics = metrics_calculator.calculate_attack_success_rate(attacks_data, time_range_days)
        attack_success_rate = AttackSuccessRate(**attack_metrics)
        
        # Calculate model drift (simplified)
        model_drift = []
        
        # Calculate cost metrics
        cost_metrics_data = metrics_calculator.calculate_cost_metrics(usage_data, time_range_days)
        cost_metrics = CostMetrics(**cost_metrics_data)
        
        # Calculate system effectiveness
        effectiveness_data = metrics_calculator.calculate_system_effectiveness(performance_data, time_range_days)
        system_effectiveness = SystemEffectiveness(**effectiveness_data)
        
        # Generate recommendations
        recommendations = []
        if include_recommendations:
            if attack_success_rate.success_rate > 0.8:
                recommendations.append("High attack success rate detected - consider model retraining")
            if cost_metrics.cost_per_prediction > 0.01:
                recommendations.append("High cost per prediction - optimize model efficiency")
            if system_effectiveness.overall_effectiveness < 0.7:
                recommendations.append("Low system effectiveness - review model performance")
        
        return BusinessKPI(
            timestamp=datetime.now(),
            attack_success_rate=attack_success_rate,
            model_drift=model_drift,
            cost_metrics=cost_metrics,
            system_effectiveness=system_effectiveness,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Failed to get business KPIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/attack-success-rate", response_model=AttackSuccessRate)
async def get_attack_success_rate(
    time_range_days: int = Query(30, description="Time range in days"),
    model_name: str = Query(None, description="Filter by model name"),
    category: str = Query(None, description="Filter by attack category")
):
    """Get attack success rate metrics"""
    try:
        # Mock data - in real implementation, fetch from database
        attacks_data = []
        
        # Filter by model name if provided
        if model_name:
            attacks_data = [attack for attack in attacks_data if attack.get('model_name') == model_name]
        
        # Filter by category if provided
        if category:
            attacks_data = [attack for attack in attacks_data if attack.get('category') == category]
        
        attack_metrics = metrics_calculator.calculate_attack_success_rate(attacks_data, time_range_days)
        return AttackSuccessRate(**attack_metrics)
        
    except Exception as e:
        logger.error(f"Failed to get attack success rate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-drift", response_model=List[ModelDriftMetrics])
async def get_model_drift():
    """Get model drift metrics for all models"""
    try:
        # Mock data - in real implementation, fetch from database
        model_drift = []
        
        return model_drift
        
    except Exception as e:
        logger.error(f"Failed to get model drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cost-metrics", response_model=CostMetrics)
async def get_cost_metrics(
    time_range_days: int = Query(30, description="Time range in days"),
    include_breakdown: bool = Query(True, description="Include cost breakdown")
):
    """Get cost metrics and analysis"""
    try:
        # Mock data - in real implementation, fetch from database
        usage_data = []
        
        cost_metrics_data = metrics_calculator.calculate_cost_metrics(usage_data, time_range_days)
        return CostMetrics(**cost_metrics_data)
        
    except Exception as e:
        logger.error(f"Failed to get cost metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system-effectiveness", response_model=SystemEffectiveness)
async def get_system_effectiveness(
    time_range_days: int = Query(30, description="Time range in days"),
    include_metrics: List[str] = Query(["accuracy", "response_time", "availability"], description="Metrics to include")
):
    """Get system effectiveness metrics"""
    try:
        # Mock data - in real implementation, fetch from database
        performance_data = []
        
        effectiveness_data = metrics_calculator.calculate_system_effectiveness(performance_data, time_range_days)
        return SystemEffectiveness(**effectiveness_data)
        
    except Exception as e:
        logger.error(f"Failed to get system effectiveness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations", response_model=List[str])
async def get_recommendations():
    """Get AI-powered recommendations for system optimization"""
    try:
        # Mock recommendations - in real implementation, use AI/ML to generate
        recommendations = [
            "Consider retraining models with recent attack patterns",
            "Optimize model inference for cost reduction",
            "Implement additional monitoring for model drift",
            "Review and update security policies based on attack trends"
        ]
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
