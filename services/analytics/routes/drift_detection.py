"""
Drift Detection Routes
API endpoints for drift detection functionality
"""

import logging
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime, timedelta

from services.drift_detection import drift_detector, DriftConfig
from services.email_notifications import email_service
from services.model_promotion import model_promotion_service, PromotionCriteria
from database.connection import db_manager

logger = logging.getLogger(__name__)
router = APIRouter()

class DriftDetectionRequest(BaseModel):
    """Request model for drift detection"""
    current_data: List[Dict[str, Any]]
    reference_data: Optional[List[Dict[str, Any]]] = None
    feature_columns: Optional[List[str]] = None

class ModelDriftRequest(BaseModel):
    """Request model for model drift detection"""
    current_predictions: List[Dict[str, Any]]
    reference_predictions: Optional[List[Dict[str, Any]]] = None

class BaselineRequest(BaseModel):
    """Request model for baseline establishment"""
    model_name: str
    reference_data: Optional[List[Dict[str, Any]]] = None
    hours: Optional[int] = 168  # 7 days default
    current_labels: Optional[List[str]] = None
    reference_labels: Optional[List[str]] = None

class DriftConfigRequest(BaseModel):
    """Request model for drift configuration"""
    ks_threshold: Optional[float] = None
    chi2_threshold: Optional[float] = None
    psi_threshold: Optional[float] = None
    accuracy_drop_threshold: Optional[float] = None
    f1_drop_threshold: Optional[float] = None

@router.post("/data-drift")
async def detect_data_drift(request: DriftDetectionRequest):
    """Detect data drift between reference and current data"""
    try:
        # Convert data to DataFrame
        current_df = pd.DataFrame(request.current_data)
        reference_df = pd.DataFrame(request.reference_data) if request.reference_data else None
        
        # Perform drift detection
        results = drift_detector.detect_data_drift(
            current_data=current_df,
            reference_data=reference_df,
            feature_columns=request.feature_columns
        )
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        return results
        
    except Exception as e:
        logger.error(f"Error detecting data drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model-drift")
async def detect_model_drift(request: ModelDriftRequest):
    """Detect model performance drift"""
    try:
        # Perform model drift detection
        results = drift_detector.detect_model_drift(
            current_predictions=request.current_predictions,
            reference_predictions=request.reference_predictions,
            current_labels=request.current_labels,
            reference_labels=request.reference_labels
        )
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        
        return results
        
    except Exception as e:
        logger.error(f"Error detecting model drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/set-reference")
async def set_reference_data(
    data: List[Dict[str, Any]],
    predictions: Optional[List[Dict[str, Any]]] = None
):
    """Set reference data for drift detection"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Set reference data
        drift_detector.set_reference_data(df, predictions)
        
        return {
            "message": "Reference data set successfully",
            "samples": len(df),
            "features": len(df.columns),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error setting reference data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_drift_history(days: int = Query(30, ge=1, le=365)):
    """Get drift detection history"""
    try:
        history = drift_detector.get_drift_history(days)
        
        return {
            "period_days": days,
            "total_checks": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error getting drift history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_drift_summary(days: int = Query(30, ge=1, le=365)):
    """Get drift detection summary"""
    try:
        summary = drift_detector.get_drift_summary(days)
        return summary
        
    except Exception as e:
        logger.error(f"Error getting drift summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_drift_config():
    """Get current drift detection configuration"""
    try:
        config = drift_detector.config
        
        return {
            "ks_threshold": config.ks_threshold,
            "chi2_threshold": config.chi2_threshold,
            "psi_threshold": config.psi_threshold,
            "psi_minor_threshold": config.psi_minor_threshold,
            "psi_moderate_threshold": config.psi_moderate_threshold,
            "psi_severe_threshold": config.psi_severe_threshold,
            "accuracy_drop_threshold": config.accuracy_drop_threshold,
            "f1_drop_threshold": config.f1_drop_threshold,
            "reference_window_days": config.reference_window_days,
            "detection_window_days": config.detection_window_days,
            "min_samples": config.min_samples
        }
        
    except Exception as e:
        logger.error(f"Error getting drift config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/config")
async def update_drift_config(request: DriftConfigRequest):
    """Update drift detection configuration"""
    try:
        config = drift_detector.config
        
        # Update configuration
        if request.ks_threshold is not None:
            config.ks_threshold = request.ks_threshold
        if request.chi2_threshold is not None:
            config.chi2_threshold = request.chi2_threshold
        if request.psi_threshold is not None:
            config.psi_threshold = request.psi_threshold
        if request.accuracy_drop_threshold is not None:
            config.accuracy_drop_threshold = request.accuracy_drop_threshold
        if request.f1_drop_threshold is not None:
            config.f1_drop_threshold = request.f1_drop_threshold
        
        return {
            "message": "Drift detection configuration updated successfully",
            "config": {
                "ks_threshold": config.ks_threshold,
                "chi2_threshold": config.chi2_threshold,
                "psi_threshold": config.psi_threshold,
                "accuracy_drop_threshold": config.accuracy_drop_threshold,
                "f1_drop_threshold": config.f1_drop_threshold
            }
        }
        
    except Exception as e:
        logger.error(f"Error updating drift config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_drift_alerts(days: int = Query(7, ge=1, le=30)):
    """Get drift alerts for the last N days"""
    try:
        history = drift_detector.get_drift_history(days)
        
        alerts = []
        for result in history:
            # Check for data drift alerts
            if "drift_summary" in result:
                drift_summary = result["drift_summary"]
                if drift_summary.get("total_drifted_features", 0) > 0:
                    alerts.append({
                        "type": "data_drift",
                        "timestamp": result["timestamp"],
                        "severity": "high" if drift_summary.get("severe_drift_features") else "medium",
                        "message": f"{drift_summary.get('total_drifted_features', 0)} features drifted",
                        "details": drift_summary
                    })
            
            # Check for model drift alerts
            if result.get("overall_model_drift", False):
                alerts.append({
                    "type": "model_drift",
                    "timestamp": result["timestamp"],
                    "severity": "high",
                    "message": "Model performance drift detected",
                    "details": result.get("performance_drift", {})
                })
        
        return {
            "period_days": days,
            "total_alerts": len(alerts),
            "alerts": alerts
        }
        
    except Exception as e:
        logger.error(f"Error getting drift alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-drift")
async def test_drift_detection():
    """Test drift detection with sample data"""
    try:
        import numpy as np
        
        # Generate sample reference data
        np.random.seed(42)
        reference_data = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.normal(5, 2, 1000),
            "feature3": np.random.choice(["A", "B", "C"], 1000)
        })
        
        # Generate sample current data with slight drift
        current_data = pd.DataFrame({
            "feature1": np.random.normal(0.2, 1.1, 1000),  # Slight drift
            "feature2": np.random.normal(5.5, 2.2, 1000),  # Slight drift
            "feature3": np.random.choice(["A", "B", "C"], 1000, p=[0.4, 0.3, 0.3])  # Slight drift
        })
        
        # Set reference data
        drift_detector.set_reference_data(reference_data)
        
        # Test drift detection
        results = drift_detector.detect_data_drift(current_data)
        
        return {
            "message": "Drift detection test completed",
            "test_results": results,
            "reference_samples": len(reference_data),
            "current_samples": len(current_data)
        }
        
    except Exception as e:
        import traceback
        error_detail = f"Error testing drift detection: {str(e)}"
        logger.error(f"{error_detail}\nTraceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)

class ModelPerformanceDriftRequest(BaseModel):
    """Request model for model performance drift detection"""
    old_model_predictions: List[Dict[str, Any]]
    new_model_predictions: List[Dict[str, Any]]
    ground_truth: Optional[List[str]] = None

class CheckAndRetrainRequest(BaseModel):
    """Request model for check-and-retrain endpoint"""
    model_name: str
    current_data: List[Dict[str, Any]]
    reference_data: Optional[List[Dict[str, Any]]] = None
    training_data_path: str = "latest"
    feature_columns: Optional[List[str]] = None

class ModelPromotionRequest(BaseModel):
    """Request model for model promotion"""
    model_name: str
    version: str
    force: bool = False
    test_data: Optional[List[Dict[str, Any]]] = None

class ModelEvaluationRequest(BaseModel):
    """Request model for model evaluation"""
    model_name: str
    version: str
    test_data: Optional[List[Dict[str, Any]]] = None

class ComprehensiveMetricsRequest(BaseModel):
    """Request model for comprehensive metrics calculation"""
    y_true: List[str]
    y_pred: List[str]
    y_prob: Optional[List[float]] = None
    model_name: str = "unknown"

@router.post("/model-performance-drift")
async def detect_model_performance_drift(request: ModelPerformanceDriftRequest):
    """
    Compare performance between old and new models on the same data
    
    This endpoint helps determine if a newly trained model performs better
    than the current production model.
    """
    try:
        # Detect model performance drift
        drift_results = drift_detector.detect_model_performance_drift(
            old_model_predictions=request.old_model_predictions,
            new_model_predictions=request.new_model_predictions,
            ground_truth=request.ground_truth
        )
        
        if "error" in drift_results:
            raise HTTPException(status_code=400, detail=drift_results["error"])
        
        return {
            "model_performance_drift": drift_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        error_detail = f"Error detecting model performance drift: {str(e)}"
        logger.error(f"{error_detail}\nTraceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/check-and-retrain")
async def check_drift_and_retrain(request: CheckAndRetrainRequest):
    """
    Check for drift and automatically trigger retraining if needed
    
    This is the main endpoint for production drift monitoring with auto-remediation
    """
    try:
        from database.connection import db_manager
        
        # Convert data to DataFrame
        current_df = pd.DataFrame(request.current_data)
        reference_df = pd.DataFrame(request.reference_data) if request.reference_data else None
        
        # Detect data drift
        drift_results = drift_detector.detect_data_drift(
            current_data=current_df,
            reference_data=reference_df,
            feature_columns=request.feature_columns
        )
        
        if "error" in drift_results:
            raise HTTPException(status_code=400, detail=drift_results["error"])
        
        # Use transaction for drift detection and retraining operations
        async with db_manager.transaction() as conn:
            # Store drift detection results
            drift_id = await conn.fetchval(
                """
                INSERT INTO analytics.drift_detections 
                (model_name, drift_score, drift_type, detected_at, details)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
                """,
                request.model_name,
                drift_results.get("overall_drift_score", 0.0),
                "data_drift",
                datetime.now(),
                json.dumps(drift_results)
            )
            
            # Get model predictions for performance comparison
            model_performance_drift = None
            if request.model_name:
                try:
                    # Get predictions from current model
                    import httpx
                    async with httpx.AsyncClient() as client:
                        # Convert data to text format for model prediction
                        texts = current_df['text'].tolist() if 'text' in current_df.columns else []
                        if texts:
                            # Get predictions from model-api
                            response = await client.post(
                                "http://model-api:8000/predict",
                                json={"text": texts[0], "models": [request.model_name]},  # Test with first text
                                timeout=30.0
                            )
                            if response.status_code == 200:
                                current_predictions = [response.json()]
                                
                                # For now, use the same predictions as reference (in real scenario, would get from different model)
                                # This is a placeholder - in production, you'd compare with a baseline model
                                reference_predictions = current_predictions
                            
                            # Detect model performance drift
                            model_performance_drift = drift_detector.detect_model_performance_drift(
                                old_model_predictions=reference_predictions,
                                new_model_predictions=current_predictions,
                                ground_truth=None  # No ground truth available in drift detection
                            )
                except Exception as e:
                    logger.warning(f"Could not get model predictions for drift detection: {e}")
            
            # Check if retraining should be triggered
            retrain_result = await drift_detector.trigger_retraining_if_drift(
                drift_results=drift_results,
                model_name=request.model_name,
                training_data_path=request.training_data_path
            )
            
            # If retraining was triggered, log the event
            if retrain_result.get("retraining_triggered"):
                await conn.execute(
                    """
                    INSERT INTO analytics.retrain_events 
                    (model_name, trigger_reason, drift_id, job_id, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    request.model_name,
                    "drift_detected",
                    drift_id,
                    retrain_result.get("job_id"),
                    datetime.now()
                )
        
        return {
            "drift_detection": drift_results,
            "model_performance_drift": model_performance_drift,
            "auto_retraining": retrain_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        error_detail = f"Error in check-and-retrain: {str(e)}"
        logger.error(f"{error_detail}\nTraceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/evaluate-model")
async def evaluate_model_for_promotion(request: ModelEvaluationRequest):
    """
    Evaluate a model for promotion from Staging to Production
    
    This endpoint runs comprehensive evaluation including:
    - Performance comparison with current production model
    - Statistical significance testing
    - Drift detection analysis
    - Confidence stability assessment
    """
    try:
        evaluation = await model_promotion_service.evaluate_model_for_promotion(
            model_name=request.model_name,
            version=request.version,
            test_data=request.test_data
        )
        
        from services.model_promotion import convert_numpy_types
        
        return {
            "evaluation": convert_numpy_types({
                "status": evaluation.status.value,
                "score": evaluation.score,
                "criteria_met": evaluation.criteria_met,
                "metrics": evaluation.metrics,
                "reasons": evaluation.reasons,
                "recommendations": evaluation.recommendations,
                "timestamp": evaluation.timestamp.isoformat()
            }),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        error_detail = f"Error evaluating model: {str(e)}"
        logger.error(f"{error_detail}\nTraceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/promote-model")
async def promote_model(request: ModelPromotionRequest):
    """
    Promote a model from Staging to Production
    
    This endpoint:
    1. Evaluates the model against promotion criteria (unless force=true)
    2. Promotes the model in MLflow from Staging to Production
    3. Notifies model-api to reload the promoted model
    4. Sends promotion notification email
    """
    try:
        promotion_result = await model_promotion_service.promote_model(
            model_name=request.model_name,
            version=request.version,
            force=request.force
        )
        
        return {
            "promotion": promotion_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        error_detail = f"Error promoting model: {str(e)}"
        logger.error(f"{error_detail}\nTraceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/promotion-history")
async def get_promotion_history(model_name: Optional[str] = None):
    """
    Get promotion evaluation history
    
    Args:
        model_name: Optional model name to filter history
    """
    try:
        history = model_promotion_service.get_evaluation_history(model_name)
        
        return {
            "history": history,
            "total_evaluations": len(history),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        error_detail = f"Error getting promotion history: {str(e)}"
        logger.error(f"{error_detail}\nTraceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/promotion-criteria")
async def get_promotion_criteria():
    """
    Get current model promotion criteria configuration
    """
    try:
        criteria = model_promotion_service.criteria
        
        return {
            "criteria": {
                "performance": {
                    "min_accuracy_improvement": criteria.min_accuracy_improvement,
                    "min_f1_improvement": criteria.min_f1_improvement,
                    "min_precision_improvement": criteria.min_precision_improvement,
                    "min_recall_improvement": criteria.min_recall_improvement
                },
                "statistical": {
                    "max_p_value": criteria.max_p_value,
                    "min_sample_size": criteria.min_sample_size
                },
                "drift": {
                    "max_psi_value": criteria.max_psi_value,
                    "min_prediction_agreement": criteria.min_prediction_agreement
                },
                "confidence": {
                    "max_confidence_variance": criteria.max_confidence_variance
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        error_detail = f"Error getting promotion criteria: {str(e)}"
        logger.error(f"{error_detail}\nTraceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/test-email")
async def test_email_notification():
    """Test email notification service"""
    try:
        # Create dummy drift results for testing
        test_drift_results = {
            "timestamp": datetime.now().isoformat(),
            "total_features": 2,
            "drifted_features": ["length", "word_count"],
            "drift_summary": {
                "total_drifted_features": 2,
                "drift_percentage": 100.0,
                "severe_drift_features": ["length", "word_count"],
                "moderate_drift_features": [],
                "minor_drift_features": []
            },
            "statistical_tests": {
                "length": {
                    "feature": "length",
                    "is_drifted": True,
                    "drift_severity": "severe",
                    "ks_statistic": 0.75,
                    "ks_pvalue": 9.987382295089846e-28,
                    "psi_value": 22.335889101047535,
                    "reference_mean": 28.65,
                    "current_mean": 43.15
                },
                "word_count": {
                    "feature": "word_count",
                    "is_drifted": True,
                    "drift_severity": "severe",
                    "ks_statistic": 0.5,
                    "ks_pvalue": 1.0024645454361508e-11,
                    "psi_value": 33.28329407013193,
                    "reference_mean": 6.25,
                    "current_mean": 8.25
                }
            }
        }
        
        # Send test drift alert
        drift_success = email_service.send_drift_alert(test_drift_results, "Test Model")
        
        # Create dummy performance results for testing
        test_performance_results = {
            "timestamp": datetime.now().isoformat(),
            "prediction_agreement": 0.8,
            "confidence_change": 0.15,
            "is_improved": True,
            "recommendation": "use_new_model",
            "performance_metrics": {
                "old_model": {"accuracy": 0.85, "f1_score": 0.82},
                "new_model": {"accuracy": 0.92, "f1_score": 0.89},
                "improvements": {"accuracy": 0.07, "f1_score": 0.07}
            }
        }
        
        # Send test performance alert
        performance_success = email_service.send_model_performance_alert(test_performance_results, "Test Model")
        
        return {
            "status": "success",
            "message": "Email notifications tested",
            "drift_alert_sent": drift_success,
            "performance_alert_sent": performance_success,
            "dummy_mode": email_service.dummy_mode,
            "recipients": email_service.recipient_emails,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error testing email notifications: {e}")
        raise HTTPException(status_code=500, detail=f"Email test failed: {str(e)}")

@router.get("/recent-results")
async def get_recent_drift_results(
    model_name: Optional[str] = Query(None, description="Model name to filter by"),
    hours: int = Query(24, description="Hours to look back for results")
):
    """Get recent drift detection results for auto-retrain service"""
    try:
        # Get drift history
        drift_history = drift_detector.get_drift_history(hours)
        
        # Filter by model if specified
        if model_name:
            drift_history = [d for d in drift_history if d.get('model_name') == model_name]
        
        if not drift_history:
            return {
                "data_drift_score": 0.0,
                "model_drift_score": 0.0,
                "performance_drop": 0.0,
                "last_check": None,
                "total_checks": 0
            }
        
        # Get the most recent result
        latest_result = drift_history[0]
        
        # Calculate average drift scores
        data_drift_scores = [d.get('data_drift_score', 0.0) for d in drift_history if 'data_drift_score' in d]
        model_drift_scores = [d.get('model_drift_score', 0.0) for d in drift_history if 'model_drift_score' in d]
        performance_drops = [d.get('performance_drop', 0.0) for d in drift_history if 'performance_drop' in d]
        
        return {
            "data_drift_score": max(data_drift_scores) if data_drift_scores else 0.0,
            "model_drift_score": max(model_drift_scores) if model_drift_scores else 0.0,
            "performance_drop": max(performance_drops) if performance_drops else 0.0,
            "last_check": latest_result.get('timestamp'),
            "total_checks": len(drift_history),
            "model_name": model_name,
            "hours_looked_back": hours
        }
        
    except Exception as e:
        logger.error(f"Error getting recent drift results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def drift_detection_health():
    """Health check for drift detection service"""
    try:
        # Check if reference data is available
        has_reference = drift_detector.reference_data.get('data') is not None
        
        # Get recent drift history
        recent_history = drift_detector.get_drift_history(1)
        
        return {
            "status": "healthy",
            "has_reference_data": has_reference,
            "recent_checks": len(recent_history),
            "config": {
                "ks_threshold": drift_detector.config.ks_threshold,
                "psi_threshold": drift_detector.config.psi_threshold
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking drift detection health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.post("/comprehensive-metrics")
async def calculate_comprehensive_metrics(request: ComprehensiveMetricsRequest):
    """Calculate comprehensive performance metrics for model evaluation"""
    try:
        logger.info(f"📊 [METRICS] Calculating comprehensive metrics for {request.model_name}")
        
        metrics = drift_detector.calculate_comprehensive_metrics(
            y_true=request.y_true,
            y_pred=request.y_pred,
            y_prob=request.y_prob,
            model_name=request.model_name
        )
        
        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ [METRICS] Failed to calculate comprehensive metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/production-inference")
async def get_production_inference_data(
    hours: int = Query(24, description="Hours of data to retrieve"),
    model_name: Optional[str] = Query(None, description="Specific model to filter by")
):
    """Get production inference data for retraining"""
    try:
        logger.info(f"🔍 [PRODUCTION-DATA] Retrieving production inference data for last {hours} hours")
        
        # Build query for production inference data
        query = """
            SELECT input_text, prediction, confidence, created_at, model_name
            FROM predictions
            WHERE created_at > NOW() - INTERVAL '%s hours'
        """
        params = [hours]
        
        if model_name:
            query += " AND model_name = %s"
            params.append(model_name)
        
        query += " ORDER BY created_at DESC LIMIT 10000"
        
        # Execute query
        async with db_manager.get_connection() as conn:
            rows = await conn.fetch(query, *params)
            
            if not rows:
                logger.warning("⚠️ [PRODUCTION-DATA] No production inference data found")
                return {"s3_path": None, "message": "No production data available"}
            
            # Convert to DataFrame for processing
            data = []
            for row in rows:
                data.append({
                    "input_text": row["input_text"],
                    "prediction": row["prediction"],
                    "confidence": float(row["confidence"]) if row["confidence"] else 0.0,
                    "created_at": row["created_at"].isoformat(),
                    "model_name": row["model_name"]
                })
            
            # Create training data file
            df = pd.DataFrame(data)
            
            # Generate filename with timestamp
            timestamp = int(datetime.now().timestamp())
            filename = f"production_inference_data_{timestamp}.jsonl"
            
            # Save to temporary file (in production, this would be saved to S3)
            import json
            import os
            
            temp_dir = "/tmp/retrain_data"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, filename)
            
            with open(file_path, 'w') as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row) + '\n')
            
            # In production, upload to S3 and return S3 path
            # For now, return local path
            s3_path = f"s3://ml-security/retrain-data/{filename}"
            
            logger.info(f"✅ [PRODUCTION-DATA] Generated training data: {len(data)} samples")
            
        return {
            "s3_path": s3_path,
            "local_path": file_path,
            "sample_count": len(data),
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name
        }
        
    except Exception as e:
        logger.error(f"❌ [PRODUCTION-DATA] Failed to get production inference data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/drift/production-data")
async def detect_drift_with_production_data(
    model_name: str = Query(..., description="Model name to check for drift"),
    hours: int = Query(24, description="Hours of production data to analyze")
):
    """Detect drift using actual production inference data with sliding window"""
    try:
        logger.info(f"🔍 [DRIFT] Detecting drift for {model_name} using production data")
        
        # Use the enhanced drift detection with production data
        drift_results = await drift_detector.detect_data_drift_with_production_data(model_name, hours)
        
        if "error" in drift_results:
            raise HTTPException(status_code=400, detail=drift_results["error"])
        
        # Store drift results in history
        drift_detector.drift_history.append({
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "drift_detected": len(drift_results.get("drifted_features", [])) > 0,
            "drift_score": drift_results.get("overall_drift_score", 0.0),
            "data_source": "production_inference"
        })
        
        # Send email notification if drift detected
        if drift_results.get("drift_detected", False):
            try:
                await email_service.send_drift_alert(
                    model_name=model_name,
                    drift_score=drift_results.get("overall_drift_score", 0.0),
                    drifted_features=drift_results.get("drifted_features", []),
                    timestamp=drift_results.get("timestamp")
                )
            except Exception as e:
                logger.warning(f"Failed to send drift alert email: {e}")
        
        return drift_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [DRIFT] Failed to detect drift with production data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/baseline/establish")
async def establish_baseline(request: BaselineRequest):
    """Establish baseline metrics for drift detection"""
    try:
        logger.info(f"📊 [BASELINE] Establishing baseline for {request.model_name}")
        
        # Convert reference data to DataFrame if provided
        reference_data = None
        if request.reference_data:
            reference_data = pd.DataFrame(request.reference_data)
        
        # Establish baseline
        result = await drift_detector.establish_baseline(
            model_name=request.model_name,
            reference_data=reference_data
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error establishing baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/baseline/status/{model_name}")
async def get_baseline_status(model_name: str):
    """Get baseline status for a model"""
    try:
        status = drift_detector.get_baseline_status(model_name)
        return status
        
    except Exception as e:
        logger.error(f"Error getting baseline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/baseline/list")
async def list_baselines():
    """List all established baselines"""
    try:
        baselines = {}
        for model_name in drift_detector.baseline_metrics.keys():
            baselines[model_name] = drift_detector.get_baseline_status(model_name)
        
        return {
            "baselines": baselines,
            "total_count": len(baselines)
        }
        
    except Exception as e:
        logger.error(f"Error listing baselines: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/baseline/{model_name}")
async def delete_baseline(model_name: str):
    """Delete baseline for a model"""
    try:
        if model_name not in drift_detector.baseline_metrics:
            raise HTTPException(status_code=404, detail=f"Baseline not found for {model_name}")
        
        # Remove baseline data
        del drift_detector.baseline_metrics[model_name]
        if model_name in drift_detector.reference_data:
            del drift_detector.reference_data[model_name]
        
        logger.info(f"🗑️ [BASELINE] Deleted baseline for {model_name}")
        
        return {"message": f"Baseline deleted for {model_name}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e))
