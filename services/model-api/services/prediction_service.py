"""
Prediction Service - Core prediction logic
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from models.requests import PredictionRequest
from models.responses import PredictionResponse

logger = logging.getLogger(__name__)


class PredictionService:
    """Handles prediction logic and ensemble methods"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def predict_single(self, text: str, model_name: str) -> Dict[str, Any]:
        """Make prediction using a single model"""
        if model_name not in self.model_manager.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.model_manager.models[model_name]
        if not model.loaded:
            raise ValueError(f"Model {model_name} is not loaded")
        
        start_time = time.time()
        result = model.predict(text)
        processing_time = (time.time() - start_time) * 1000
        
        result.update({
            "processing_time_ms": processing_time,
            "timestamp": datetime.now()
        })
        
        return result
    
    def predict_ensemble(self, text: str, models: List[str]) -> Dict[str, Any]:
        """Make prediction using ensemble of models"""
        if not models:
            raise ValueError("No models specified for ensemble")
        
        predictions = {}
        total_confidence = 0
        total_processing_time = 0
        
        for model_name in models:
            try:
                pred = self.predict_single(text, model_name)
                predictions[model_name] = pred
                total_confidence += pred["confidence"]
                total_processing_time += pred["processing_time_ms"]
            except Exception as e:
                logger.warning(f"Error with model {model_name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions from any model")
        
        # Simple average ensemble
        avg_confidence = total_confidence / len(predictions)
        
        # Find most confident prediction
        best_model = max(predictions.keys(), key=lambda k: predictions[k]["confidence"])
        best_prediction = predictions[best_model]
        
        return {
            "prediction": best_prediction["prediction"],
            "confidence": avg_confidence,
            "probabilities": best_prediction["probabilities"],  # Simplified
            "model_predictions": predictions,
            "ensemble_used": True,
            "processing_time_ms": total_processing_time,
            "timestamp": datetime.now()
        }
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Main prediction method"""
        start_time = time.time()
        
        try:
            if request.ensemble and request.models and len(request.models) > 1:
                # Ensemble prediction
                result = self.predict_ensemble(request.text, request.models)
            else:
                # Single model prediction
                if request.models and len(request.models) > 0:
                    model_name = request.models[0]
                else:
                    # Use best available model
                    available_models = self.model_manager.get_available_models()
                    if not available_models:
                        raise ValueError("No models available")
                    model_name = available_models[0]
                
                result = self.predict_single(request.text, model_name)
                result["model_predictions"] = {model_name: result}
                result["ensemble_used"] = False
            
            # Add request text and final processing time
            result["text"] = request.text
            result["processing_time_ms"] = (time.time() - start_time) * 1000
            
            return PredictionResponse(**result)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
