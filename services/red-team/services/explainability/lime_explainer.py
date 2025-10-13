"""
LIME Explainer for ML Security
Local Interpretable Model-agnostic Explanations for vulnerability analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import lime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for ML security
    Provides local explanations for model predictions
    """
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize LIME explainer
        
        Args:
            model: Target model to analyze
            tokenizer: Tokenizer for the model
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Initialize LIME explainer
        try:
            self.explainer = LimeTextExplainer(
                class_names=['safe', 'attack'],
                random_state=42
            )
            logger.info("✅ LIME explainer initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize LIME explainer: {e}")
            self.explainer = None
    
    def _model_predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Wrapper function for LIME explainer
        
        Args:
            texts: List of input texts
            
        Returns:
            Model prediction probabilities as numpy array
        """
        try:
            predictions = []
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1)
                    predictions.append(probs.cpu().numpy()[0])
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return np.array([[0.5, 0.5]] * len(texts))  # Fallback
    
    def explain_vulnerability(self, text: str, attack_result: Dict, 
                            num_features: int = 20, num_samples: int = 5000) -> Dict:
        """
        Explain vulnerability using LIME
        
        Args:
            text: Input text that was attacked
            attack_result: Result of the attack
            num_features: Number of features to explain
            num_samples: Number of samples for LIME
            
        Returns:
            LIME explanation dictionary
        """
        try:
            if self.explainer is None:
                return {"error": "LIME explainer not available"}
            
            # Generate LIME explanation
            explanation = self.explainer.explain_instance(
                text,
                self._model_predict_proba,
                num_features=num_features,
                num_samples=num_samples
            )
            
            # Extract explanation data
            explanation_data = self._extract_explanation_data(explanation, text, attack_result)
            
            logger.debug(f"LIME explanation generated for text: {text[:50]}...")
            return explanation_data
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {"error": str(e), "method": "LIME"}
    
    def _extract_explanation_data(self, explanation, text: str, attack_result: Dict) -> Dict:
        """
        Extract data from LIME explanation
        
        Args:
            explanation: LIME explanation object
            text: Original text
            attack_result: Attack result information
            
        Returns:
            Extracted explanation data
        """
        try:
            # Get feature weights
            feature_weights = explanation.as_list()
            
            # Separate positive and negative features
            positive_features = [f for f in feature_weights if f[1] > 0]
            negative_features = [f for f in feature_weights if f[1] < 0]
            
            # Sort by absolute weight
            positive_features.sort(key=lambda x: abs(x[1]), reverse=True)
            negative_features.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Calculate explanation metrics
            explanation_score = explanation.score
            prediction = explanation.predict_proba
            
            # Generate explanation
            explanation_data = {
                "method": "LIME",
                "text": text,
                "attack_detected": attack_result.get("detected", False),
                "explanation_score": explanation_score,
                "prediction_probabilities": {
                    "safe": float(prediction[0]),
                    "attack": float(prediction[1])
                },
                "top_positive_features": positive_features[:10],
                "top_negative_features": negative_features[:10],
                "all_features": feature_weights,
                "feature_analysis": self._analyze_features(feature_weights, attack_result),
                "vulnerability_insights": self._generate_vulnerability_insights(
                    positive_features, negative_features, attack_result
                ),
                "visualizations": self._generate_lime_visualizations(explanation, text)
            }
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"Explanation data extraction failed: {e}")
            return {"error": str(e), "method": "LIME"}
    
    def _analyze_features(self, feature_weights: List[Tuple], attack_result: Dict) -> Dict:
        """
        Analyze feature weights for patterns
        
        Args:
            feature_weights: List of (feature, weight) tuples
            attack_result: Attack result information
            
        Returns:
            Feature analysis dictionary
        """
        try:
            if not feature_weights:
                return {"error": "No features to analyze"}
            
            weights = [w for _, w in feature_weights]
            
            analysis = {
                "total_features": len(feature_weights),
                "positive_features": len([w for w in weights if w > 0]),
                "negative_features": len([w for w in weights if w < 0]),
                "max_weight": max(weights) if weights else 0,
                "min_weight": min(weights) if weights else 0,
                "avg_weight": np.mean(weights) if weights else 0,
                "weight_std": np.std(weights) if weights else 0,
                "weight_range": max(weights) - min(weights) if weights else 0
            }
            
            # Calculate feature importance distribution
            abs_weights = [abs(w) for w in weights]
            analysis["importance_distribution"] = {
                "high_importance": len([w for w in abs_weights if w > 0.1]),
                "medium_importance": len([w for w in abs_weights if 0.05 < w <= 0.1]),
                "low_importance": len([w for w in abs_weights if w <= 0.05])
            }
            
            # Identify key patterns
            analysis["patterns"] = self._identify_feature_patterns(feature_weights, attack_result)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Feature analysis failed: {e}")
            return {"error": str(e)}
    
    def _identify_feature_patterns(self, feature_weights: List[Tuple], attack_result: Dict) -> List[Dict]:
        """
        Identify patterns in feature weights
        
        Args:
            feature_weights: List of (feature, weight) tuples
            attack_result: Attack result information
            
        Returns:
            List of identified patterns
        """
        patterns = []
        
        if not feature_weights:
            return patterns
        
        # Pattern 1: Dominant features
        weights = [w for _, w in feature_weights]
        max_weight = max(abs(w) for w in weights)
        dominant_features = [f for f, w in feature_weights if abs(w) > max_weight * 0.5]
        
        if len(dominant_features) > 0:
            patterns.append({
                "type": "dominant_features",
                "description": f"Found {len(dominant_features)} dominant features",
                "features": dominant_features,
                "count": len(dominant_features)
            })
        
        # Pattern 2: Balanced features
        positive_count = len([w for w in weights if w > 0])
        negative_count = len([w for w in weights if w < 0])
        total_count = len(weights)
        
        if abs(positive_count - negative_count) / total_count < 0.2:
            patterns.append({
                "type": "balanced_features",
                "description": "Features are relatively balanced between positive and negative",
                "positive_ratio": positive_count / total_count,
                "negative_ratio": negative_count / total_count
            })
        
        # Pattern 3: Attack-specific patterns
        if not attack_result.get("detected", False):
            # Attack succeeded - look for suspicious patterns
            suspicious_features = [f for f, w in feature_weights if "inject" in f.lower() or "bypass" in f.lower()]
            if suspicious_features:
                patterns.append({
                    "type": "suspicious_features",
                    "description": "Found features with suspicious keywords",
                    "features": suspicious_features,
                    "count": len(suspicious_features)
                })
        
        return patterns
    
    def _generate_vulnerability_insights(self, positive_features: List[Tuple], 
                                       negative_features: List[Tuple], 
                                       attack_result: Dict) -> Dict:
        """
        Generate insights about vulnerability based on features
        
        Args:
            positive_features: Features with positive weights
            negative_features: Features with negative weights
            attack_result: Attack result information
            
        Returns:
            Vulnerability insights dictionary
        """
        try:
            insights = {
                "attack_success_factors": [],
                "defense_factors": [],
                "risk_assessment": "UNKNOWN",
                "recommendations": []
            }
            
            # Analyze positive features (contribute to attack)
            for feature, weight in positive_features[:5]:  # Top 5
                insights["attack_success_factors"].append({
                    "feature": feature,
                    "weight": weight,
                    "contribution": "high" if abs(weight) > 0.1 else "medium" if abs(weight) > 0.05 else "low"
                })
            
            # Analyze negative features (contribute to defense)
            for feature, weight in negative_features[:5]:  # Top 5
                insights["defense_factors"].append({
                    "feature": feature,
                    "weight": weight,
                    "contribution": "high" if abs(weight) > 0.1 else "medium" if abs(weight) > 0.05 else "low"
                })
            
            # Risk assessment
            if not attack_result.get("detected", False):
                # Attack succeeded
                if len(positive_features) > len(negative_features):
                    insights["risk_assessment"] = "HIGH"
                    insights["recommendations"].append("Model is vulnerable - strengthen detection mechanisms")
                else:
                    insights["risk_assessment"] = "MEDIUM"
                    insights["recommendations"].append("Model shows some vulnerability - review feature importance")
            else:
                # Attack detected
                insights["risk_assessment"] = "LOW"
                insights["recommendations"].append("Model successfully detected attack - maintain current defenses")
            
            # Additional recommendations based on feature analysis
            if len(positive_features) > 0:
                insights["recommendations"].append("Focus on features that contribute to attack success")
            
            if len(negative_features) > 0:
                insights["recommendations"].append("Strengthen features that contribute to defense")
            
            return insights
            
        except Exception as e:
            logger.error(f"Vulnerability insights generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_lime_visualizations(self, explanation, text: str) -> Dict:
        """
        Generate visualization data for LIME analysis
        
        Args:
            explanation: LIME explanation object
            text: Original text
            
        Returns:
            Dictionary of visualization data
        """
        try:
            visualizations = {}
            
            # Bar plot data
            visualizations["bar_plot"] = self._generate_bar_plot_data(explanation)
            
            # Text highlighting data
            visualizations["text_highlighting"] = self._generate_text_highlighting_data(explanation, text)
            
            # Feature importance plot
            visualizations["feature_importance"] = self._generate_feature_importance_data(explanation)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"LIME visualization generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_bar_plot_data(self, explanation) -> Dict:
        """Generate bar plot data for LIME features"""
        try:
            feature_weights = explanation.as_list()
            
            # Sort by absolute weight
            sorted_features = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)
            
            bar_data = []
            for i, (feature, weight) in enumerate(sorted_features[:20]):  # Top 20
                bar_data.append({
                    "feature": feature,
                    "weight": float(weight),
                    "abs_weight": float(abs(weight)),
                    "rank": i + 1,
                    "color": "red" if weight > 0 else "blue"
                })
            
            return {
                "data": bar_data,
                "title": "LIME Feature Importance",
                "x_label": "Features",
                "y_label": "Weight"
            }
            
        except Exception as e:
            logger.error(f"Bar plot data generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_text_highlighting_data(self, explanation, text: str) -> Dict:
        """Generate text highlighting data"""
        try:
            feature_weights = explanation.as_list()
            
            # Create highlighting data
            highlighting_data = []
            words = text.split()
            
            for i, word in enumerate(words):
                # Find features that contain this word
                word_features = []
                for feature, weight in feature_weights:
                    if word.lower() in feature.lower():
                        word_features.append({
                            "feature": feature,
                            "weight": float(weight)
                        })
                
                if word_features:
                    # Calculate average weight for this word
                    avg_weight = np.mean([f["weight"] for f in word_features])
                    highlighting_data.append({
                        "word": word,
                        "position": i,
                        "weight": float(avg_weight),
                        "color": self._get_highlight_color(avg_weight),
                        "features": word_features
                    })
                else:
                    highlighting_data.append({
                        "word": word,
                        "position": i,
                        "weight": 0.0,
                        "color": "gray",
                        "features": []
                    })
            
            return {
                "data": highlighting_data,
                "text": text,
                "title": "Text Highlighting by LIME Weights"
            }
            
        except Exception as e:
            logger.error(f"Text highlighting data generation failed: {e}")
            return {"error": str(e)}
    
    def _get_highlight_color(self, weight: float) -> str:
        """Get color for highlighting based on weight"""
        if weight > 0.1:
            return "red"
        elif weight > 0.05:
            return "orange"
        elif weight > 0:
            return "yellow"
        elif weight > -0.05:
            return "lightblue"
        elif weight > -0.1:
            return "blue"
        else:
            return "darkblue"
    
    def _generate_feature_importance_data(self, explanation) -> Dict:
        """Generate feature importance visualization data"""
        try:
            feature_weights = explanation.as_list()
            
            # Separate positive and negative
            positive = [(f, w) for f, w in feature_weights if w > 0]
            negative = [(f, w) for f, w in feature_weights if w < 0]
            
            # Sort by absolute weight
            positive.sort(key=lambda x: abs(x[1]), reverse=True)
            negative.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return {
                "positive_features": [{"feature": f, "weight": float(w)} for f, w in positive[:10]],
                "negative_features": [{"feature": f, "weight": float(w)} for f, w in negative[:10]],
                "title": "Feature Importance Distribution"
            }
            
        except Exception as e:
            logger.error(f"Feature importance data generation failed: {e}")
            return {"error": str(e)}
    
    def compare_explanations(self, original_text: str, adversarial_text: str) -> Dict:
        """
        Compare LIME explanations between original and adversarial text
        
        Args:
            original_text: Original text
            adversarial_text: Adversarial text
            
        Returns:
            Comparison analysis
        """
        try:
            # Get explanations for both texts
            orig_explanation = self.explain_vulnerability(original_text, {"detected": True})
            adv_explanation = self.explain_vulnerability(adversarial_text, {"detected": False})
            
            # Compare explanations
            comparison = {
                "original_explanation": orig_explanation,
                "adversarial_explanation": adv_explanation,
                "feature_changes": self._analyze_feature_changes(
                    orig_explanation.get("all_features", []),
                    adv_explanation.get("all_features", [])
                ),
                "explanation_shift": self._calculate_explanation_shift(orig_explanation, adv_explanation)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Explanation comparison failed: {e}")
            return {"error": str(e)}
    
    def _analyze_feature_changes(self, orig_features: List[Tuple], adv_features: List[Tuple]) -> Dict:
        """Analyze changes in feature weights between explanations"""
        try:
            # Create feature maps
            orig_map = {f: w for f, w in orig_features}
            adv_map = {f: w for f, w in adv_features}
            
            changes = {
                "added_features": [],
                "removed_features": [],
                "modified_features": [],
                "unchanged_features": []
            }
            
            all_features = set(orig_map.keys()) | set(adv_map.keys())
            
            for feature in all_features:
                if feature in orig_map and feature in adv_map:
                    # Feature exists in both
                    orig_weight = orig_map[feature]
                    adv_weight = adv_map[feature]
                    
                    if abs(orig_weight - adv_weight) > 0.01:  # Significant change
                        changes["modified_features"].append({
                            "feature": feature,
                            "original_weight": orig_weight,
                            "adversarial_weight": adv_weight,
                            "change": adv_weight - orig_weight,
                            "change_percentage": ((adv_weight - orig_weight) / abs(orig_weight)) * 100 if orig_weight != 0 else 0
                        })
                    else:
                        changes["unchanged_features"].append(feature)
                        
                elif feature in orig_map:
                    changes["removed_features"].append({
                        "feature": feature,
                        "weight": orig_map[feature]
                    })
                else:
                    changes["added_features"].append({
                        "feature": feature,
                        "weight": adv_map[feature]
                    })
            
            return changes
            
        except Exception as e:
            logger.error(f"Feature change analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_explanation_shift(self, orig_explanation: Dict, adv_explanation: Dict) -> Dict:
        """Calculate how the explanation shifted between original and adversarial"""
        try:
            orig_score = orig_explanation.get("explanation_score", 0)
            adv_score = adv_explanation.get("explanation_score", 0)
            
            orig_pred = orig_explanation.get("prediction_probabilities", {})
            adv_pred = adv_explanation.get("prediction_probabilities", {})
            
            shift = {
                "explanation_score_change": adv_score - orig_score,
                "prediction_change": {
                    "safe_prob_change": adv_pred.get("safe", 0) - orig_pred.get("safe", 0),
                    "attack_prob_change": adv_pred.get("attack", 0) - orig_pred.get("attack", 0)
                },
                "confidence_change": abs(adv_pred.get("attack", 0) - orig_pred.get("attack", 0)),
                "shift_magnitude": abs(adv_score - orig_score)
            }
            
            return shift
            
        except Exception as e:
            logger.error(f"Explanation shift calculation failed: {e}")
            return {"error": str(e)}
