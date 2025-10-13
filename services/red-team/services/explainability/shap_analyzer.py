"""
SHAP Analyzer for ML Security
Game-theoretic explainability for vulnerability analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    SHAP (SHapley Additive exPlanations) analyzer for ML security
    Provides game-theoretic explanations for model vulnerabilities
    """
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize SHAP analyzer
        
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
        
        # Initialize SHAP explainer
        try:
            self.explainer = shap.Explainer(self._model_predict, self.tokenizer)
            logger.info("✅ SHAP explainer initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def _model_predict(self, texts: List[str]) -> np.ndarray:
        """
        Wrapper function for SHAP explainer
        
        Args:
            texts: List of input texts
            
        Returns:
            Model predictions as numpy array
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
            return np.zeros((len(texts), 2))  # Fallback
    
    def explain_vulnerability(self, text: str, attack_result: Dict) -> Dict:
        """
        Explain why a vulnerability exists using SHAP
        
        Args:
            text: Input text that was attacked
            attack_result: Result of the attack
            
        Returns:
            SHAP explanation dictionary
        """
        try:
            if self.explainer is None:
                return {"error": "SHAP explainer not available"}
            
            # Calculate SHAP values
            shap_values = self.explainer([text])
            
            # Tokenize text for word-level analysis
            tokens = self.tokenizer.tokenize(text)
            
            # Extract word-level SHAP values
            word_importance = self._extract_word_importance(shap_values, tokens)
            
            # Generate explanation
            explanation = {
                "method": "SHAP",
                "text": text,
                "attack_detected": attack_result.get("detected", False),
                "word_contributions": word_importance,
                "attack_success_factors": self._identify_success_factors(word_importance, attack_result),
                "vulnerability_analysis": self._analyze_vulnerability(word_importance, attack_result),
                "visualizations": self._generate_visualizations(shap_values, text, tokens)
            }
            
            logger.debug(f"SHAP explanation generated for text: {text[:50]}...")
            return explanation
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {"error": str(e), "method": "SHAP"}
    
    def _extract_word_importance(self, shap_values, tokens: List[str]) -> Dict:
        """
        Extract word-level importance from SHAP values
        
        Args:
            shap_values: SHAP values from explainer
            tokens: Tokenized text
            
        Returns:
            Dictionary of word importance
        """
        try:
            # Get SHAP values for the prediction class
            if len(shap_values.shape) > 2:
                # Multi-class case - use the predicted class
                class_shap_values = shap_values[0, :, 1]  # Assuming binary classification
            else:
                class_shap_values = shap_values[0]
            
            # Align with tokens (handle subword tokenization)
            word_importance = []
            token_idx = 0
            
            for i, token in enumerate(tokens):
                if token.startswith('##'):
                    # Subword token - add to previous word
                    if word_importance:
                        word_importance[-1]["shap_value"] += class_shap_values[i]
                        word_importance[-1]["token"] += token[2:]  # Remove ##
                else:
                    # New word
                    word_importance.append({
                        "word": token,
                        "token": token,
                        "shap_value": float(class_shap_values[i]),
                        "position": token_idx
                    })
                    token_idx += 1
            
            return {
                "words": word_importance,
                "total_importance": sum(abs(w["shap_value"]) for w in word_importance),
                "max_importance": max(abs(w["shap_value"]) for w in word_importance) if word_importance else 0
            }
            
        except Exception as e:
            logger.error(f"Word importance extraction failed: {e}")
            return {"words": [], "total_importance": 0, "max_importance": 0}
    
    def _identify_success_factors(self, word_importance: Dict, attack_result: Dict) -> List[Dict]:
        """
        Identify factors that contributed to attack success
        
        Args:
            word_importance: Word importance from SHAP
            attack_result: Attack result information
            
        Returns:
            List of success factors
        """
        factors = []
        
        if not attack_result.get("detected", False):
            # Attack succeeded - identify contributing words
            words = word_importance.get("words", [])
            
            # Sort by absolute SHAP value
            sorted_words = sorted(words, key=lambda x: abs(x["shap_value"]), reverse=True)
            
            # Identify top contributing words
            for i, word_info in enumerate(sorted_words[:5]):  # Top 5
                if abs(word_info["shap_value"]) > 0.1:  # Threshold
                    factors.append({
                        "word": word_info["word"],
                        "position": word_info["position"],
                        "contribution": word_info["shap_value"],
                        "rank": i + 1,
                        "type": "high_contribution" if word_info["shap_value"] > 0 else "high_negative_contribution"
                    })
        
        return factors
    
    def _analyze_vulnerability(self, word_importance: Dict, attack_result: Dict) -> Dict:
        """
        Analyze vulnerability patterns
        
        Args:
            word_importance: Word importance from SHAP
            attack_result: Attack result information
            
        Returns:
            Vulnerability analysis
        """
        words = word_importance.get("words", [])
        
        if not words:
            return {"error": "No words to analyze"}
        
        # Calculate statistics
        positive_contributions = [w for w in words if w["shap_value"] > 0]
        negative_contributions = [w for w in words if w["shap_value"] < 0]
        
        analysis = {
            "total_words": len(words),
            "positive_contributions": len(positive_contributions),
            "negative_contributions": len(negative_contributions),
            "contribution_ratio": len(positive_contributions) / len(words) if words else 0,
            "vulnerability_score": self._calculate_vulnerability_score(word_importance, attack_result),
            "risk_level": self._assess_risk_level(word_importance, attack_result)
        }
        
        # Identify patterns
        analysis["patterns"] = self._identify_patterns(words, attack_result)
        
        return analysis
    
    def _calculate_vulnerability_score(self, word_importance: Dict, attack_result: Dict) -> float:
        """
        Calculate vulnerability score based on SHAP values
        
        Args:
            word_importance: Word importance from SHAP
            attack_result: Attack result information
            
        Returns:
            Vulnerability score (0-1)
        """
        try:
            words = word_importance.get("words", [])
            if not words:
                return 0.0
            
            # Base score from attack success
            base_score = 0.0 if attack_result.get("detected", False) else 0.5
            
            # Add score based on SHAP value distribution
            shap_values = [abs(w["shap_value"]) for w in words]
            max_shap = max(shap_values) if shap_values else 0
            avg_shap = np.mean(shap_values) if shap_values else 0
            
            # Vulnerability increases with high SHAP values
            shap_score = min(max_shap * 2, 0.5)  # Cap at 0.5
            
            return min(base_score + shap_score, 1.0)
            
        except Exception as e:
            logger.error(f"Vulnerability score calculation failed: {e}")
            return 0.0
    
    def _assess_risk_level(self, word_importance: Dict, attack_result: Dict) -> str:
        """
        Assess risk level based on analysis
        
        Args:
            word_importance: Word importance from SHAP
            attack_result: Attack result information
            
        Returns:
            Risk level string
        """
        vulnerability_score = self._calculate_vulnerability_score(word_importance, attack_result)
        
        if vulnerability_score >= 0.8:
            return "CRITICAL"
        elif vulnerability_score >= 0.6:
            return "HIGH"
        elif vulnerability_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _identify_patterns(self, words: List[Dict], attack_result: Dict) -> List[Dict]:
        """
        Identify patterns in word contributions
        
        Args:
            words: List of word information
            attack_result: Attack result information
            
        Returns:
            List of identified patterns
        """
        patterns = []
        
        if not words:
            return patterns
        
        # Pattern 1: High concentration of importance
        high_importance_words = [w for w in words if abs(w["shap_value"]) > 0.2]
        if len(high_importance_words) > len(words) * 0.3:
            patterns.append({
                "type": "high_concentration",
                "description": "High concentration of important words",
                "count": len(high_importance_words),
                "percentage": len(high_importance_words) / len(words)
            })
        
        # Pattern 2: Position-based patterns
        early_words = [w for w in words[:len(words)//3] if abs(w["shap_value"]) > 0.1]
        if len(early_words) > 0:
            patterns.append({
                "type": "early_importance",
                "description": "Important words appear early in text",
                "count": len(early_words)
            })
        
        # Pattern 3: Negative contributions
        negative_words = [w for w in words if w["shap_value"] < -0.1]
        if len(negative_words) > 0:
            patterns.append({
                "type": "negative_contributions",
                "description": "Words with negative contributions to prediction",
                "count": len(negative_words)
            })
        
        return patterns
    
    def _generate_visualizations(self, shap_values, text: str, tokens: List[str]) -> Dict:
        """
        Generate visualization data for SHAP analysis
        
        Args:
            shap_values: SHAP values
            text: Original text
            tokens: Tokenized text
            
        Returns:
            Dictionary of visualization data
        """
        try:
            visualizations = {}
            
            # Waterfall plot data
            visualizations["waterfall"] = self._generate_waterfall_data(shap_values, tokens)
            
            # Force plot data
            visualizations["force"] = self._generate_force_plot_data(shap_values, tokens)
            
            # Bar plot data
            visualizations["bar"] = self._generate_bar_plot_data(shap_values, tokens)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_waterfall_data(self, shap_values, tokens: List[str]) -> Dict:
        """Generate waterfall plot data"""
        try:
            if len(shap_values.shape) > 2:
                values = shap_values[0, :, 1]  # Binary classification
            else:
                values = shap_values[0]
            
            # Create waterfall data
            waterfall_data = []
            cumulative = 0.0
            
            for i, (token, value) in enumerate(zip(tokens, values)):
                cumulative += value
                waterfall_data.append({
                    "token": token,
                    "value": float(value),
                    "cumulative": float(cumulative),
                    "position": i
                })
            
            return {
                "data": waterfall_data,
                "base_value": 0.0,
                "output_value": float(cumulative)
            }
            
        except Exception as e:
            logger.error(f"Waterfall data generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_force_plot_data(self, shap_values, tokens: List[str]) -> Dict:
        """Generate force plot data"""
        try:
            if len(shap_values.shape) > 2:
                values = shap_values[0, :, 1]
            else:
                values = shap_values[0]
            
            # Create force plot data
            force_data = []
            for token, value in zip(tokens, values):
                force_data.append({
                    "token": token,
                    "value": float(value),
                    "color": "red" if value > 0 else "blue"
                })
            
            return {
                "data": force_data,
                "base_value": 0.0
            }
            
        except Exception as e:
            logger.error(f"Force plot data generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_bar_plot_data(self, shap_values, tokens: List[str]) -> Dict:
        """Generate bar plot data"""
        try:
            if len(shap_values.shape) > 2:
                values = shap_values[0, :, 1]
            else:
                values = shap_values[0]
            
            # Sort by absolute value
            sorted_data = sorted(zip(tokens, values), key=lambda x: abs(x[1]), reverse=True)
            
            bar_data = []
            for token, value in sorted_data[:20]:  # Top 20
                bar_data.append({
                    "token": token,
                    "value": float(value),
                    "abs_value": float(abs(value))
                })
            
            return {
                "data": bar_data,
                "title": "Top Contributing Tokens"
            }
            
        except Exception as e:
            logger.error(f"Bar plot data generation failed: {e}")
            return {"error": str(e)}
    
    def compare_attacks(self, original_text: str, adversarial_text: str) -> Dict:
        """
        Compare SHAP explanations between original and adversarial text
        
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
            
            # Compare word importance
            orig_words = orig_explanation.get("word_contributions", {}).get("words", [])
            adv_words = adv_explanation.get("word_contributions", {}).get("words", [])
            
            comparison = {
                "original_explanation": orig_explanation,
                "adversarial_explanation": adv_explanation,
                "changes": self._analyze_changes(orig_words, adv_words),
                "vulnerability_shift": self._calculate_vulnerability_shift(orig_explanation, adv_explanation)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Attack comparison failed: {e}")
            return {"error": str(e)}
    
    def _analyze_changes(self, orig_words: List[Dict], adv_words: List[Dict]) -> Dict:
        """Analyze changes between original and adversarial explanations"""
        try:
            # Create word maps for comparison
            orig_map = {w["word"]: w for w in orig_words}
            adv_map = {w["word"]: w for w in adv_words}
            
            changes = {
                "added_words": [],
                "removed_words": [],
                "modified_words": [],
                "unchanged_words": []
            }
            
            # Find changes
            all_words = set(orig_map.keys()) | set(adv_map.keys())
            
            for word in all_words:
                if word in orig_map and word in adv_map:
                    # Word exists in both
                    orig_val = orig_map[word]["shap_value"]
                    adv_val = adv_map[word]["shap_value"]
                    
                    if abs(orig_val - adv_val) > 0.05:  # Significant change
                        changes["modified_words"].append({
                            "word": word,
                            "original_value": orig_val,
                            "adversarial_value": adv_val,
                            "change": adv_val - orig_val
                        })
                    else:
                        changes["unchanged_words"].append(word)
                        
                elif word in orig_map:
                    changes["removed_words"].append(word)
                else:
                    changes["added_words"].append(word)
            
            return changes
            
        except Exception as e:
            logger.error(f"Change analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_vulnerability_shift(self, orig_explanation: Dict, adv_explanation: Dict) -> Dict:
        """Calculate how vulnerability shifted between explanations"""
        try:
            orig_score = orig_explanation.get("vulnerability_analysis", {}).get("vulnerability_score", 0)
            adv_score = adv_explanation.get("vulnerability_analysis", {}).get("vulnerability_score", 0)
            
            shift = adv_score - orig_score
            
            return {
                "original_score": orig_score,
                "adversarial_score": adv_score,
                "shift": shift,
                "shift_direction": "increased" if shift > 0 else "decreased" if shift < 0 else "unchanged",
                "shift_magnitude": abs(shift)
            }
            
        except Exception as e:
            logger.error(f"Vulnerability shift calculation failed: {e}")
            return {"error": str(e)}
