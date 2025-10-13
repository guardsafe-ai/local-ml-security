"""
Robustness Metrics for ML Security
Research-grade evaluation metrics for model robustness
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import spearmanr, pearsonr
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
import json

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    logger.warning("Could not download NLTK data. Some metrics may not work.")


class RobustnessMetrics:
    """
    Comprehensive robustness metrics for ML security evaluation
    Implements research-grade metrics for adversarial attack assessment
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize robustness metrics calculator
        
        Args:
            device: Device to run on
        """
        self.device = device
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_sim = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_sim = None
        
        logger.info("✅ Robustness metrics calculator initialized")
    
    def calculate_comprehensive_metrics(self, 
                                      original_texts: List[str],
                                      adversarial_texts: List[str],
                                      original_predictions: List[int],
                                      adversarial_predictions: List[int],
                                      original_confidences: List[float],
                                      adversarial_confidences: List[float]) -> Dict:
        """
        Calculate comprehensive robustness metrics
        
        Args:
            original_texts: List of original texts
            adversarial_texts: List of adversarial texts
            original_predictions: List of original predictions
            adversarial_predictions: List of adversarial predictions
            original_confidences: List of original confidences
            adversarial_confidences: List of adversarial confidences
            
        Returns:
            Comprehensive metrics dictionary
        """
        try:
            metrics = {
                "attack_success_metrics": self._calculate_attack_success_metrics(
                    original_predictions, adversarial_predictions
                ),
                "semantic_preservation_metrics": self._calculate_semantic_preservation_metrics(
                    original_texts, adversarial_texts
                ),
                "perturbation_metrics": self._calculate_perturbation_metrics(
                    original_texts, adversarial_texts
                ),
                "confidence_metrics": self._calculate_confidence_metrics(
                    original_confidences, adversarial_confidences
                ),
                "robustness_score": 0.0,
                "overall_assessment": "UNKNOWN"
            }
            
            # Calculate overall robustness score
            metrics["robustness_score"] = self._calculate_overall_robustness_score(metrics)
            
            # Generate overall assessment
            metrics["overall_assessment"] = self._generate_overall_assessment(metrics)
            
            logger.info(f"Comprehensive metrics calculated: robustness_score={metrics['robustness_score']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Comprehensive metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_attack_success_metrics(self, original_predictions: List[int], 
                                        adversarial_predictions: List[int]) -> Dict:
        """Calculate attack success metrics"""
        try:
            if len(original_predictions) != len(adversarial_predictions):
                raise ValueError("Prediction lists must have same length")
            
            # Attack Success Rate (ASR)
            successful_attacks = sum(1 for orig, adv in zip(original_predictions, adversarial_predictions) 
                                   if orig != adv)
            total_attacks = len(original_predictions)
            asr = successful_attacks / total_attacks if total_attacks > 0 else 0.0
            
            # Attack Success Rate by Class
            class_asr = {}
            unique_classes = set(original_predictions)
            for class_label in unique_classes:
                class_indices = [i for i, pred in enumerate(original_predictions) if pred == class_label]
                if class_indices:
                    class_successful = sum(1 for i in class_indices 
                                         if original_predictions[i] != adversarial_predictions[i])
                    class_asr[class_label] = class_successful / len(class_indices)
            
            # Attack Consistency (how often attacks succeed)
            attack_consistency = asr
            
            return {
                "attack_success_rate": asr,
                "successful_attacks": successful_attacks,
                "total_attacks": total_attacks,
                "class_attack_success_rates": class_asr,
                "attack_consistency": attack_consistency,
                "attack_effectiveness": "HIGH" if asr > 0.8 else "MEDIUM" if asr > 0.5 else "LOW"
            }
            
        except Exception as e:
            logger.error(f"Attack success metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_semantic_preservation_metrics(self, original_texts: List[str], 
                                               adversarial_texts: List[str]) -> Dict:
        """Calculate semantic preservation metrics"""
        try:
            if len(original_texts) != len(adversarial_texts):
                raise ValueError("Text lists must have same length")
            
            metrics = {
                "sentence_similarity": [],
                "bleu_scores": [],
                "meteor_scores": [],
                "nist_scores": [],
                "semantic_preservation_score": 0.0
            }
            
            # Calculate metrics for each text pair
            for orig, adv in zip(original_texts, adversarial_texts):
                # Sentence similarity
                if self.sentence_sim:
                    sim_score = self._calculate_sentence_similarity(orig, adv)
                    metrics["sentence_similarity"].append(sim_score)
                
                # BLEU score
                bleu_score = self._calculate_bleu_score(orig, adv)
                metrics["bleu_scores"].append(bleu_score)
                
                # METEOR score
                meteor_score = self._calculate_meteor_score(orig, adv)
                metrics["meteor_scores"].append(meteor_score)
                
                # NIST score
                nist_score = self._calculate_nist_score(orig, adv)
                metrics["nist_scores"].append(nist_score)
            
            # Calculate averages
            if metrics["sentence_similarity"]:
                metrics["avg_sentence_similarity"] = np.mean(metrics["sentence_similarity"])
            else:
                metrics["avg_sentence_similarity"] = 0.0
            
            metrics["avg_bleu_score"] = np.mean(metrics["bleu_scores"])
            metrics["avg_meteor_score"] = np.mean(metrics["meteor_scores"])
            metrics["avg_nist_score"] = np.mean(metrics["nist_scores"])
            
            # Overall semantic preservation score
            semantic_components = [
                metrics["avg_sentence_similarity"],
                metrics["avg_bleu_score"],
                metrics["avg_meteor_score"],
                metrics["avg_nist_score"]
            ]
            metrics["semantic_preservation_score"] = np.mean(semantic_components)
            
            # Quality assessment
            if metrics["semantic_preservation_score"] > 0.8:
                metrics["semantic_quality"] = "HIGH"
            elif metrics["semantic_preservation_score"] > 0.6:
                metrics["semantic_quality"] = "MEDIUM"
            else:
                metrics["semantic_quality"] = "LOW"
            
            return metrics
            
        except Exception as e:
            logger.error(f"Semantic preservation metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_perturbation_metrics(self, original_texts: List[str], 
                                      adversarial_texts: List[str]) -> Dict:
        """Calculate perturbation metrics"""
        try:
            if len(original_texts) != len(adversarial_texts):
                raise ValueError("Text lists must have same length")
            
            metrics = {
                "word_edit_distances": [],
                "character_edit_distances": [],
                "perturbation_ratios": [],
                "perturbation_efficiency": 0.0
            }
            
            # Calculate metrics for each text pair
            for orig, adv in zip(original_texts, adversarial_texts):
                # Word-level edit distance
                word_ed = self._calculate_word_edit_distance(orig, adv)
                metrics["word_edit_distances"].append(word_ed)
                
                # Character-level edit distance
                char_ed = self._calculate_character_edit_distance(orig, adv)
                metrics["character_edit_distances"].append(char_ed)
                
                # Perturbation ratio
                orig_words = len(orig.split())
                perturbation_ratio = word_ed / orig_words if orig_words > 0 else 0
                metrics["perturbation_ratios"].append(perturbation_ratio)
            
            # Calculate averages
            metrics["avg_word_edit_distance"] = np.mean(metrics["word_edit_distances"])
            metrics["avg_character_edit_distance"] = np.mean(metrics["character_edit_distances"])
            metrics["avg_perturbation_ratio"] = np.mean(metrics["perturbation_ratios"])
            
            # Perturbation efficiency (lower is better)
            metrics["perturbation_efficiency"] = 1.0 - metrics["avg_perturbation_ratio"]
            
            # Efficiency assessment
            if metrics["avg_perturbation_ratio"] < 0.1:
                metrics["efficiency_rating"] = "EXCELLENT"
            elif metrics["avg_perturbation_ratio"] < 0.2:
                metrics["efficiency_rating"] = "GOOD"
            elif metrics["avg_perturbation_ratio"] < 0.3:
                metrics["efficiency_rating"] = "FAIR"
            else:
                metrics["efficiency_rating"] = "POOR"
            
            return metrics
            
        except Exception as e:
            logger.error(f"Perturbation metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_confidence_metrics(self, original_confidences: List[float], 
                                    adversarial_confidences: List[float]) -> Dict:
        """Calculate confidence-based metrics"""
        try:
            if len(original_confidences) != len(adversarial_confidences):
                raise ValueError("Confidence lists must have same length")
            
            metrics = {
                "confidence_drops": [],
                "confidence_correlations": [],
                "confidence_stability": 0.0
            }
            
            # Calculate confidence drops
            for orig_conf, adv_conf in zip(original_confidences, adversarial_confidences):
                confidence_drop = orig_conf - adv_conf
                metrics["confidence_drops"].append(confidence_drop)
            
            # Calculate correlation
            if len(original_confidences) > 1:
                correlation, _ = pearsonr(original_confidences, adversarial_confidences)
                metrics["confidence_correlations"].append(correlation)
            
            # Calculate averages
            metrics["avg_confidence_drop"] = np.mean(metrics["confidence_drops"])
            metrics["avg_confidence_correlation"] = np.mean(metrics["confidence_correlations"]) if metrics["confidence_correlations"] else 0.0
            
            # Confidence stability (higher is better)
            metrics["confidence_stability"] = 1.0 - abs(metrics["avg_confidence_drop"])
            
            # Stability assessment
            if metrics["confidence_stability"] > 0.8:
                metrics["stability_rating"] = "HIGH"
            elif metrics["confidence_stability"] > 0.6:
                metrics["stability_rating"] = "MEDIUM"
            else:
                metrics["stability_rating"] = "LOW"
            
            return metrics
            
        except Exception as e:
            logger.error(f"Confidence metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_sentence_similarity(self, text1: str, text2: str) -> float:
        """Calculate sentence similarity using sentence transformer"""
        try:
            if self.sentence_sim is None:
                return 0.0
            
            embeddings = self.sentence_sim.encode([text1, text2])
            similarity = F.cosine_similarity(
                torch.tensor(embeddings[0]).unsqueeze(0),
                torch.tensor(embeddings[1]).unsqueeze(0)
            ).item()
            
            return similarity
            
        except Exception as e:
            logger.error(f"Sentence similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score"""
        try:
            reference_tokens = reference.split()
            candidate_tokens = candidate.split()
            
            # Use smoothing function to handle zero counts
            smoothing = SmoothingFunction().method1
            bleu = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
            
            return bleu
            
        except Exception as e:
            logger.error(f"BLEU score calculation failed: {e}")
            return 0.0
    
    def _calculate_meteor_score(self, reference: str, candidate: str) -> float:
        """Calculate METEOR score"""
        try:
            reference_tokens = reference.split()
            candidate_tokens = candidate.split()
            
            meteor = meteor_score([reference_tokens], candidate_tokens)
            return meteor
            
        except Exception as e:
            logger.error(f"METEOR score calculation failed: {e}")
            return 0.0
    
    def _calculate_nist_score(self, reference: str, candidate: str) -> float:
        """Calculate NIST score"""
        try:
            reference_tokens = reference.split()
            candidate_tokens = candidate.split()
            
            nist = sentence_nist([reference_tokens], candidate_tokens)
            return nist
            
        except Exception as e:
            logger.error(f"NIST score calculation failed: {e}")
            return 0.0
    
    def _calculate_word_edit_distance(self, text1: str, text2: str) -> int:
        """Calculate word-level edit distance"""
        try:
            words1 = text1.split()
            words2 = text2.split()
            
            # Dynamic programming for edit distance
            m, n = len(words1), len(words2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            # Initialize base cases
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            # Fill the dp table
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if words1[i-1] == words2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            
            return dp[m][n]
            
        except Exception as e:
            logger.error(f"Word edit distance calculation failed: {e}")
            return 0
    
    def _calculate_character_edit_distance(self, text1: str, text2: str) -> int:
        """Calculate character-level edit distance"""
        try:
            m, n = len(text1), len(text2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            # Initialize base cases
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            # Fill the dp table
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if text1[i-1] == text2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            
            return dp[m][n]
            
        except Exception as e:
            logger.error(f"Character edit distance calculation failed: {e}")
            return 0
    
    def _calculate_overall_robustness_score(self, metrics: Dict) -> float:
        """Calculate overall robustness score"""
        try:
            # Weighted combination of different metrics
            weights = {
                "attack_success": 0.3,
                "semantic_preservation": 0.3,
                "perturbation_efficiency": 0.2,
                "confidence_stability": 0.2
            }
            
            # Extract component scores
            attack_success = metrics.get("attack_success_metrics", {}).get("attack_success_rate", 0.0)
            semantic_preservation = metrics.get("semantic_preservation_metrics", {}).get("semantic_preservation_score", 0.0)
            perturbation_efficiency = metrics.get("perturbation_metrics", {}).get("perturbation_efficiency", 0.0)
            confidence_stability = metrics.get("confidence_metrics", {}).get("confidence_stability", 0.0)
            
            # Calculate weighted score
            robustness_score = (
                weights["attack_success"] * attack_success +
                weights["semantic_preservation"] * semantic_preservation +
                weights["perturbation_efficiency"] * perturbation_efficiency +
                weights["confidence_stability"] * confidence_stability
            )
            
            return min(max(robustness_score, 0.0), 1.0)  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Overall robustness score calculation failed: {e}")
            return 0.0
    
    def _generate_overall_assessment(self, metrics: Dict) -> str:
        """Generate overall assessment based on metrics"""
        try:
            robustness_score = metrics.get("robustness_score", 0.0)
            
            if robustness_score >= 0.8:
                return "EXCELLENT"
            elif robustness_score >= 0.6:
                return "GOOD"
            elif robustness_score >= 0.4:
                return "FAIR"
            else:
                return "POOR"
                
        except Exception as e:
            logger.error(f"Overall assessment generation failed: {e}")
            return "UNKNOWN"
    
    def calculate_transferability_metrics(self, 
                                        attack_texts: List[str],
                                        models: List[Any],
                                        model_names: List[str]) -> Dict:
        """
        Calculate transferability metrics across multiple models
        
        Args:
            attack_texts: List of attack texts
            models: List of models to test
            model_names: List of model names
            
        Returns:
            Transferability metrics dictionary
        """
        try:
            if len(models) != len(model_names):
                raise ValueError("Models and model names must have same length")
            
            transferability_metrics = {
                "model_performance": {},
                "cross_model_correlation": {},
                "transferability_score": 0.0,
                "transferability_assessment": "UNKNOWN"
            }
            
            # Test each model
            for model, model_name in zip(models, model_names):
                # This would need to be implemented based on your model interface
                # For now, return placeholder
                transferability_metrics["model_performance"][model_name] = {
                    "success_rate": 0.0,
                    "avg_confidence": 0.0
                }
            
            # Calculate transferability score
            transferability_metrics["transferability_score"] = 0.0  # Placeholder
            
            return transferability_metrics
            
        except Exception as e:
            logger.error(f"Transferability metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def generate_robustness_report(self, metrics: Dict) -> str:
        """Generate a comprehensive robustness report"""
        try:
            report = {
                "summary": {
                    "robustness_score": metrics.get("robustness_score", 0.0),
                    "overall_assessment": metrics.get("overall_assessment", "UNKNOWN"),
                    "timestamp": str(np.datetime64('now'))
                },
                "detailed_metrics": metrics,
                "recommendations": self._generate_recommendations(metrics)
            }
            
            return json.dumps(report, indent=2)
            
        except Exception as e:
            logger.error(f"Robustness report generation failed: {e}")
            return json.dumps({"error": str(e)})
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate recommendations based on metrics"""
        try:
            recommendations = []
            
            # Attack success recommendations
            attack_metrics = metrics.get("attack_success_metrics", {})
            asr = attack_metrics.get("attack_success_rate", 0.0)
            
            if asr > 0.8:
                recommendations.append("High attack success rate detected - strengthen model defenses")
            elif asr < 0.2:
                recommendations.append("Low attack success rate - model shows good robustness")
            
            # Semantic preservation recommendations
            semantic_metrics = metrics.get("semantic_preservation_metrics", {})
            semantic_score = semantic_metrics.get("semantic_preservation_score", 0.0)
            
            if semantic_score < 0.6:
                recommendations.append("Poor semantic preservation - consider attack refinement")
            
            # Perturbation efficiency recommendations
            perturbation_metrics = metrics.get("perturbation_metrics", {})
            perturbation_ratio = perturbation_metrics.get("avg_perturbation_ratio", 0.0)
            
            if perturbation_ratio > 0.3:
                recommendations.append("High perturbation ratio - consider more efficient attack methods")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Error generating recommendations"]
