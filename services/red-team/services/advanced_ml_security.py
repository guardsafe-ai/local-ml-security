"""
Advanced ML Security Integration
Main integration module for world-class ML security platform
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from datetime import datetime

# Import all advanced components
from .adversarial_ml import (
    GradientBasedAttacks, TextFoolerAttack, BERTAttack, HotFlipAttack,
    UniversalTriggerGenerator, EmbeddingPerturbation
)
from .explainability import (
    SHAPAnalyzer, LIMEExplainer, IntegratedGradientsAnalyzer, AttentionAnalyzer
)
from .evaluation import RobustnessMetrics

logger = logging.getLogger(__name__)


class AdvancedMLSecurityPlatform:
    """
    World-class ML Security Platform
    Integrates all advanced ML security components for comprehensive testing
    """
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the advanced ML security platform
        
        Args:
            model: Target model to test
            tokenizer: Tokenizer for the model
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Initialize all components
        self._initialize_components()
        
        logger.info("✅ Advanced ML Security Platform initialized")
    
    def _initialize_components(self):
        """Initialize all security components"""
        try:
            # Adversarial ML components
            self.gradient_attacks = GradientBasedAttacks(self.model, self.tokenizer, self.device)
            self.textfooler = TextFoolerAttack(self.model, self.tokenizer, device=self.device)
            self.bert_attack = BERTAttack(self.model, self.tokenizer, self.device)
            self.hotflip = HotFlipAttack(self.model, self.tokenizer, self.device)
            self.universal_triggers = UniversalTriggerGenerator(self.model, self.tokenizer, self.device)
            self.embedding_perturbation = EmbeddingPerturbation(self.model, self.tokenizer, self.device)
            
            # Explainability components
            self.shap_analyzer = SHAPAnalyzer(self.model, self.tokenizer, self.device)
            self.lime_explainer = LIMEExplainer(self.model, self.tokenizer, self.device)
            self.integrated_gradients = IntegratedGradientsAnalyzer(self.model, self.tokenizer, self.device)
            self.attention_analyzer = AttentionAnalyzer(self.model, self.tokenizer, self.device)
            
            # Evaluation components
            self.robustness_metrics = RobustnessMetrics(self.device)
            
            logger.info("✅ All security components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize security components: {e}")
            raise
    
    async def comprehensive_security_test(self, 
                                        text: str,
                                        attack_categories: List[str] = None,
                                        explainability_methods: List[str] = None,
                                        evaluation_metrics: List[str] = None) -> Dict:
        """
        Perform comprehensive security testing
        
        Args:
            text: Input text to test
            attack_categories: List of attack categories to test
            explainability_methods: List of explainability methods to use
            evaluation_metrics: List of evaluation metrics to calculate
            
        Returns:
            Comprehensive security test results
        """
        try:
            if attack_categories is None:
                attack_categories = ["gradient", "word_level", "universal_triggers"]
            
            if explainability_methods is None:
                explainability_methods = ["shap", "lime", "integrated_gradients", "attention"]
            
            if evaluation_metrics is None:
                evaluation_metrics = ["robustness", "semantic_preservation", "transferability"]
            
            logger.info(f"Starting comprehensive security test for text: {text[:50]}...")
            
            # Initialize results
            results = {
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "attack_results": {},
                "explainability_results": {},
                "evaluation_results": {},
                "overall_assessment": "UNKNOWN"
            }
            
            # Perform attacks
            attack_results = await self._perform_attacks(text, attack_categories)
            results["attack_results"] = attack_results
            
            # Generate explanations
            explainability_results = await self._generate_explanations(text, attack_results, explainability_methods)
            results["explainability_results"] = explainability_results
            
            # Calculate evaluation metrics
            evaluation_results = await self._calculate_evaluation_metrics(text, attack_results, evaluation_metrics)
            results["evaluation_results"] = evaluation_results
            
            # Generate overall assessment
            results["overall_assessment"] = self._generate_overall_assessment(attack_results, evaluation_results)
            
            logger.info(f"Comprehensive security test completed: {results['overall_assessment']}")
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive security test failed: {e}")
            return {"error": str(e), "text": text}
    
    async def _perform_attacks(self, text: str, attack_categories: List[str]) -> Dict:
        """Perform various types of attacks"""
        try:
            attack_results = {}
            
            for category in attack_categories:
                if category == "gradient":
                    # FGSM attack
                    fgsm_text, fgsm_norm, fgsm_info = self.gradient_attacks.fgsm_attack(text)
                    attack_results["fgsm"] = {
                        "adversarial_text": fgsm_text,
                        "perturbation_norm": fgsm_norm,
                        "attack_info": fgsm_info
                    }
                    
                    # PGD attack
                    pgd_text, pgd_norm, pgd_info = self.gradient_attacks.pgd_attack(text)
                    attack_results["pgd"] = {
                        "adversarial_text": pgd_text,
                        "perturbation_norm": pgd_norm,
                        "attack_info": pgd_info
                    }
                    
                    # C&W attack
                    cw_text, cw_norm, cw_info = self.gradient_attacks.carlini_wagner_attack(text)
                    attack_results["carlini_wagner"] = {
                        "adversarial_text": cw_text,
                        "perturbation_norm": cw_norm,
                        "attack_info": cw_info
                    }
                
                elif category == "word_level":
                    # TextFooler attack
                    tf_text, tf_info = self.textfooler.attack(text)
                    attack_results["textfooler"] = {
                        "adversarial_text": tf_text,
                        "attack_info": tf_info
                    }
                    
                    # BERT-Attack
                    ba_text, ba_info = self.bert_attack.attack(text)
                    attack_results["bert_attack"] = {
                        "adversarial_text": ba_text,
                        "attack_info": ba_info
                    }
                    
                    # HotFlip attack
                    hf_text, hf_info = self.hotflip.attack(text)
                    attack_results["hotflip"] = {
                        "adversarial_text": hf_text,
                        "attack_info": hf_info
                    }
                
                elif category == "universal_triggers":
                    # Generate universal trigger
                    trigger_text, trigger_info = self.universal_triggers.generate_universal_trigger(
                        trigger_length=5,
                        num_iterations=100,
                        dataset_samples=[text]
                    )
                    attack_results["universal_trigger"] = {
                        "trigger_text": trigger_text,
                        "generation_info": trigger_info
                    }
            
            return attack_results
            
        except Exception as e:
            logger.error(f"Attack performance failed: {e}")
            return {"error": str(e)}
    
    async def _generate_explanations(self, text: str, attack_results: Dict, 
                                   explainability_methods: List[str]) -> Dict:
        """Generate explanations for attacks"""
        try:
            explainability_results = {}
            
            for method in explainability_methods:
                if method == "shap":
                    # SHAP explanation
                    shap_explanation = self.shap_analyzer.explain_vulnerability(text, {"detected": False})
                    explainability_results["shap"] = shap_explanation
                
                elif method == "lime":
                    # LIME explanation
                    lime_explanation = self.lime_explainer.explain_vulnerability(text, {"detected": False})
                    explainability_results["lime"] = lime_explanation
                
                elif method == "integrated_gradients":
                    # Integrated Gradients explanation
                    ig_explanation = self.integrated_gradients.explain_vulnerability(text, {"detected": False})
                    explainability_results["integrated_gradients"] = ig_explanation
                
                elif method == "attention":
                    # Attention analysis
                    attention_explanation = self.attention_analyzer.analyze_attention_patterns(text, {"detected": False})
                    explainability_results["attention"] = attention_explanation
            
            return explainability_results
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return {"error": str(e)}
    
    async def _calculate_evaluation_metrics(self, text: str, attack_results: Dict, 
                                          evaluation_metrics: List[str]) -> Dict:
        """Calculate evaluation metrics"""
        try:
            evaluation_results = {}
            
            # Prepare data for metrics calculation
            original_texts = [text]
            adversarial_texts = []
            original_predictions = []
            adversarial_predictions = []
            original_confidences = []
            adversarial_confidences = []
            
            # Get original prediction
            orig_pred, orig_conf = self._get_model_prediction(text)
            original_predictions.append(orig_pred)
            original_confidences.append(orig_conf)
            
            # Collect adversarial examples
            for attack_name, attack_result in attack_results.items():
                if "adversarial_text" in attack_result:
                    adv_text = attack_result["adversarial_text"]
                    adversarial_texts.append(adv_text)
                    
                    # Get adversarial prediction
                    adv_pred, adv_conf = self._get_model_prediction(adv_text)
                    adversarial_predictions.append(adv_pred)
                    adversarial_confidences.append(adv_conf)
            
            if not adversarial_texts:
                return {"error": "No adversarial examples generated"}
            
            # Calculate comprehensive metrics
            if "robustness" in evaluation_metrics:
                robustness_metrics = self.robustness_metrics.calculate_comprehensive_metrics(
                    original_texts * len(adversarial_texts),  # Repeat for each adversarial example
                    adversarial_texts,
                    original_predictions * len(adversarial_texts),
                    adversarial_predictions,
                    original_confidences * len(adversarial_texts),
                    adversarial_confidences
                )
                evaluation_results["robustness"] = robustness_metrics
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def _get_model_prediction(self, text: str) -> Tuple[int, float]:
        """Get model prediction and confidence"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probs, dim=-1).item()
                confidence = torch.max(probs, dim=-1)[0].item()
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return 0, 0.0
    
    def _generate_overall_assessment(self, attack_results: Dict, evaluation_results: Dict) -> str:
        """Generate overall security assessment"""
        try:
            # Count successful attacks
            successful_attacks = 0
            total_attacks = 0
            
            for attack_name, attack_result in attack_results.items():
                if "adversarial_text" in attack_result:
                    total_attacks += 1
                    # Check if attack was successful (simplified)
                    if "attack_success" in attack_result.get("attack_info", {}):
                        if attack_result["attack_info"]["attack_success"]:
                            successful_attacks += 1
            
            # Calculate success rate
            success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0.0
            
            # Get robustness score
            robustness_score = 0.0
            if "robustness" in evaluation_results:
                robustness_score = evaluation_results["robustness"].get("robustness_score", 0.0)
            
            # Generate assessment
            if success_rate > 0.8 and robustness_score < 0.3:
                return "CRITICAL_VULNERABILITY"
            elif success_rate > 0.5 and robustness_score < 0.5:
                return "HIGH_VULNERABILITY"
            elif success_rate > 0.2 and robustness_score < 0.7:
                return "MEDIUM_VULNERABILITY"
            elif success_rate < 0.2 and robustness_score > 0.7:
                return "LOW_VULNERABILITY"
            else:
                return "MODERATE_VULNERABILITY"
                
        except Exception as e:
            logger.error(f"Overall assessment generation failed: {e}")
            return "UNKNOWN"
    
    async def batch_security_test(self, 
                                texts: List[str],
                                attack_categories: List[str] = None,
                                explainability_methods: List[str] = None) -> List[Dict]:
        """
        Perform security testing on a batch of texts
        
        Args:
            texts: List of texts to test
            attack_categories: List of attack categories
            explainability_methods: List of explainability methods
            
        Returns:
            List of security test results
        """
        try:
            results = []
            
            for i, text in enumerate(texts):
                logger.info(f"Processing text {i+1}/{len(texts)}: {text[:50]}...")
                
                result = await self.comprehensive_security_test(
                    text, attack_categories, explainability_methods
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch security test failed: {e}")
            return [{"error": str(e)} for _ in texts]
    
    def generate_security_report(self, results: List[Dict]) -> Dict:
        """Generate comprehensive security report"""
        try:
            # Aggregate results
            total_tests = len(results)
            successful_attacks = 0
            vulnerability_levels = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "MODERATE": 0}
            
            for result in results:
                if "overall_assessment" in result:
                    assessment = result["overall_assessment"]
                    if assessment in vulnerability_levels:
                        vulnerability_levels[assessment] += 1
                
                # Count successful attacks
                if "attack_results" in result:
                    for attack_name, attack_result in result["attack_results"].items():
                        if "attack_info" in attack_result:
                            if attack_result["attack_info"].get("attack_success", False):
                                successful_attacks += 1
            
            # Calculate overall metrics
            overall_success_rate = successful_attacks / total_tests if total_tests > 0 else 0.0
            
            # Generate report
            report = {
                "summary": {
                    "total_tests": total_tests,
                    "successful_attacks": successful_attacks,
                    "overall_success_rate": overall_success_rate,
                    "vulnerability_distribution": vulnerability_levels,
                    "timestamp": datetime.now().isoformat()
                },
                "detailed_results": results,
                "recommendations": self._generate_security_recommendations(vulnerability_levels, overall_success_rate)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Security report generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_security_recommendations(self, vulnerability_levels: Dict, success_rate: float) -> List[str]:
        """Generate security recommendations based on results"""
        try:
            recommendations = []
            
            # Critical vulnerabilities
            if vulnerability_levels["CRITICAL"] > 0:
                recommendations.append("CRITICAL: Immediate action required - multiple critical vulnerabilities detected")
                recommendations.append("Implement comprehensive model hardening and defense mechanisms")
            
            # High vulnerabilities
            if vulnerability_levels["HIGH"] > 0:
                recommendations.append("HIGH: Significant vulnerabilities detected - prioritize security improvements")
                recommendations.append("Consider adversarial training and robust model architectures")
            
            # Overall success rate
            if success_rate > 0.7:
                recommendations.append("High attack success rate - strengthen model defenses")
            elif success_rate < 0.3:
                recommendations.append("Low attack success rate - model shows good robustness")
            
            # General recommendations
            recommendations.extend([
                "Regular security testing and monitoring recommended",
                "Implement continuous adversarial training",
                "Consider ensemble methods for improved robustness",
                "Monitor for new attack patterns and techniques"
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Security recommendation generation failed: {e}")
            return ["Error generating recommendations"]
