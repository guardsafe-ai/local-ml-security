"""
Integrated Gradients Analyzer for ML Security
Axiomatic attribution method for vulnerability analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


class IntegratedGradientsAnalyzer:
    """
    Integrated Gradients (Sundararajan et al., 2017) for ML security
    Provides axiomatic attribution for model predictions
    """
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Integrated Gradients analyzer
        
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
        
        logger.info("✅ Integrated Gradients analyzer initialized")
    
    def explain_vulnerability(self, text: str, attack_result: Dict, 
                            steps: int = 50, baseline: str = None) -> Dict:
        """
        Explain vulnerability using Integrated Gradients
        
        Args:
            text: Input text that was attacked
            attack_result: Result of the attack
            steps: Number of integration steps
            baseline: Baseline text (if None, uses empty string)
            
        Returns:
            Integrated Gradients explanation dictionary
        """
        try:
            if baseline is None:
                baseline = ""
            
            # Get attributions
            attributions = self._compute_integrated_gradients(text, baseline, steps)
            
            # Tokenize text for analysis
            tokens = self.tokenizer.tokenize(text)
            
            # Generate explanation
            explanation = {
                "method": "Integrated Gradients",
                "text": text,
                "baseline": baseline,
                "steps": steps,
                "attack_detected": attack_result.get("detected", False),
                "attributions": attributions,
                "token_attributions": self._compute_token_attributions(attributions, tokens),
                "vulnerability_analysis": self._analyze_vulnerability_attributions(attributions, attack_result),
                "visualizations": self._generate_attribution_visualizations(attributions, text, tokens)
            }
            
            logger.debug(f"Integrated Gradients explanation generated for text: {text[:50]}...")
            return explanation
            
        except Exception as e:
            logger.error(f"Integrated Gradients explanation failed: {e}")
            return {"error": str(e), "method": "Integrated Gradients"}
    
    def _compute_integrated_gradients(self, text: str, baseline: str, steps: int) -> torch.Tensor:
        """
        Compute Integrated Gradients for the given text
        
        Args:
            text: Input text
            baseline: Baseline text
            steps: Number of integration steps
            
        Returns:
            Attribution tensor
        """
        try:
            # Tokenize input and baseline
            input_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            baseline_tokens = self.tokenizer(baseline, return_tensors="pt", padding=True, truncation=True)
            
            # Ensure same length by padding
            max_len = max(input_tokens.input_ids.shape[1], baseline_tokens.input_ids.shape[1])
            
            input_ids = F.pad(input_tokens.input_ids, (0, max_len - input_tokens.input_ids.shape[1]))
            baseline_ids = F.pad(baseline_tokens.input_ids, (0, max_len - baseline_tokens.input_ids.shape[1]))
            attention_mask = F.pad(input_tokens.attention_mask, (0, max_len - input_tokens.attention_mask.shape[1]))
            
            # Move to device
            input_ids = input_ids.to(self.device)
            baseline_ids = baseline_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Get embeddings
            input_embeddings = self.model.get_input_embeddings()(input_ids)
            baseline_embeddings = self.model.get_input_embeddings()(baseline_ids)
            
            # Create interpolation steps
            alphas = torch.linspace(0, 1, steps + 1, device=self.device)
            
            # Compute gradients for each step
            gradients = []
            for alpha in alphas:
                # Interpolate embeddings
                interpolated_embeddings = baseline_embeddings + alpha * (input_embeddings - baseline_embeddings)
                interpolated_embeddings.requires_grad = True
                
                # Forward pass
                outputs = self.model(inputs_embeds=interpolated_embeddings, attention_mask=attention_mask)
                
                # Get prediction for attack class (assuming binary classification)
                target_class = 1  # Attack class
                target_logit = outputs.logits[0, target_class]
                
                # Backward pass
                target_logit.backward()
                
                # Store gradient
                gradients.append(interpolated_embeddings.grad.clone())
            
            # Compute integrated gradients
            # Average gradients across steps
            avg_gradients = torch.mean(torch.stack(gradients), dim=0)
            
            # Multiply by input difference
            input_diff = input_embeddings - baseline_embeddings
            attributions = input_diff * avg_gradients
            
            return attributions
            
        except Exception as e:
            logger.error(f"Integrated gradients computation failed: {e}")
            return torch.zeros((1, 512, 768))  # Fallback shape
    
    def _compute_token_attributions(self, attributions: torch.Tensor, tokens: List[str]) -> List[Dict]:
        """
        Compute token-level attributions from embedding attributions
        
        Args:
            attributions: Attribution tensor
            tokens: List of tokens
            
        Returns:
            List of token attribution dictionaries
        """
        try:
            # Sum attributions across embedding dimensions
            token_attributions = torch.sum(attributions[0], dim=1)  # [seq_len]
            
            # Align with tokens
            token_data = []
            for i, token in enumerate(tokens):
                if i < token_attributions.shape[0]:
                    attribution = token_attributions[i].item()
                    token_data.append({
                        "token": token,
                        "attribution": attribution,
                        "abs_attribution": abs(attribution),
                        "position": i
                    })
                else:
                    token_data.append({
                        "token": token,
                        "attribution": 0.0,
                        "abs_attribution": 0.0,
                        "position": i
                    })
            
            return token_data
            
        except Exception as e:
            logger.error(f"Token attribution computation failed: {e}")
            return []
    
    def _analyze_vulnerability_attributions(self, attributions: torch.Tensor, attack_result: Dict) -> Dict:
        """
        Analyze vulnerability based on attributions
        
        Args:
            attributions: Attribution tensor
            attack_result: Attack result information
            
        Returns:
            Vulnerability analysis dictionary
        """
        try:
            # Compute attribution statistics
            attribution_values = attributions.flatten()
            
            analysis = {
                "total_attribution": torch.sum(attribution_values).item(),
                "max_attribution": torch.max(attribution_values).item(),
                "min_attribution": torch.min(attribution_values).item(),
                "mean_attribution": torch.mean(attribution_values).item(),
                "std_attribution": torch.std(attribution_values).item(),
                "positive_attribution": torch.sum(attribution_values[attribution_values > 0]).item(),
                "negative_attribution": torch.sum(attribution_values[attribution_values < 0]).item(),
                "attribution_ratio": self._calculate_attribution_ratio(attributions),
                "vulnerability_score": self._calculate_vulnerability_score(attributions, attack_result),
                "risk_level": self._assess_risk_level(attributions, attack_result)
            }
            
            # Identify attribution patterns
            analysis["patterns"] = self._identify_attribution_patterns(attributions, attack_result)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Vulnerability attribution analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_attribution_ratio(self, attributions: torch.Tensor) -> float:
        """Calculate ratio of positive to negative attributions"""
        try:
            positive = torch.sum(attributions[attributions > 0])
            negative = torch.sum(attributions[attributions < 0])
            
            if negative == 0:
                return float('inf') if positive > 0 else 0.0
            
            return (positive / abs(negative)).item()
            
        except Exception as e:
            logger.error(f"Attribution ratio calculation failed: {e}")
            return 0.0
    
    def _calculate_vulnerability_score(self, attributions: torch.Tensor, attack_result: Dict) -> float:
        """Calculate vulnerability score based on attributions"""
        try:
            # Base score from attack success
            base_score = 0.0 if attack_result.get("detected", False) else 0.3
            
            # Add score based on attribution magnitude
            attribution_magnitude = torch.mean(torch.abs(attributions)).item()
            attribution_score = min(attribution_magnitude * 10, 0.7)  # Cap at 0.7
            
            return min(base_score + attribution_score, 1.0)
            
        except Exception as e:
            logger.error(f"Vulnerability score calculation failed: {e}")
            return 0.0
    
    def _assess_risk_level(self, attributions: torch.Tensor, attack_result: Dict) -> str:
        """Assess risk level based on attributions and attack result"""
        try:
            vulnerability_score = self._calculate_vulnerability_score(attributions, attack_result)
            
            if vulnerability_score >= 0.8:
                return "CRITICAL"
            elif vulnerability_score >= 0.6:
                return "HIGH"
            elif vulnerability_score >= 0.4:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception as e:
            logger.error(f"Risk level assessment failed: {e}")
            return "UNKNOWN"
    
    def _identify_attribution_patterns(self, attributions: torch.Tensor, attack_result: Dict) -> List[Dict]:
        """Identify patterns in attributions"""
        try:
            patterns = []
            
            # Pattern 1: High attribution concentration
            attribution_values = attributions.flatten()
            high_attributions = attribution_values[torch.abs(attribution_values) > 0.1]
            
            if len(high_attributions) > len(attribution_values) * 0.2:
                patterns.append({
                    "type": "high_concentration",
                    "description": "High concentration of significant attributions",
                    "count": len(high_attributions),
                    "percentage": len(high_attributions) / len(attribution_values)
                })
            
            # Pattern 2: Attribution distribution
            positive_attributions = attribution_values[attribution_values > 0]
            negative_attributions = attribution_values[attribution_values < 0]
            
            if len(positive_attributions) > len(negative_attributions) * 1.5:
                patterns.append({
                    "type": "positive_bias",
                    "description": "Attributions are predominantly positive",
                    "positive_ratio": len(positive_attributions) / len(attribution_values)
                })
            elif len(negative_attributions) > len(positive_attributions) * 1.5:
                patterns.append({
                    "type": "negative_bias",
                    "description": "Attributions are predominantly negative",
                    "negative_ratio": len(negative_attributions) / len(attribution_values)
                })
            
            # Pattern 3: Attack-specific patterns
            if not attack_result.get("detected", False):
                # Attack succeeded - look for suspicious attribution patterns
                max_attribution = torch.max(torch.abs(attribution_values)).item()
                if max_attribution > 0.5:
                    patterns.append({
                        "type": "high_impact",
                        "description": "Very high attribution values detected",
                        "max_attribution": max_attribution
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Attribution pattern identification failed: {e}")
            return []
    
    def _generate_attribution_visualizations(self, attributions: torch.Tensor, 
                                           text: str, tokens: List[str]) -> Dict:
        """Generate visualization data for attributions"""
        try:
            visualizations = {}
            
            # Token-level attribution plot
            visualizations["token_attributions"] = self._generate_token_attribution_plot(attributions, tokens)
            
            # Attribution heatmap
            visualizations["attribution_heatmap"] = self._generate_attribution_heatmap(attributions)
            
            # Attribution distribution
            visualizations["attribution_distribution"] = self._generate_attribution_distribution(attributions)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Attribution visualization generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_token_attribution_plot(self, attributions: torch.Tensor, tokens: List[str]) -> Dict:
        """Generate token-level attribution plot data"""
        try:
            # Sum attributions across embedding dimensions
            token_attributions = torch.sum(attributions[0], dim=1)
            
            plot_data = []
            for i, (token, attribution) in enumerate(zip(tokens, token_attributions)):
                plot_data.append({
                    "token": token,
                    "attribution": float(attribution),
                    "abs_attribution": float(abs(attribution)),
                    "position": i,
                    "color": "red" if attribution > 0 else "blue"
                })
            
            return {
                "data": plot_data,
                "title": "Token Attribution Values",
                "x_label": "Token Position",
                "y_label": "Attribution Value"
            }
            
        except Exception as e:
            logger.error(f"Token attribution plot generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_attribution_heatmap(self, attributions: torch.Tensor) -> Dict:
        """Generate attribution heatmap data"""
        try:
            # Reshape attributions for heatmap
            heatmap_data = attributions[0].cpu().numpy()  # [seq_len, embedding_dim]
            
            return {
                "data": heatmap_data.tolist(),
                "title": "Attribution Heatmap",
                "x_label": "Embedding Dimension",
                "y_label": "Token Position"
            }
            
        except Exception as e:
            logger.error(f"Attribution heatmap generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_attribution_distribution(self, attributions: torch.Tensor) -> Dict:
        """Generate attribution distribution data"""
        try:
            attribution_values = attributions.flatten().cpu().numpy()
            
            # Create histogram data
            hist, bins = np.histogram(attribution_values, bins=20)
            
            distribution_data = []
            for i in range(len(hist)):
                distribution_data.append({
                    "bin_center": float((bins[i] + bins[i+1]) / 2),
                    "count": int(hist[i]),
                    "bin_start": float(bins[i]),
                    "bin_end": float(bins[i+1])
                })
            
            return {
                "data": distribution_data,
                "title": "Attribution Distribution",
                "x_label": "Attribution Value",
                "y_label": "Count",
                "mean": float(np.mean(attribution_values)),
                "std": float(np.std(attribution_values))
            }
            
        except Exception as e:
            logger.error(f"Attribution distribution generation failed: {e}")
            return {"error": str(e)}
    
    def compare_attributions(self, original_text: str, adversarial_text: str) -> Dict:
        """
        Compare attributions between original and adversarial text
        
        Args:
            original_text: Original text
            adversarial_text: Adversarial text
            
        Returns:
            Attribution comparison dictionary
        """
        try:
            # Get attributions for both texts
            orig_attributions = self._compute_integrated_gradients(original_text, "", 50)
            adv_attributions = self._compute_integrated_gradients(adversarial_text, "", 50)
            
            # Compare attributions
            comparison = {
                "original_attributions": orig_attributions.cpu().numpy().tolist(),
                "adversarial_attributions": adv_attributions.cpu().numpy().tolist(),
                "attribution_difference": (adv_attributions - orig_attributions).cpu().numpy().tolist(),
                "attribution_change_analysis": self._analyze_attribution_changes(orig_attributions, adv_attributions)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Attribution comparison failed: {e}")
            return {"error": str(e)}
    
    def _analyze_attribution_changes(self, orig_attributions: torch.Tensor, 
                                   adv_attributions: torch.Tensor) -> Dict:
        """Analyze changes in attributions between original and adversarial"""
        try:
            # Calculate difference
            attribution_diff = adv_attributions - orig_attributions
            
            # Calculate statistics
            analysis = {
                "mean_change": torch.mean(attribution_diff).item(),
                "max_change": torch.max(attribution_diff).item(),
                "min_change": torch.min(attribution_diff).item(),
                "std_change": torch.std(attribution_diff).item(),
                "total_change": torch.sum(torch.abs(attribution_diff)).item(),
                "change_ratio": torch.sum(torch.abs(attribution_diff)).item() / torch.sum(torch.abs(orig_attributions)).item() if torch.sum(torch.abs(orig_attributions)) > 0 else 0
            }
            
            # Identify significant changes
            significant_changes = attribution_diff[torch.abs(attribution_diff) > 0.1]
            analysis["significant_changes"] = {
                "count": len(significant_changes),
                "percentage": len(significant_changes) / attribution_diff.numel(),
                "mean_magnitude": torch.mean(torch.abs(significant_changes)).item() if len(significant_changes) > 0 else 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Attribution change analysis failed: {e}")
            return {"error": str(e)}
