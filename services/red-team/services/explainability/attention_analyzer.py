"""
Attention Analyzer for ML Security
Transformer attention visualization and analysis for vulnerability detection
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """
    Attention Analyzer for transformer models
    Analyzes attention patterns to understand model behavior and vulnerabilities
    """
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize attention analyzer
        
        Args:
            model: Transformer model to analyze
            tokenizer: Tokenizer for the model
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Get model configuration
        self.num_layers = getattr(model.config, 'num_hidden_layers', 12)
        self.num_heads = getattr(model.config, 'num_attention_heads', 12)
        self.hidden_size = getattr(model.config, 'hidden_size', 768)
        
        logger.info(f"✅ Attention analyzer initialized: {self.num_layers} layers, {self.num_heads} heads")
    
    def analyze_attention_patterns(self, text: str, attack_result: Dict) -> Dict:
        """
        Analyze attention patterns for vulnerability detection
        
        Args:
            text: Input text to analyze
            attack_result: Result of the attack
            
        Returns:
            Attention analysis dictionary
        """
        try:
            # Get attention weights
            attention_weights = self._get_attention_weights(text)
            
            # Tokenize text
            tokens = self.tokenizer.tokenize(text)
            
            # Analyze patterns
            analysis = {
                "method": "Attention Analysis",
                "text": text,
                "tokens": tokens,
                "attack_detected": attack_result.get("detected", False),
                "attention_weights": attention_weights,
                "layer_analysis": self._analyze_layer_attention(attention_weights),
                "head_analysis": self._analyze_head_attention(attention_weights),
                "token_importance": self._calculate_token_importance(attention_weights, tokens),
                "attention_flow": self._trace_attention_flow(attention_weights, tokens),
                "vulnerability_insights": self._identify_attention_vulnerabilities(attention_weights, attack_result),
                "visualizations": self._generate_attention_visualizations(attention_weights, tokens)
            }
            
            logger.debug(f"Attention analysis completed for text: {text[:50]}...")
            return analysis
            
        except Exception as e:
            logger.error(f"Attention analysis failed: {e}")
            return {"error": str(e), "method": "Attention Analysis"}
    
    def _get_attention_weights(self, text: str) -> List[torch.Tensor]:
        """
        Get attention weights from all layers
        
        Args:
            text: Input text
            
        Returns:
            List of attention weight tensors for each layer
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass with attention output
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
                attention_weights = outputs.attentions  # List of [batch, heads, seq_len, seq_len]
            
            return attention_weights
            
        except Exception as e:
            logger.error(f"Attention weight extraction failed: {e}")
            return []
    
    def _analyze_layer_attention(self, attention_weights: List[torch.Tensor]) -> Dict:
        """
        Analyze attention patterns across layers
        
        Args:
            attention_weights: List of attention weight tensors
            
        Returns:
            Layer-wise analysis dictionary
        """
        try:
            if not attention_weights:
                return {"error": "No attention weights available"}
            
            layer_analysis = {
                "num_layers": len(attention_weights),
                "layer_statistics": [],
                "attention_entropy": [],
                "attention_diversity": []
            }
            
            for layer_idx, attention in enumerate(attention_weights):
                # Average across heads and batch
                avg_attention = torch.mean(attention, dim=(0, 1))  # [seq_len, seq_len]
                
                # Calculate statistics
                layer_stats = {
                    "layer": layer_idx,
                    "max_attention": torch.max(avg_attention).item(),
                    "min_attention": torch.min(avg_attention).item(),
                    "mean_attention": torch.mean(avg_attention).item(),
                    "std_attention": torch.std(avg_attention).item(),
                    "attention_sparsity": self._calculate_attention_sparsity(avg_attention)
                }
                
                # Calculate entropy
                entropy = self._calculate_attention_entropy(avg_attention)
                layer_analysis["attention_entropy"].append(entropy)
                
                # Calculate diversity
                diversity = self._calculate_attention_diversity(avg_attention)
                layer_analysis["attention_diversity"].append(diversity)
                
                layer_analysis["layer_statistics"].append(layer_stats)
            
            # Overall layer analysis
            layer_analysis["entropy_trend"] = self._analyze_entropy_trend(layer_analysis["attention_entropy"])
            layer_analysis["diversity_trend"] = self._analyze_diversity_trend(layer_analysis["attention_diversity"])
            
            return layer_analysis
            
        except Exception as e:
            logger.error(f"Layer attention analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_head_attention(self, attention_weights: List[torch.Tensor]) -> Dict:
        """
        Analyze attention patterns across heads
        
        Args:
            attention_weights: List of attention weight tensors
            
        Returns:
            Head-wise analysis dictionary
        """
        try:
            if not attention_weights:
                return {"error": "No attention weights available"}
            
            head_analysis = {
                "num_heads": attention_weights[0].shape[1],
                "head_statistics": [],
                "head_specialization": [],
                "head_interaction": []
            }
            
            # Analyze each head across all layers
            for head_idx in range(attention_weights[0].shape[1]):
                head_attentions = []
                
                for layer_idx, attention in enumerate(attention_weights):
                    head_attention = attention[0, head_idx]  # [seq_len, seq_len]
                    head_attentions.append(head_attention)
                
                # Calculate head statistics
                head_stats = {
                    "head": head_idx,
                    "avg_attention": torch.mean(torch.stack(head_attentions)).item(),
                    "attention_variance": torch.var(torch.stack(head_attentions)).item(),
                    "specialization_score": self._calculate_head_specialization(head_attentions)
                }
                
                head_analysis["head_statistics"].append(head_stats)
                head_analysis["head_specialization"].append(head_stats["specialization_score"])
            
            # Calculate head interactions
            head_analysis["head_interaction"] = self._calculate_head_interactions(attention_weights)
            
            return head_analysis
            
        except Exception as e:
            logger.error(f"Head attention analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_token_importance(self, attention_weights: List[torch.Tensor], tokens: List[str]) -> List[Dict]:
        """
        Calculate token importance based on attention patterns
        
        Args:
            attention_weights: List of attention weight tensors
            tokens: List of tokens
            
        Returns:
            List of token importance dictionaries
        """
        try:
            if not attention_weights:
                return []
            
            # Average attention across all layers and heads
            all_attention = torch.stack(attention_weights)  # [layers, batch, heads, seq_len, seq_len]
            avg_attention = torch.mean(all_attention, dim=(0, 1, 2))  # [seq_len, seq_len]
            
            # Calculate importance for each token
            token_importance = []
            for i, token in enumerate(tokens):
                if i < avg_attention.shape[0]:
                    # Sum of attention received by this token
                    attention_received = torch.sum(avg_attention[:, i]).item()
                    
                    # Sum of attention given by this token
                    attention_given = torch.sum(avg_attention[i, :]).item()
                    
                    # Self-attention
                    self_attention = avg_attention[i, i].item()
                    
                    token_importance.append({
                        "token": token,
                        "position": i,
                        "attention_received": attention_received,
                        "attention_given": attention_given,
                        "self_attention": self_attention,
                        "importance_score": (attention_received + attention_given) / 2,
                        "attention_balance": attention_given / (attention_received + 1e-8)
                    })
            
            # Sort by importance
            token_importance.sort(key=lambda x: x["importance_score"], reverse=True)
            
            return token_importance
            
        except Exception as e:
            logger.error(f"Token importance calculation failed: {e}")
            return []
    
    def _trace_attention_flow(self, attention_weights: List[torch.Tensor], tokens: List[str]) -> Dict:
        """
        Trace attention flow through the model
        
        Args:
            attention_weights: List of attention weight tensors
            tokens: List of tokens
            
        Returns:
            Attention flow analysis dictionary
        """
        try:
            if not attention_weights:
                return {"error": "No attention weights available"}
            
            flow_analysis = {
                "attention_paths": [],
                "attention_cycles": [],
                "attention_bottlenecks": [],
                "flow_statistics": {}
            }
            
            # Trace attention paths
            for layer_idx, attention in enumerate(attention_weights):
                layer_attention = torch.mean(attention, dim=(0, 1))  # [seq_len, seq_len]
                
                # Find strongest attention connections
                for i in range(layer_attention.shape[0]):
                    for j in range(layer_attention.shape[1]):
                        if layer_attention[i, j] > 0.1:  # Threshold
                            flow_analysis["attention_paths"].append({
                                "layer": layer_idx,
                                "from_token": tokens[i] if i < len(tokens) else f"token_{i}",
                                "to_token": tokens[j] if j < len(tokens) else f"token_{j}",
                                "strength": layer_attention[i, j].item(),
                                "from_position": i,
                                "to_position": j
                            })
            
            # Sort by strength
            flow_analysis["attention_paths"].sort(key=lambda x: x["strength"], reverse=True)
            
            # Find attention cycles
            flow_analysis["attention_cycles"] = self._find_attention_cycles(attention_weights, tokens)
            
            # Find bottlenecks
            flow_analysis["attention_bottlenecks"] = self._find_attention_bottlenecks(attention_weights, tokens)
            
            return flow_analysis
            
        except Exception as e:
            logger.error(f"Attention flow tracing failed: {e}")
            return {"error": str(e)}
    
    def _identify_attention_vulnerabilities(self, attention_weights: List[torch.Tensor], 
                                          attack_result: Dict) -> Dict:
        """
        Identify vulnerabilities based on attention patterns
        
        Args:
            attention_weights: List of attention weight tensors
            attack_result: Attack result information
            
        Returns:
            Vulnerability insights dictionary
        """
        try:
            vulnerabilities = {
                "attention_anomalies": [],
                "vulnerability_indicators": [],
                "risk_assessment": "UNKNOWN",
                "recommendations": []
            }
            
            if not attention_weights:
                return vulnerabilities
            
            # Check for attention anomalies
            vulnerabilities["attention_anomalies"] = self._detect_attention_anomalies(attention_weights)
            
            # Check vulnerability indicators
            vulnerabilities["vulnerability_indicators"] = self._check_vulnerability_indicators(
                attention_weights, attack_result
            )
            
            # Assess risk
            vulnerabilities["risk_assessment"] = self._assess_attention_risk(
                attention_weights, attack_result
            )
            
            # Generate recommendations
            vulnerabilities["recommendations"] = self._generate_attention_recommendations(
                attention_weights, attack_result
            )
            
            return vulnerabilities
            
        except Exception as e:
            logger.error(f"Attention vulnerability identification failed: {e}")
            return {"error": str(e)}
    
    def _calculate_attention_sparsity(self, attention: torch.Tensor) -> float:
        """Calculate sparsity of attention matrix"""
        try:
            # Count non-zero elements
            non_zero = torch.count_nonzero(attention)
            total = attention.numel()
            return 1.0 - (non_zero.float() / total).item()
        except Exception as e:
            logger.error(f"Attention sparsity calculation failed: {e}")
            return 0.0
    
    def _calculate_attention_entropy(self, attention: torch.Tensor) -> float:
        """Calculate entropy of attention distribution"""
        try:
            # Flatten attention matrix
            flat_attention = attention.flatten()
            
            # Normalize to probabilities
            probs = F.softmax(flat_attention, dim=0)
            
            # Calculate entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            return entropy.item()
        except Exception as e:
            logger.error(f"Attention entropy calculation failed: {e}")
            return 0.0
    
    def _calculate_attention_diversity(self, attention: torch.Tensor) -> float:
        """Calculate diversity of attention patterns"""
        try:
            # Calculate variance across attention matrix
            diversity = torch.var(attention).item()
            return diversity
        except Exception as e:
            logger.error(f"Attention diversity calculation failed: {e}")
            return 0.0
    
    def _calculate_head_specialization(self, head_attentions: List[torch.Tensor]) -> float:
        """Calculate specialization score for a head"""
        try:
            if not head_attentions:
                return 0.0
            
            # Calculate variance across layers for this head
            stacked_attentions = torch.stack(head_attentions)
            specialization = torch.var(stacked_attentions).item()
            return specialization
        except Exception as e:
            logger.error(f"Head specialization calculation failed: {e}")
            return 0.0
    
    def _calculate_head_interactions(self, attention_weights: List[torch.Tensor]) -> List[Dict]:
        """Calculate interactions between attention heads"""
        try:
            interactions = []
            
            if not attention_weights:
                return interactions
            
            # Use first layer for head interaction analysis
            layer_attention = attention_weights[0][0]  # [heads, seq_len, seq_len]
            num_heads = layer_attention.shape[0]
            
            for i in range(num_heads):
                for j in range(i + 1, num_heads):
                    head_i = layer_attention[i]
                    head_j = layer_attention[j]
                    
                    # Calculate correlation between heads
                    correlation = torch.corrcoef(torch.stack([
                        head_i.flatten(), head_j.flatten()
                    ]))[0, 1].item()
                    
                    interactions.append({
                        "head_1": i,
                        "head_2": j,
                        "correlation": correlation,
                        "interaction_strength": abs(correlation)
                    })
            
            # Sort by interaction strength
            interactions.sort(key=lambda x: x["interaction_strength"], reverse=True)
            
            return interactions
            
        except Exception as e:
            logger.error(f"Head interaction calculation failed: {e}")
            return []
    
    def _analyze_entropy_trend(self, entropy_values: List[float]) -> Dict:
        """Analyze entropy trend across layers"""
        try:
            if len(entropy_values) < 2:
                return {"trend": "insufficient_data"}
            
            # Calculate trend
            x = np.arange(len(entropy_values))
            y = np.array(entropy_values)
            
            # Linear regression
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            
            if slope > 0.01:
                trend = "increasing"
            elif slope < -0.01:
                trend = "decreasing"
            else:
                trend = "stable"
            
            return {
                "trend": trend,
                "slope": slope,
                "entropy_values": entropy_values
            }
            
        except Exception as e:
            logger.error(f"Entropy trend analysis failed: {e}")
            return {"trend": "error"}
    
    def _analyze_diversity_trend(self, diversity_values: List[float]) -> Dict:
        """Analyze diversity trend across layers"""
        try:
            if len(diversity_values) < 2:
                return {"trend": "insufficient_data"}
            
            # Calculate trend
            x = np.arange(len(diversity_values))
            y = np.array(diversity_values)
            
            # Linear regression
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            
            if slope > 0.01:
                trend = "increasing"
            elif slope < -0.01:
                trend = "decreasing"
            else:
                trend = "stable"
            
            return {
                "trend": trend,
                "slope": slope,
                "diversity_values": diversity_values
            }
            
        except Exception as e:
            logger.error(f"Diversity trend analysis failed: {e}")
            return {"trend": "error"}
    
    def _find_attention_cycles(self, attention_weights: List[torch.Tensor], tokens: List[str]) -> List[Dict]:
        """Find attention cycles in the model"""
        try:
            cycles = []
            
            if not attention_weights:
                return cycles
            
            # Use first layer for cycle detection
            attention = attention_weights[0][0]  # [heads, seq_len, seq_len]
            avg_attention = torch.mean(attention, dim=0)  # [seq_len, seq_len]
            
            # Find cycles of length 2 and 3
            for i in range(avg_attention.shape[0]):
                for j in range(avg_attention.shape[1]):
                    if i != j and avg_attention[i, j] > 0.1:
                        # Check for cycle back
                        if avg_attention[j, i] > 0.1:
                            cycles.append({
                                "type": "2_cycle",
                                "tokens": [tokens[i] if i < len(tokens) else f"token_{i}",
                                         tokens[j] if j < len(tokens) else f"token_{j}"],
                                "strength": (avg_attention[i, j] + avg_attention[j, i]) / 2,
                                "positions": [i, j]
                            })
            
            return cycles
            
        except Exception as e:
            logger.error(f"Attention cycle detection failed: {e}")
            return []
    
    def _find_attention_bottlenecks(self, attention_weights: List[torch.Tensor], tokens: List[str]) -> List[Dict]:
        """Find attention bottlenecks"""
        try:
            bottlenecks = []
            
            if not attention_weights:
                return bottlenecks
            
            # Average attention across all layers and heads
            all_attention = torch.stack(attention_weights)
            avg_attention = torch.mean(all_attention, dim=(0, 1, 2))  # [seq_len, seq_len]
            
            # Find tokens with high incoming attention
            for i in range(avg_attention.shape[0]):
                incoming_attention = torch.sum(avg_attention[:, i]).item()
                if incoming_attention > 0.5:  # Threshold
                    bottlenecks.append({
                        "token": tokens[i] if i < len(tokens) else f"token_{i}",
                        "position": i,
                        "incoming_attention": incoming_attention,
                        "type": "high_incoming"
                    })
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Attention bottleneck detection failed: {e}")
            return []
    
    def _detect_attention_anomalies(self, attention_weights: List[torch.Tensor]) -> List[Dict]:
        """Detect attention anomalies"""
        try:
            anomalies = []
            
            if not attention_weights:
                return anomalies
            
            # Check for extremely high attention values
            for layer_idx, attention in enumerate(attention_weights):
                max_attention = torch.max(attention).item()
                if max_attention > 0.9:  # Very high attention
                    anomalies.append({
                        "type": "high_attention",
                        "layer": layer_idx,
                        "value": max_attention,
                        "description": "Extremely high attention value detected"
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Attention anomaly detection failed: {e}")
            return []
    
    def _check_vulnerability_indicators(self, attention_weights: List[torch.Tensor], 
                                      attack_result: Dict) -> List[Dict]:
        """Check for vulnerability indicators in attention patterns"""
        try:
            indicators = []
            
            if not attention_weights:
                return indicators
            
            # Check if attack was not detected
            if not attack_result.get("detected", False):
                # Look for attention patterns that might indicate vulnerability
                avg_attention = torch.mean(torch.stack(attention_weights), dim=(0, 1, 2))
                
                # Check for uniform attention (might indicate lack of focus)
                attention_std = torch.std(avg_attention).item()
                if attention_std < 0.05:
                    indicators.append({
                        "type": "uniform_attention",
                        "value": attention_std,
                        "description": "Attention is too uniform, may indicate lack of focus"
                    })
            
            return indicators
            
        except Exception as e:
            logger.error(f"Vulnerability indicator check failed: {e}")
            return []
    
    def _assess_attention_risk(self, attention_weights: List[torch.Tensor], attack_result: Dict) -> str:
        """Assess risk level based on attention patterns"""
        try:
            if not attention_weights:
                return "UNKNOWN"
            
            risk_score = 0.0
            
            # Base risk from attack success
            if not attack_result.get("detected", False):
                risk_score += 0.5
            
            # Add risk from attention patterns
            avg_attention = torch.mean(torch.stack(attention_weights), dim=(0, 1, 2))
            attention_std = torch.std(avg_attention).item()
            
            if attention_std < 0.05:
                risk_score += 0.3  # Uniform attention is risky
            
            if risk_score >= 0.8:
                return "HIGH"
            elif risk_score >= 0.5:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception as e:
            logger.error(f"Attention risk assessment failed: {e}")
            return "UNKNOWN"
    
    def _generate_attention_recommendations(self, attention_weights: List[torch.Tensor], 
                                          attack_result: Dict) -> List[str]:
        """Generate recommendations based on attention analysis"""
        try:
            recommendations = []
            
            if not attention_weights:
                return recommendations
            
            # Check for uniform attention
            avg_attention = torch.mean(torch.stack(attention_weights), dim=(0, 1, 2))
            attention_std = torch.std(avg_attention).item()
            
            if attention_std < 0.05:
                recommendations.append("Model shows uniform attention patterns - consider attention regularization")
            
            # Check for attack success
            if not attack_result.get("detected", False):
                recommendations.append("Model failed to detect attack - strengthen attention mechanisms")
                recommendations.append("Consider attention-based defense mechanisms")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Attention recommendation generation failed: {e}")
            return []
    
    def _generate_attention_visualizations(self, attention_weights: List[torch.Tensor], 
                                         tokens: List[str]) -> Dict:
        """Generate visualization data for attention analysis"""
        try:
            visualizations = {}
            
            if not attention_weights:
                return {"error": "No attention weights available"}
            
            # Attention heatmap
            visualizations["attention_heatmap"] = self._generate_attention_heatmap(attention_weights, tokens)
            
            # Layer-wise attention
            visualizations["layer_attention"] = self._generate_layer_attention_plot(attention_weights)
            
            # Head attention analysis
            visualizations["head_attention"] = self._generate_head_attention_plot(attention_weights)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Attention visualization generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_attention_heatmap(self, attention_weights: List[torch.Tensor], tokens: List[str]) -> Dict:
        """Generate attention heatmap data"""
        try:
            # Use first layer for heatmap
            attention = attention_weights[0][0]  # [heads, seq_len, seq_len]
            avg_attention = torch.mean(attention, dim=0)  # [seq_len, seq_len]
            
            return {
                "data": avg_attention.cpu().numpy().tolist(),
                "tokens": tokens,
                "title": "Attention Heatmap",
                "x_label": "Key Tokens",
                "y_label": "Query Tokens"
            }
            
        except Exception as e:
            logger.error(f"Attention heatmap generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_layer_attention_plot(self, attention_weights: List[torch.Tensor]) -> Dict:
        """Generate layer-wise attention plot data"""
        try:
            layer_data = []
            
            for layer_idx, attention in enumerate(attention_weights):
                avg_attention = torch.mean(attention, dim=(0, 1))  # [seq_len, seq_len]
                
                layer_data.append({
                    "layer": layer_idx,
                    "mean_attention": torch.mean(avg_attention).item(),
                    "max_attention": torch.max(avg_attention).item(),
                    "attention_entropy": self._calculate_attention_entropy(avg_attention)
                })
            
            return {
                "data": layer_data,
                "title": "Layer-wise Attention Analysis",
                "x_label": "Layer",
                "y_label": "Attention Value"
            }
            
        except Exception as e:
            logger.error(f"Layer attention plot generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_head_attention_plot(self, attention_weights: List[torch.Tensor]) -> Dict:
        """Generate head-wise attention plot data"""
        try:
            head_data = []
            
            if not attention_weights:
                return {"error": "No attention weights available"}
            
            # Use first layer for head analysis
            attention = attention_weights[0][0]  # [heads, seq_len, seq_len]
            
            for head_idx in range(attention.shape[0]):
                head_attention = attention[head_idx]
                
                head_data.append({
                    "head": head_idx,
                    "mean_attention": torch.mean(head_attention).item(),
                    "max_attention": torch.max(head_attention).item(),
                    "attention_entropy": self._calculate_attention_entropy(head_attention)
                })
            
            return {
                "data": head_data,
                "title": "Head-wise Attention Analysis",
                "x_label": "Head",
                "y_label": "Attention Value"
            }
            
        except Exception as e:
            logger.error(f"Head attention plot generation failed: {e}")
            return {"error": str(e)}
