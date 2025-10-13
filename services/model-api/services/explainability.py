"""
Model Explainability Service
Provides SHAP values and attention visualization for model debugging
"""

import logging
import numpy as np
import torch
import shap
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Handles model explainability using SHAP and attention visualization"""
    
    def __init__(self, max_cache_size: int = 10):
        self.explainers = {}  # Cache for SHAP explainers
        self.background_data = {}  # Background data for SHAP
        self.max_cache_size = max_cache_size
        self._access_times = {}  # Track access times for LRU eviction
        
    def explain_prediction(self, model, tokenizer, text: str, 
                          method: str = "shap", max_length: int = 512) -> Dict[str, Any]:
        """
        Explain a single prediction
        
        Args:
            model: PyTorch model
            tokenizer: Tokenizer for the model
            text: Input text to explain
            method: Explanation method ("shap", "attention", "both")
            max_length: Maximum sequence length
            
        Returns:
            Explanation results
        """
        try:
            results = {
                "text": text,
                "method": method,
                "explanations": {}
            }
            
            if method in ["shap", "both"]:
                shap_explanation = self._get_shap_explanation(model, tokenizer, text, max_length)
                results["explanations"]["shap"] = shap_explanation
            
            if method in ["attention", "both"]:
                attention_explanation = self._get_attention_explanation(model, tokenizer, text, max_length)
                results["explanations"]["attention"] = attention_explanation
            
            return results
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {"error": str(e)}
    
    def _get_shap_explanation(self, model, tokenizer, text: str, max_length: int) -> Dict[str, Any]:
        """Get SHAP explanation for a prediction"""
        try:
            # Tokenize input
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move to same device as model
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Get input tokens
            input_ids = inputs["input_ids"][0]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # Create SHAP explainer if not cached
            model_name = getattr(model, 'model_name', 'unknown')
            if model_name not in self.explainers:
                self._manage_cache_size(model_name)
                self.explainers[model_name] = self._create_shap_explainer(model, tokenizer)
            
            self._update_access_time(model_name)
            explainer = self.explainers[model_name]
            
            # Get SHAP values
            shap_values = explainer(input_ids.unsqueeze(0))
            
            # Process SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first output
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities, dim=-1)[0].item()
            
            # Create explanation
            explanation = {
                "tokens": tokens,
                "shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "feature_importance": self._calculate_feature_importance(tokens, shap_values),
                "top_features": self._get_top_features(tokens, shap_values, top_k=10)
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error getting SHAP explanation: {e}")
            return {"error": str(e)}
    
    def _get_attention_explanation(self, model, tokenizer, text: str, max_length: int) -> Dict[str, Any]:
        """Get attention weights explanation"""
        try:
            # Tokenize input
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move to same device as model
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Get input tokens
            input_ids = inputs["input_ids"][0]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # Get attention weights
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
                attentions = outputs.attentions  # List of attention tensors
                
                # Get prediction
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities, dim=-1)[0].item()
            
            # Process attention weights
            attention_analysis = self._analyze_attention_weights(attentions, tokens)
            
            explanation = {
                "tokens": tokens,
                "attention_weights": attention_analysis,
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "attention_visualization": self._create_attention_heatmap(attentions, tokens)
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error getting attention explanation: {e}")
            return {"error": str(e)}
    
    def _create_shap_explainer(self, model, tokenizer):
        """Create SHAP explainer for the model - IMPROVED IMPLEMENTATION"""
        try:
            # Create more diverse background data (50-100 samples)
            background_texts = [
                "This is a normal text",
                "Please help me with this",
                "I need assistance",
                "Can you explain this",
                "What is the weather like",
                "How does this work",
                "Tell me about it",
                "I want to know more",
                "Can you help me understand",
                "What should I do",
                "This is important",
                "I need to know",
                "Please explain",
                "How can I help",
                "What do you think",
                "I'm not sure",
                "That's interesting",
                "I understand now",
                "Thank you for explaining",
                "That makes sense",
                "I see what you mean",
                "That's helpful",
                "I appreciate it",
                "Good to know",
                "I'll remember that",
                "That's useful information",
                "I didn't know that",
                "That's good to know",
                "I'll keep that in mind",
                "That's very helpful",
                "I understand better now",
                "That clarifies things",
                "I see the point",
                "That's clear now",
                "I get it now",
                "That's very useful",
                "I'll use that",
                "That's good advice",
                "I'll try that",
                "That sounds good",
                "I agree with that",
                "That's a good point",
                "I think so too",
                "That's right",
                "I believe that",
                "That's correct",
                "I know that",
                "I've heard that",
                "That's true",
                "I think that's right"
            ]
            
            # Tokenize background texts with consistent padding
            background_tokens = []
            max_length = 512
            
            for text in background_texts:
                inputs = tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding='max_length',  # Consistent padding
                    truncation=True, 
                    max_length=max_length
                )
                background_tokens.append(inputs["input_ids"][0])
            
            # Convert to tensor for consistent shape
            background_tokens = torch.stack(background_tokens)
            
            # Create SHAP explainer with improved model wrapper
            def model_predict(input_ids):
                """Wrapper function for SHAP - handles variable batch sizes"""
                with torch.no_grad():
                    if hasattr(model, 'device'):
                        input_ids = input_ids.to(model.device)
                    
                    # Ensure input_ids is 2D
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    
                    # Create attention mask
                    attention_mask = (input_ids != tokenizer.pad_token_id).long()
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                    
                    return probabilities.cpu().numpy()
            
            # Use PartitionExplainer for text (more appropriate than generic Explainer)
            try:
                explainer = shap.PartitionExplainer(model_predict, background_tokens)
            except:
                # Fallback to regular Explainer if PartitionExplainer fails
                explainer = shap.Explainer(model_predict, background_tokens)
            
            return explainer
            
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {e}")
            return None
    
    def _analyze_attention_weights(self, attentions, tokens):
        """Analyze attention weights across layers"""
        try:
            # Get attention from last layer
            last_attention = attentions[-1]  # Shape: (batch, heads, seq_len, seq_len)
            
            # Average across heads
            avg_attention = last_attention.mean(dim=1)[0]  # Shape: (seq_len, seq_len)
            
            # Get attention to [CLS] token (first token)
            cls_attention = avg_attention[0, :]  # Shape: (seq_len,)
            
            # Get attention from [CLS] token
            cls_from_attention = avg_attention[:, 0]  # Shape: (seq_len,)
            
            # Calculate token importance
            token_importance = (cls_attention + cls_from_attention) / 2
            
            # Create attention pairs
            attention_pairs = []
            for i in range(len(tokens)):
                for j in range(len(tokens)):
                    if i != j and avg_attention[i, j] > 0.1:  # Threshold for significant attention
                        attention_pairs.append({
                            "from_token": tokens[i],
                            "to_token": tokens[j],
                            "attention_weight": float(avg_attention[i, j])
                        })
            
            # Sort by attention weight
            attention_pairs.sort(key=lambda x: x["attention_weight"], reverse=True)
            
            return {
                "cls_attention_to_tokens": cls_attention.tolist(),
                "tokens_attention_to_cls": cls_from_attention.tolist(),
                "token_importance": token_importance.tolist(),
                "top_attention_pairs": attention_pairs[:20],  # Top 20 pairs
                "attention_statistics": {
                    "max_attention": float(avg_attention.max()),
                    "min_attention": float(avg_attention.min()),
                    "mean_attention": float(avg_attention.mean()),
                    "std_attention": float(avg_attention.std())
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing attention weights: {e}")
            return {"error": str(e)}
    
    def _create_attention_heatmap(self, attentions, tokens):
        """Create attention heatmap visualization"""
        fig = None
        try:
            # Get attention from last layer
            last_attention = attentions[-1]
            avg_attention = last_attention.mean(dim=1)[0]  # Average across heads
            
            # Create heatmap
            fig = plt.figure(figsize=(12, 8))
            sns.heatmap(
                avg_attention.cpu().numpy(),
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='Blues',
                cbar=True
            )
            plt.title('Attention Heatmap')
            plt.xlabel('Key Tokens')
            plt.ylabel('Query Tokens')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error creating attention heatmap: {e}")
            return None
        finally:
            # Always close the figure to prevent memory leaks
            if fig is not None:
                plt.close(fig)
    
    def _calculate_feature_importance(self, tokens, shap_values):
        """Calculate feature importance from SHAP values"""
        try:
            # Get absolute SHAP values
            abs_shap = np.abs(shap_values)
            
            # Calculate importance scores
            importance_scores = []
            for i, token in enumerate(tokens):
                if i < len(abs_shap):
                    importance_scores.append({
                        "token": token,
                        "importance": float(abs_shap[i]),
                        "position": i
                    })
            
            # Sort by importance
            importance_scores.sort(key=lambda x: x["importance"], reverse=True)
            
            return importance_scores
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return []
    
    def _get_top_features(self, tokens, shap_values, top_k=10):
        """Get top K most important features"""
        try:
            feature_importance = self._calculate_feature_importance(tokens, shap_values)
            return feature_importance[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting top features: {e}")
            return []
    
    def explain_batch(self, model, tokenizer, texts: List[str], 
                     method: str = "shap") -> List[Dict[str, Any]]:
        """Explain a batch of predictions"""
        try:
            explanations = []
            
            for text in texts:
                explanation = self.explain_prediction(model, tokenizer, text, method)
                explanations.append(explanation)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error explaining batch: {e}")
            return [{"error": str(e)} for _ in texts]
    
    def get_model_insights(self, model, tokenizer, sample_texts: List[str]) -> Dict[str, Any]:
        """Get overall model insights from sample texts"""
        try:
            insights = {
                "model_analysis": {},
                "feature_analysis": {},
                "attention_analysis": {}
            }
            
            # Analyze multiple samples
            all_explanations = []
            for text in sample_texts:
                explanation = self.explain_prediction(model, tokenizer, text, "both")
                all_explanations.append(explanation)
            
            # Analyze SHAP patterns
            if "shap" in all_explanations[0].get("explanations", {}):
                shap_insights = self._analyze_shap_patterns(all_explanations)
                insights["feature_analysis"] = shap_insights
            
            # Analyze attention patterns
            if "attention" in all_explanations[0].get("explanations", {}):
                attention_insights = self._analyze_attention_patterns(all_explanations)
                insights["attention_analysis"] = attention_insights
            
            # Overall model insights
            insights["model_analysis"] = {
                "total_samples": len(sample_texts),
                "successful_explanations": len([e for e in all_explanations if "error" not in e]),
                "average_confidence": np.mean([
                    e.get("explanations", {}).get("shap", {}).get("confidence", 0)
                    for e in all_explanations if "error" not in e
                ])
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting model insights: {e}")
            return {"error": str(e)}
    
    def _analyze_shap_patterns(self, explanations):
        """Analyze SHAP patterns across multiple samples"""
        try:
            all_features = []
            all_importances = []
            
            for explanation in explanations:
                if "shap" in explanation.get("explanations", {}):
                    shap_data = explanation["explanations"]["shap"]
                    if "feature_importance" in shap_data:
                        all_features.extend(shap_data["feature_importance"])
                        all_importances.extend([f["importance"] for f in shap_data["feature_importance"]])
            
            if not all_features:
                return {"error": "No SHAP data available"}
            
            # Find most important features across all samples
            feature_importance_map = {}
            for feature in all_features:
                token = feature["token"]
                if token not in feature_importance_map:
                    feature_importance_map[token] = []
                feature_importance_map[token].append(feature["importance"])
            
            # Calculate average importance per token
            avg_importance = {
                token: np.mean(importances)
                for token, importances in feature_importance_map.items()
            }
            
            # Sort by average importance
            top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            
            return {
                "top_important_tokens": top_features,
                "importance_statistics": {
                    "mean_importance": np.mean(all_importances),
                    "std_importance": np.std(all_importances),
                    "max_importance": np.max(all_importances),
                    "min_importance": np.min(all_importances)
                },
                "total_unique_tokens": len(feature_importance_map)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing SHAP patterns: {e}")
            return {"error": str(e)}
    
    def _analyze_attention_patterns(self, explanations):
        """Analyze attention patterns across multiple samples"""
        try:
            all_attention_stats = []
            
            for explanation in explanations:
                if "attention" in explanation.get("explanations", {}):
                    attention_data = explanation["explanations"]["attention"]
                    if "attention_statistics" in attention_data:
                        all_attention_stats.append(attention_data["attention_statistics"])
            
            if not all_attention_stats:
                return {"error": "No attention data available"}
            
            # Aggregate attention statistics
            avg_max_attention = np.mean([s["max_attention"] for s in all_attention_stats])
            avg_mean_attention = np.mean([s["mean_attention"] for s in all_attention_stats])
            avg_std_attention = np.mean([s["std_attention"] for s in all_attention_stats])
            
            return {
                "attention_statistics": {
                    "average_max_attention": avg_max_attention,
                    "average_mean_attention": avg_mean_attention,
                    "average_std_attention": avg_std_attention
                },
                "samples_analyzed": len(all_attention_stats)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing attention patterns: {e}")
            return {"error": str(e)}
    
    def _evict_oldest(self):
        """Evict oldest explainer from cache"""
        if not self._access_times:
            return
        
        # Find oldest access time
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        # Remove from all caches
        if oldest_key in self.explainers:
            del self.explainers[oldest_key]
        if oldest_key in self.background_data:
            del self.background_data[oldest_key]
        if oldest_key in self._access_times:
            del self._access_times[oldest_key]
        
        logger.info(f"Evicted explainer for model: {oldest_key}")
    
    def _update_access_time(self, model_name: str):
        """Update access time for LRU eviction"""
        import time
        self._access_times[model_name] = time.time()
    
    def _manage_cache_size(self, model_name: str):
        """Manage cache size using LRU eviction"""
        if len(self.explainers) >= self.max_cache_size and model_name not in self.explainers:
            self._evict_oldest()
    
    def clear_cache(self):
        """Clear explainer cache"""
        self.explainers.clear()
        self.background_data.clear()
        self._access_times.clear()
        logger.info("Explainability cache cleared")

# Global explainer instance
explainer = ModelExplainer()
