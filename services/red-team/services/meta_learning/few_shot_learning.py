"""
Few-Shot Learning for Attack Pattern Recognition
Implements few-shot learning techniques for rapid attack adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class FewShotAttackLearner:
    """
    Few-shot learning system for attack pattern recognition
    Implements Prototypical Networks and Matching Networks for rapid adaptation
    """
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize few-shot attack learner
        
        Args:
            embedding_dim: Dimension of input embeddings
            device: Device to run on
        """
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Prototype storage for each attack category
        self.prototypes = defaultdict(list)
        self.support_embeddings = defaultdict(list)
        self.attack_categories = set()
        
        # Few-shot learning parameters
        self.support_size = 5  # Number of support examples per class
        self.query_size = 10   # Number of query examples per class
        self.num_ways = 2      # Number of classes (safe vs attack)
        
        logger.info(f"✅ Few-Shot Attack Learner initialized: embedding_dim={embedding_dim}")
    
    def add_support_examples(self, 
                           category: str, 
                           examples: List[Dict[str, Any]], 
                           labels: List[int]) -> None:
        """
        Add support examples for few-shot learning
        
        Args:
            category: Attack category
            examples: List of example patterns
            labels: Corresponding labels
        """
        try:
            for example, label in zip(examples, labels):
                # Extract features from example
                features = self._extract_features(example)
                
                self.support_embeddings[category].append({
                    'features': features,
                    'label': label,
                    'example': example
                })
            
            self.attack_categories.add(category)
            logger.info(f"Added {len(examples)} support examples for category: {category}")
            
        except Exception as e:
            logger.error(f"Failed to add support examples: {e}")
    
    def compute_prototypes(self, category: str) -> Dict[str, np.ndarray]:
        """
        Compute prototypes for a given category using Prototypical Networks
        
        Args:
            category: Attack category
            
        Returns:
            Dictionary of prototypes for each class
        """
        try:
            if category not in self.support_embeddings:
                return {}
            
            # Group examples by label
            class_examples = defaultdict(list)
            for example in self.support_embeddings[category]:
                label = example['label']
                features = example['features']
                class_examples[label].append(features)
            
            # Compute prototypes (mean of support examples for each class)
            prototypes = {}
            for label, examples in class_examples.items():
                if examples:
                    prototype = np.mean(examples, axis=0)
                    prototypes[label] = prototype
            
            # Store prototypes
            self.prototypes[category] = prototypes
            
            logger.info(f"Computed prototypes for category {category}: {list(prototypes.keys())}")
            return prototypes
            
        except Exception as e:
            logger.error(f"Prototype computation failed: {e}")
            return {}
    
    def few_shot_classification(self, 
                              query_examples: List[Dict[str, Any]], 
                              category: str,
                              method: str = 'prototypical') -> List[Dict[str, Any]]:
        """
        Perform few-shot classification on query examples
        
        Args:
            query_examples: List of query examples to classify
            category: Attack category
            method: Classification method ('prototypical' or 'matching')
            
        Returns:
            List of classification results
        """
        try:
            if category not in self.prototypes:
                # Compute prototypes if not available
                self.compute_prototypes(category)
            
            if category not in self.prototypes:
                return [{"error": f"No prototypes available for category: {category}"} for _ in query_examples]
            
            results = []
            
            for example in query_examples:
                # Extract features from query example
                query_features = self._extract_features(example)
                
                if method == 'prototypical':
                    prediction = self._prototypical_classification(query_features, category)
                elif method == 'matching':
                    prediction = self._matching_network_classification(query_features, category)
                else:
                    prediction = {"error": f"Unknown method: {method}"}
                
                results.append({
                    'example': example,
                    'prediction': prediction,
                    'method': method,
                    'category': category
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Few-shot classification failed: {e}")
            return [{"error": str(e)} for _ in query_examples]
    
    def _prototypical_classification(self, 
                                   query_features: np.ndarray, 
                                   category: str) -> Dict[str, Any]:
        """
        Perform classification using Prototypical Networks
        
        Args:
            query_features: Features of query example
            category: Attack category
            
        Returns:
            Classification result
        """
        try:
            prototypes = self.prototypes[category]
            
            if not prototypes:
                return {"error": "No prototypes available"}
            
            # Compute distances to prototypes
            distances = {}
            similarities = {}
            
            for label, prototype in prototypes.items():
                # Euclidean distance
                distance = np.linalg.norm(query_features - prototype)
                distances[label] = distance
                
                # Cosine similarity
                similarity = cosine_similarity([query_features], [prototype])[0][0]
                similarities[label] = similarity
            
            # Find closest prototype
            predicted_label = min(distances.keys(), key=lambda x: distances[x])
            confidence = similarities[predicted_label]
            
            return {
                'predicted_label': predicted_label,
                'confidence': float(confidence),
                'distances': {str(k): float(v) for k, v in distances.items()},
                'similarities': {str(k): float(v) for k, v in similarities.items()}
            }
            
        except Exception as e:
            logger.error(f"Prototypical classification failed: {e}")
            return {"error": str(e)}
    
    def _matching_network_classification(self, 
                                       query_features: np.ndarray, 
                                       category: str) -> Dict[str, Any]:
        """
        Perform classification using Matching Networks
        
        Args:
            query_features: Features of query example
            category: Attack category
            
        Returns:
            Classification result
        """
        try:
            if category not in self.support_embeddings:
                return {"error": "No support examples available"}
            
            # Compute similarities to all support examples
            similarities = []
            labels = []
            
            for support_example in self.support_embeddings[category]:
                support_features = support_example['features']
                similarity = cosine_similarity([query_features], [support_features])[0][0]
                similarities.append(similarity)
                labels.append(support_example['label'])
            
            # Weighted voting based on similarities
            label_weights = defaultdict(float)
            for similarity, label in zip(similarities, labels):
                label_weights[label] += similarity
            
            # Find label with highest weight
            predicted_label = max(label_weights.keys(), key=lambda x: label_weights[x])
            confidence = label_weights[predicted_label] / sum(label_weights.values())
            
            return {
                'predicted_label': predicted_label,
                'confidence': float(confidence),
                'label_weights': {str(k): float(v) for k, v in label_weights.items()},
                'support_similarities': [float(s) for s in similarities]
            }
            
        except Exception as e:
            logger.error(f"Matching network classification failed: {e}")
            return {"error": str(e)}
    
    def _extract_features(self, example: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from an example
        
        Args:
            example: Example pattern
            
        Returns:
            Feature vector
        """
        try:
            # Extract text from example
            if isinstance(example, dict):
                text = example.get('text', example.get('pattern', ''))
            else:
                text = str(example)
            
            # Simple feature extraction (replace with proper embedding)
            # In practice, use a pre-trained model like BERT
            features = self._text_to_features(text)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(self.embedding_dim)
    
    def _text_to_features(self, text: str) -> np.ndarray:
        """
        Convert text to feature vector
        
        Args:
            text: Input text
            
        Returns:
            Feature vector
        """
        try:
            # Simple bag-of-words features (replace with proper embedding)
            words = text.lower().split()
            
            # Create a simple feature vector
            features = np.zeros(self.embedding_dim)
            
            # Character-level features
            for i, char in enumerate(text[:self.embedding_dim]):
                features[i] = ord(char) % 100
            
            # Word-level features (if space available)
            if len(words) > 0:
                word_features = np.array([len(word) for word in words[:100]])
                if len(word_features) < self.embedding_dim - 100:
                    features[100:100+len(word_features)] = word_features
            
            # Normalize features
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except Exception as e:
            logger.error(f"Text to features conversion failed: {e}")
            return np.zeros(self.embedding_dim)
    
    def meta_learning_episode(self, 
                            support_examples: List[Dict[str, Any]], 
                            support_labels: List[int],
                            query_examples: List[Dict[str, Any]], 
                            query_labels: List[int],
                            category: str) -> Dict[str, Any]:
        """
        Perform a meta-learning episode
        
        Args:
            support_examples: Support set examples
            support_labels: Support set labels
            query_examples: Query set examples
            query_labels: Query set labels
            category: Attack category
            
        Returns:
            Episode results
        """
        try:
            # Add support examples
            self.add_support_examples(category, support_examples, support_labels)
            
            # Compute prototypes
            prototypes = self.compute_prototypes(category)
            
            # Classify query examples
            query_results = self.few_shot_classification(query_examples, category)
            
            # Calculate accuracy
            correct_predictions = 0
            total_predictions = len(query_results)
            
            for result, true_label in zip(query_results, query_labels):
                if 'prediction' in result and 'predicted_label' in result['prediction']:
                    predicted_label = result['prediction']['predicted_label']
                    if predicted_label == true_label:
                        correct_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            episode_results = {
                'category': category,
                'support_size': len(support_examples),
                'query_size': len(query_examples),
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'prototypes_computed': len(prototypes),
                'query_results': query_results
            }
            
            logger.info(f"Meta-learning episode completed: category={category}, accuracy={accuracy:.4f}")
            return episode_results
            
        except Exception as e:
            logger.error(f"Meta-learning episode failed: {e}")
            return {"error": str(e)}
    
    def get_category_stats(self, category: str) -> Dict[str, Any]:
        """
        Get statistics for a specific category
        
        Args:
            category: Attack category
            
        Returns:
            Category statistics
        """
        try:
            if category not in self.support_embeddings:
                return {"error": f"Category {category} not found"}
            
            examples = self.support_embeddings[category]
            
            # Count examples by label
            label_counts = defaultdict(int)
            for example in examples:
                label_counts[example['label']] += 1
            
            # Calculate label distribution
            total_examples = len(examples)
            label_distribution = {str(k): v/total_examples for k, v in label_counts.items()}
            
            stats = {
                'category': category,
                'total_examples': total_examples,
                'label_counts': dict(label_counts),
                'label_distribution': label_distribution,
                'has_prototypes': category in self.prototypes,
                'prototype_count': len(self.prototypes.get(category, {}))
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Category stats computation failed: {e}")
            return {"error": str(e)}
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        Get global statistics across all categories
        
        Returns:
            Global statistics
        """
        try:
            total_examples = sum(len(examples) for examples in self.support_embeddings.values())
            total_categories = len(self.attack_categories)
            
            category_stats = {}
            for category in self.attack_categories:
                category_stats[category] = self.get_category_stats(category)
            
            global_stats = {
                'total_categories': total_categories,
                'total_examples': total_examples,
                'categories': list(self.attack_categories),
                'category_stats': category_stats,
                'avg_examples_per_category': total_examples / total_categories if total_categories > 0 else 0
            }
            
            return global_stats
            
        except Exception as e:
            logger.error(f"Global stats computation failed: {e}")
            return {"error": str(e)}
    
    def clear_category(self, category: str) -> None:
        """
        Clear all data for a specific category
        
        Args:
            category: Category to clear
        """
        try:
            if category in self.support_embeddings:
                del self.support_embeddings[category]
            
            if category in self.prototypes:
                del self.prototypes[category]
            
            self.attack_categories.discard(category)
            
            logger.info(f"Cleared category: {category}")
            
        except Exception as e:
            logger.error(f"Failed to clear category {category}: {e}")
    
    def clear_all(self) -> None:
        """Clear all data"""
        try:
            self.support_embeddings.clear()
            self.prototypes.clear()
            self.attack_categories.clear()
            
            logger.info("Cleared all few-shot learning data")
            
        except Exception as e:
            logger.error(f"Failed to clear all data: {e}")
