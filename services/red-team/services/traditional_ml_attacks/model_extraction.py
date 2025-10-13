"""
Model Extraction Attacks for Traditional ML Models
Black-box model stealing attacks for sklearn, XGBoost, and other traditional ML models
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb

logger = logging.getLogger(__name__)


class ModelExtractionType(Enum):
    """Types of model extraction attacks"""
    QUERY_BASED = "query_based"
    SYNTHETIC_DATA = "synthetic_data"
    TRANSFER_LEARNING = "transfer_learning"
    ADVERSARIAL_QUERIES = "adversarial_queries"
    ACTIVE_LEARNING = "active_learning"


@dataclass
class ModelExtractionResult:
    """Result of model extraction attack"""
    success: bool
    attack_type: ModelExtractionType
    extraction_accuracy: float
    original_accuracy: float
    similarity_score: float
    queries_used: int
    extraction_time: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelExtractionAttacks:
    """
    Model extraction attacks for traditional ML models
    Focuses on stealing model functionality through black-box queries
    """
    
    def __init__(self):
        """Initialize model extraction attack generator"""
        self.extraction_strategies = self._load_extraction_strategies()
        self.query_generators = self._load_query_generators()
        
        logger.info("✅ Initialized ModelExtractionAttacks")
    
    def _load_extraction_strategies(self) -> List[Dict[str, Any]]:
        """Load model extraction strategies"""
        return [
            {
                "name": "random_queries",
                "description": "Use random queries to extract model",
                "efficiency": "low",
                "stealth": "high"
            },
            {
                "name": "boundary_queries",
                "description": "Query decision boundaries to extract model",
                "efficiency": "high",
                "stealth": "medium"
            },
            {
                "name": "gradient_based",
                "description": "Use gradient information to extract model",
                "efficiency": "high",
                "stealth": "low"
            },
            {
                "name": "synthetic_data",
                "description": "Generate synthetic data for extraction",
                "efficiency": "medium",
                "stealth": "high"
            },
            {
                "name": "active_learning",
                "description": "Use active learning for efficient extraction",
                "efficiency": "high",
                "stealth": "medium"
            }
        ]
    
    def _load_query_generators(self) -> List[str]:
        """Load query generation methods"""
        return [
            "random_sampling",
            "boundary_search",
            "gradient_ascent",
            "synthetic_generation",
            "active_selection"
        ]
    
    async def query_based_extraction(self, 
                                   target_model: Any,
                                   X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   max_queries: int = 1000,
                                   query_strategy: str = "random_sampling") -> ModelExtractionResult:
        """
        Query-based model extraction
        Extract model by querying it with various inputs
        """
        try:
            logger.info(f"Starting query-based extraction with {max_queries} queries")
            
            start_time = asyncio.get_event_loop().time()
            
            # Generate queries based on strategy
            if query_strategy == "random_sampling":
                queries = self._generate_random_queries(X_test, max_queries)
            elif query_strategy == "boundary_search":
                queries = self._generate_boundary_queries(X_test, max_queries)
            elif query_strategy == "gradient_ascent":
                queries = self._generate_gradient_queries(target_model, X_test, max_queries)
            elif query_strategy == "synthetic_generation":
                queries = self._generate_synthetic_queries(X_test, max_queries)
            else:
                queries = self._generate_random_queries(X_test, max_queries)
            
            # Query target model
            target_predictions = []
            for query in queries:
                try:
                    pred = target_model.predict(query.reshape(1, -1))[0]
                    target_predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Query failed: {e}")
                    target_predictions.append(0)  # Default prediction
            
            # Train surrogate model
            surrogate_model = self._create_surrogate_model(target_model)
            surrogate_model.fit(queries, target_predictions)
            
            # Evaluate extraction accuracy
            surrogate_pred = surrogate_model.predict(X_test)
            extraction_accuracy = accuracy_score(y_test, surrogate_pred)
            
            # Get original model accuracy for comparison
            original_pred = target_model.predict(X_test)
            original_accuracy = accuracy_score(y_test, original_pred)
            
            # Calculate similarity score
            similarity_score = self._calculate_model_similarity(
                target_model, surrogate_model, X_test
            )
            
            extraction_time = asyncio.get_event_loop().time() - start_time
            
            return ModelExtractionResult(
                success=extraction_accuracy > 0.8,  # 80% accuracy threshold
                attack_type=ModelExtractionType.QUERY_BASED,
                extraction_accuracy=extraction_accuracy,
                original_accuracy=original_accuracy,
                similarity_score=similarity_score,
                queries_used=len(queries),
                extraction_time=extraction_time,
                metadata={
                    "query_strategy": query_strategy,
                    "max_queries": max_queries,
                    "surrogate_model_type": type(surrogate_model).__name__
                }
            )
            
        except Exception as e:
            logger.error(f"Query-based extraction failed: {e}")
            return ModelExtractionResult(
                success=False,
                attack_type=ModelExtractionType.QUERY_BASED,
                extraction_accuracy=0.0,
                original_accuracy=0.0,
                similarity_score=0.0,
                queries_used=0,
                extraction_time=0.0,
                metadata={"error": str(e)}
            )
    
    async def synthetic_data_extraction(self, 
                                      target_model: Any,
                                      X_test: np.ndarray,
                                      y_test: np.ndarray,
                                      synthetic_samples: int = 1000,
                                      data_generation_method: str = "gaussian") -> ModelExtractionResult:
        """
        Synthetic data extraction
        Generate synthetic data to extract model functionality
        """
        try:
            logger.info(f"Starting synthetic data extraction with {synthetic_samples} samples")
            
            start_time = asyncio.get_event_loop().time()
            
            # Generate synthetic data
            if data_generation_method == "gaussian":
                synthetic_X = np.random.normal(0, 1, (synthetic_samples, X_test.shape[1]))
            elif data_generation_method == "uniform":
                synthetic_X = np.random.uniform(-1, 1, (synthetic_samples, X_test.shape[1]))
            elif data_generation_method == "bootstrap":
                # Bootstrap from test data
                indices = np.random.choice(len(X_test), synthetic_samples, replace=True)
                synthetic_X = X_test[indices]
            else:
                synthetic_X = np.random.normal(0, 1, (synthetic_samples, X_test.shape[1]))
            
            # Query target model with synthetic data
            synthetic_y = target_model.predict(synthetic_X)
            
            # Train surrogate model
            surrogate_model = self._create_surrogate_model(target_model)
            surrogate_model.fit(synthetic_X, synthetic_y)
            
            # Evaluate extraction accuracy
            surrogate_pred = surrogate_model.predict(X_test)
            extraction_accuracy = accuracy_score(y_test, surrogate_pred)
            
            # Get original model accuracy
            original_pred = target_model.predict(X_test)
            original_accuracy = accuracy_score(y_test, original_pred)
            
            # Calculate similarity score
            similarity_score = self._calculate_model_similarity(
                target_model, surrogate_model, X_test
            )
            
            extraction_time = asyncio.get_event_loop().time() - start_time
            
            return ModelExtractionResult(
                success=extraction_accuracy > 0.8,
                attack_type=ModelExtractionType.SYNTHETIC_DATA,
                extraction_accuracy=extraction_accuracy,
                original_accuracy=original_accuracy,
                similarity_score=similarity_score,
                queries_used=synthetic_samples,
                extraction_time=extraction_time,
                metadata={
                    "data_generation_method": data_generation_method,
                    "synthetic_samples": synthetic_samples,
                    "surrogate_model_type": type(surrogate_model).__name__
                }
            )
            
        except Exception as e:
            logger.error(f"Synthetic data extraction failed: {e}")
            return ModelExtractionResult(
                success=False,
                attack_type=ModelExtractionType.SYNTHETIC_DATA,
                extraction_accuracy=0.0,
                original_accuracy=0.0,
                similarity_score=0.0,
                queries_used=0,
                extraction_time=0.0,
                metadata={"error": str(e)}
            )
    
    async def transfer_learning_extraction(self, 
                                         target_model: Any,
                                         X_test: np.ndarray,
                                         y_test: np.ndarray,
                                         source_data: np.ndarray,
                                         source_labels: np.ndarray,
                                         transfer_method: str = "fine_tuning") -> ModelExtractionResult:
        """
        Transfer learning extraction
        Use transfer learning to extract model functionality
        """
        try:
            logger.info(f"Starting transfer learning extraction with method {transfer_method}")
            
            start_time = asyncio.get_event_loop().time()
            
            # Query target model with source data
            target_predictions = target_model.predict(source_data)
            
            # Create surrogate model
            surrogate_model = self._create_surrogate_model(target_model)
            
            if transfer_method == "fine_tuning":
                # Fine-tune on target predictions
                surrogate_model.fit(source_data, target_predictions)
            elif transfer_method == "feature_extraction":
                # Extract features and train classifier
                # This is a simplified implementation
                surrogate_model.fit(source_data, target_predictions)
            else:
                # Default to fine-tuning
                surrogate_model.fit(source_data, target_predictions)
            
            # Evaluate extraction accuracy
            surrogate_pred = surrogate_model.predict(X_test)
            extraction_accuracy = accuracy_score(y_test, surrogate_pred)
            
            # Get original model accuracy
            original_pred = target_model.predict(X_test)
            original_accuracy = accuracy_score(y_test, original_pred)
            
            # Calculate similarity score
            similarity_score = self._calculate_model_similarity(
                target_model, surrogate_model, X_test
            )
            
            extraction_time = asyncio.get_event_loop().time() - start_time
            
            return ModelExtractionResult(
                success=extraction_accuracy > 0.8,
                attack_type=ModelExtractionType.TRANSFER_LEARNING,
                extraction_accuracy=extraction_accuracy,
                original_accuracy=original_accuracy,
                similarity_score=similarity_score,
                queries_used=len(source_data),
                extraction_time=extraction_time,
                metadata={
                    "transfer_method": transfer_method,
                    "source_samples": len(source_data),
                    "surrogate_model_type": type(surrogate_model).__name__
                }
            )
            
        except Exception as e:
            logger.error(f"Transfer learning extraction failed: {e}")
            return ModelExtractionResult(
                success=False,
                attack_type=ModelExtractionType.TRANSFER_LEARNING,
                extraction_accuracy=0.0,
                original_accuracy=0.0,
                similarity_score=0.0,
                queries_used=0,
                extraction_time=0.0,
                metadata={"error": str(e)}
            )
    
    async def adversarial_queries_extraction(self, 
                                           target_model: Any,
                                           X_test: np.ndarray,
                                           y_test: np.ndarray,
                                           max_queries: int = 1000,
                                           adversarial_method: str = "fgsm") -> ModelExtractionResult:
        """
        Adversarial queries extraction
        Use adversarial examples to extract model functionality
        """
        try:
            logger.info(f"Starting adversarial queries extraction with {max_queries} queries")
            
            start_time = asyncio.get_event_loop().time()
            
            # Generate adversarial queries
            adversarial_queries = self._generate_adversarial_queries(
                target_model, X_test, max_queries, adversarial_method
            )
            
            # Query target model
            target_predictions = target_model.predict(adversarial_queries)
            
            # Train surrogate model
            surrogate_model = self._create_surrogate_model(target_model)
            surrogate_model.fit(adversarial_queries, target_predictions)
            
            # Evaluate extraction accuracy
            surrogate_pred = surrogate_model.predict(X_test)
            extraction_accuracy = accuracy_score(y_test, surrogate_pred)
            
            # Get original model accuracy
            original_pred = target_model.predict(X_test)
            original_accuracy = accuracy_score(y_test, original_pred)
            
            # Calculate similarity score
            similarity_score = self._calculate_model_similarity(
                target_model, surrogate_model, X_test
            )
            
            extraction_time = asyncio.get_event_loop().time() - start_time
            
            return ModelExtractionResult(
                success=extraction_accuracy > 0.8,
                attack_type=ModelExtractionType.ADVERSARIAL_QUERIES,
                extraction_accuracy=extraction_accuracy,
                original_accuracy=original_accuracy,
                similarity_score=similarity_score,
                queries_used=len(adversarial_queries),
                extraction_time=extraction_time,
                metadata={
                    "adversarial_method": adversarial_method,
                    "max_queries": max_queries,
                    "surrogate_model_type": type(surrogate_model).__name__
                }
            )
            
        except Exception as e:
            logger.error(f"Adversarial queries extraction failed: {e}")
            return ModelExtractionResult(
                success=False,
                attack_type=ModelExtractionType.ADVERSARIAL_QUERIES,
                extraction_accuracy=0.0,
                original_accuracy=0.0,
                similarity_score=0.0,
                queries_used=0,
                extraction_time=0.0,
                metadata={"error": str(e)}
            )
    
    async def active_learning_extraction(self, 
                                       target_model: Any,
                                       X_test: np.ndarray,
                                       y_test: np.ndarray,
                                       max_queries: int = 1000,
                                       uncertainty_method: str = "entropy") -> ModelExtractionResult:
        """
        Active learning extraction
        Use active learning to efficiently extract model functionality
        """
        try:
            logger.info(f"Starting active learning extraction with {max_queries} queries")
            
            start_time = asyncio.get_event_loop().time()
            
            # Initialize with random samples
            initial_samples = min(100, max_queries // 10)
            queries = self._generate_random_queries(X_test, initial_samples)
            target_predictions = target_model.predict(queries)
            
            # Active learning loop
            for i in range(initial_samples, max_queries):
                # Train surrogate model on current data
                surrogate_model = self._create_surrogate_model(target_model)
                surrogate_model.fit(queries, target_predictions)
                
                # Generate candidate queries
                candidate_queries = self._generate_random_queries(X_test, 100)
                
                # Select most uncertain queries
                if uncertainty_method == "entropy":
                    uncertainties = self._calculate_entropy_uncertainty(
                        surrogate_model, candidate_queries
                    )
                elif uncertainty_method == "margin":
                    uncertainties = self._calculate_margin_uncertainty(
                        surrogate_model, candidate_queries
                    )
                else:
                    uncertainties = self._calculate_entropy_uncertainty(
                        surrogate_model, candidate_queries
                    )
                
                # Select most uncertain query
                most_uncertain_idx = np.argmax(uncertainties)
                new_query = candidate_queries[most_uncertain_idx]
                
                # Query target model
                new_prediction = target_model.predict(new_query.reshape(1, -1))[0]
                
                # Add to training data
                queries = np.vstack([queries, new_query])
                target_predictions = np.append(target_predictions, new_prediction)
            
            # Final surrogate model
            surrogate_model = self._create_surrogate_model(target_model)
            surrogate_model.fit(queries, target_predictions)
            
            # Evaluate extraction accuracy
            surrogate_pred = surrogate_model.predict(X_test)
            extraction_accuracy = accuracy_score(y_test, surrogate_pred)
            
            # Get original model accuracy
            original_pred = target_model.predict(X_test)
            original_accuracy = accuracy_score(y_test, original_pred)
            
            # Calculate similarity score
            similarity_score = self._calculate_model_similarity(
                target_model, surrogate_model, X_test
            )
            
            extraction_time = asyncio.get_event_loop().time() - start_time
            
            return ModelExtractionResult(
                success=extraction_accuracy > 0.8,
                attack_type=ModelExtractionType.ACTIVE_LEARNING,
                extraction_accuracy=extraction_accuracy,
                original_accuracy=original_accuracy,
                similarity_score=similarity_score,
                queries_used=len(queries),
                extraction_time=extraction_time,
                metadata={
                    "uncertainty_method": uncertainty_method,
                    "max_queries": max_queries,
                    "surrogate_model_type": type(surrogate_model).__name__
                }
            )
            
        except Exception as e:
            logger.error(f"Active learning extraction failed: {e}")
            return ModelExtractionResult(
                success=False,
                attack_type=ModelExtractionType.ACTIVE_LEARNING,
                extraction_accuracy=0.0,
                original_accuracy=0.0,
                similarity_score=0.0,
                queries_used=0,
                extraction_time=0.0,
                metadata={"error": str(e)}
            )
    
    def _generate_random_queries(self, X_test: np.ndarray, n_queries: int) -> np.ndarray:
        """Generate random queries"""
        # Generate queries based on test data distribution
        mean = np.mean(X_test, axis=0)
        std = np.std(X_test, axis=0)
        queries = np.random.normal(mean, std, (n_queries, X_test.shape[1]))
        return queries
    
    def _generate_boundary_queries(self, X_test: np.ndarray, n_queries: int) -> np.ndarray:
        """Generate queries near decision boundaries"""
        # Simplified boundary search
        queries = []
        for _ in range(n_queries):
            # Sample two points and interpolate
            idx1, idx2 = np.random.choice(len(X_test), 2, replace=False)
            alpha = np.random.random()
            query = alpha * X_test[idx1] + (1 - alpha) * X_test[idx2]
            queries.append(query)
        return np.array(queries)
    
    def _generate_gradient_queries(self, target_model: Any, X_test: np.ndarray, n_queries: int) -> np.ndarray:
        """Generate queries using gradient information"""
        # Simplified gradient-based query generation
        queries = []
        for _ in range(n_queries):
            # Start with random point
            query = np.random.normal(0, 1, X_test.shape[1])
            # Add small random perturbation
            query += np.random.normal(0, 0.1, X_test.shape[1])
            queries.append(query)
        return np.array(queries)
    
    def _generate_synthetic_queries(self, X_test: np.ndarray, n_queries: int) -> np.ndarray:
        """Generate synthetic queries"""
        # Generate queries using various distributions
        queries = []
        for _ in range(n_queries):
            if np.random.random() < 0.5:
                # Gaussian distribution
                query = np.random.normal(0, 1, X_test.shape[1])
            else:
                # Uniform distribution
                query = np.random.uniform(-1, 1, X_test.shape[1])
            queries.append(query)
        return np.array(queries)
    
    def _generate_adversarial_queries(self, 
                                    target_model: Any, 
                                    X_test: np.ndarray, 
                                    n_queries: int,
                                    method: str) -> np.ndarray:
        """Generate adversarial queries"""
        # Simplified adversarial query generation
        queries = []
        for _ in range(n_queries):
            # Start with test sample
            base_sample = X_test[np.random.randint(len(X_test))]
            # Add adversarial perturbation
            if method == "fgsm":
                perturbation = np.random.normal(0, 0.1, base_sample.shape)
            else:
                perturbation = np.random.normal(0, 0.05, base_sample.shape)
            
            query = base_sample + perturbation
            queries.append(query)
        return np.array(queries)
    
    def _create_surrogate_model(self, target_model: Any) -> Any:
        """Create surrogate model based on target model type"""
        # Create similar model to target
        if hasattr(target_model, 'n_estimators'):  # Tree-based model
            return RandomForestClassifier(n_estimators=50, random_state=42)
        elif hasattr(target_model, 'C'):  # SVM
            return SVC(kernel='rbf', random_state=42)
        elif hasattr(target_model, 'max_iter'):  # Logistic regression
            return LogisticRegression(random_state=42)
        else:
            # Default to random forest
            return RandomForestClassifier(n_estimators=50, random_state=42)
    
    def _calculate_model_similarity(self, 
                                  target_model: Any, 
                                  surrogate_model: Any, 
                                  X_test: np.ndarray) -> float:
        """Calculate similarity between target and surrogate models"""
        try:
            # Get predictions from both models
            target_pred = target_model.predict(X_test)
            surrogate_pred = surrogate_model.predict(X_test)
            
            # Calculate agreement rate
            agreement_rate = np.mean(target_pred == surrogate_pred)
            return agreement_rate
            
        except Exception as e:
            logger.warning(f"Failed to calculate model similarity: {e}")
            return 0.0
    
    def _calculate_entropy_uncertainty(self, model: Any, queries: np.ndarray) -> np.ndarray:
        """Calculate entropy-based uncertainty"""
        try:
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(queries)
                # Calculate entropy
                entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                return entropy
            else:
                # Fallback to random uncertainty
                return np.random.random(len(queries))
        except Exception as e:
            logger.warning(f"Failed to calculate entropy uncertainty: {e}")
            return np.random.random(len(queries))
    
    def _calculate_margin_uncertainty(self, model: Any, queries: np.ndarray) -> np.ndarray:
        """Calculate margin-based uncertainty"""
        try:
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(queries)
                # Calculate margin (difference between top two probabilities)
                sorted_probs = np.sort(probs, axis=1)
                margin = sorted_probs[:, -1] - sorted_probs[:, -2]
                return 1 - margin  # Lower margin = higher uncertainty
            else:
                # Fallback to random uncertainty
                return np.random.random(len(queries))
        except Exception as e:
            logger.warning(f"Failed to calculate margin uncertainty: {e}")
            return np.random.random(len(queries))
    
    async def run_comprehensive_extraction_attacks(self, 
                                                 target_model: Any,
                                                 X_test: np.ndarray,
                                                 y_test: np.ndarray) -> Dict[str, ModelExtractionResult]:
        """Run comprehensive model extraction attacks"""
        results = {}
        
        # Run all attack types
        attack_methods = [
            ("query_based", self.query_based_extraction),
            ("synthetic_data", self.synthetic_data_extraction),
            ("adversarial_queries", self.adversarial_queries_extraction),
            ("active_learning", self.active_learning_extraction)
        ]
        
        for attack_name, attack_method in attack_methods:
            try:
                result = await attack_method(target_model, X_test, y_test)
                results[attack_name] = result
                
            except Exception as e:
                logger.error(f"Model extraction attack {attack_name} failed: {e}")
                results[attack_name] = ModelExtractionResult(
                    success=False,
                    attack_type=ModelExtractionType.QUERY_BASED,
                    extraction_accuracy=0.0,
                    original_accuracy=0.0,
                    similarity_score=0.0,
                    queries_used=0,
                    extraction_time=0.0,
                    metadata={"error": str(e)}
                )
        
        return results
