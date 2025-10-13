"""
Anomaly Detection
Implements behavioral anomaly detection for unusual model responses
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class AnomalyMethod(Enum):
    """Anomaly detection methods"""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    DBSCAN = "dbscan"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    MAHALANOBIS_DISTANCE = "mahalanobis_distance"
    AUTOENCODER = "autoencoder"
    STATISTICAL = "statistical"


class AnomalyType(Enum):
    """Types of anomalies"""
    POINT_ANOMALY = "point_anomaly"
    CONTEXTUAL_ANOMALY = "contextual_anomaly"
    COLLECTIVE_ANOMALY = "collective_anomaly"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"


@dataclass
class AnomalyData:
    """Anomaly data structure"""
    input_data: np.ndarray
    anomaly_scores: np.ndarray
    anomaly_labels: np.ndarray
    anomaly_method: AnomalyMethod
    anomaly_type: AnomalyType
    threshold: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnomalyAnalysis:
    """Anomaly analysis results"""
    anomaly_method: AnomalyMethod
    anomaly_type: AnomalyType
    n_anomalies: int
    anomaly_ratio: float
    anomaly_severity: Dict[str, float]
    anomaly_patterns: List[str]
    insights: List[str]
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AnomalyDetector:
    """
    Anomaly Detector
    Implements behavioral anomaly detection for unusual model responses
    """
    
    def __init__(self):
        """Initialize anomaly detector"""
        self.anomaly_data: List[AnomalyData] = []
        self.analysis_results: List[AnomalyAnalysis] = []
        self.detection_models: Dict[str, Any] = {}
        
        logger.info("✅ Initialized Anomaly Detector")
    
    async def detect_anomalies(self, 
                             data: np.ndarray,
                             methods: List[AnomalyMethod] = None,
                             contamination: float = 0.1) -> List[AnomalyData]:
        """
        Detect anomalies in data using specified methods
        """
        try:
            logger.info(f"Detecting anomalies in data shape: {data.shape}")
            
            if methods is None:
                methods = [AnomalyMethod.ISOLATION_FOREST, AnomalyMethod.ONE_CLASS_SVM, AnomalyMethod.STATISTICAL]
            
            anomaly_results = []
            
            for method in methods:
                try:
                    anomaly_data = await self._detect_anomalies_method(data, method, contamination)
                    if anomaly_data:
                        anomaly_results.append(anomaly_data)
                        self.anomaly_data.append(anomaly_data)
                except Exception as e:
                    logger.warning(f"Anomaly detection failed for {method.value}: {e}")
                    continue
            
            logger.info(f"Detected anomalies using {len(anomaly_results)} methods")
            return anomaly_results
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    async def _detect_anomalies_method(self, 
                                     data: np.ndarray, 
                                     method: AnomalyMethod, 
                                     contamination: float) -> Optional[AnomalyData]:
        """Detect anomalies using specific method"""
        try:
            if method == AnomalyMethod.ISOLATION_FOREST:
                return await self._detect_isolation_forest(data, contamination)
            elif method == AnomalyMethod.ONE_CLASS_SVM:
                return await self._detect_one_class_svm(data, contamination)
            elif method == AnomalyMethod.DBSCAN:
                return await self._detect_dbscan(data)
            elif method == AnomalyMethod.LOCAL_OUTLIER_FACTOR:
                return await self._detect_local_outlier_factor(data, contamination)
            elif method == AnomalyMethod.MAHALANOBIS_DISTANCE:
                return await self._detect_mahalanobis_distance(data, contamination)
            elif method == AnomalyMethod.AUTOENCODER:
                return await self._detect_autoencoder(data, contamination)
            elif method == AnomalyMethod.STATISTICAL:
                return await self._detect_statistical(data, contamination)
            else:
                logger.warning(f"Unknown anomaly detection method: {method}")
                return None
                
        except Exception as e:
            logger.error(f"Anomaly detection failed for {method.value}: {e}")
            return None
    
    async def _detect_isolation_forest(self, data: np.ndarray, contamination: float) -> AnomalyData:
        """Detect anomalies using Isolation Forest"""
        try:
            # Flatten data if needed
            if len(data.shape) > 2:
                data_flat = data.reshape(data.shape[0], -1)
            else:
                data_flat = data
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = iso_forest.fit_predict(data_flat)
            anomaly_scores = iso_forest.decision_function(data_flat)
            
            # Convert labels: -1 for anomalies, 1 for normal
            anomaly_labels = (anomaly_labels == -1).astype(int)
            
            # Determine threshold
            threshold = np.percentile(anomaly_scores, contamination * 100)
            
            return AnomalyData(
                input_data=data,
                anomaly_scores=anomaly_scores,
                anomaly_labels=anomaly_labels,
                anomaly_method=AnomalyMethod.ISOLATION_FOREST,
                anomaly_type=AnomalyType.POINT_ANOMALY,
                threshold=threshold,
                metadata={
                    "contamination": contamination,
                    "n_estimators": iso_forest.n_estimators,
                    "detected_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Isolation Forest detection failed: {e}")
            raise
    
    async def _detect_one_class_svm(self, data: np.ndarray, contamination: float) -> AnomalyData:
        """Detect anomalies using One-Class SVM"""
        try:
            # Flatten data if needed
            if len(data.shape) > 2:
                data_flat = data.reshape(data.shape[0], -1)
            else:
                data_flat = data
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_flat)
            
            # Fit One-Class SVM
            nu = contamination  # Fraction of outliers
            oc_svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
            anomaly_labels = oc_svm.fit_predict(data_scaled)
            anomaly_scores = oc_svm.decision_function(data_scaled)
            
            # Convert labels: -1 for anomalies, 1 for normal
            anomaly_labels = (anomaly_labels == -1).astype(int)
            
            # Determine threshold
            threshold = np.percentile(anomaly_scores, contamination * 100)
            
            return AnomalyData(
                input_data=data,
                anomaly_scores=anomaly_scores,
                anomaly_labels=anomaly_labels,
                anomaly_method=AnomalyMethod.ONE_CLASS_SVM,
                anomaly_type=AnomalyType.POINT_ANOMALY,
                threshold=threshold,
                metadata={
                    "contamination": contamination,
                    "nu": nu,
                    "kernel": oc_svm.kernel,
                    "detected_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"One-Class SVM detection failed: {e}")
            raise
    
    async def _detect_dbscan(self, data: np.ndarray) -> AnomalyData:
        """Detect anomalies using DBSCAN"""
        try:
            # Flatten data if needed
            if len(data.shape) > 2:
                data_flat = data.reshape(data.shape[0], -1)
            else:
                data_flat = data
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_flat)
            
            # Fit DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(data_scaled)
            
            # Anomalies are points with label -1
            anomaly_labels = (cluster_labels == -1).astype(int)
            
            # Calculate anomaly scores based on distance to nearest cluster
            anomaly_scores = np.zeros(len(data_flat))
            
            for i, label in enumerate(cluster_labels):
                if label == -1:  # Anomaly
                    # Distance to nearest non-anomaly point
                    non_anomaly_mask = cluster_labels != -1
                    if np.any(non_anomaly_mask):
                        distances = np.linalg.norm(data_scaled[i] - data_scaled[non_anomaly_mask], axis=1)
                        anomaly_scores[i] = -np.min(distances)  # Negative for anomalies
                    else:
                        anomaly_scores[i] = -1.0
                else:
                    # Distance to cluster center
                    cluster_mask = cluster_labels == label
                    if np.sum(cluster_mask) > 1:
                        cluster_center = np.mean(data_scaled[cluster_mask], axis=0)
                        anomaly_scores[i] = np.linalg.norm(data_scaled[i] - cluster_center)
                    else:
                        anomaly_scores[i] = 0.0
            
            # Determine threshold
            threshold = np.percentile(anomaly_scores, 90)  # Top 10% as anomalies
            
            return AnomalyData(
                input_data=data,
                anomaly_scores=anomaly_scores,
                anomaly_labels=anomaly_labels,
                anomaly_method=AnomalyMethod.DBSCAN,
                anomaly_type=AnomalyType.COLLECTIVE_ANOMALY,
                threshold=threshold,
                metadata={
                    "eps": dbscan.eps,
                    "min_samples": dbscan.min_samples,
                    "n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                    "detected_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"DBSCAN detection failed: {e}")
            raise
    
    async def _detect_local_outlier_factor(self, data: np.ndarray, contamination: float) -> AnomalyData:
        """Detect anomalies using Local Outlier Factor"""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            # Flatten data if needed
            if len(data.shape) > 2:
                data_flat = data.reshape(data.shape[0], -1)
            else:
                data_flat = data
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_flat)
            
            # Fit Local Outlier Factor
            n_neighbors = min(20, len(data_scaled) - 1)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            anomaly_labels = lof.fit_predict(data_scaled)
            anomaly_scores = lof.negative_outlier_factor_
            
            # Convert labels: -1 for anomalies, 1 for normal
            anomaly_labels = (anomaly_labels == -1).astype(int)
            
            # Determine threshold
            threshold = np.percentile(anomaly_scores, contamination * 100)
            
            return AnomalyData(
                input_data=data,
                anomaly_scores=anomaly_scores,
                anomaly_labels=anomaly_labels,
                anomaly_method=AnomalyMethod.LOCAL_OUTLIER_FACTOR,
                anomaly_type=AnomalyType.CONTEXTUAL_ANOMALY,
                threshold=threshold,
                metadata={
                    "contamination": contamination,
                    "n_neighbors": n_neighbors,
                    "detected_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Local Outlier Factor detection failed: {e}")
            raise
    
    async def _detect_mahalanobis_distance(self, data: np.ndarray, contamination: float) -> AnomalyData:
        """Detect anomalies using Mahalanobis distance"""
        try:
            # Flatten data if needed
            if len(data.shape) > 2:
                data_flat = data.reshape(data.shape[0], -1)
            else:
                data_flat = data
            
            # Calculate mean and covariance
            mean = np.mean(data_flat, axis=0)
            cov = np.cov(data_flat.T)
            
            # Add small value to diagonal for numerical stability
            cov += np.eye(cov.shape[0]) * 1e-6
            
            # Calculate Mahalanobis distances
            try:
                inv_cov = np.linalg.inv(cov)
                diff = data_flat - mean
                mahalanobis_distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
            except np.linalg.LinAlgError:
                # Fallback to Euclidean distance if covariance is singular
                mahalanobis_distances = np.linalg.norm(data_flat - mean, axis=1)
            
            # Determine threshold
            threshold = np.percentile(mahalanobis_distances, (1 - contamination) * 100)
            
            # Anomaly labels
            anomaly_labels = (mahalanobis_distances > threshold).astype(int)
            
            # Anomaly scores (negative for anomalies)
            anomaly_scores = -mahalanobis_distances
            
            return AnomalyData(
                input_data=data,
                anomaly_scores=anomaly_scores,
                anomaly_labels=anomaly_labels,
                anomaly_method=AnomalyMethod.MAHALANOBIS_DISTANCE,
                anomaly_type=AnomalyType.POINT_ANOMALY,
                threshold=threshold,
                metadata={
                    "contamination": contamination,
                    "mean": mean.tolist(),
                    "cov_shape": cov.shape,
                    "detected_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Mahalanobis distance detection failed: {e}")
            raise
    
    async def _detect_autoencoder(self, data: np.ndarray, contamination: float) -> AnomalyData:
        """Detect anomalies using Autoencoder"""
        try:
            # Flatten data if needed
            if len(data.shape) > 2:
                data_flat = data.reshape(data.shape[0], -1)
            else:
                data_flat = data
            
            # Standardize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_flat)
            
            # Simple autoencoder architecture
            input_dim = data_scaled.shape[1]
            encoding_dim = max(1, input_dim // 4)
            
            # Create autoencoder
            class SimpleAutoencoder(nn.Module):
                def __init__(self, input_dim, encoding_dim):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, encoding_dim * 2),
                        nn.ReLU(),
                        nn.Linear(encoding_dim * 2, encoding_dim)
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(encoding_dim, encoding_dim * 2),
                        nn.ReLU(),
                        nn.Linear(encoding_dim * 2, input_dim)
                    )
                
                def forward(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return decoded
            
            # Train autoencoder
            autoencoder = SimpleAutoencoder(input_dim, encoding_dim)
            optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Convert to tensor
            data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
            
            # Training loop
            n_epochs = 100
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                reconstructed = autoencoder(data_tensor)
                loss = criterion(reconstructed, data_tensor)
                loss.backward()
                optimizer.step()
            
            # Calculate reconstruction errors
            with torch.no_grad():
                reconstructed = autoencoder(data_tensor)
                reconstruction_errors = torch.mean((data_tensor - reconstructed) ** 2, dim=1).numpy()
            
            # Determine threshold
            threshold = np.percentile(reconstruction_errors, (1 - contamination) * 100)
            
            # Anomaly labels
            anomaly_labels = (reconstruction_errors > threshold).astype(int)
            
            # Anomaly scores (negative for anomalies)
            anomaly_scores = -reconstruction_errors
            
            return AnomalyData(
                input_data=data,
                anomaly_scores=anomaly_scores,
                anomaly_labels=anomaly_labels,
                anomaly_method=AnomalyMethod.AUTOENCODER,
                anomaly_type=AnomalyType.BEHAVIORAL_ANOMALY,
                threshold=threshold,
                metadata={
                    "contamination": contamination,
                    "input_dim": input_dim,
                    "encoding_dim": encoding_dim,
                    "n_epochs": n_epochs,
                    "detected_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Autoencoder detection failed: {e}")
            raise
    
    async def _detect_statistical(self, data: np.ndarray, contamination: float) -> AnomalyData:
        """Detect anomalies using statistical methods"""
        try:
            # Flatten data if needed
            if len(data.shape) > 2:
                data_flat = data.reshape(data.shape[0], -1)
            else:
                data_flat = data
            
            # Calculate statistical measures
            mean = np.mean(data_flat, axis=0)
            std = np.std(data_flat, axis=0)
            
            # Z-score based anomaly detection
            z_scores = np.abs((data_flat - mean) / (std + 1e-8))
            max_z_scores = np.max(z_scores, axis=1)
            
            # Determine threshold (e.g., 3 standard deviations)
            threshold = 3.0
            
            # Anomaly labels
            anomaly_labels = (max_z_scores > threshold).astype(int)
            
            # Anomaly scores
            anomaly_scores = -max_z_scores
            
            return AnomalyData(
                input_data=data,
                anomaly_scores=anomaly_scores,
                anomaly_labels=anomaly_labels,
                anomaly_method=AnomalyMethod.STATISTICAL,
                anomaly_type=AnomalyType.POINT_ANOMALY,
                threshold=threshold,
                metadata={
                    "contamination": contamination,
                    "threshold_z_score": threshold,
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "detected_at": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Statistical detection failed: {e}")
            raise
    
    async def analyze_anomalies(self, anomaly_data: List[AnomalyData]) -> List[AnomalyAnalysis]:
        """
        Analyze anomaly detection results
        """
        try:
            logger.info(f"Analyzing {len(anomaly_data)} anomaly detection results")
            
            analysis_results = []
            
            for anomaly in anomaly_data:
                analysis = await self._analyze_single_anomaly(anomaly)
                if analysis:
                    analysis_results.append(analysis)
                    self.analysis_results.append(analysis)
            
            logger.info(f"Completed anomaly analysis: {len(analysis_results)} results")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Anomaly analysis failed: {e}")
            return []
    
    async def _analyze_single_anomaly(self, anomaly: AnomalyData) -> Optional[AnomalyAnalysis]:
        """Analyze single anomaly detection result"""
        try:
            # Calculate basic statistics
            n_anomalies = np.sum(anomaly.anomaly_labels)
            anomaly_ratio = n_anomalies / len(anomaly.anomaly_labels)
            
            # Calculate anomaly severity
            anomaly_scores = anomaly.anomaly_scores
            anomaly_severity = {
                "mean_score": float(np.mean(anomaly_scores)),
                "std_score": float(np.std(anomaly_scores)),
                "min_score": float(np.min(anomaly_scores)),
                "max_score": float(np.max(anomaly_scores)),
                "score_range": float(np.max(anomaly_scores) - np.min(anomaly_scores))
            }
            
            # Generate patterns and insights
            patterns = self._generate_anomaly_patterns(anomaly)
            insights = self._generate_anomaly_insights(anomaly, anomaly_ratio, anomaly_severity)
            
            # Calculate confidence
            confidence = self._calculate_anomaly_confidence(anomaly, anomaly_ratio)
            
            return AnomalyAnalysis(
                anomaly_method=anomaly.anomaly_method,
                anomaly_type=anomaly.anomaly_type,
                n_anomalies=int(n_anomalies),
                anomaly_ratio=float(anomaly_ratio),
                anomaly_severity=anomaly_severity,
                anomaly_patterns=patterns,
                insights=insights,
                confidence=confidence,
                metadata={
                    "input_shape": anomaly.input_data.shape,
                    "threshold": anomaly.threshold,
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Single anomaly analysis failed: {e}")
            return None
    
    def _generate_anomaly_patterns(self, anomaly: AnomalyData) -> List[str]:
        """Generate anomaly patterns"""
        try:
            patterns = []
            
            # Analyze anomaly distribution
            anomaly_labels = anomaly.anomaly_labels
            anomaly_scores = anomaly.anomaly_scores
            
            # Check for clustering of anomalies
            anomaly_indices = np.where(anomaly_labels == 1)[0]
            if len(anomaly_indices) > 1:
                # Check if anomalies are clustered
                consecutive_anomalies = np.sum(np.diff(anomaly_indices) == 1)
                if consecutive_anomalies > len(anomaly_indices) * 0.3:
                    patterns.append("Clustered anomalies detected")
                else:
                    patterns.append("Scattered anomalies detected")
            
            # Check for extreme scores
            if np.max(anomaly_scores) > np.mean(anomaly_scores) + 3 * np.std(anomaly_scores):
                patterns.append("Extreme anomaly scores detected")
            
            # Check for uniform distribution
            if np.std(anomaly_scores) < 0.1:
                patterns.append("Uniform anomaly scores")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Anomaly pattern generation failed: {e}")
            return []
    
    def _generate_anomaly_insights(self, 
                                 anomaly: AnomalyData, 
                                 anomaly_ratio: float, 
                                 anomaly_severity: Dict[str, float]) -> List[str]:
        """Generate insights from anomaly analysis"""
        try:
            insights = []
            
            # Analyze anomaly ratio
            if anomaly_ratio > 0.2:
                insights.append("High anomaly ratio - system may be under attack")
            elif anomaly_ratio < 0.01:
                insights.append("Very low anomaly ratio - system appears normal")
            else:
                insights.append("Moderate anomaly ratio - some unusual behavior detected")
            
            # Analyze anomaly severity
            if anomaly_severity["max_score"] > anomaly_severity["mean_score"] + 2 * anomaly_severity["std_score"]:
                insights.append("Some anomalies are extremely severe")
            
            if anomaly_severity["score_range"] < 0.1:
                insights.append("Anomaly scores are very similar")
            
            # Analyze method-specific insights
            if anomaly.anomaly_method == AnomalyMethod.ISOLATION_FOREST:
                insights.append("Isolation Forest: Anomalies are isolated from normal data")
            elif anomaly.anomaly_method == AnomalyMethod.ONE_CLASS_SVM:
                insights.append("One-Class SVM: Anomalies are outside the learned boundary")
            elif anomaly.anomaly_method == AnomalyMethod.DBSCAN:
                insights.append("DBSCAN: Anomalies are not part of any cluster")
            elif anomaly.anomaly_method == AnomalyMethod.AUTOENCODER:
                insights.append("Autoencoder: Anomalies have high reconstruction error")
            elif anomaly.anomaly_method == AnomalyMethod.STATISTICAL:
                insights.append("Statistical: Anomalies are statistical outliers")
            
            return insights
            
        except Exception as e:
            logger.error(f"Anomaly insight generation failed: {e}")
            return []
    
    def _calculate_anomaly_confidence(self, anomaly: AnomalyData, anomaly_ratio: float) -> float:
        """Calculate confidence in anomaly detection"""
        try:
            # Base confidence from method reliability
            method_confidence = {
                AnomalyMethod.ISOLATION_FOREST: 0.8,
                AnomalyMethod.ONE_CLASS_SVM: 0.7,
                AnomalyMethod.DBSCAN: 0.6,
                AnomalyMethod.LOCAL_OUTLIER_FACTOR: 0.7,
                AnomalyMethod.MAHALANOBIS_DISTANCE: 0.8,
                AnomalyMethod.AUTOENCODER: 0.6,
                AnomalyMethod.STATISTICAL: 0.5
            }.get(anomaly.anomaly_method, 0.5)
            
            # Adjust based on anomaly ratio
            if 0.05 <= anomaly_ratio <= 0.15:  # Reasonable anomaly ratio
                ratio_adjustment = 1.0
            elif anomaly_ratio < 0.05:  # Very low anomaly ratio
                ratio_adjustment = 0.8
            elif anomaly_ratio > 0.15:  # High anomaly ratio
                ratio_adjustment = 0.9
            else:
                ratio_adjustment = 0.7
            
            # Adjust based on score distribution
            scores = anomaly.anomaly_scores
            if np.std(scores) > 0.1:  # Good score separation
                score_adjustment = 1.0
            else:  # Poor score separation
                score_adjustment = 0.8
            
            confidence = method_confidence * ratio_adjustment * score_adjustment
            
            return float(min(confidence, 1.0))
            
        except Exception as e:
            logger.error(f"Anomaly confidence calculation failed: {e}")
            return 0.5
    
    async def compare_anomaly_methods(self, anomaly_data: List[AnomalyData]) -> Dict[str, Any]:
        """Compare different anomaly detection methods"""
        try:
            if len(anomaly_data) < 2:
                return {}
            
            # Group by method
            methods = defaultdict(list)
            for anomaly in anomaly_data:
                methods[anomaly.anomaly_method.value].append(anomaly)
            
            # Compute comparison metrics
            comparison = {
                "methods_compared": list(methods.keys()),
                "method_statistics": {},
                "agreement_analysis": {},
                "performance_comparison": {}
            }
            
            # Compute statistics for each method
            for method, anomalies in methods.items():
                if not anomalies:
                    continue
                
                # Combine results
                all_labels = np.concatenate([a.anomaly_labels for a in anomalies])
                all_scores = np.concatenate([a.anomaly_scores for a in anomalies])
                
                comparison["method_statistics"][method] = {
                    "n_anomalies": int(np.sum(all_labels)),
                    "anomaly_ratio": float(np.mean(all_labels)),
                    "mean_score": float(np.mean(all_scores)),
                    "std_score": float(np.std(all_scores)),
                    "count": len(anomalies)
                }
            
            # Compute agreement between methods
            if len(anomaly_data) >= 2:
                # Compare first two methods
                method1 = anomaly_data[0]
                method2 = anomaly_data[1]
                
                # Compute agreement
                agreement = np.mean(method1.anomaly_labels == method2.anomaly_labels)
                
                # Compute correlation
                correlation = np.corrcoef(method1.anomaly_scores, method2.anomaly_scores)[0, 1]
                
                comparison["agreement_analysis"] = {
                    "agreement": float(agreement),
                    "correlation": float(correlation)
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Anomaly method comparison failed: {e}")
            return {}
    
    async def export_anomaly_data(self, format: str = "json") -> str:
        """Export anomaly analysis data"""
        try:
            if format.lower() == "json":
                data = {
                    "anomaly_data": [a.__dict__ for a in self.anomaly_data],
                    "analysis_results": [r.__dict__ for r in self.analysis_results],
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Anomaly data export failed: {e}")
            return ""
