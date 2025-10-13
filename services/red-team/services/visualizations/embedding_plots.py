"""
Embedding Plotter
Creates visualizations of text embeddings, attack patterns, and model behavior in high-dimensional spaces.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    from sklearn.manifold import TSNE, UMAP
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    # Fallback for environments without required packages
    class plt:
        @staticmethod
        def figure(*args, **kwargs): return None
        @staticmethod
        def subplot(*args, **kwargs): return None
        @staticmethod
        def scatter(*args, **kwargs): pass
        @staticmethod
        def plot(*args, **kwargs): pass
        @staticmethod
        def show(*args, **kwargs): pass
        @staticmethod
        def savefig(*args, **kwargs): pass
        @staticmethod
        def close(*args, **kwargs): pass
        @staticmethod
        def colorbar(*args, **kwargs): pass
        @staticmethod
        def title(*args, **kwargs): pass
        @staticmethod
        def xlabel(*args, **kwargs): pass
        @staticmethod
        def ylabel(*args, **kwargs): pass
        @staticmethod
        def legend(*args, **kwargs): pass
    class sns:
        @staticmethod
        def scatterplot(*args, **kwargs): pass
        @staticmethod
        def heatmap(*args, **kwargs): pass
    class TSNE:
        def __init__(self, *args, **kwargs): pass
        def fit_transform(self, *args, **kwargs): return np.array([])
    class UMAP:
        def __init__(self, *args, **kwargs): pass
        def fit_transform(self, *args, **kwargs): return np.array([])
    class PCA:
        def __init__(self, *args, **kwargs): pass
        def fit_transform(self, *args, **kwargs): return np.array([])
    class KMeans:
        def __init__(self, *args, **kwargs): pass
        def fit_predict(self, *args, **kwargs): return np.array([])
    class DBSCAN:
        def __init__(self, *args, **kwargs): pass
        def fit_predict(self, *args, **kwargs): return np.array([])

logger = logging.getLogger(__name__)

class ReductionMethod(Enum):
    """Dimensionality reduction methods"""
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"

class PlotType(Enum):
    """Types of plots"""
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    CLUSTER = "cluster"
    SIMILARITY = "similarity"
    TRAJECTORY = "trajectory"

@dataclass
class EmbeddingPoint:
    """Point in embedding space"""
    text: str
    embedding: np.ndarray
    label: str = ""
    category: str = ""
    metadata: Dict[str, Any] = None

@dataclass
class PlotConfig:
    """Plot configuration"""
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 300
    colormap: str = "viridis"
    alpha: float = 0.7
    s: int = 50
    show_legend: bool = True
    show_colorbar: bool = True

class EmbeddingPlotter:
    """Creates visualizations of embeddings and high-dimensional data"""
    
    def __init__(self):
        self.embeddings: List[EmbeddingPoint] = []
        self.reduction_methods = {
            ReductionMethod.PCA: PCA,
            ReductionMethod.TSNE: TSNE,
            ReductionMethod.UMAP: UMAP
        }
    
    def add_embedding(self, text: str, embedding: np.ndarray, 
                     label: str = "", category: str = "", 
                     metadata: Dict[str, Any] = None):
        """Add embedding point"""
        point = EmbeddingPoint(
            text=text,
            embedding=embedding,
            label=label,
            category=category,
            metadata=metadata or {}
        )
        self.embeddings.append(point)
    
    def add_embeddings_batch(self, texts: List[str], embeddings: np.ndarray,
                           labels: List[str] = None, categories: List[str] = None,
                           metadata: List[Dict[str, Any]] = None):
        """Add multiple embeddings at once"""
        if labels is None:
            labels = [""] * len(texts)
        if categories is None:
            categories = [""] * len(texts)
        if metadata is None:
            metadata = [{}] * len(texts)
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            self.add_embedding(
                text=text,
                embedding=embedding,
                label=labels[i],
                category=categories[i],
                metadata=metadata[i]
            )
    
    def reduce_dimensions(self, method: ReductionMethod = ReductionMethod.UMAP,
                         n_components: int = 2, **kwargs) -> np.ndarray:
        """Reduce dimensionality of embeddings"""
        try:
            if not self.embeddings:
                logger.warning("No embeddings to reduce")
                return np.array([])
            
            # Extract embeddings
            embeddings_matrix = np.array([point.embedding for point in self.embeddings])
            
            # Apply reduction method
            if method == ReductionMethod.PCA:
                reducer = PCA(n_components=n_components, **kwargs)
            elif method == ReductionMethod.TSNE:
                reducer = TSNE(n_components=n_components, **kwargs)
            elif method == ReductionMethod.UMAP:
                reducer = UMAP(n_components=n_components, **kwargs)
            else:
                raise ValueError(f"Unsupported reduction method: {method}")
            
            reduced_embeddings = reducer.fit_transform(embeddings_matrix)
            
            # Update embedding points with reduced dimensions
            for i, point in enumerate(self.embeddings):
                point.embedding = reduced_embeddings[i]
            
            return reduced_embeddings
            
        except Exception as e:
            logger.error(f"Failed to reduce dimensions: {e}")
            return np.array([])
    
    def create_scatter_plot(self, config: PlotConfig = None, 
                          color_by: str = "category",
                          size_by: str = None) -> str:
        """Create scatter plot of embeddings"""
        try:
            if not self.embeddings:
                logger.warning("No embeddings to plot")
                return ""
            
            config = config or PlotConfig()
            
            # Extract data
            x = [point.embedding[0] for point in self.embeddings]
            y = [point.embedding[1] for point in self.embeddings]
            
            # Create figure
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Determine colors
            if color_by == "category":
                categories = [point.category for point in self.embeddings]
                unique_categories = list(set(categories))
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
                color_map = {cat: colors[i] for i, cat in enumerate(unique_categories)}
                colors = [color_map[cat] for cat in categories]
            elif color_by == "label":
                labels = [point.label for point in self.embeddings]
                unique_labels = list(set(labels))
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
                colors = [color_map[label] for label in labels]
            else:
                colors = config.colormap
            
            # Determine sizes
            if size_by:
                sizes = [point.metadata.get(size_by, 50) for point in self.embeddings]
            else:
                sizes = config.s
            
            # Create scatter plot
            scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=config.alpha, 
                               cmap=config.colormap if isinstance(colors, str) else None)
            
            # Add labels
            ax.set_title(config.title or "Embedding Scatter Plot")
            ax.set_xlabel(config.xlabel or "Dimension 1")
            ax.set_ylabel(config.ylabel or "Dimension 2")
            
            # Add legend
            if config.show_legend and color_by in ["category", "label"]:
                handles = []
                labels_list = []
                for i, (cat, color) in enumerate(color_map.items()):
                    handles.append(plt.scatter([], [], c=color, s=50))
                    labels_list.append(cat)
                ax.legend(handles, labels_list)
            
            # Add colorbar
            if config.show_colorbar and isinstance(colors, str):
                plt.colorbar(scatter)
            
            # Save plot
            filename = f"embedding_scatter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create scatter plot: {e}")
            return ""
    
    def create_heatmap(self, similarity_matrix: np.ndarray = None,
                      labels: List[str] = None, config: PlotConfig = None) -> str:
        """Create heatmap of similarity matrix"""
        try:
            config = config or PlotConfig()
            
            # Calculate similarity matrix if not provided
            if similarity_matrix is None:
                if not self.embeddings:
                    logger.warning("No embeddings to create heatmap")
                    return ""
                
                embeddings_matrix = np.array([point.embedding for point in self.embeddings])
                similarity_matrix = cosine_similarity(embeddings_matrix)
                labels = [point.text[:20] for point in self.embeddings]
            
            # Create figure
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Create heatmap
            im = ax.imshow(similarity_matrix, cmap=config.colormap, aspect='auto')
            
            # Add labels
            if labels:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_yticklabels(labels)
            
            # Add colorbar
            if config.show_colorbar:
                plt.colorbar(im, ax=ax)
            
            # Add title
            ax.set_title(config.title or "Similarity Heatmap")
            
            # Save plot
            filename = f"similarity_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create heatmap: {e}")
            return ""
    
    def create_cluster_plot(self, n_clusters: int = 5, 
                          method: str = "kmeans", config: PlotConfig = None) -> str:
        """Create cluster visualization"""
        try:
            if not self.embeddings:
                logger.warning("No embeddings to cluster")
                return ""
            
            config = config or PlotConfig()
            
            # Extract embeddings
            embeddings_matrix = np.array([point.embedding for point in self.embeddings])
            
            # Perform clustering
            if method == "kmeans":
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = clusterer.fit_predict(embeddings_matrix)
            elif method == "dbscan":
                clusterer = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = clusterer.fit_predict(embeddings_matrix)
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            # Create figure
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Create scatter plot with cluster colors
            scatter = ax.scatter(embeddings_matrix[:, 0], embeddings_matrix[:, 1],
                               c=cluster_labels, cmap=config.colormap, 
                               alpha=config.alpha, s=config.s)
            
            # Add labels
            ax.set_title(config.title or f"Cluster Plot ({method.upper()})")
            ax.set_xlabel(config.xlabel or "Dimension 1")
            ax.set_ylabel(config.ylabel or "Dimension 2")
            
            # Add colorbar
            if config.show_colorbar:
                plt.colorbar(scatter, label="Cluster")
            
            # Save plot
            filename = f"cluster_plot_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create cluster plot: {e}")
            return ""
    
    def create_trajectory_plot(self, trajectories: List[List[np.ndarray]], 
                             labels: List[str] = None, config: PlotConfig = None) -> str:
        """Create trajectory plot for attack evolution"""
        try:
            config = config or PlotConfig()
            
            # Create figure
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Plot trajectories
            colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
            
            for i, trajectory in enumerate(trajectories):
                if len(trajectory) < 2:
                    continue
                
                # Extract coordinates
                x = [point[0] for point in trajectory]
                y = [point[1] for point in trajectory]
                
                # Plot trajectory
                ax.plot(x, y, color=colors[i], alpha=config.alpha, linewidth=2)
                
                # Mark start and end points
                ax.scatter(x[0], y[0], color=colors[i], s=100, marker='o', 
                          label=f"Start {labels[i] if labels else i}")
                ax.scatter(x[-1], y[-1], color=colors[i], s=100, marker='s',
                          label=f"End {labels[i] if labels else i}")
            
            # Add labels
            ax.set_title(config.title or "Attack Trajectory Plot")
            ax.set_xlabel(config.xlabel or "Dimension 1")
            ax.set_ylabel(config.ylabel or "Dimension 2")
            
            # Add legend
            if config.show_legend:
                ax.legend()
            
            # Save plot
            filename = f"trajectory_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create trajectory plot: {e}")
            return ""
    
    def create_attack_pattern_visualization(self, attack_patterns: List[Dict[str, Any]]) -> str:
        """Create visualization of attack patterns"""
        try:
            # Extract embeddings from attack patterns
            texts = []
            embeddings = []
            labels = []
            categories = []
            
            for pattern in attack_patterns:
                texts.append(pattern.get("text", ""))
                embeddings.append(pattern.get("embedding", np.array([])))
                labels.append(pattern.get("attack_type", "unknown"))
                categories.append(pattern.get("category", "attack"))
            
            # Add to plotter
            self.add_embeddings_batch(texts, np.array(embeddings), labels, categories)
            
            # Reduce dimensions
            self.reduce_dimensions(ReductionMethod.UMAP, n_components=2)
            
            # Create scatter plot
            config = PlotConfig(
                title="Attack Pattern Visualization",
                xlabel="UMAP Dimension 1",
                ylabel="UMAP Dimension 2",
                color_by="category"
            )
            
            return self.create_scatter_plot(config)
            
        except Exception as e:
            logger.error(f"Failed to create attack pattern visualization: {e}")
            return ""
    
    def create_model_behavior_plot(self, model_outputs: List[Dict[str, Any]]) -> str:
        """Create visualization of model behavior"""
        try:
            # Extract model outputs
            texts = []
            embeddings = []
            labels = []
            categories = []
            
            for output in model_outputs:
                texts.append(output.get("input_text", ""))
                embeddings.append(output.get("hidden_states", np.array([])))
                labels.append(output.get("prediction", "unknown"))
                categories.append(output.get("confidence", "high"))
            
            # Add to plotter
            self.add_embeddings_batch(texts, np.array(embeddings), labels, categories)
            
            # Reduce dimensions
            self.reduce_dimensions(ReductionMethod.TSNE, n_components=2)
            
            # Create scatter plot
            config = PlotConfig(
                title="Model Behavior Visualization",
                xlabel="t-SNE Dimension 1",
                ylabel="t-SNE Dimension 2",
                color_by="labels"
            )
            
            return self.create_scatter_plot(config)
            
        except Exception as e:
            logger.error(f"Failed to create model behavior plot: {e}")
            return ""
    
    def calculate_embedding_statistics(self) -> Dict[str, Any]:
        """Calculate statistics of embeddings"""
        try:
            if not self.embeddings:
                return {}
            
            # Extract embeddings
            embeddings_matrix = np.array([point.embedding for point in self.embeddings])
            
            # Calculate statistics
            stats = {
                "total_embeddings": len(self.embeddings),
                "embedding_dimension": embeddings_matrix.shape[1],
                "mean_embedding": np.mean(embeddings_matrix, axis=0).tolist(),
                "std_embedding": np.std(embeddings_matrix, axis=0).tolist(),
                "min_embedding": np.min(embeddings_matrix, axis=0).tolist(),
                "max_embedding": np.max(embeddings_matrix, axis=0).tolist(),
                "categories": list(set(point.category for point in self.embeddings)),
                "labels": list(set(point.label for point in self.embeddings))
            }
            
            # Calculate pairwise similarities
            if len(embeddings_matrix) > 1:
                similarity_matrix = cosine_similarity(embeddings_matrix)
                stats["mean_similarity"] = float(np.mean(similarity_matrix))
                stats["std_similarity"] = float(np.std(similarity_matrix))
                stats["min_similarity"] = float(np.min(similarity_matrix))
                stats["max_similarity"] = float(np.max(similarity_matrix))
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate embedding statistics: {e}")
            return {}
    
    def find_similar_embeddings(self, query_embedding: np.ndarray, 
                              top_k: int = 5) -> List[Tuple[EmbeddingPoint, float]]:
        """Find most similar embeddings to query"""
        try:
            if not self.embeddings:
                return []
            
            # Calculate similarities
            similarities = []
            for point in self.embeddings:
                similarity = cosine_similarity([query_embedding], [point.embedding])[0][0]
                similarities.append((point, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {e}")
            return []
    
    def export_embeddings(self, format_type: str = "json") -> str:
        """Export embeddings data"""
        try:
            data = {
                "embeddings": [
                    {
                        "text": point.text,
                        "embedding": point.embedding.tolist(),
                        "label": point.label,
                        "category": point.category,
                        "metadata": point.metadata
                    }
                    for point in self.embeddings
                ],
                "statistics": self.calculate_embedding_statistics(),
                "exported_at": datetime.now().isoformat()
            }
            
            if format_type == "json":
                import json
                return json.dumps(data, indent=2)
            else:
                return str(data)
                
        except Exception as e:
            logger.error(f"Failed to export embeddings: {e}")
            return "{}"
    
    def clear_embeddings(self):
        """Clear all embeddings"""
        self.embeddings.clear()
        logger.info("Cleared all embeddings")
