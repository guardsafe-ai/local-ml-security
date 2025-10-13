"""
Heatmap Generator
Creates various types of heatmaps for security analysis, attack patterns, and model behavior.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Rectangle
    import pandas as pd
except ImportError:
    # Fallback for environments without required packages
    class plt:
        @staticmethod
        def figure(*args, **kwargs): return None
        @staticmethod
        def subplot(*args, **kwargs): return None
        @staticmethod
        def imshow(*args, **kwargs): pass
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
        def xticks(*args, **kwargs): pass
        @staticmethod
        def yticks(*args, **kwargs): pass
        @staticmethod
        def setp(*args, **kwargs): pass
    class sns:
        @staticmethod
        def heatmap(*args, **kwargs): pass
        @staticmethod
        def clustermap(*args, **kwargs): pass
    class pd:
        @staticmethod
        def DataFrame(*args, **kwargs): return None

logger = logging.getLogger(__name__)

class HeatmapType(Enum):
    """Types of heatmaps"""
    SIMILARITY = "similarity"
    ATTACK_MATRIX = "attack_matrix"
    VULNERABILITY = "vulnerability"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    CORRELATION = "correlation"
    CONFUSION = "confusion"
    RISK = "risk"

class ColorScheme(Enum):
    """Color schemes for heatmaps"""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    RED_BLUE = "RdBu_r"
    BLUE_WHITE_RED = "bwr"
    COOLWARM = "coolwarm"
    SEISMIC = "seismic"
    BINARY = "binary"
    GRAY = "gray"

@dataclass
class HeatmapConfig:
    """Heatmap configuration"""
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 300
    colormap: ColorScheme = ColorScheme.VIRIDIS
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    center: Optional[float] = None
    annotate: bool = True
    fmt: str = ".2f"
    cbar_kws: Dict[str, Any] = None
    xticklabels: bool = True
    yticklabels: bool = True
    rotation: int = 45

class HeatmapGenerator:
    """Generates various types of heatmaps for security analysis"""
    
    def __init__(self):
        self.data_cache: Dict[str, np.ndarray] = {}
        self.color_schemes = {
            ColorScheme.VIRIDIS: "viridis",
            ColorScheme.PLASMA: "plasma",
            ColorScheme.INFERNO: "inferno",
            ColorScheme.MAGMA: "magma",
            ColorScheme.RED_BLUE: "RdBu_r",
            ColorScheme.BLUE_WHITE_RED: "bwr",
            ColorScheme.COOLWARM: "coolwarm",
            ColorScheme.SEISMIC: "seismic",
            ColorScheme.BINARY: "binary",
            ColorScheme.GRAY: "gray"
        }
    
    def create_similarity_heatmap(self, similarity_matrix: np.ndarray,
                                labels: List[str] = None,
                                config: HeatmapConfig = None) -> str:
        """Create similarity heatmap"""
        try:
            config = config or HeatmapConfig()
            
            # Create figure
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Create heatmap
            im = ax.imshow(similarity_matrix, cmap=self.color_schemes[config.colormap],
                          vmin=config.vmin, vmax=config.vmax, aspect='auto')
            
            # Add labels
            if labels:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=config.rotation, ha='right')
                ax.set_yticklabels(labels)
            
            # Add annotations
            if config.annotate:
                for i in range(len(similarity_matrix)):
                    for j in range(len(similarity_matrix[i])):
                        text = ax.text(j, i, f"{similarity_matrix[i, j]:{config.fmt}}",
                                     ha="center", va="center", color="black")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Similarity Score", rotation=270, labelpad=20)
            
            # Add title and labels
            ax.set_title(config.title or "Similarity Heatmap")
            ax.set_xlabel(config.xlabel or "Items")
            ax.set_ylabel(config.ylabel or "Items")
            
            # Save plot
            filename = f"similarity_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create similarity heatmap: {e}")
            return ""
    
    def create_attack_matrix_heatmap(self, attack_data: List[Dict[str, Any]],
                                   config: HeatmapConfig = None) -> str:
        """Create attack matrix heatmap"""
        try:
            config = config or HeatmapConfig()
            
            # Extract attack types and models
            attack_types = list(set(item.get("attack_type", "unknown") for item in attack_data))
            model_types = list(set(item.get("model_type", "unknown") for item in attack_data))
            
            # Create matrix
            matrix = np.zeros((len(attack_types), len(model_types)))
            
            for item in attack_data:
                attack_idx = attack_types.index(item.get("attack_type", "unknown"))
                model_idx = model_types.index(item.get("model_type", "unknown"))
                success_rate = item.get("success_rate", 0.0)
                matrix[attack_idx, model_idx] = success_rate
            
            # Create figure
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Create heatmap
            im = ax.imshow(matrix, cmap=self.color_schemes[config.colormap],
                          vmin=0, vmax=1, aspect='auto')
            
            # Add labels
            ax.set_xticks(range(len(model_types)))
            ax.set_yticks(range(len(attack_types)))
            ax.set_xticklabels(model_types, rotation=config.rotation, ha='right')
            ax.set_yticklabels(attack_types)
            
            # Add annotations
            if config.annotate:
                for i in range(len(attack_types)):
                    for j in range(len(model_types)):
                        text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                                     ha="center", va="center", color="white" if matrix[i, j] > 0.5 else "black")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Success Rate", rotation=270, labelpad=20)
            
            # Add title and labels
            ax.set_title(config.title or "Attack Success Rate Matrix")
            ax.set_xlabel(config.xlabel or "Model Types")
            ax.set_ylabel(config.ylabel or "Attack Types")
            
            # Save plot
            filename = f"attack_matrix_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create attack matrix heatmap: {e}")
            return ""
    
    def create_vulnerability_heatmap(self, vulnerability_data: List[Dict[str, Any]],
                                   config: HeatmapConfig = None) -> str:
        """Create vulnerability heatmap"""
        try:
            config = config or HeatmapConfig()
            
            # Extract vulnerability types and severity levels
            vuln_types = list(set(item.get("vulnerability_type", "unknown") for item in vulnerability_data))
            severity_levels = ["low", "medium", "high", "critical"]
            
            # Create matrix
            matrix = np.zeros((len(vuln_types), len(severity_levels)))
            
            for item in vulnerability_data:
                vuln_idx = vuln_types.index(item.get("vulnerability_type", "unknown"))
                severity = item.get("severity", "medium").lower()
                if severity in severity_levels:
                    severity_idx = severity_levels.index(severity)
                    count = item.get("count", 1)
                    matrix[vuln_idx, severity_idx] += count
            
            # Create figure
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Create heatmap
            im = ax.imshow(matrix, cmap=self.color_schemes[config.colormap],
                          vmin=0, vmax=np.max(matrix), aspect='auto')
            
            # Add labels
            ax.set_xticks(range(len(severity_levels)))
            ax.set_yticks(range(len(vuln_types)))
            ax.set_xticklabels(severity_levels, rotation=config.rotation, ha='right')
            ax.set_yticklabels(vuln_types)
            
            # Add annotations
            if config.annotate:
                for i in range(len(vuln_types)):
                    for j in range(len(severity_levels)):
                        if matrix[i, j] > 0:
                            text = ax.text(j, i, f"{int(matrix[i, j])}",
                                         ha="center", va="center", color="white" if matrix[i, j] > np.max(matrix)/2 else "black")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Count", rotation=270, labelpad=20)
            
            # Add title and labels
            ax.set_title(config.title or "Vulnerability Distribution")
            ax.set_xlabel(config.xlabel or "Severity Level")
            ax.set_ylabel(config.ylabel or "Vulnerability Type")
            
            # Save plot
            filename = f"vulnerability_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create vulnerability heatmap: {e}")
            return ""
    
    def create_compliance_heatmap(self, compliance_data: List[Dict[str, Any]],
                                config: HeatmapConfig = None) -> str:
        """Create compliance heatmap"""
        try:
            config = config or HeatmapConfig()
            
            # Extract frameworks and controls
            frameworks = list(set(item.get("framework", "unknown") for item in compliance_data))
            controls = list(set(item.get("control", "unknown") for item in compliance_data))
            
            # Create matrix
            matrix = np.zeros((len(frameworks), len(controls)))
            
            for item in compliance_data:
                framework_idx = frameworks.index(item.get("framework", "unknown"))
                control_idx = controls.index(item.get("control", "unknown"))
                compliance_score = item.get("compliance_score", 0.0)
                matrix[framework_idx, control_idx] = compliance_score
            
            # Create figure
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Create heatmap
            im = ax.imshow(matrix, cmap=self.color_schemes[config.colormap],
                          vmin=0, vmax=1, aspect='auto')
            
            # Add labels
            ax.set_xticks(range(len(controls)))
            ax.set_yticks(range(len(frameworks)))
            ax.set_xticklabels(controls, rotation=config.rotation, ha='right')
            ax.set_yticklabels(frameworks)
            
            # Add annotations
            if config.annotate:
                for i in range(len(frameworks)):
                    for j in range(len(controls)):
                        text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                                     ha="center", va="center", color="white" if matrix[i, j] > 0.5 else "black")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Compliance Score", rotation=270, labelpad=20)
            
            # Add title and labels
            ax.set_title(config.title or "Compliance Matrix")
            ax.set_xlabel(config.xlabel or "Controls")
            ax.set_ylabel(config.ylabel or "Frameworks")
            
            # Save plot
            filename = f"compliance_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create compliance heatmap: {e}")
            return ""
    
    def create_performance_heatmap(self, performance_data: List[Dict[str, Any]],
                                 config: HeatmapConfig = None) -> str:
        """Create performance heatmap"""
        try:
            config = config or HeatmapConfig()
            
            # Extract metrics and models
            metrics = list(set(item.get("metric", "unknown") for item in performance_data))
            models = list(set(item.get("model", "unknown") for item in performance_data))
            
            # Create matrix
            matrix = np.zeros((len(metrics), len(models)))
            
            for item in performance_data:
                metric_idx = metrics.index(item.get("metric", "unknown"))
                model_idx = models.index(item.get("model", "unknown"))
                value = item.get("value", 0.0)
                matrix[metric_idx, model_idx] = value
            
            # Create figure
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Create heatmap
            im = ax.imshow(matrix, cmap=self.color_schemes[config.colormap],
                          vmin=config.vmin, vmax=config.vmax, aspect='auto')
            
            # Add labels
            ax.set_xticks(range(len(models)))
            ax.set_yticks(range(len(metrics)))
            ax.set_xticklabels(models, rotation=config.rotation, ha='right')
            ax.set_yticklabels(metrics)
            
            # Add annotations
            if config.annotate:
                for i in range(len(metrics)):
                    for j in range(len(models)):
                        text = ax.text(j, i, f"{matrix[i, j]:{config.fmt}}",
                                     ha="center", va="center", color="white" if matrix[i, j] > np.mean(matrix) else "black")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Performance Value", rotation=270, labelpad=20)
            
            # Add title and labels
            ax.set_title(config.title or "Performance Matrix")
            ax.set_xlabel(config.xlabel or "Models")
            ax.set_ylabel(config.ylabel or "Metrics")
            
            # Save plot
            filename = f"performance_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create performance heatmap: {e}")
            return ""
    
    def create_correlation_heatmap(self, data: np.ndarray, labels: List[str] = None,
                                 config: HeatmapConfig = None) -> str:
        """Create correlation heatmap"""
        try:
            config = config or HeatmapConfig()
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(data.T)
            
            # Create figure
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Create heatmap
            im = ax.imshow(correlation_matrix, cmap=self.color_schemes[config.colormap],
                          vmin=-1, vmax=1, aspect='auto')
            
            # Add labels
            if labels:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=config.rotation, ha='right')
                ax.set_yticklabels(labels)
            
            # Add annotations
            if config.annotate:
                for i in range(len(correlation_matrix)):
                    for j in range(len(correlation_matrix[i])):
                        text = ax.text(j, i, f"{correlation_matrix[i, j]:.2f}",
                                     ha="center", va="center", color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Correlation", rotation=270, labelpad=20)
            
            # Add title and labels
            ax.set_title(config.title or "Correlation Matrix")
            ax.set_xlabel(config.xlabel or "Variables")
            ax.set_ylabel(config.ylabel or "Variables")
            
            # Save plot
            filename = f"correlation_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create correlation heatmap: {e}")
            return ""
    
    def create_confusion_matrix_heatmap(self, confusion_matrix: np.ndarray,
                                      labels: List[str] = None,
                                      config: HeatmapConfig = None) -> str:
        """Create confusion matrix heatmap"""
        try:
            config = config or HeatmapConfig()
            
            # Create figure
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Create heatmap
            im = ax.imshow(confusion_matrix, cmap=self.color_schemes[config.colormap],
                          vmin=0, vmax=np.max(confusion_matrix), aspect='auto')
            
            # Add labels
            if labels:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=config.rotation, ha='right')
                ax.set_yticklabels(labels)
            
            # Add annotations
            if config.annotate:
                for i in range(len(confusion_matrix)):
                    for j in range(len(confusion_matrix[i])):
                        text = ax.text(j, i, f"{int(confusion_matrix[i, j])}",
                                     ha="center", va="center", color="white" if confusion_matrix[i, j] > np.max(confusion_matrix)/2 else "black")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Count", rotation=270, labelpad=20)
            
            # Add title and labels
            ax.set_title(config.title or "Confusion Matrix")
            ax.set_xlabel(config.xlabel or "Predicted")
            ax.set_ylabel(config.ylabel or "Actual")
            
            # Save plot
            filename = f"confusion_matrix_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create confusion matrix heatmap: {e}")
            return ""
    
    def create_risk_heatmap(self, risk_data: List[Dict[str, Any]],
                          config: HeatmapConfig = None) -> str:
        """Create risk heatmap"""
        try:
            config = config or HeatmapConfig()
            
            # Extract risk categories and levels
            categories = list(set(item.get("category", "unknown") for item in risk_data))
            risk_levels = ["low", "medium", "high", "critical"]
            
            # Create matrix
            matrix = np.zeros((len(categories), len(risk_levels)))
            
            for item in risk_data:
                category_idx = categories.index(item.get("category", "unknown"))
                risk_level = item.get("risk_level", "medium").lower()
                if risk_level in risk_levels:
                    risk_idx = risk_levels.index(risk_level)
                    risk_score = item.get("risk_score", 0.0)
                    matrix[category_idx, risk_idx] += risk_score
            
            # Create figure
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Create heatmap
            im = ax.imshow(matrix, cmap=self.color_schemes[config.colormap],
                          vmin=0, vmax=np.max(matrix), aspect='auto')
            
            # Add labels
            ax.set_xticks(range(len(risk_levels)))
            ax.set_yticks(range(len(categories)))
            ax.set_xticklabels(risk_levels, rotation=config.rotation, ha='right')
            ax.set_yticklabels(categories)
            
            # Add annotations
            if config.annotate:
                for i in range(len(categories)):
                    for j in range(len(risk_levels)):
                        if matrix[i, j] > 0:
                            text = ax.text(j, i, f"{matrix[i, j]:.1f}",
                                         ha="center", va="center", color="white" if matrix[i, j] > np.max(matrix)/2 else "black")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Risk Score", rotation=270, labelpad=20)
            
            # Add title and labels
            ax.set_title(config.title or "Risk Distribution")
            ax.set_xlabel(config.xlabel or "Risk Level")
            ax.set_ylabel(config.ylabel or "Category")
            
            # Save plot
            filename = f"risk_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create risk heatmap: {e}")
            return ""
    
    def create_clustered_heatmap(self, data: np.ndarray, labels: List[str] = None,
                               config: HeatmapConfig = None) -> str:
        """Create clustered heatmap using seaborn"""
        try:
            config = config or HeatmapConfig()
            
            # Create DataFrame
            if labels:
                df = pd.DataFrame(data, columns=labels, index=labels)
            else:
                df = pd.DataFrame(data)
            
            # Create figure
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Create clustered heatmap
            sns.heatmap(df, cmap=self.color_schemes[config.colormap],
                       vmin=config.vmin, vmax=config.vmax,
                       center=config.center, annot=config.annotate,
                       fmt=config.fmt, cbar_kws=config.cbar_kws,
                       xticklabels=config.xticklabels,
                       yticklabels=config.yticklabels,
                       ax=ax)
            
            # Add title
            ax.set_title(config.title or "Clustered Heatmap")
            
            # Save plot
            filename = f"clustered_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create clustered heatmap: {e}")
            return ""
    
    def create_custom_heatmap(self, data: np.ndarray, labels: List[str] = None,
                            config: HeatmapConfig = None) -> str:
        """Create custom heatmap with advanced features"""
        try:
            config = config or HeatmapConfig()
            
            # Create figure
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            
            # Create heatmap
            im = ax.imshow(data, cmap=self.color_schemes[config.colormap],
                          vmin=config.vmin, vmax=config.vmax, aspect='auto')
            
            # Add labels
            if labels:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=config.rotation, ha='right')
                ax.set_yticklabels(labels)
            
            # Add annotations
            if config.annotate:
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        text = ax.text(j, i, f"{data[i, j]:{config.fmt}}",
                                     ha="center", va="center", color="white" if data[i, j] > np.mean(data) else "black")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Value", rotation=270, labelpad=20)
            
            # Add title and labels
            ax.set_title(config.title or "Custom Heatmap")
            ax.set_xlabel(config.xlabel or "X Axis")
            ax.set_ylabel(config.ylabel or "Y Axis")
            
            # Save plot
            filename = f"custom_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create custom heatmap: {e}")
            return ""
    
    def export_heatmap_data(self, data: np.ndarray, labels: List[str] = None,
                          format_type: str = "json") -> str:
        """Export heatmap data"""
        try:
            export_data = {
                "data": data.tolist(),
                "labels": labels or [],
                "shape": data.shape,
                "min_value": float(np.min(data)),
                "max_value": float(np.max(data)),
                "mean_value": float(np.mean(data)),
                "std_value": float(np.std(data)),
                "exported_at": datetime.now().isoformat()
            }
            
            if format_type == "json":
                import json
                return json.dumps(export_data, indent=2)
            else:
                return str(export_data)
                
        except Exception as e:
            logger.error(f"Failed to export heatmap data: {e}")
            return "{}"
