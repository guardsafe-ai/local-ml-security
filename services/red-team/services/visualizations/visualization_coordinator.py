"""
Visualization Coordinator
Coordinates all visualization components and provides unified interface for creating visualizations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json

from .attack_trees import AttackTreeVisualizer
from .embedding_plots import EmbeddingPlotter
from .heatmaps import HeatmapGenerator
from .security_dashboard import SecurityDashboard, DashboardType

logger = logging.getLogger(__name__)

class VisualizationType(Enum):
    """Types of visualizations"""
    ATTACK_TREE = "attack_tree"
    EMBEDDING_PLOT = "embedding_plot"
    HEATMAP = "heatmap"
    SECURITY_DASHBOARD = "security_dashboard"

class VisualizationConfig:
    """Configuration for visualizations"""
    def __init__(self, 
                 output_format: str = "png",
                 dpi: int = 300,
                 figsize: Tuple[int, int] = (12, 8),
                 theme: str = "white",
                 save_path: str = None):
        self.output_format = output_format
        self.dpi = dpi
        self.figsize = figsize
        self.theme = theme
        self.save_path = save_path or f"./visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

class VisualizationCoordinator:
    """Coordinates all visualization components"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        
        # Initialize components
        self.attack_tree_visualizer = AttackTreeVisualizer()
        self.embedding_plotter = EmbeddingPlotter()
        self.heatmap_generator = HeatmapGenerator()
        self.security_dashboard = SecurityDashboard()
        
        # Visualization cache
        self.visualization_cache: Dict[str, str] = {}
        
    def create_attack_tree_visualization(self, attack_scenarios: List[Dict[str, Any]]) -> str:
        """Create attack tree visualization"""
        try:
            # Generate attack tree
            filename = self.attack_tree_visualizer.create_attack_tree(attack_scenarios)
            
            # Cache result
            cache_key = f"attack_tree_{hash(str(attack_scenarios))}"
            self.visualization_cache[cache_key] = filename
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create attack tree visualization: {e}")
            return ""
    
    def create_embedding_visualization(self, embeddings: List[Dict[str, Any]], 
                                     plot_type: str = "scatter") -> str:
        """Create embedding visualization"""
        try:
            # Add embeddings to plotter
            for emb in embeddings:
                self.embedding_plotter.add_embedding(
                    text=emb.get("text", ""),
                    embedding=emb.get("embedding", []),
                    label=emb.get("label", ""),
                    category=emb.get("category", ""),
                    metadata=emb.get("metadata", {})
                )
            
            # Reduce dimensions
            self.embedding_plotter.reduce_dimensions()
            
            # Create visualization
            if plot_type == "scatter":
                filename = self.embedding_plotter.create_scatter_plot()
            elif plot_type == "heatmap":
                filename = self.embedding_plotter.create_heatmap()
            elif plot_type == "cluster":
                filename = self.embedding_plotter.create_cluster_plot()
            else:
                filename = self.embedding_plotter.create_scatter_plot()
            
            # Cache result
            cache_key = f"embedding_{plot_type}_{hash(str(embeddings))}"
            self.visualization_cache[cache_key] = filename
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create embedding visualization: {e}")
            return ""
    
    def create_heatmap_visualization(self, data: Dict[str, Any], 
                                   heatmap_type: str = "similarity") -> str:
        """Create heatmap visualization"""
        try:
            if heatmap_type == "similarity":
                filename = self.heatmap_generator.create_similarity_heatmap(
                    data.get("matrix", []),
                    data.get("labels", [])
                )
            elif heatmap_type == "attack_matrix":
                filename = self.heatmap_generator.create_attack_matrix_heatmap(
                    data.get("attack_data", [])
                )
            elif heatmap_type == "vulnerability":
                filename = self.heatmap_generator.create_vulnerability_heatmap(
                    data.get("vulnerability_data", [])
                )
            elif heatmap_type == "compliance":
                filename = self.heatmap_generator.create_compliance_heatmap(
                    data.get("compliance_data", [])
                )
            elif heatmap_type == "performance":
                filename = self.heatmap_generator.create_performance_heatmap(
                    data.get("performance_data", [])
                )
            elif heatmap_type == "correlation":
                filename = self.heatmap_generator.create_correlation_heatmap(
                    data.get("matrix", []),
                    data.get("labels", [])
                )
            elif heatmap_type == "risk":
                filename = self.heatmap_generator.create_risk_heatmap(
                    data.get("risk_data", [])
                )
            else:
                filename = self.heatmap_generator.create_custom_heatmap(
                    data.get("matrix", []),
                    data.get("labels", [])
                )
            
            # Cache result
            cache_key = f"heatmap_{heatmap_type}_{hash(str(data))}"
            self.visualization_cache[cache_key] = filename
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create heatmap visualization: {e}")
            return ""
    
    def create_security_dashboard(self, security_data: Dict[str, Any], 
                                dashboard_type: str = "executive") -> str:
        """Create security dashboard"""
        try:
            if dashboard_type == "executive":
                filename = self.security_dashboard.create_executive_dashboard(security_data)
            elif dashboard_type == "technical":
                filename = self.security_dashboard.create_technical_dashboard(security_data)
            elif dashboard_type == "operational":
                filename = self.security_dashboard.create_operational_dashboard(security_data)
            elif dashboard_type == "compliance":
                filename = self.security_dashboard.create_compliance_dashboard(security_data)
            else:
                filename = self.security_dashboard.create_executive_dashboard(security_data)
            
            # Cache result
            cache_key = f"dashboard_{dashboard_type}_{hash(str(security_data))}"
            self.visualization_cache[cache_key] = filename
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to create security dashboard: {e}")
            return ""
    
    def create_comprehensive_report(self, report_data: Dict[str, Any]) -> str:
        """Create comprehensive security report with multiple visualizations"""
        try:
            report_visualizations = []
            
            # Attack tree
            if "attack_scenarios" in report_data:
                attack_tree = self.create_attack_tree_visualization(
                    report_data["attack_scenarios"]
                )
                if attack_tree:
                    report_visualizations.append({
                        "type": "attack_tree",
                        "title": "Attack Tree Analysis",
                        "filename": attack_tree
                    })
            
            # Embedding plots
            if "embeddings" in report_data:
                embedding_plot = self.create_embedding_visualization(
                    report_data["embeddings"],
                    plot_type="scatter"
                )
                if embedding_plot:
                    report_visualizations.append({
                        "type": "embedding_plot",
                        "title": "Attack Pattern Analysis",
                        "filename": embedding_plot
                    })
            
            # Heatmaps
            if "attack_matrix" in report_data:
                attack_heatmap = self.create_heatmap_visualization(
                    {"attack_data": report_data["attack_matrix"]},
                    heatmap_type="attack_matrix"
                )
                if attack_heatmap:
                    report_visualizations.append({
                        "type": "heatmap",
                        "title": "Attack Success Matrix",
                        "filename": attack_heatmap
                    })
            
            if "vulnerability_data" in report_data:
                vuln_heatmap = self.create_heatmap_visualization(
                    {"vulnerability_data": report_data["vulnerability_data"]},
                    heatmap_type="vulnerability"
                )
                if vuln_heatmap:
                    report_visualizations.append({
                        "type": "heatmap",
                        "title": "Vulnerability Distribution",
                        "filename": vuln_heatmap
                    })
            
            # Security dashboard
            dashboard = self.create_security_dashboard(
                report_data,
                dashboard_type="executive"
            )
            if dashboard:
                report_visualizations.append({
                    "type": "dashboard",
                    "title": "Executive Security Dashboard",
                    "filename": dashboard
                })
            
            # Save report metadata
            report_metadata = {
                "created_at": datetime.now().isoformat(),
                "visualizations": report_visualizations,
                "total_visualizations": len(report_visualizations),
                "report_data_keys": list(report_data.keys())
            }
            
            metadata_filename = f"report_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metadata_filename, 'w') as f:
                json.dump(report_metadata, f, indent=2)
            
            return metadata_filename
            
        except Exception as e:
            logger.error(f"Failed to create comprehensive report: {e}")
            return ""
    
    def create_attack_analysis_visualization(self, attack_data: Dict[str, Any]) -> str:
        """Create comprehensive attack analysis visualization"""
        try:
            visualizations = []
            
            # Attack success matrix
            if "attack_results" in attack_data:
                attack_matrix = self._create_attack_matrix(attack_data["attack_results"])
                heatmap = self.create_heatmap_visualization(
                    {"attack_data": attack_matrix},
                    heatmap_type="attack_matrix"
                )
                if heatmap:
                    visualizations.append(heatmap)
            
            # Attack timeline
            if "attack_timeline" in attack_data:
                timeline_data = attack_data["attack_timeline"]
                # Create timeline visualization (simplified)
                timeline_viz = self._create_timeline_visualization(timeline_data)
                if timeline_viz:
                    visualizations.append(timeline_viz)
            
            # Vulnerability distribution
            if "vulnerabilities" in attack_data:
                vuln_data = attack_data["vulnerabilities"]
                vuln_heatmap = self.create_heatmap_visualization(
                    {"vulnerability_data": vuln_data},
                    heatmap_type="vulnerability"
                )
                if vuln_heatmap:
                    visualizations.append(vuln_heatmap)
            
            # Attack tree
            if "attack_scenarios" in attack_data:
                attack_tree = self.create_attack_tree_visualization(
                    attack_data["attack_scenarios"]
                )
                if attack_tree:
                    visualizations.append(attack_tree)
            
            return visualizations[0] if visualizations else ""
            
        except Exception as e:
            logger.error(f"Failed to create attack analysis visualization: {e}")
            return ""
    
    def create_model_behavior_visualization(self, model_data: Dict[str, Any]) -> str:
        """Create model behavior visualization"""
        try:
            visualizations = []
            
            # Model performance heatmap
            if "performance_metrics" in model_data:
                perf_heatmap = self.create_heatmap_visualization(
                    {"performance_data": model_data["performance_metrics"]},
                    heatmap_type="performance"
                )
                if perf_heatmap:
                    visualizations.append(perf_heatmap)
            
            # Model embeddings
            if "embeddings" in model_data:
                embedding_plot = self.create_embedding_visualization(
                    model_data["embeddings"],
                    plot_type="scatter"
                )
                if embedding_plot:
                    visualizations.append(embedding_plot)
            
            # Model confusion matrix
            if "confusion_matrix" in model_data:
                conf_matrix = self.heatmap_generator.create_confusion_matrix_heatmap(
                    model_data["confusion_matrix"],
                    model_data.get("labels", [])
                )
                if conf_matrix:
                    visualizations.append(conf_matrix)
            
            return visualizations[0] if visualizations else ""
            
        except Exception as e:
            logger.error(f"Failed to create model behavior visualization: {e}")
            return ""
    
    def create_compliance_visualization(self, compliance_data: Dict[str, Any]) -> str:
        """Create compliance visualization"""
        try:
            visualizations = []
            
            # Compliance heatmap
            if "compliance_matrix" in compliance_data:
                comp_heatmap = self.create_heatmap_visualization(
                    {"compliance_data": compliance_data["compliance_matrix"]},
                    heatmap_type="compliance"
                )
                if comp_heatmap:
                    visualizations.append(comp_heatmap)
            
            # Compliance dashboard
            dashboard = self.create_security_dashboard(
                compliance_data,
                dashboard_type="compliance"
            )
            if dashboard:
                visualizations.append(dashboard)
            
            return visualizations[0] if visualizations else ""
            
        except Exception as e:
            logger.error(f"Failed to create compliance visualization: {e}")
            return ""
    
    def _create_attack_matrix(self, attack_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create attack matrix from results"""
        try:
            # Group by attack type and model type
            matrix_data = []
            attack_types = set()
            model_types = set()
            
            for result in attack_results:
                attack_type = result.get("attack_type", "unknown")
                model_type = result.get("model_type", "unknown")
                success_rate = result.get("success_rate", 0.0)
                
                attack_types.add(attack_type)
                model_types.add(model_type)
                
                matrix_data.append({
                    "attack_type": attack_type,
                    "model_type": model_type,
                    "success_rate": success_rate
                })
            
            return matrix_data
            
        except Exception as e:
            logger.error(f"Failed to create attack matrix: {e}")
            return []
    
    def _create_timeline_visualization(self, timeline_data: List[Dict[str, Any]]) -> str:
        """Create timeline visualization"""
        try:
            # This is a simplified timeline visualization
            # In a full implementation, you would create a proper timeline chart
            return ""
            
        except Exception as e:
            logger.error(f"Failed to create timeline visualization: {e}")
            return ""
    
    def get_visualization_statistics(self) -> Dict[str, Any]:
        """Get visualization statistics"""
        try:
            stats = {
                "total_visualizations": len(self.visualization_cache),
                "cached_visualizations": list(self.visualization_cache.keys()),
                "components": {
                    "attack_tree_visualizer": True,
                    "embedding_plotter": True,
                    "heatmap_generator": True,
                    "security_dashboard": True
                },
                "created_at": datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get visualization statistics: {e}")
            return {}
    
    def clear_cache(self):
        """Clear visualization cache"""
        self.visualization_cache.clear()
        logger.info("Visualization cache cleared")
    
    def export_visualization_data(self, format_type: str = "json") -> str:
        """Export visualization data"""
        try:
            data = {
                "config": {
                    "output_format": self.config.output_format,
                    "dpi": self.config.dpi,
                    "figsize": self.config.figsize,
                    "theme": self.config.theme,
                    "save_path": self.config.save_path
                },
                "cached_visualizations": self.visualization_cache,
                "statistics": self.get_visualization_statistics(),
                "exported_at": datetime.now().isoformat()
            }
            
            if format_type == "json":
                return json.dumps(data, indent=2)
            else:
                return str(data)
                
        except Exception as e:
            logger.error(f"Failed to export visualization data: {e}")
            return "{}"
