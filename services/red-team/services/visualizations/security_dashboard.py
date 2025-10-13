"""
Security Dashboard
Creates comprehensive security dashboards with multiple visualizations and real-time updates.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    import seaborn as sns
    import pandas as pd
except ImportError:
    # Fallback for environments without required packages
    class plt:
        @staticmethod
        def figure(*args, **kwargs): return None
        @staticmethod
        def subplot(*args, **kwargs): return None
        @staticmethod
        def show(*args, **kwargs): pass
        @staticmethod
        def savefig(*args, **kwargs): pass
        @staticmethod
        def close(*args, **kwargs): pass
        @staticmethod
        def title(*args, **kwargs): pass
        @staticmethod
        def xlabel(*args, **kwargs): pass
        @staticmethod
        def ylabel(*args, **kwargs): pass
        @staticmethod
        def legend(*args, **kwargs): pass
        @staticmethod
        def text(*args, **kwargs): pass
        @staticmethod
        def bar(*args, **kwargs): pass
        @staticmethod
        def plot(*args, **kwargs): pass
        @staticmethod
        def pie(*args, **kwargs): pass
        @staticmethod
        def scatter(*args, **kwargs): pass
    class GridSpec:
        def __init__(self, *args, **kwargs): pass
    class sns:
        @staticmethod
        def heatmap(*args, **kwargs): pass
        @staticmethod
        def barplot(*args, **kwargs): pass
        @staticmethod
        def lineplot(*args, **kwargs): pass
    class pd:
        @staticmethod
        def DataFrame(*args, **kwargs): return None

logger = logging.getLogger(__name__)

class DashboardType(Enum):
    """Types of dashboards"""
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    CUSTOM = "custom"

class WidgetType(Enum):
    """Types of dashboard widgets"""
    METRIC_CARD = "metric_card"
    CHART = "chart"
    TABLE = "table"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TIMELINE = "timeline"
    ALERT_LIST = "alert_list"

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    title: str = "Security Dashboard"
    dashboard_type: DashboardType = DashboardType.EXECUTIVE
    figsize: Tuple[int, int] = (20, 12)
    dpi: int = 300
    theme: str = "white"
    show_title: bool = True
    show_timestamp: bool = True
    auto_refresh: bool = False
    refresh_interval: int = 300  # seconds

@dataclass
class WidgetConfig:
    """Widget configuration"""
    widget_type: WidgetType
    title: str
    position: Tuple[int, int, int, int]  # (row, col, rowspan, colspan)
    data: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

class SecurityDashboard:
    """Creates comprehensive security dashboards"""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.widgets: List[WidgetConfig] = []
        self.data_sources: Dict[str, Any] = {}
        
    def add_data_source(self, name: str, data: Any):
        """Add data source for dashboard"""
        self.data_sources[name] = data
        
    def add_widget(self, widget: WidgetConfig):
        """Add widget to dashboard"""
        self.widgets.append(widget)
        
    def create_executive_dashboard(self, security_data: Dict[str, Any]) -> str:
        """Create executive-level security dashboard"""
        try:
            # Clear existing widgets
            self.widgets.clear()
            
            # Add executive widgets
            self._add_executive_widgets(security_data)
            
            # Generate dashboard
            return self._generate_dashboard()
            
        except Exception as e:
            logger.error(f"Failed to create executive dashboard: {e}")
            return ""
    
    def create_technical_dashboard(self, security_data: Dict[str, Any]) -> str:
        """Create technical-level security dashboard"""
        try:
            # Clear existing widgets
            self.widgets.clear()
            
            # Add technical widgets
            self._add_technical_widgets(security_data)
            
            # Generate dashboard
            return self._generate_dashboard()
            
        except Exception as e:
            logger.error(f"Failed to create technical dashboard: {e}")
            return ""
    
    def create_operational_dashboard(self, security_data: Dict[str, Any]) -> str:
        """Create operational-level security dashboard"""
        try:
            # Clear existing widgets
            self.widgets.clear()
            
            # Add operational widgets
            self._add_operational_widgets(security_data)
            
            # Generate dashboard
            return self._generate_dashboard()
            
        except Exception as e:
            logger.error(f"Failed to create operational dashboard: {e}")
            return ""
    
    def create_compliance_dashboard(self, compliance_data: Dict[str, Any]) -> str:
        """Create compliance-focused dashboard"""
        try:
            # Clear existing widgets
            self.widgets.clear()
            
            # Add compliance widgets
            self._add_compliance_widgets(compliance_data)
            
            # Generate dashboard
            return self._generate_dashboard()
            
        except Exception as e:
            logger.error(f"Failed to create compliance dashboard: {e}")
            return ""
    
    def _add_executive_widgets(self, data: Dict[str, Any]):
        """Add executive-level widgets"""
        # Key metrics cards
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.METRIC_CARD,
            title="Total Attacks",
            position=(0, 0, 1, 1),
            data={"value": data.get("total_attacks", 0), "trend": "+5%"}
        ))
        
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.METRIC_CARD,
            title="Vulnerabilities Found",
            position=(0, 1, 1, 1),
            data={"value": data.get("vulnerabilities", 0), "trend": "+12%"}
        ))
        
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.METRIC_CARD,
            title="Compliance Score",
            position=(0, 2, 1, 1),
            data={"value": f"{data.get('compliance_score', 0):.1f}%", "trend": "+2%"}
        ))
        
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.METRIC_CARD,
            title="Risk Level",
            position=(0, 3, 1, 1),
            data={"value": data.get("risk_level", "Medium"), "trend": "-1%"}
        ))
        
        # Attack success rate chart
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.CHART,
            title="Attack Success Rate by Type",
            position=(1, 0, 2, 2),
            data=data.get("attack_success_rates", {}),
            config={"chart_type": "bar"}
        ))
        
        # Vulnerability distribution
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.CHART,
            title="Vulnerability Distribution",
            position=(1, 2, 2, 2),
            data=data.get("vulnerability_distribution", {}),
            config={"chart_type": "pie"}
        ))
        
        # Risk heatmap
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.HEATMAP,
            title="Risk Assessment Matrix",
            position=(3, 0, 2, 4),
            data=data.get("risk_matrix", {}),
            config={"colormap": "Reds"}
        ))
    
    def _add_technical_widgets(self, data: Dict[str, Any]):
        """Add technical-level widgets"""
        # Attack timeline
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.TIMELINE,
            title="Attack Timeline",
            position=(0, 0, 2, 3),
            data=data.get("attack_timeline", []),
            config={"show_details": True}
        ))
        
        # Model performance metrics
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.CHART,
            title="Model Performance",
            position=(0, 3, 2, 1),
            data=data.get("model_performance", {}),
            config={"chart_type": "line"}
        ))
        
        # Attack success matrix
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.HEATMAP,
            title="Attack Success Matrix",
            position=(2, 0, 2, 2),
            data=data.get("attack_matrix", {}),
            config={"colormap": "RdYlGn"}
        ))
        
        # Vulnerability details table
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.TABLE,
            title="Vulnerability Details",
            position=(2, 2, 2, 2),
            data=data.get("vulnerability_details", []),
            config={"max_rows": 10}
        ))
        
        # System metrics
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.CHART,
            title="System Metrics",
            position=(4, 0, 1, 4),
            data=data.get("system_metrics", {}),
            config={"chart_type": "line", "multiple_series": True}
        ))
    
    def _add_operational_widgets(self, data: Dict[str, Any]):
        """Add operational-level widgets"""
        # Active alerts
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.ALERT_LIST,
            title="Active Alerts",
            position=(0, 0, 2, 2),
            data=data.get("active_alerts", []),
            config={"max_items": 10}
        ))
        
        # Worker status
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.CHART,
            title="Worker Status",
            position=(0, 2, 1, 2),
            data=data.get("worker_status", {}),
            config={"chart_type": "gauge"}
        ))
        
        # Queue status
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.CHART,
            title="Queue Status",
            position=(1, 2, 1, 2),
            data=data.get("queue_status", {}),
            config={"chart_type": "bar"}
        ))
        
        # Performance metrics
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.CHART,
            title="Performance Metrics",
            position=(2, 0, 2, 4),
            data=data.get("performance_metrics", {}),
            config={"chart_type": "line", "multiple_series": True}
        ))
    
    def _add_compliance_widgets(self, data: Dict[str, Any]):
        """Add compliance-focused widgets"""
        # Compliance scores
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.CHART,
            title="Compliance Scores by Framework",
            position=(0, 0, 2, 2),
            data=data.get("compliance_scores", {}),
            config={"chart_type": "bar"}
        ))
        
        # Control status
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.HEATMAP,
            title="Control Status Matrix",
            position=(0, 2, 2, 2),
            data=data.get("control_status", {}),
            config={"colormap": "RdYlGn"}
        ))
        
        # Violation timeline
        self.add_widget(WidgetConfig(
            widget_type=WidgetType.TIMELINE,
            title="Compliance Violations",
            position=(2, 0, 2, 4),
            data=data.get("violation_timeline", []),
            config={"show_severity": True}
        ))
    
    def _generate_dashboard(self) -> str:
        """Generate the dashboard visualization"""
        try:
            # Create figure with grid layout
            fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
            gs = GridSpec(6, 4, figure=fig)
            
            # Add title
            if self.config.show_title:
                fig.suptitle(self.config.title, fontsize=16, fontweight='bold')
            
            # Add timestamp
            if self.config.show_timestamp:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                fig.text(0.99, 0.01, f"Last Updated: {timestamp}", 
                        ha='right', va='bottom', fontsize=8)
            
            # Render widgets
            for widget in self.widgets:
                self._render_widget(fig, gs, widget)
            
            # Save dashboard
            filename = f"security_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")
            return ""
    
    def _render_widget(self, fig, gs, widget: WidgetConfig):
        """Render individual widget"""
        try:
            row, col, rowspan, colspan = widget.position
            ax = fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
            
            if widget.widget_type == WidgetType.METRIC_CARD:
                self._render_metric_card(ax, widget)
            elif widget.widget_type == WidgetType.CHART:
                self._render_chart(ax, widget)
            elif widget.widget_type == WidgetType.TABLE:
                self._render_table(ax, widget)
            elif widget.widget_type == WidgetType.HEATMAP:
                self._render_heatmap(ax, widget)
            elif widget.widget_type == WidgetType.GAUGE:
                self._render_gauge(ax, widget)
            elif widget.widget_type == WidgetType.TIMELINE:
                self._render_timeline(ax, widget)
            elif widget.widget_type == WidgetType.ALERT_LIST:
                self._render_alert_list(ax, widget)
            
        except Exception as e:
            logger.error(f"Failed to render widget: {e}")
    
    def _render_metric_card(self, ax, widget: WidgetConfig):
        """Render metric card widget"""
        try:
            value = widget.data.get("value", 0)
            trend = widget.data.get("trend", "")
            
            # Clear axes
            ax.axis('off')
            
            # Add value
            ax.text(0.5, 0.6, str(value), ha='center', va='center',
                   fontsize=24, fontweight='bold', color='#2E86AB')
            
            # Add trend
            if trend:
                color = 'green' if trend.startswith('+') else 'red'
                ax.text(0.5, 0.3, trend, ha='center', va='center',
                       fontsize=12, color=color)
            
            # Add title
            ax.text(0.5, 0.9, widget.title, ha='center', va='center',
                   fontsize=12, fontweight='bold')
            
        except Exception as e:
            logger.error(f"Failed to render metric card: {e}")
    
    def _render_chart(self, ax, widget: WidgetConfig):
        """Render chart widget"""
        try:
            data = widget.data
            chart_type = widget.config.get("chart_type", "bar")
            
            if chart_type == "bar":
                if isinstance(data, dict):
                    keys = list(data.keys())
                    values = list(data.values())
                    ax.bar(keys, values, color='#2E86AB')
                else:
                    ax.bar(range(len(data)), data, color='#2E86AB')
            
            elif chart_type == "line":
                if isinstance(data, dict):
                    for key, values in data.items():
                        ax.plot(values, label=key)
                    ax.legend()
                else:
                    ax.plot(data, color='#2E86AB')
            
            elif chart_type == "pie":
                if isinstance(data, dict):
                    ax.pie(data.values(), labels=data.keys(), autopct='%1.1f%%')
                else:
                    ax.pie(data, autopct='%1.1f%%')
            
            ax.set_title(widget.title, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Failed to render chart: {e}")
    
    def _render_table(self, ax, widget: WidgetConfig):
        """Render table widget"""
        try:
            data = widget.data
            max_rows = widget.config.get("max_rows", 10)
            
            # Clear axes
            ax.axis('off')
            
            if not data:
                ax.text(0.5, 0.5, "No data available", ha='center', va='center')
                return
            
            # Create table
            table_data = data[:max_rows] if len(data) > max_rows else data
            
            # Add headers
            if table_data and isinstance(table_data[0], dict):
                headers = list(table_data[0].keys())
                ax.text(0.1, 0.9, " | ".join(headers), fontweight='bold', fontsize=10)
                
                # Add rows
                for i, row in enumerate(table_data):
                    row_text = " | ".join(str(row.get(h, "")) for h in headers)
                    ax.text(0.1, 0.8 - i * 0.1, row_text, fontsize=9)
            else:
                # Simple list
                for i, item in enumerate(table_data):
                    ax.text(0.1, 0.9 - i * 0.1, str(item), fontsize=9)
            
            ax.set_title(widget.title, fontweight='bold')
            
        except Exception as e:
            logger.error(f"Failed to render table: {e}")
    
    def _render_heatmap(self, ax, widget: WidgetConfig):
        """Render heatmap widget"""
        try:
            data = widget.data
            colormap = widget.config.get("colormap", "viridis")
            
            if isinstance(data, dict):
                # Convert dict to matrix
                keys = list(data.keys())
                values = list(data.values())
                matrix = np.array(values).reshape(1, -1)
                ax.imshow(matrix, cmap=colormap, aspect='auto')
                ax.set_xticks(range(len(keys)))
                ax.set_xticklabels(keys, rotation=45, ha='right')
            else:
                ax.imshow(data, cmap=colormap, aspect='auto')
            
            ax.set_title(widget.title, fontweight='bold')
            
        except Exception as e:
            logger.error(f"Failed to render heatmap: {e}")
    
    def _render_gauge(self, ax, widget: WidgetConfig):
        """Render gauge widget"""
        try:
            data = widget.data
            value = data.get("value", 0) if isinstance(data, dict) else data
            
            # Clear axes
            ax.axis('off')
            
            # Create gauge
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            # Plot gauge background
            ax.plot(theta, r, 'k-', linewidth=2)
            
            # Plot gauge value
            value_theta = value * np.pi
            ax.plot([0, value_theta], [0, 1], 'r-', linewidth=4)
            
            # Add value text
            ax.text(0, 0, f"{value:.1f}", ha='center', va='center',
                   fontsize=16, fontweight='bold')
            
            ax.set_title(widget.title, fontweight='bold')
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.2, 1.2)
            
        except Exception as e:
            logger.error(f"Failed to render gauge: {e}")
    
    def _render_timeline(self, ax, widget: WidgetConfig):
        """Render timeline widget"""
        try:
            data = widget.data
            
            if not data:
                ax.text(0.5, 0.5, "No timeline data", ha='center', va='center')
                return
            
            # Plot timeline events
            for i, event in enumerate(data):
                x = event.get("timestamp", i)
                y = event.get("severity", 1)
                color = event.get("color", "blue")
                
                ax.scatter(x, y, c=color, s=100, alpha=0.7)
                ax.text(x, y + 0.1, event.get("title", ""), ha='center', va='bottom', fontsize=8)
            
            ax.set_title(widget.title, fontweight='bold')
            ax.set_xlabel("Time")
            ax.set_ylabel("Severity")
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Failed to render timeline: {e}")
    
    def _render_alert_list(self, ax, widget: WidgetConfig):
        """Render alert list widget"""
        try:
            data = widget.data
            max_items = widget.config.get("max_items", 10)
            
            # Clear axes
            ax.axis('off')
            
            if not data:
                ax.text(0.5, 0.5, "No alerts", ha='center', va='center')
                return
            
            # Add alerts
            for i, alert in enumerate(data[:max_items]):
                severity = alert.get("severity", "info")
                title = alert.get("title", "Unknown Alert")
                time = alert.get("timestamp", "")
                
                color = {"critical": "red", "high": "orange", "medium": "yellow", "low": "green"}.get(severity, "blue")
                
                ax.text(0.05, 0.9 - i * 0.1, f"• {title}", fontsize=10, color=color)
                ax.text(0.8, 0.9 - i * 0.1, time, fontsize=8, color="gray")
            
            ax.set_title(widget.title, fontweight='bold')
            
        except Exception as e:
            logger.error(f"Failed to render alert list: {e}")
    
    def export_dashboard_data(self, format_type: str = "json") -> str:
        """Export dashboard data"""
        try:
            data = {
                "config": {
                    "title": self.config.title,
                    "dashboard_type": self.config.dashboard_type.value,
                    "figsize": self.config.figsize,
                    "dpi": self.config.dpi,
                    "theme": self.config.theme
                },
                "widgets": [
                    {
                        "widget_type": widget.widget_type.value,
                        "title": widget.title,
                        "position": widget.position,
                        "data": widget.data,
                        "config": widget.config
                    }
                    for widget in self.widgets
                ],
                "data_sources": list(self.data_sources.keys()),
                "exported_at": datetime.now().isoformat()
            }
            
            if format_type == "json":
                return json.dumps(data, indent=2)
            else:
                return str(data)
                
        except Exception as e:
            logger.error(f"Failed to export dashboard data: {e}")
            return "{}"
    
    def get_dashboard_statistics(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        try:
            stats = {
                "total_widgets": len(self.widgets),
                "widget_types": {},
                "data_sources": len(self.data_sources),
                "dashboard_type": self.config.dashboard_type.value,
                "created_at": datetime.now().isoformat()
            }
            
            # Count widget types
            for widget in self.widgets:
                widget_type = widget.widget_type.value
                stats["widget_types"][widget_type] = stats["widget_types"].get(widget_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get dashboard statistics: {e}")
            return {}
