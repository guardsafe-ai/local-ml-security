"""
Monitoring Service - Visualization
Creates visualizations for monitoring data
"""

import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class VisualizationService:
    """Creates visualizations for monitoring data"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#8c564b'
        }
    
    def create_model_loading_chart(self, data: List[Dict[str, Any]]) -> go.Figure:
        """Create model loading status chart"""
        try:
            if not data:
                # Return empty chart
                fig = go.Figure()
                fig.add_annotation(
                    text="No model loading data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            df = pd.DataFrame(data)
            
            # Create pie chart for model status
            status_counts = df['status'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.3,
                marker_colors=[self.color_palette['success'] if status == 'loaded' 
                              else self.color_palette['warning'] for status in status_counts.index]
            )])
            
            fig.update_layout(
                title="Model Loading Status",
                showlegend=True,
                height=400
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating model loading chart: {e}")
            return go.Figure()
    
    def create_training_progress_chart(self, data: List[Dict[str, Any]]) -> go.Figure:
        """Create training progress chart"""
        try:
            if not data:
                # Return empty chart
                fig = go.Figure()
                fig.add_annotation(
                    text="No training data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            df = pd.DataFrame(data)
            
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Training Progress', 'Loss & Accuracy'),
                vertical_spacing=0.1
            )
            
            # Training progress bar
            for idx, row in df.iterrows():
                fig.add_trace(
                    go.Bar(
                        x=[row['model_name']],
                        y=[row['progress']],
                        name=f"Job {row['job_id'][:8]}",
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Loss and accuracy
            fig.add_trace(
                go.Scatter(
                    x=df['model_name'],
                    y=df['loss'],
                    mode='lines+markers',
                    name='Loss',
                    line=dict(color=self.color_palette['warning'])
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['model_name'],
                    y=df['accuracy'],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color=self.color_palette['success']),
                    yaxis='y2'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Training Progress",
                height=600,
                showlegend=True
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating training progress chart: {e}")
            return go.Figure()
    
    def create_system_metrics_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create system metrics chart"""
        try:
            if not data:
                return go.Figure()
            
            # Create gauge charts for system metrics
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]],
                subplot_titles=('CPU Usage', 'Memory Usage', 'Disk Usage', 'Network I/O')
            )
            
            # CPU Usage
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=data['cpu_usage'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CPU %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': self.color_palette['primary']},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}]}
            ), row=1, col=1)
            
            # Memory Usage
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=data['memory_usage'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Memory %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': self.color_palette['secondary']},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}]}
            ), row=1, col=2)
            
            # Disk Usage
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=data['disk_usage'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Disk %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': self.color_palette['info']},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}]}
            ), row=2, col=1)
            
            # Network I/O
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=data['network_io'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Network I/O"},
                gauge={'axis': {'range': [None, 1000]},
                       'bar': {'color': self.color_palette['light']}}
            ), row=2, col=2)
            
            fig.update_layout(
                title="System Metrics",
                height=500,
                showlegend=False
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating system metrics chart: {e}")
            return go.Figure()
    
    def create_service_health_chart(self, data: List[Dict[str, Any]]) -> go.Figure:
        """Create service health chart"""
        try:
            if not data:
                return go.Figure()
            
            df = pd.DataFrame(data)
            
            # Create horizontal bar chart for service health
            fig = go.Figure()
            
            # Color mapping for status
            color_map = {
                'healthy': self.color_palette['success'],
                'unhealthy': self.color_palette['warning'],
                'unknown': self.color_palette['light']
            }
            
            colors = [color_map.get(status, self.color_palette['light']) for status in df['status']]
            
            fig.add_trace(go.Bar(
                y=df['service_name'],
                x=df['response_time'],
                orientation='h',
                marker_color=colors,
                text=df['status'],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Service Health & Response Times",
                xaxis_title="Response Time (ms)",
                yaxis_title="Service",
                height=400
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating service health chart: {e}")
            return go.Figure()
    
    def create_alerts_timeline(self, data: List[Dict[str, Any]]) -> go.Figure:
        """Create alerts timeline chart"""
        try:
            if not data:
                # Return empty chart
                fig = go.Figure()
                fig.add_annotation(
                    text="No alerts available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create timeline chart
            fig = go.Figure()
            
            # Color mapping for severity
            severity_colors = {
                'critical': self.color_palette['warning'],
                'high': self.color_palette['secondary'],
                'medium': self.color_palette['info'],
                'low': self.color_palette['light']
            }
            
            for severity in df['severity'].unique():
                severity_data = df[df['severity'] == severity]
                fig.add_trace(go.Scatter(
                    x=severity_data['timestamp'],
                    y=severity_data['service'],
                    mode='markers',
                    name=severity.title(),
                    marker=dict(
                        color=severity_colors.get(severity, self.color_palette['light']),
                        size=10
                    ),
                    text=severity_data['message'],
                    hovertemplate='<b>%{y}</b><br>%{x}<br>%{text}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Alerts Timeline",
                xaxis_title="Time",
                yaxis_title="Service",
                height=400
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating alerts timeline: {e}")
            return go.Figure()
