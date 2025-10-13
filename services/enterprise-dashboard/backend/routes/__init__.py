"""
Enterprise Dashboard Backend - Routes Package
"""

from .health import router as health_router
from .dashboard import router as dashboard_router
from .models import router as models_router
from .training import router as training_router
from .red_team import router as red_team_router
from .analytics import router as analytics_router
from .business_metrics import router as business_metrics_router
from .data_privacy import router as data_privacy_router
from .mlflow import router as mlflow_router
from .monitoring import router as monitoring_router
from .websocket import router as websocket_router
