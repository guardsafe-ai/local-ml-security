"""
API Routes for Analytics Service
"""

from .health import router as health_router
from .red_team import router as red_team_router
from .model_performance import router as model_performance_router
from .analytics import router as analytics_router

__all__ = [
    "health_router",
    "red_team_router", 
    "model_performance_router",
    "analytics_router"
]
