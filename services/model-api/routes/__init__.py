"""
API Routes for Model API Service
"""

from .health import router as health_router
from .models import router as models_router
from .predictions import router as predictions_router

__all__ = [
    "health_router",
    "models_router",
    "predictions_router"
]
