"""
Training Service - Routes Package
"""

from .health import router as health_router
from .models import router as models_router
from .training import router as training_router
from .data import router as data_router
