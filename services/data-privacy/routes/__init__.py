# Data Privacy Routes Package
from .health import router as health_router
from .privacy import router as privacy_router

__all__ = ['health_router', 'privacy_router']
