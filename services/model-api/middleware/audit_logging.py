"""
Audit Logging Middleware for Model API Service
Logs all model lifecycle events for compliance and tracking
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Import the correct AuditLogger from services
from services.audit_logger import AuditLogger as ServiceAuditLogger

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically log model-related operations"""
    
    def __init__(self, app: ASGIApp, audit_logger: ServiceAuditLogger):
        super().__init__(app)
        self.audit_logger = audit_logger
        logger.info("✅ AuditLoggingMiddleware initialized")

    async def dispatch(self, request: Request, call_next):
        # Extract user info (in production, this would come from auth)
        user_id = request.headers.get("X-User-ID", "system")
        ip_address = request.client.host if request.client else "unknown"
        
        # Process request
        response = await call_next(request)
        
        # Log model-related operations in background (non-blocking)
        # This ensures audit logging never blocks the HTTP response
        try:
            # Use asyncio.create_task to run in background
            import asyncio
            asyncio.create_task(self._log_model_operations(
                request=request,
                response=response,
                user_id=user_id,
                ip_address=ip_address
            ))
        except Exception as e:
            # Even if background logging fails, don't affect the response
            logger.debug(f"Background audit logging failed: {e}")
        
        return response
    
    async def _log_model_operations(
        self, 
        request: Request, 
        response: Response, 
        user_id: str, 
        ip_address: str
    ):
        """Log model-related operations based on endpoint - COMPLETELY NON-BLOCKING"""
        
        try:
            path = request.url.path
            method = request.method
            
            # Model loading operations
            if path.startswith("/models/") and method == "POST":
                if "load" in path:
                    await self._log_model_load(request, response, user_id, ip_address)
                elif "unload" in path:
                    await self._log_model_unload(request, response, user_id, ip_address)
                elif "reload" in path:
                    await self._log_model_reload(request, response, user_id, ip_address)
            
            # Prediction operations
            elif path.startswith("/predict") and method == "POST":
                await self._log_prediction(request, response, user_id, ip_address)
        except Exception as e:
            # Never let audit logging errors affect the application
            logger.debug(f"Audit logging failed (non-critical): {e}")
    
    async def _log_model_load(self, request: Request, response: Response, user_id: str, ip_address: str):
        """Log model loading events - SAFE AND NON-BLOCKING"""
        try:
            body = await request.json() if hasattr(request, 'json') else {}
            model_name = body.get("model_name", "unknown")
            version = body.get("version", "latest")
            
            from services.audit_logger import AuditEventType
            await self.audit_logger.log_event(
                event_type=AuditEventType.MODEL_LOAD,
                user_id=user_id,
                resource=f"model:{model_name}",
                action="load",
                details={
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "method": request.method,
                    "version": version
                }
            )
        except Exception as e:
            # Never let audit logging errors affect the application
            logger.debug(f"Model load audit logging failed (non-critical): {e}")
    
    async def _log_model_unload(self, request: Request, response: Response, user_id: str, ip_address: str):
        """Log model unloading events - SAFE AND NON-BLOCKING"""
        try:
            body = await request.json() if hasattr(request, 'json') else {}
            model_name = body.get("model_name", "unknown")
            
            from services.audit_logger import AuditEventType
            await self.audit_logger.log_event(
                event_type=AuditEventType.MODEL_UNLOAD,
                user_id=user_id,
                resource=f"model:{model_name}",
                action="unload",
                details={
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "method": request.method
                }
            )
        except Exception as e:
            logger.debug(f"Model unload audit logging failed (non-critical): {e}")
    
    async def _log_model_reload(self, request: Request, response: Response, user_id: str, ip_address: str):
        """Log model reload events - SAFE AND NON-BLOCKING"""
        try:
            model_name = request.path_params.get("model_name", "unknown")
            
            from services.audit_logger import AuditEventType
            await self.audit_logger.log_event(
                event_type=AuditEventType.MODEL_LOAD,  # Use MODEL_LOAD for reload
                user_id=user_id,
                resource=f"model:{model_name}",
                action="reload",
                details={
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "method": request.method
                }
            )
        except Exception as e:
            logger.debug(f"Model reload audit logging failed (non-critical): {e}")
    
    async def _log_prediction(self, request: Request, response: Response, user_id: str, ip_address: str):
        """Log prediction events - SAFE AND NON-BLOCKING"""
        try:
            body = await request.json() if hasattr(request, 'json') else {}
            models = body.get("models", [])
            text_length = len(body.get("text", ""))
            
            from services.audit_logger import AuditEventType
            await self.audit_logger.log_event(
                event_type=AuditEventType.MODEL_PREDICT,
                user_id=user_id,
                resource=f"model:{models[0] if models else 'ensemble'}",
                action="predict",
                details={
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "method": request.method,
                    "models_used": models,
                    "text_length": text_length,
                    "ensemble": body.get("ensemble", False)
                }
            )
        except Exception as e:
            logger.debug(f"Prediction audit logging failed (non-critical): {e}")
    
    async def _log_model_management(self, request: Request, response: Response, user_id: str, ip_address: str):
        """Log model management events - SAFE AND NON-BLOCKING"""
        try:
            model_name = request.path_params.get("model_name", "unknown")
            operation = "updated" if request.method == "PUT" else "deleted"
            
            from services.audit_logger import AuditEventType
            await self.audit_logger.log_event(
                event_type=AuditEventType.ADMIN_ACTION,  # Use ADMIN_ACTION for model management
                user_id=user_id,
                resource=f"model:{model_name}",
                action=operation,
                details={
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "method": request.method
                }
            )
        except Exception as e:
            logger.debug(f"Model management audit logging failed (non-critical): {e}")

# Global audit logger instance
audit_logger = ServiceAuditLogger()
