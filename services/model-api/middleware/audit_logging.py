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

class AuditLogger:
    """Log all model lifecycle events for compliance"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
    
    async def log_model_event(
        self,
        event_type: str,  # deployed, promoted, archived, rolled_back, loaded, unloaded
        model_name: str,
        version: str = None,
        user_id: str = None,
        ip_address: str = None,
        metadata: Dict[str, Any] = None
    ):
        """Log model lifecycle event"""
        
        # Log to structured logger
        logger.info(f"Model audit event: {event_type} - {model_name} v{version} by {user_id} at {datetime.now().isoformat()}")
        
        # Store in database if available
        if self.db_manager:
            try:
                await self.db_manager.execute(
                    """
                    INSERT INTO model_audit_log 
                    (event_type, model_name, version, user_id, ip_address, metadata, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    event_type, model_name, version, user_id, 
                    ip_address, json.dumps(metadata or {}), datetime.now()
                )
            except Exception as e:
                logger.error(f"Failed to store audit event in database: {e}")

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically log model-related operations"""
    
    def __init__(self, app: ASGIApp, audit_logger: AuditLogger):
        super().__init__(app)
        self.audit_logger = audit_logger
        logger.info("✅ AuditLoggingMiddleware initialized")

    async def dispatch(self, request: Request, call_next):
        # Extract user info (in production, this would come from auth)
        user_id = request.headers.get("X-User-ID", "system")
        ip_address = request.client.host if request.client else "unknown"
        
        # Process request
        response = await call_next(request)
        
        # Log model-related operations
        await self._log_model_operations(
            request=request,
            response=response,
            user_id=user_id,
            ip_address=ip_address
        )
        
        return response
    
    async def _log_model_operations(
        self, 
        request: Request, 
        response: Response, 
        user_id: str, 
        ip_address: str
    ):
        """Log model-related operations based on endpoint"""
        
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
        
        # Model management operations
        elif path.startswith("/models/") and method in ["PUT", "DELETE"]:
            await self._log_model_management(request, response, user_id, ip_address)
    
    async def _log_model_load(self, request: Request, response: Response, user_id: str, ip_address: str):
        """Log model loading events"""
        try:
            body = await request.json() if hasattr(request, 'json') else {}
            model_name = body.get("model_name", "unknown")
            version = body.get("version", "latest")
            
            event_type = "loaded" if response.status_code == 200 else "load_failed"
            
            await self.audit_logger.log_model_event(
                event_type=event_type,
                model_name=model_name,
                version=version,
                user_id=user_id,
                ip_address=ip_address,
                metadata={
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "method": request.method
                }
            )
        except Exception as e:
            logger.error(f"Failed to log model load event: {e}")
    
    async def _log_model_unload(self, request: Request, response: Response, user_id: str, ip_address: str):
        """Log model unloading events"""
        try:
            body = await request.json() if hasattr(request, 'json') else {}
            model_name = body.get("model_name", "unknown")
            
            event_type = "unloaded" if response.status_code == 200 else "unload_failed"
            
            await self.audit_logger.log_model_event(
                event_type=event_type,
                model_name=model_name,
                user_id=user_id,
                ip_address=ip_address,
                metadata={
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "method": request.method
                }
            )
        except Exception as e:
            logger.error(f"Failed to log model unload event: {e}")
    
    async def _log_model_reload(self, request: Request, response: Response, user_id: str, ip_address: str):
        """Log model reload events"""
        try:
            model_name = request.path_params.get("model_name", "unknown")
            
            event_type = "reloaded" if response.status_code == 200 else "reload_failed"
            
            await self.audit_logger.log_model_event(
                event_type=event_type,
                model_name=model_name,
                user_id=user_id,
                ip_address=ip_address,
                metadata={
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "method": request.method
                }
            )
        except Exception as e:
            logger.error(f"Failed to log model reload event: {e}")
    
    async def _log_prediction(self, request: Request, response: Response, user_id: str, ip_address: str):
        """Log prediction events"""
        try:
            body = await request.json() if hasattr(request, 'json') else {}
            models = body.get("models", [])
            text_length = len(body.get("text", ""))
            
            event_type = "prediction_success" if response.status_code == 200 else "prediction_failed"
            
            await self.audit_logger.log_model_event(
                event_type=event_type,
                model_name=models[0] if models else "ensemble",
                user_id=user_id,
                ip_address=ip_address,
                metadata={
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "method": request.method,
                    "models_used": models,
                    "text_length": text_length,
                    "ensemble": body.get("ensemble", False)
                }
            )
        except Exception as e:
            logger.error(f"Failed to log prediction event: {e}")
    
    async def _log_model_management(self, request: Request, response: Response, user_id: str, ip_address: str):
        """Log model management events"""
        try:
            model_name = request.path_params.get("model_name", "unknown")
            operation = "updated" if request.method == "PUT" else "deleted"
            
            event_type = f"model_{operation}" if response.status_code == 200 else f"model_{operation}_failed"
            
            await self.audit_logger.log_model_event(
                event_type=event_type,
                model_name=model_name,
                user_id=user_id,
                ip_address=ip_address,
                metadata={
                    "endpoint": request.url.path,
                    "status_code": response.status_code,
                    "method": request.method
                }
            )
        except Exception as e:
            logger.error(f"Failed to log model management event: {e}")

# Global audit logger instance
audit_logger = AuditLogger()
