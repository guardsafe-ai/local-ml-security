"""
Idempotency Middleware
Prevents duplicate processing of identical requests using idempotency keys
"""

import hashlib
import json
import redis
import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

logger = logging.getLogger(__name__)

class IdempotencyMiddleware:
    """Prevent duplicate processing of identical requests"""
    
    def __init__(self, redis_client: redis.Redis, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl  # 1 hour default
    
    async def __call__(self, request: Request, call_next):
        # Only check POST/PUT/PATCH requests
        if request.method not in ["POST", "PUT", "PATCH"]:
            return await call_next(request)
        
        # Get idempotency key from header
        idempotency_key = request.headers.get("Idempotency-Key")
        
        if idempotency_key:
            # Check if we've seen this request before
            cache_key = f"idempotency:{idempotency_key}"
            cached_response = self.redis.get(cache_key)
            
            if cached_response:
                # Return cached response
                logger.info(f"Returning cached response for idempotency key: {idempotency_key}")
                return JSONResponse(
                    content=json.loads(cached_response),
                    status_code=200,
                    headers={"X-Idempotent-Replay": "true"}
                )
            
            # Process request
            response = await call_next(request)
            
            # Cache successful responses
            if 200 <= response.status_code < 300:
                try:
                    # Get response body
                    response_body = b""
                    async for chunk in response.body_iterator:
                        response_body += chunk
                    
                    # Cache the response
                    self.redis.setex(
                        cache_key,
                        self.ttl,
                        response_body.decode('utf-8')
                    )
                    
                    # Return response with body
                    return JSONResponse(
                        content=json.loads(response_body.decode('utf-8')),
                        status_code=response.status_code,
                        headers=dict(response.headers)
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache response for idempotency key {idempotency_key}: {e}")
                    return response
            
            return response
        
        return await call_next(request)
