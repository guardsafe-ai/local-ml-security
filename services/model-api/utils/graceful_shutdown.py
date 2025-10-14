"""
Graceful Shutdown Handler
Handles graceful shutdown of the service with proper resource cleanup
"""

import signal
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class GracefulShutdown:
    """Handles graceful shutdown of the service"""
    
    def __init__(self, app):
        self.app = app
        self.shutdown_event = asyncio.Event()
        self.grace_period = 30  # 30 seconds grace period
        self.shutdown_started = False
    
    async def shutdown_handler(self, signum: int, frame):
        """Handle shutdown signals"""
        if self.shutdown_started:
            logger.warning(f"Shutdown already in progress, ignoring signal {signum}")
            return
        
        self.shutdown_started = True
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        
        # Stop accepting new requests
        logger.info("Stopping request acceptance...")
        
        # Wait for in-flight requests to complete
        logger.info(f"Waiting for in-flight requests to complete (max {self.grace_period}s)...")
        try:
            await asyncio.wait_for(self.shutdown_event.wait(), timeout=self.grace_period)
        except asyncio.TimeoutError:
            logger.warning(f"Grace period of {self.grace_period}s exceeded, forcing shutdown")
        
        # Close database connections
        if hasattr(self.app.state, 'db'):
            logger.info("Closing database connections...")
            try:
                await self.app.state.db.disconnect()
            except Exception as e:
                logger.error(f"Error closing database connections: {e}")
        
        # Close HTTP clients
        if hasattr(self.app.state, 'http_client'):
            logger.info("Closing HTTP clients...")
            try:
                await self.app.state.http_client.close()
            except Exception as e:
                logger.error(f"Error closing HTTP clients: {e}")
        
        # Close Redis connections
        if hasattr(self.app.state, 'redis'):
            logger.info("Closing Redis connections...")
            try:
                await self.app.state.redis.close()
            except Exception as e:
                logger.error(f"Error closing Redis connections: {e}")
        
        # Close model cache connections
        if hasattr(self.app.state, 'model_cache'):
            logger.info("Closing model cache connections...")
            try:
                await self.app.state.model_cache.close()
            except Exception as e:
                logger.error(f"Error closing model cache: {e}")
        
        logger.info("Shutdown complete")
        self.shutdown_event.set()
    
    def register_handlers(self):
        """Register signal handlers"""
        signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(self.shutdown_handler(s, f)))
        signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(self.shutdown_handler(s, f)))
        logger.info("Signal handlers registered")
    
    async def wait_for_shutdown(self):
        """Wait for shutdown to complete"""
        await self.shutdown_event.wait()