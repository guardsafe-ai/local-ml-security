"""
Enterprise Dashboard Backend - WebSocket Manager
Handles WebSocket connections and real-time communication
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from models.requests import WebSocketMessage
from models.responses import WebSocketResponse

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasting"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Store connection metadata
        self.connection_metadata[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": datetime.now(),
            "last_activity": datetime.now()
        }
        
        logger.info(f"WebSocket connected: {self.connection_metadata[websocket]['client_id']}")
        
        # Send welcome message
        await self.send_personal_message(
            WebSocketResponse(
                type="connection_established",
                data={"message": "Connected to ML Security Dashboard", "client_id": self.connection_metadata[websocket]['client_id']},
                timestamp=datetime.now()
            ).model_dump(),
            websocket
        )

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            client_id = self.connection_metadata.get(websocket, {}).get("client_id", "unknown")
            self.active_connections.remove(websocket)
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]
            logger.info(f"WebSocket disconnected: {client_id}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            if websocket in self.active_connections:
                await websocket.send_text(json.dumps(message, default=str))
                # Update last activity
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["last_activity"] = datetime.now()
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            # Remove failed connection
            self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        if not self.active_connections:
            return
        
        # Create a copy of connections to avoid modification during iteration
        connections_to_remove = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, default=str))
                # Update last activity
                if connection in self.connection_metadata:
                    self.connection_metadata[connection]["last_activity"] = datetime.now()
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")
                connections_to_remove.append(connection)
        
        # Remove failed connections
        for connection in connections_to_remove:
            self.disconnect(connection)

    async def broadcast_to_type(self, message: Dict[str, Any], message_type: str):
        """Broadcast a specific type of message"""
        message["type"] = message_type
        await self.broadcast(message)

    async def send_system_update(self, update_type: str, data: Dict[str, Any]):
        """Send system update to all clients"""
        message = WebSocketResponse(
            type="system_update",
            data={
                "update_type": update_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            },
            timestamp=datetime.now()
        ).model_dump()
        
        await self.broadcast(message)

    async def send_training_update(self, job_id: str, status: str, progress: float = None, message: str = None):
        """Send training job update"""
        data = {
            "job_id": job_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        if progress is not None:
            data["progress"] = progress
        
        if message:
            data["message"] = message
        
        await self.send_system_update("training_update", data)

    async def send_red_team_update(self, test_id: str, status: str, results: Dict[str, Any] = None):
        """Send red team test update"""
        data = {
            "test_id": test_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        if results:
            data["results"] = results
        
        await self.send_system_update("red_team_update", data)

    async def send_model_update(self, model_name: str, action: str, status: str, details: Dict[str, Any] = None):
        """Send model operation update"""
        data = {
            "model_name": model_name,
            "action": action,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            data["details"] = details
        
        await self.send_system_update("model_update", data)

    async def send_alert(self, alert_type: str, message: str, severity: str = "info", service: str = None):
        """Send system alert"""
        data = {
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        
        if service:
            data["service"] = service
        
        await self.send_system_update("alert", data)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)

    def get_connection_info(self) -> List[Dict[str, Any]]:
        """Get information about active connections"""
        connections = []
        for websocket, metadata in self.connection_metadata.items():
            connections.append({
                "client_id": metadata["client_id"],
                "connected_at": metadata["connected_at"].isoformat(),
                "last_activity": metadata["last_activity"].isoformat()
            })
        return connections

    async def cleanup_inactive_connections(self, max_inactivity_minutes: int = 30):
        """Clean up inactive connections"""
        current_time = datetime.now()
        inactive_connections = []
        
        for websocket, metadata in self.connection_metadata.items():
            last_activity = metadata["last_activity"]
            inactivity_minutes = (current_time - last_activity).total_seconds() / 60
            
            if inactivity_minutes > max_inactivity_minutes:
                inactive_connections.append(websocket)
        
        for connection in inactive_connections:
            logger.info(f"Cleaning up inactive connection: {self.connection_metadata[connection]['client_id']}")
            self.disconnect(connection)


# Global connection manager instance
manager = ConnectionManager()
