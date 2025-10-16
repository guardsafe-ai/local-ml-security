"""
Enterprise Dashboard Backend - WebSocket Routes
Real-time communication for dashboard updates
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.dashboard_service import DashboardService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize dashboard service
dashboard_service = DashboardService()

# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = set()
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    def subscribe(self, websocket: WebSocket, topic: str):
        if websocket in self.subscriptions:
            self.subscriptions[websocket].add(topic)
            logger.info(f"Subscribed to {topic}")

    def unsubscribe(self, websocket: WebSocket, topic: str):
        if websocket in self.subscriptions:
            self.subscriptions[websocket].discard(topic)
            logger.info(f"Unsubscribed from {topic}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str, topic: str = None):
        """Broadcast message to all connections or specific topic subscribers"""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                # If topic specified, only send to subscribers
                if topic and connection in self.subscriptions:
                    if topic not in self.subscriptions[connection]:
                        continue
                
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)

        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_metrics_update(self):
        """Broadcast real-time metrics update"""
        try:
            metrics = await dashboard_service.get_dashboard_metrics()
            health_status = await dashboard_service.api_client.get_all_services_health()
            
            # Convert datetime objects to ISO format for JSON serialization
            serialized_health_status = []
            for service in health_status:
                serialized_service = service.copy()
                if 'last_check' in serialized_service and hasattr(serialized_service['last_check'], 'isoformat'):
                    serialized_service['last_check'] = serialized_service['last_check'].isoformat()
                serialized_health_status.append(serialized_service)
            
            message = {
                "type": "metrics_update",
                "data": {
                    "metrics": {
                        "total_models": metrics.total_models,
                        "active_jobs": metrics.active_jobs,
                        "total_attacks": metrics.total_attacks,
                        "detection_rate": metrics.detection_rate,
                        "system_health": metrics.system_health,
                        "last_updated": metrics.last_updated.isoformat()
                    },
                    "services": serialized_health_status,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            await self.broadcast(json.dumps(message), "metrics")
        except Exception as e:
            logger.error(f"Error broadcasting metrics update: {e}")

    async def broadcast_system_alert(self, alert_type: str, message: str, severity: str = "info"):
        """Broadcast system alert"""
        alert_message = {
            "type": "alert",
            "data": {
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        await self.broadcast(json.dumps(alert_message), "alerts")

# Global connection manager
manager = ConnectionManager()
_broadcast_task_started = False

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time dashboard updates"""
    global _broadcast_task_started
    
    # Start background task on first connection
    if not _broadcast_task_started:
        asyncio.create_task(periodic_metrics_broadcast())
        _broadcast_task_started = True
        logger.info("Started periodic metrics broadcast task")
    
    await manager.connect(websocket)
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                topic = message.get("topic", "all")
                manager.subscribe(websocket, topic)
                
                # Send confirmation
                await manager.send_personal_message(
                    json.dumps({
                        "type": "subscription_confirmed",
                        "topic": topic,
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
                
            elif message.get("type") == "unsubscribe":
                topic = message.get("topic", "all")
                manager.unsubscribe(websocket, topic)
                
            elif message.get("type") == "ping":
                # Respond to ping with pong
                await manager.send_personal_message(
                    json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Background task for periodic updates
async def periodic_metrics_broadcast():
    """Background task to broadcast metrics every 5 seconds"""
    while True:
        try:
            await manager.broadcast_metrics_update()
            await asyncio.sleep(5)  # Broadcast every 5 seconds
        except Exception as e:
            logger.error(f"Error in periodic metrics broadcast: {e}")
            await asyncio.sleep(5)

# Background task will be started when the first WebSocket connection is made