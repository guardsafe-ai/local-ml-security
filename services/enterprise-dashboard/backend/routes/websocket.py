"""
Enterprise Dashboard Backend - WebSocket Routes
Real-time communication endpoints
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from models.requests import WebSocketMessage
from models.responses import WebSocketResponse
from services.websocket_manager import manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            try:
                message_data = json.loads(data)
                message = WebSocketMessage(**message_data)
                
                # Handle different message types
                if message.type == "ping":
                    await manager.send_personal_message(
                        WebSocketResponse(
                            type="pong",
                            data={"timestamp": datetime.now().isoformat()},
                            timestamp=datetime.now()
                        ).model_dump(),
                        websocket
                    )
                
                elif message.type == "subscribe":
                    # Handle subscription to specific updates
                    await manager.send_personal_message(
                        WebSocketResponse(
                            type="subscribed",
                            data={"subscription": message.data.get("type", "all")},
                            timestamp=datetime.now()
                        ).model_dump(),
                        websocket
                    )
                
                elif message.type == "get_status":
                    # Send current system status
                    status_data = {
                        "active_connections": manager.get_connection_count(),
                        "timestamp": datetime.now().isoformat()
                    }
                    await manager.send_personal_message(
                        WebSocketResponse(
                            type="status",
                            data=status_data,
                            timestamp=datetime.now()
                        ).model_dump(),
                        websocket
                    )
                
                else:
                    # Echo back unknown message types
                    await manager.send_personal_message(
                        WebSocketResponse(
                            type="echo",
                            data={"original_message": message_data},
                            timestamp=datetime.now()
                        ).model_dump(),
                        websocket
                    )
                    
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    WebSocketResponse(
                        type="error",
                        data={"message": "Invalid JSON format"},
                        timestamp=datetime.now(),
                        success=False
                    ).model_dump(),
                    websocket
                )
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await manager.send_personal_message(
                    WebSocketResponse(
                        type="error",
                        data={"message": "Internal server error"},
                        timestamp=datetime.now(),
                        success=False
                    ).model_dump(),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.get("/ws/status")
async def get_websocket_status():
    """Get WebSocket connection status"""
    try:
        return {
            "active_connections": manager.get_connection_count(),
            "connections": manager.get_connection_info(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get WebSocket status: {e}")
        return {
            "active_connections": 0,
            "connections": [],
            "timestamp": datetime.now().isoformat()
        }
