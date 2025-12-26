"""
Evidence Suite - WebSocket Support
Real-time analysis progress updates.
"""
import asyncio
import json
from typing import Dict, Set, Optional
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from loguru import logger

from api.auth import get_current_user
from core.database import User


class ConnectionManager:
    """
    WebSocket connection manager for real-time updates.

    Features:
    - Per-evidence subscriptions
    - Per-case subscriptions
    - Broadcast to all connections
    - User-specific messages
    """

    def __init__(self):
        # Active connections: websocket -> user_id
        self.active_connections: Dict[WebSocket, Optional[str]] = {}

        # Subscriptions: evidence_id -> set of websockets
        self.evidence_subscriptions: Dict[str, Set[WebSocket]] = {}

        # Case subscriptions: case_id -> set of websockets
        self.case_subscriptions: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections[websocket] = user_id
        logger.info(f"WebSocket connected: {user_id or 'anonymous'}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        user_id = self.active_connections.pop(websocket, None)
        logger.info(f"WebSocket disconnected: {user_id or 'anonymous'}")

        # Remove from all subscriptions
        for subs in self.evidence_subscriptions.values():
            subs.discard(websocket)
        for subs in self.case_subscriptions.values():
            subs.discard(websocket)

    def subscribe_evidence(self, websocket: WebSocket, evidence_id: str):
        """Subscribe to evidence updates."""
        if evidence_id not in self.evidence_subscriptions:
            self.evidence_subscriptions[evidence_id] = set()
        self.evidence_subscriptions[evidence_id].add(websocket)
        logger.debug(f"Subscribed to evidence: {evidence_id}")

    def unsubscribe_evidence(self, websocket: WebSocket, evidence_id: str):
        """Unsubscribe from evidence updates."""
        if evidence_id in self.evidence_subscriptions:
            self.evidence_subscriptions[evidence_id].discard(websocket)

    def subscribe_case(self, websocket: WebSocket, case_id: str):
        """Subscribe to case updates."""
        if case_id not in self.case_subscriptions:
            self.case_subscriptions[case_id] = set()
        self.case_subscriptions[case_id].add(websocket)
        logger.debug(f"Subscribed to case: {case_id}")

    def unsubscribe_case(self, websocket: WebSocket, case_id: str):
        """Unsubscribe from case updates."""
        if case_id in self.case_subscriptions:
            self.case_subscriptions[case_id].discard(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connections."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)

    async def notify_evidence_update(
        self,
        evidence_id: str,
        event_type: str,
        data: dict
    ):
        """Notify all subscribers of evidence update."""
        if evidence_id not in self.evidence_subscriptions:
            return

        message = {
            "type": "evidence_update",
            "event": event_type,
            "evidence_id": evidence_id,
            "data": data
        }

        disconnected = []
        for websocket in self.evidence_subscriptions[evidence_id]:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)

        # Clean up disconnected
        for ws in disconnected:
            self.disconnect(ws)

    async def notify_case_update(
        self,
        case_id: str,
        event_type: str,
        data: dict
    ):
        """Notify all subscribers of case update."""
        if case_id not in self.case_subscriptions:
            return

        message = {
            "type": "case_update",
            "event": event_type,
            "case_id": case_id,
            "data": data
        }

        disconnected = []
        for websocket in self.case_subscriptions[case_id]:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)

        # Clean up
        for ws in disconnected:
            self.disconnect(ws)

    async def notify_analysis_progress(
        self,
        evidence_id: str,
        stage: str,
        progress: float,
        details: Optional[dict] = None
    ):
        """Notify subscribers of analysis progress."""
        await self.notify_evidence_update(
            evidence_id,
            "analysis_progress",
            {
                "stage": stage,
                "progress": progress,
                "details": details or {}
            }
        )

    async def notify_analysis_complete(
        self,
        evidence_id: str,
        result: dict
    ):
        """Notify subscribers that analysis is complete."""
        await self.notify_evidence_update(
            evidence_id,
            "analysis_complete",
            result
        )

    async def notify_analysis_error(
        self,
        evidence_id: str,
        error: str
    ):
        """Notify subscribers of analysis error."""
        await self.notify_evidence_update(
            evidence_id,
            "analysis_error",
            {"error": error}
        )


# Global connection manager
manager = ConnectionManager()


# Router
router = APIRouter(prefix="/ws", tags=["WebSocket"])


@router.websocket("/updates")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Message format (incoming):
    {
        "action": "subscribe_evidence" | "unsubscribe_evidence" |
                  "subscribe_case" | "unsubscribe_case" | "ping",
        "evidence_id": "...",  // for evidence actions
        "case_id": "..."       // for case actions
    }

    Message format (outgoing):
    {
        "type": "evidence_update" | "case_update" | "pong",
        "event": "analysis_progress" | "analysis_complete" | "analysis_error" | ...,
        "evidence_id": "...",
        "data": {...}
    }
    """
    await manager.connect(websocket)

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "ping":
                await manager.send_personal_message({"type": "pong"}, websocket)

            elif action == "subscribe_evidence":
                evidence_id = data.get("evidence_id")
                if evidence_id:
                    manager.subscribe_evidence(websocket, evidence_id)
                    await manager.send_personal_message({
                        "type": "subscribed",
                        "target": "evidence",
                        "id": evidence_id
                    }, websocket)

            elif action == "unsubscribe_evidence":
                evidence_id = data.get("evidence_id")
                if evidence_id:
                    manager.unsubscribe_evidence(websocket, evidence_id)

            elif action == "subscribe_case":
                case_id = data.get("case_id")
                if case_id:
                    manager.subscribe_case(websocket, case_id)
                    await manager.send_personal_message({
                        "type": "subscribed",
                        "target": "case",
                        "id": case_id
                    }, websocket)

            elif action == "unsubscribe_case":
                case_id = data.get("case_id")
                if case_id:
                    manager.unsubscribe_case(websocket, case_id)

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Helper function to get the manager
def get_ws_manager() -> ConnectionManager:
    """Get WebSocket connection manager."""
    return manager
