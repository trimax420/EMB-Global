from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Optional
import json
from datetime import datetime, timedelta

from app.core.database import get_db
from app.schemas.alert import Alert, AlertUpdate
from app.models.alert import Alert as AlertModel

router = APIRouter()

# In-memory store for websocket connections
connected_clients = set()

@router.get("/", response_model=List[Alert])
async def get_alerts(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    alert_type: Optional[str] = None,
    severity: Optional[int] = None,
    acknowledged: Optional[bool] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get alerts with optional filtering
    """
    query = db.query(AlertModel)
    
    if start_time:
        query = query.filter(AlertModel.timestamp >= start_time)
    if end_time:
        query = query.filter(AlertModel.timestamp <= end_time)
    if alert_type:
        query = query.filter(AlertModel.alert_type == alert_type)
    if severity is not None:
        query = query.filter(AlertModel.severity >= severity)
    if acknowledged is not None:
        query = query.filter(AlertModel.acknowledged == acknowledged)
    
    alerts = query.order_by(AlertModel.timestamp.desc()).limit(limit).all()
    return alerts

@router.put("/{alert_id}", response_model=Alert)
async def update_alert(
    alert_id: int,
    alert_update: AlertUpdate,
    db: Session = Depends(get_db)
):
    """
    Update alert status (e.g., acknowledge alert)
    """
    alert = db.query(AlertModel).filter(AlertModel.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    # Update alert fields
    if alert_update.acknowledged is not None:
        alert.acknowledged = alert_update.acknowledged
        if alert_update.acknowledged:
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = alert_update.acknowledged_by
    
    db.commit()
    db.refresh(alert)
    return alert

@router.get("/stats/", response_model=dict)
async def get_alert_stats(
    days: int = Query(7, description="Number of days to include in stats"),
    db: Session = Depends(get_db)
):
    """
    Get alert statistics for the dashboard
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get total counts by type
    query = db.query(
        AlertModel.alert_type,
        db.func.count(AlertModel.id)
    ).filter(
        AlertModel.timestamp >= start_date,
        AlertModel.timestamp <= end_date
    ).group_by(
        AlertModel.alert_type
    )
    
    type_counts = dict(query.all())
    
    # Get counts by severity
    query = db.query(
        AlertModel.severity,
        db.func.count(AlertModel.id)
    ).filter(
        AlertModel.timestamp >= start_date,
        AlertModel.timestamp <= end_date
    ).group_by(
        AlertModel.severity
    )
    
    severity_counts = dict(query.all())
    
    # Get daily counts
    daily_query = db.query(
        db.func.date_trunc('day', AlertModel.timestamp).label('day'),
        db.func.count(AlertModel.id)
    ).filter(
        AlertModel.timestamp >= start_date,
        AlertModel.timestamp <= end_date
    ).group_by('day').order_by('day')
    
    daily_counts = [
        {"date": day.strftime("%Y-%m-%d"), "count": count}
        for day, count in daily_query.all()
    ]
    
    return {
        "by_type": type_counts,
        "by_severity": severity_counts,
        "daily_counts": daily_counts,
        "total": sum(type_counts.values())
    }

@router.websocket("/ws")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alerts
    """
    await websocket.accept()
    connected_clients.add(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

# Function to broadcast new alerts to all connected clients
async def broadcast_alert(alert: dict):
    """Broadcast alert to all connected WebSocket clients"""
    for client in connected_clients:
        try:
            await client.send_json(alert)
        except:
            connected_clients.remove(client)