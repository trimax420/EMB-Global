from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from app.core.database import get_db
from app.schemas.analytics import ZoneAnalytics, AggregateAnalytics
from app.models.analytics import ZoneAnalytics as ZoneAnalyticsModel
from app.models.analytics import AnalyticsAggregates

router = APIRouter()

@router.get("/zones/", response_model=List[ZoneAnalytics])
async def get_zone_analytics(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    zone_id: Optional[int] = None,
    time_period: str = "hourly",
    db: Session = Depends(get_db)
):
    """
    Get zone analytics with optional filtering
    """
    query = db.query(ZoneAnalyticsModel)
    
    if start_time:
        query = query.filter(ZoneAnalyticsModel.timestamp >= start_time)
    if end_time:
        query = query.filter(ZoneAnalyticsModel.timestamp <= end_time)
    if zone_id:
        query = query.filter(ZoneAnalyticsModel.zone_id == zone_id)
    
    query = query.filter(ZoneAnalyticsModel.time_period == time_period)
    
    analytics = query.order_by(ZoneAnalyticsModel.timestamp.desc()).all()
    return analytics

@router.get("/heat-map/", response_model=List[dict])
async def get_heat_map_data(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    hour: Optional[int] = Query(None, description="Hour (0-23) for hourly heat map"),
    db: Session = Depends(get_db)
):
    """
    Get heat map data for visualization
    """
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    query = db.query(
        ZoneAnalyticsModel.zone_id,
        ZoneAnalyticsModel.heat_level
    )
    
    query = query.filter(ZoneAnalyticsModel.timestamp >= target_date)
    query = query.filter(ZoneAnalyticsModel.timestamp < target_date + timedelta(days=1))
    
    if hour is not None:
        if not (0 <= hour <= 23):
            raise HTTPException(status_code=400, detail="Hour must be between 0 and 23")
        
        start_time = datetime.combine(target_date, datetime.min.time().replace(hour=hour))
        end_time = start_time + timedelta(hours=1)
        
        query = query.filter(ZoneAnalyticsModel.timestamp >= start_time)
        query = query.filter(ZoneAnalyticsModel.timestamp < end_time)
    
    # Group by zone_id and get the average heat level
    results = query.all()
    
    # Convert to list of dictionaries
    heat_map_data = [
        {"zone_id": zone_id, "heat_level": heat_level}
        for zone_id, heat_level in results
    ]
    
    return heat_map_data

@router.get("/aggregates/", response_model=List[AggregateAnalytics])
async def get_aggregate_analytics(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    aggregation: str = Query("daily", description="Aggregation level: 'daily' or 'hourly'"),
    zone_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Get aggregated analytics for the dashboard
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    query = db.query(AnalyticsAggregates)
    query = query.filter(AnalyticsAggregates.date >= start)
    query = query.filter(AnalyticsAggregates.date <= end)
    
    if aggregation == "hourly":
        query = query.filter(AnalyticsAggregates.hour.isnot(None))
    else:
        query = query.filter(AnalyticsAggregates.hour.is_(None))
    
    if zone_id:
        query = query.filter(AnalyticsAggregates.zone_id == zone_id)
    
    results = query.order_by(AnalyticsAggregates.date, AnalyticsAggregates.hour).all()
    return results