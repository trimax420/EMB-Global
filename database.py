from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime, ForeignKey, Text, select
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database URL with error checking
try:
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'security_dashboard')

    DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    logger.info(f"Connecting to database at {DB_HOST}:{DB_PORT}/{DB_NAME}")
except Exception as e:
    logger.error(f"Error configuring database URL: {str(e)}")
    raise

# Create async engine with error handling
try:
    engine = create_async_engine(DATABASE_URL, echo=True)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    Base = declarative_base()
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Error creating database engine: {str(e)}")
    raise

# Models
class Video(Base):
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    file_path = Column(String(500), nullable=False)
    status = Column(String(20), nullable=False)  # pending, processing, completed, failed
    upload_time = Column(DateTime, nullable=False, default=datetime.now)
    processing_time = Column(DateTime)
    duration = Column(Float)  # in seconds
    frame_count = Column(Integer)
    fps = Column(Integer)
    resolution = Column(String(50))  # e.g., "1920x1080"
    detection_type = Column(String(50))  # theft, loitering, face_detection
    processed_file_path = Column(String(500))
    thumbnail_path = Column(String(500))

class Detection(Base):
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    frame_number = Column(Integer, nullable=False)
    detection_type = Column(String(50), nullable=False)  # theft, loitering, face
    confidence = Column(Float, nullable=False)
    bbox = Column(JSON, nullable=False)  # [x1, y1, x2, y2]
    class_name = Column(String(50))  # for object detection
    image_path = Column(String(500))  # path to the frame image
    detection_metadata = Column(JSON)  # additional detection data

class Incident(Base):
    __tablename__ = "incidents"
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    location = Column(String(100), nullable=False)
    type = Column(String(50), nullable=False)
    description = Column(Text)
    image = Column(String(500))
    video_url = Column(String(500))
    severity = Column(String(20), nullable=False)
    detection_ids = Column(JSON)  # List of detection IDs that triggered this incident

class MallStructure(Base):
    __tablename__ = "mall_structure"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    crowd_density = Column(Integer, nullable=False)
    bounds = Column(JSON, nullable=False)
    fill_color = Column(String(20), nullable=False)

class BillingActivity(Base):
    __tablename__ = "billing_activity"
    
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String(50), unique=True, nullable=False)
    customer_id = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    products = Column(JSON, nullable=False)
    total_amount = Column(Float, nullable=False)
    status = Column(String(50), nullable=False)
    suspicious_activity = Column(Boolean, default=False)
    skipped_items = Column(JSON, default=list)

class CustomerData(Base):
    __tablename__ = "customer_data"
    
    id = Column(Integer, primary_key=True)
    image_url = Column(String(500))
    gender = Column(String(20), nullable=False)
    entry_time = Column(String(20), nullable=False)
    entry_date = Column(String(20), nullable=False)
    age_group = Column(String(20), nullable=False)
    clothing_color = Column(String(20))
    notes = Column(Text)

# Database operations
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with async_session() as session:
        # Initialize sample videos if empty
        result = await session.execute(select(Video))
        if not result.scalars().first():
            videos_data = [
                Video(
                    name="Sample Theft Video",
                    file_path="videos/sample_theft.mp4",
                    status="pending",
                    detection_type="theft",
                    upload_time=datetime.now()
                ),
                Video(
                    name="Sample Loitering Video",
                    file_path="videos/sample_loitering.mp4",
                    status="pending",
                    detection_type="loitering",
                    upload_time=datetime.now()
                ),
                Video(
                    name="Sample Face Detection Video",
                    file_path="videos/sample_faces.mp4",
                    status="pending",
                    detection_type="face_detection",
                    upload_time=datetime.now()
                )
            ]
            session.add_all(videos_data)
            await session.commit()

# CRUD Operations
async def get_all_videos():
    async with async_session() as session:
        result = await session.execute(select(Video))
        return [dict(id=v.id, name=v.name, status=v.status, 
                    upload_time=v.upload_time.isoformat(),
                    detection_type=v.detection_type,
                    processed_file_path=v.processed_file_path,
                    thumbnail_path=v.thumbnail_path)
                for v in result.scalars()]

async def get_video_detections(video_id: int):
    async with async_session() as session:
        query = select(Detection).where(Detection.video_id == video_id)
        result = await session.execute(query)
        return [dict(id=d.id, timestamp=d.timestamp.isoformat(),
                    frame_number=d.frame_number,
                    detection_type=d.detection_type,
                    confidence=d.confidence,
                    bbox=d.bbox,
                    class_name=d.class_name,
                    image_path=d.image_path,
                    detection_metadata=d.detection_metadata)
                for d in result.scalars()]

async def get_incidents(recent=False):
    async with async_session() as session:
        if recent:
            time_threshold = datetime.now() - timedelta(hours=24)
            query = select(Incident).where(Incident.timestamp >= time_threshold)
        else:
            query = select(Incident)
        result = await session.execute(query)
        return [dict(id=i.id, timestamp=i.timestamp.isoformat(),
                    location=i.location, type=i.type,
                    description=i.description, image=i.image,
                    video_url=i.video_url, severity=i.severity,
                    detection_ids=i.detection_ids)
                for i in result.scalars()]

async def add_video(video_data: dict):
    async with async_session() as session:
        video = Video(**video_data)
        session.add(video)
        await session.commit()
        return video.id

async def add_detection(detection_data: dict):
    async with async_session() as session:
        detection = Detection(**detection_data)
        session.add(detection)
        await session.commit()
        return detection.id

async def add_incident(incident_data: dict):
    async with async_session() as session:
        incident = Incident(**incident_data)
        session.add(incident)
        await session.commit()
        return incident.id

async def update_video_status(video_id: int, status: str, processed_path: str = None):
    async with async_session() as session:
        video = await session.get(Video, video_id)
        if video:
            video.status = status
            if processed_path:
                video.processed_file_path = processed_path
            if status == "completed":
                video.processing_time = datetime.now()
            await session.commit()
            return True
        return False

async def get_mall_structure():
    async with async_session() as session:
        result = await session.execute(select(MallStructure))
        return [dict(id=m.id, name=m.name, crowd_density=m.crowd_density,
                    bounds=m.bounds, fill_color=m.fill_color)
                for m in result.scalars()]

async def get_billing_activity(filter_type="all"):
    async with async_session() as session:
        if filter_type == "suspicious":
            query = select(BillingActivity).where(BillingActivity.suspicious_activity == True)
        elif filter_type == "skipped":
            query = select(BillingActivity).where(BillingActivity.skipped_items != '[]')
        else:
            query = select(BillingActivity)
        result = await session.execute(query)
        return [dict(id=b.id, transaction_id=b.transaction_id, customer_id=b.customer_id,
                    timestamp=b.timestamp.isoformat(), products=b.products,
                    total_amount=b.total_amount, status=b.status,
                    suspicious_activity=b.suspicious_activity, skipped_items=b.skipped_items)
                for b in result.scalars()]

async def get_customer_data(filters=None):
    async with async_session() as session:
        query = select(CustomerData)
        if filters:
            if filters.get("gender") and filters["gender"] != "all":
                query = query.where(CustomerData.gender == filters["gender"])
            if filters.get("date"):
                query = query.where(CustomerData.entry_date == filters["date"])
        result = await session.execute(query)
        return [dict(id=c.id, image_url=c.image_url, gender=c.gender,
                    entry_time=c.entry_time, entry_date=c.entry_date,
                    age_group=c.age_group, clothing_color=c.clothing_color,
                    notes=c.notes)
                for c in result.scalars()]

# Additional helper functions for database operations
async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close() 