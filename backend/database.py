from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Float, Boolean, JSON, DateTime, ForeignKey, Text, select
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import logging
from typing import List, Optional, Dict, Any

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
    
    # Existing fields
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"))
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    frame_number = Column(Integer, nullable=False)
    detection_type = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    bbox = Column(JSON)
    class_name = Column(String(50))
    image_path = Column(String(500))
    detection_metadata = Column(JSON, nullable=True)
    
    # Add this field to match your schema
    camera_id = Column(Integer, nullable=True)
    
    # Your other fields...
    gender = Column(String(20), nullable=True)
    age_group = Column(String(20), nullable=True)
    clothing_color = Column(String(30), nullable=True)

class Incident(Base):
    __tablename__ = "incidents"
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"))
    timestamp = Column(DateTime, nullable=False, default=datetime.now)
    location = Column(String(100))
    type = Column(String(50), nullable=False)  # loitering, theft, damage
    description = Column(String(500))
    image_path = Column(String(500))
    video_url = Column(String(500))
    severity = Column(String(20), default="medium")  # low, medium, high
    detection_ids = Column(JSON, nullable=True)  # List of detection IDs associated with this incident
    
    # New fields for incident tracking
    frame_number = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)  # Duration of incident in seconds
    confidence = Column(Float, nullable=True)  # Confidence of detection
    is_resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text, nullable=True)

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
    video_id = Column(Integer, ForeignKey("videos.id"))
    detection_id = Column(Integer, ForeignKey("detections.id"), nullable=True)
    image_url = Column(String(500))
    gender = Column(String(20))
    entry_time = Column(DateTime, nullable=False, default=datetime.now)
    entry_date = Column(String(20))
    age_group = Column(String(20))
    clothing_color = Column(String(30))
    notes = Column(String(500), nullable=True)
    face_encoding = Column(JSON, nullable=True)  # For face recognition
    is_repeat_customer = Column(Boolean, default=False)

class DemographicsSummary(Base):
    __tablename__ = "demographics_summary"
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"))
    date = Column(DateTime, nullable=False, default=datetime.now)
    male_count = Column(Integer, default=0)
    female_count = Column(Integer, default=0)
    unknown_gender_count = Column(Integer, default=0)
    
    # Age groups
    child_count = Column(Integer, default=0)
    young_count = Column(Integer, default=0)
    adult_count = Column(Integer, default=0)
    senior_count = Column(Integer, default=0)
    
    # Summary JSON for flexibility
    summary_data = Column(JSON, nullable=True)

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

async def get_detections(video_id: int = None) -> List[dict]:
    async with async_session() as session:
        if video_id:
            query = select(Detection).where(Detection.video_id == video_id)
        else:
            query = select(Detection)
        result = await session.execute(query)
        detections = result.scalars().all()
        return [detection.to_dict() for detection in detections]

async def get_incidents(recent: bool = False) -> List[dict]:
    async with async_session() as session:
        if recent:
            # Get incidents from last 24 hours
            query = select(Incident).where(
                Incident.timestamp >= datetime.now() - timedelta(days=1)
            ).order_by(Incident.timestamp.desc())
        else:
            query = select(Incident).order_by(Incident.timestamp.desc())
        result = await session.execute(query)
        incidents = result.scalars().all()
        return [incident.to_dict() for incident in incidents]

async def add_video(video_data: dict):
    async with async_session() as session:
        video = Video(**video_data)
        session.add(video)
        await session.commit()
        return video.id

async def add_detection(detection_data: dict) -> int:
    async with async_session() as session:
        detection = Detection(**detection_data)
        session.add(detection)
        await session.commit()
        return detection.id

async def add_incident(incident_data: dict) -> int:
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
                    entry_time=c.entry_time.isoformat() if c.entry_time else None,
                    entry_date=c.entry_date,
                    age_group=c.age_group, clothing_color=c.clothing_color,
                    notes=c.notes, is_repeat_customer=c.is_repeat_customer)
                for c in result.scalars()]

async def add_customer_data(customer_data: dict):
    """Add customer data to the database"""
    async with async_session() as session:
        customer = CustomerData(
            image_url=customer_data["image_url"],
            gender=customer_data["gender"],
            entry_time=customer_data["entry_time"],
            entry_date=customer_data["entry_date"],
            age_group=customer_data["age_group"],
            clothing_color=customer_data["clothing_color"],
            notes=customer_data.get("notes"),
            is_repeat_customer=customer_data.get("is_repeat_customer", False)
        )
        session.add(customer)
        await session.commit()
        return customer.id

async def add_demographics_summary(demographics_data: dict) -> int:
    async with async_session() as session:
        try:
            new_demographics = DemographicsSummary(**demographics_data)
            session.add(new_demographics)
            await session.commit()
            await session.refresh(new_demographics)
            logger.info(f"Added demographics summary with ID: {new_demographics.id}")
            return new_demographics.id
        except Exception as e:
            await session.rollback()
            logger.error(f"Error adding demographics summary: {str(e)}")
            raise

# Additional helper functions for database operations
async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

async def add_detections_bulk(detections: List[dict]):
    """Add multiple detections to the database in bulk"""
    try:
        async with async_session() as session:
            # Create Detection objects in bulk
            detection_objects = [
                Detection(
                    video_id=det["video_id"],
                    timestamp=det["timestamp"],
                    frame_number=det["frame_number"],
                    detection_type=det["detection_type"],
                    confidence=det["confidence"],
                    bbox=det["bbox"],
                    class_name=det["class_name"],
                    image_path=det["image_path"]
                )
                for det in detections
            ]
            
            # Bulk insert all detections
            session.add_all(detection_objects)
            await session.commit()
            
    except Exception as e:
        logger.error(f"Error in bulk detection insert: {str(e)}")
        raise 

async def update_incident(incident_id: int, update_data: dict):
    """Update an existing incident with new information"""
    async with async_session() as session:
        try:
            incident = await session.get(Incident, incident_id)
            if not incident:
                logger.error(f"Incident with ID {incident_id} not found")
                return False
                
            # Update incident fields
            for key, value in update_data.items():
                if hasattr(incident, key):
                    setattr(incident, key, value)
                    
            # Special handling for detection_ids if it exists
            if 'detection_ids' in update_data:
                # If the field already has detection IDs, append new ones
                if incident.detection_ids:
                    # Make sure we don't duplicate IDs
                    existing_ids = incident.detection_ids if isinstance(incident.detection_ids, list) else []
                    new_ids = update_data['detection_ids']
                    # Combine existing and new IDs without duplicates
                    combined_ids = list(set(existing_ids + new_ids))
                    incident.detection_ids = combined_ids
                else:
                    # No existing detection IDs, just set the new ones
                    incident.detection_ids = update_data['detection_ids']
            
            await session.commit()
            return True
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error updating incident: {str(e)}")
            return False
        

async def update_customer_face_encoding(customer_id: int, face_encoding, session: AsyncSession = None):
    """
    Update a customer's face encoding for future matching.
    
    Args:
        customer_id: ID of the customer to update
        face_encoding: Face encoding data (numpy array converted to list)
        session: Optional database session
    
    Returns:
        bool: True if successful, False otherwise
    """
    if session is None:
        from database import async_session, CustomerData
        async with async_session() as session:
            try:
                customer = await session.get(CustomerData, customer_id)
                if customer:
                    # Convert numpy array to list for JSON storage if needed
                    if hasattr(face_encoding, 'tolist'):
                        face_encoding = face_encoding.tolist()
                    
                    customer.face_encoding = face_encoding
                    await session.commit()
                    return True
                return False
            except Exception as e:
                await session.rollback()
                logger.error(f"Error updating customer face encoding: {str(e)}")
                return False
    else:
        try:
            from database import CustomerData
            customer = await session.get(CustomerData, customer_id)
            if customer:
                # Convert numpy array to list for JSON storage if needed
                if hasattr(face_encoding, 'tolist'):
                    face_encoding = face_encoding.tolist()
                
                customer.face_encoding = face_encoding
                await session.commit()
                return True
            return False
        except Exception as e:
            await session.rollback()
            logger.error(f"Error updating customer face encoding: {str(e)}")
            return False

# Add a function to find customers by face encoding similarity
async def find_customers_by_face(face_encoding, similarity_threshold=0.6, limit=10):
    """
    Find customers with similar face encodings.
    
    Args:
        face_encoding: Query face encoding
        similarity_threshold: Maximum distance for a match (lower is stricter)
        limit: Maximum number of results to return
    
    Returns:
        List of matching customers with similarity scores
    """
    import face_recognition
    import numpy as np
    from database import async_session, CustomerData, select
    
    async with async_session() as session:
        try:
            # Get all customers with face encodings
            query = select(CustomerData).where(CustomerData.face_encoding != None)
            result = await session.execute(query)
            customers = result.scalars().all()
            
            matches = []
            for customer in customers:
                try:
                    # Skip if no valid encoding
                    if not customer.face_encoding:
                        continue
                        
                    # Convert stored encoding back to numpy array
                    customer_encoding = np.array(customer.face_encoding)
                    
                    # Calculate face distance
                    distance = face_recognition.face_distance([customer_encoding], face_encoding)[0]
                    
                    # If distance is below threshold, consider it a match
                    if distance < similarity_threshold:
                        matches.append({
                            "customer_id": customer.id,
                            "image_url": customer.image_url,
                            "entry_time": customer.entry_time.isoformat() if customer.entry_time else None,
                            "entry_date": customer.entry_date,
                            "gender": customer.gender,
                            "age_group": customer.age_group,
                            "similarity": float(1.0 - distance)  # Convert to similarity score
                        })
                except Exception as e:
                    logger.warning(f"Error processing customer {customer.id}: {str(e)}")
                    continue
            
            # Sort by similarity (highest first) and limit results
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            return matches[:limit]
            
        except Exception as e:
            logger.error(f"Error searching customers by face: {str(e)}")
            return []