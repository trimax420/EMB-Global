from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    """Bounding box model for object detections"""
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")
    
    def to_list(self) -> List[float]:
        """Convert to [x1, y1, x2, y2] list format"""
        return [self.x1, self.y1, self.x2, self.y2]
    
    @classmethod
    def from_list(cls, bbox_list: List[float]) -> "BoundingBox":
        """Create from [x1, y1, x2, y2] list format"""
        if len(bbox_list) != 4:
            raise ValueError("Bounding box list must have 4 values")
        return cls(
            x1=bbox_list[0],
            y1=bbox_list[1],
            x2=bbox_list[2],
            y2=bbox_list[3]
        )
    
    @property
    def width(self) -> float:
        """Get width of bounding box"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Get height of bounding box"""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """Get area of bounding box"""
        return self.width * self.height
    
    @property
    def center(self) -> tuple:
        """Get center point (x, y) of bounding box"""
        return (
            self.x1 + (self.width / 2),
            self.y1 + (self.height / 2)
        ) 