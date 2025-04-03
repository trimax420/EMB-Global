import numpy as np
import cv2

def get_model(model_type):
    """
    Get a detection model by type.
    This is a simple mock implementation to use with WebRTC and other components.
    
    Args:
        model_type (str): Type of model to get (object, theft, loitering, etc.)
        
    Returns:
        dict: A mock model that can be used for testing
    """
    # Return a mock model with minimal properties
    return {
        "type": model_type,
        "name": f"{model_type}_model",
        "loaded": True,
        "mocked": True
    }

def process_frame(frame, model, detection_type='object', confidence_threshold=0.4):
    """
    Process a frame using the detection model.
    This is a simple mock implementation that generates fake detections.
    
    Args:
        frame (numpy.ndarray): Input frame to process
        model (dict): Model to use for detection
        detection_type (str): Type of detection to perform
        confidence_threshold (float): Minimum confidence threshold
        
    Returns:
        tuple: (processed_frame, detections)
    """
    if frame is None:
        return None, []
    
    # Make a copy of the frame to draw on
    processed_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Generate mock detections based on detection type
    detections = []
    
    if detection_type == 'object':
        # Add a person detection with random position
        x = int(width * 0.3)
        y = int(height * 0.3)
        w = int(width * 0.2)
        h = int(height * 0.4)
        
        confidence = 0.85
        class_name = 'person'
        
        # Draw bounding box
        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (50, 100, 200), 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(processed_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 100, 200), 2)
        
        detections.append({
            'type': 'object',
            'class_name': class_name,
            'confidence': confidence,
            'bbox': [x, y, x + w, y + h]
        })
    
    elif detection_type == 'theft':
        # Add a theft detection
        x = int(width * 0.6)
        y = int(height * 0.4)
        w = int(width * 0.2)
        h = int(height * 0.3)
        
        confidence = 0.75
        
        # Draw bounding box with red color for theft
        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 200), 2)
        label = f"theft {confidence:.2f}"
        cv2.putText(processed_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
        
        detections.append({
            'type': 'theft',
            'confidence': confidence,
            'bbox': [x, y, x + w, y + h],
            'zone': 'chest'
        })
    
    elif detection_type == 'loitering':
        # Add a loitering detection
        x = int(width * 0.5)
        y = int(height * 0.5)
        w = int(width * 0.25)
        h = int(height * 0.4)
        
        confidence = 0.80
        time_present = 15.5  # seconds
        
        # Draw bounding box with amber color for loitering
        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        label = f"loitering {confidence:.2f} ({time_present}s)"
        cv2.putText(processed_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        detections.append({
            'type': 'loitering',
            'confidence': confidence,
            'bbox': [x, y, x + w, y + h],
            'time_present': time_present
        })
    
    return processed_frame, detections

def initialize_model(model_path, model_type='object'):
    """
    Initialize a detection model.
    This is a mock implementation for testing.
    
    Args:
        model_path (str): Path to the model file
        model_type (str): Type of model to initialize
        
    Returns:
        dict: Initialized model
    """
    return get_model(model_type) 