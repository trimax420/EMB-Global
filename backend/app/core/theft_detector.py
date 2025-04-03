import cv2
import numpy as np

def process_frame_for_theft(frame, context=None, tracking_context=None):
    """
    Process a frame for theft detection.
    This is a stub implementation for testing.
    
    Args:
        frame (numpy.ndarray): The input frame
        context (dict): Detection context with parameters
        tracking_context (dict): Tracking context for tracking objects across frames
        
    Returns:
        tuple: (processed_frame, detections, updated_tracking_context)
    """
    if frame is None:
        return None, [], tracking_context
    
    # Make a copy of the frame to draw on
    processed_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Create an empty tracking context if none provided
    if tracking_context is None:
        tracking_context = {
            'hands': {},
            'faces': {},
            'objects': {},
            'suspects': {},
            'frame_count': 0
        }
    
    # Increment frame count
    tracking_context['frame_count'] = tracking_context.get('frame_count', 0) + 1
    
    # Generate a mock theft detection on every 50th frame
    detections = []
    if tracking_context['frame_count'] % 50 == 0:
        # Add a theft detection
        x = int(width * 0.6)
        y = int(height * 0.3)
        w = int(width * 0.2)
        h = int(height * 0.4)
        
        confidence = 0.75
        
        # Draw bounding box with red color for theft
        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 200), 2)
        label = f"theft {confidence:.2f}"
        cv2.putText(processed_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
        
        # Add to detections
        detections.append({
            'type': 'theft',
            'confidence': confidence,
            'bbox': [x, y, x + w, y + h],
            'zone': 'chest',
            'frame': tracking_context['frame_count']
        })
        
        # Add to suspects in tracking context
        suspect_id = f"suspect_{len(tracking_context['suspects']) + 1}"
        tracking_context['suspects'][suspect_id] = {
            'first_detected': tracking_context['frame_count'],
            'last_detected': tracking_context['frame_count'],
            'detections': [detections[-1]]
        }
    
    return processed_frame, detections, tracking_context 