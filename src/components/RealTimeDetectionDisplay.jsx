import React, { useState, useEffect, useRef } from 'react';
import { ShieldAlert, UserCheck, ShoppingBag, AlertTriangle } from 'lucide-react';
import * as websocketService from '../services/websocketService';

const RealTimeDetectionDisplay = ({ isActive, cameraId, detectionType }) => {
  const [detections, setDetections] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const inferenceSocketRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);

  // Connect to inference WebSocket when component becomes active
  useEffect(() => {
    if (isActive && cameraId) {
      try {
        // Connect to real-time inference
        inferenceSocketRef.current = websocketService.startRealTimeInference({
          camera_id: cameraId,
          detection_type: detectionType || 'all'
        });

        inferenceSocketRef.current.onopen = () => {
          setIsConnected(true);
          setError(null);
        };

        inferenceSocketRef.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'detection') {
              // Add new detection with timestamp
              const newDetection = {
                ...data.detection,
                timestamp: new Date().toISOString()
              };
              
              setDetections(prev => {
                // Keep only last 20 detections to avoid memory issues
                const updated = [newDetection, ...prev];
                return updated.slice(0, 20);
              });
              
              // Draw detection on canvas if it has bounding box
              if (data.detection.bbox && canvasRef.current) {
                drawDetection(data.detection);
              }
            }
          } catch (error) {
            console.error('Error processing inference message:', error);
          }
        };

        inferenceSocketRef.current.onerror = (error) => {
          console.error('Inference WebSocket error:', error);
          setError('Failed to connect to inference service');
          setIsConnected(false);
        };

        inferenceSocketRef.current.onclose = () => {
          setIsConnected(false);
        };

        // Start capturing video frames if we have video access
        if (videoRef.current) {
          startVideoCapture();
        }

        return () => {
          stopVideoCapture();
          if (inferenceSocketRef.current) {
            inferenceSocketRef.current.close();
            inferenceSocketRef.current = null;
          }
        };
      } catch (error) {
        console.error('Error setting up real-time detection:', error);
        setError(`Failed to start real-time detection: ${error.message}`);
      }
    }
  }, [isActive, cameraId, detectionType]);

  // Start capturing video frames for inference
  const startVideoCapture = () => {
    const captureFrame = () => {
      if (videoRef.current && canvasRef.current && inferenceSocketRef.current?.readyState === WebSocket.OPEN) {
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        const video = videoRef.current;
        
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw video frame to canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Extract image data and send for inference
        canvas.toBlob((blob) => {
          if (blob) {
            websocketService.sendFrameForInference(inferenceSocketRef.current, blob);
          }
        }, 'image/jpeg', 0.7); // Use JPEG with 70% quality for better performance
      }
      
      // Schedule next frame capture (throttle to 5 FPS for performance)
      animationFrameRef.current = setTimeout(() => {
        animationFrameRef.current = requestAnimationFrame(captureFrame);
      }, 200); // 200ms = ~5 FPS
    };
    
    captureFrame();
  };

  // Stop video frame capture
  const stopVideoCapture = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      clearTimeout(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  };

  // Draw detection bounding boxes on canvas
  const drawDetection = (detection) => {
    if (!canvasRef.current || !detection.bbox) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const [x, y, width, height] = detection.bbox;
    
    // Set style based on detection type
    let color;
    switch (detection.type || detection.class_name) {
      case 'theft':
        color = 'red';
        break;
      case 'loitering':
        color = 'orange';
        break;
      case 'person':
        color = 'green';
        break;
      default:
        color = 'blue';
    }
    
    // Draw bounding box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
    
    // Draw label
    ctx.fillStyle = color;
    ctx.font = '12px Arial';
    const label = `${detection.type || detection.class_name} ${Math.round((detection.confidence || 0) * 100)}%`;
    ctx.fillText(label, x, y > 20 ? y - 5 : y + 20);
    
    // Draw keypoints if available
    if (detection.keypoints) {
      ctx.fillStyle = 'red';
      detection.keypoints.forEach(([kpX, kpY]) => {
        if (kpX > 0 && kpY > 0) {
          ctx.beginPath();
          ctx.arc(kpX, kpY, 3, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    }
  };

  // Get icon for detection type
  const getDetectionIcon = (type) => {
    switch (type) {
      case 'theft':
        return <ShoppingBag className="text-red-500" size={16} />;
      case 'loitering':
        return <UserCheck className="text-orange-500" size={16} />;
      default:
        return <ShieldAlert className="text-blue-500" size={16} />;
    }
  };

  // Format detection time as relative
  const getRelativeTime = (timestamp) => {
    const seconds = Math.floor((new Date() - new Date(timestamp)) / 1000);
    
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex justify-between items-center mb-4">
        <h3 className="font-semibold text-lg flex items-center gap-2">
          <ShieldAlert className="text-blue-500" />
          Real-Time Detection
          {isConnected && (
            <span className="text-xs bg-green-100 text-green-600 px-2 py-0.5 rounded-full flex items-center">
              <span className="w-2 h-2 bg-green-500 rounded-full mr-1 animate-pulse"></span>
              Live
            </span>
          )}
        </h3>
        
        {error && (
          <div className="text-sm text-red-500 flex items-center gap-1">
            <AlertTriangle size={14} />
            {error}
          </div>
        )}
      </div>

      {/* Hidden video element to capture frames from */}
      <video 
        ref={videoRef}
        className="hidden"
        autoPlay
        playsInline
        muted
      >
        <source src={`/api/cameras/${cameraId}/stream`} type="video/mp4" />
      </video>

      {/* Canvas for drawing detections */}
      <div className="relative aspect-video bg-gray-900 rounded-lg overflow-hidden mb-4">
        <canvas
          ref={canvasRef}
          className="w-full h-full object-contain"
        />
        {!isConnected && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-70">
            <div className="text-white text-center">
              <AlertTriangle size={40} className="mx-auto mb-2" />
              <p className="font-semibold">Not Connected</p>
              <p className="text-sm">Waiting for video stream...</p>
            </div>
          </div>
        )}
      </div>

      {/* Detection list */}
      <div className="max-h-[200px] overflow-y-auto pr-1">
        <h4 className="font-medium text-sm mb-2 text-gray-700">Recent Detections</h4>
        
        {detections.length === 0 ? (
          <div className="text-center py-4 text-gray-500">
            <p className="text-sm">No detections yet</p>
          </div>
        ) : (
          <ul className="space-y-2">
            {detections.map((detection, index) => (
              <li key={index} className="flex items-center gap-2 border-b border-gray-100 pb-2">
                <div className="p-1 rounded-full bg-gray-100">
                  {getDetectionIcon(detection.type || detection.class_name)}
                </div>
                <div className="flex-1">
                  <div className="flex justify-between">
                    <p className="text-sm font-medium">
                      {detection.type || detection.class_name || "Object"}
                    </p>
                    <p className="text-xs text-gray-500">
                      {getRelativeTime(detection.timestamp)}
                    </p>
                  </div>
                  <p className="text-xs text-gray-600">
                    Confidence: {Math.round((detection.confidence || 0) * 100)}%
                  </p>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default RealTimeDetectionDisplay;
