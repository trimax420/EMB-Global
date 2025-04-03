import React, { useEffect, useRef, useState } from 'react';
import websocketService from '../services/websocketService';

const RealTimeDetection = ({ detectionType, isActive }) => {
  const videoRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [detections, setDetections] = useState([]);

  useEffect(() => {
    if (isActive) {
      // Connect to WebSocket when component becomes active
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/api/ws/detection/${detectionType}`;
      websocketService.connect(wsUrl);
      
      const onConnect = () => setIsConnected(true);
      const onDisconnect = () => setIsConnected(false);
      const onMessage = (data) => {
        if (data.type === 'detection') {
          setDetections(data.detections || []);
        } else if (data.type === 'frame') {
          // If the server sends image frames, display them
          if (videoRef.current && data.imageData) {
            // Handle video frame updates
            updateVideoFrame(data.imageData);
          }
        }
      };

      websocketService.on('connect', onConnect);
      websocketService.on('disconnect', onDisconnect);
      websocketService.on('message', onMessage);

      return () => {
        websocketService.off('connect', onConnect);
        websocketService.off('disconnect', onDisconnect);
        websocketService.off('message', onMessage);
        websocketService.disconnect();
      };
    }
  }, [isActive, detectionType]);

  // Function to update video frame with new image data
  const updateVideoFrame = (imageData) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      
      // Draw detection boxes
      drawDetectionBoxes(ctx, detections);
      
      // Update the video display
      if (videoRef.current) {
        videoRef.current.srcObject = canvas.captureStream();
      }
    };
    img.src = `data:image/jpeg;base64,${imageData}`;
  };

  // Function to draw detection boxes
  const drawDetectionBoxes = (ctx, detections) => {
    ctx.lineWidth = 2;
    ctx.font = '16px Arial';
    ctx.textBaseline = 'bottom';
    
    detections.forEach(detection => {
      const { x, y, width, height, confidence, label } = detection;
      
      // Determine color based on detection type
      ctx.strokeStyle = detectionType === 'theft' ? '#FF0000' : '#FFFF00';
      ctx.fillStyle = detectionType === 'theft' ? '#FF0000' : '#FFFF00';
      
      // Draw rectangle
      ctx.strokeRect(x, y, width, height);
      
      // Draw label with confidence
      const text = `${label}: ${Math.round(confidence * 100)}%`;
      ctx.fillText(text, x, y);
    });
  };

  return (
    <div className="real-time-detection">
      <div className="status-indicator">
        {isConnected ? (
          <span className="connected">Connected - Showing real-time {detectionType} detection</span>
        ) : (
          <span className="disconnected">Disconnected - Trying to connect...</span>
        )}
      </div>
      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="detection-video"
        />
      </div>
    </div>
  );
};

export default RealTimeDetection;
