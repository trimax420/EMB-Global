import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, 
  ShieldAlert, 
  UserCheck, 
  Eye, 
  Clock, 
  RefreshCw, 
  ChevronDown, 
  ShoppingBag,
  User,
  Zap,
  AlertTriangle
} from 'lucide-react';
import axios from 'axios';
import { useApi } from '../contexts/ApiContext';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

const LiveCameraComponent = ({ selectedCamera, onUpdateStats }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingType, setProcessingType] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [stats, setStats] = useState({
    people: 0,
    loitering: 0,
    theft: 0,
    objects: 0
  });
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [socketConnected, setSocketConnected] = useState(false);
  const socketRef = useRef(null);
  const resultsInterval = useRef(null);
  const videoRef = useRef(null);
  
  // Get the websocket service from API context
  const { websocket } = useApi?.() || {};

  // Connect to WebSocket for real-time updates
  useEffect(() => {
    if (!websocket?.isConnected) {
      // Set up a new connection if not available through context
      const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/ws';
      const socket = new WebSocket(wsUrl);
      
      socket.onopen = () => {
        console.log('WebSocket connected');
        setSocketConnected(true);
        socketRef.current = socket;
      };
      
      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Handle different message types from server
          if (data.type === 'processing_progress') {
            setProgress(data.progress);
          } else if (data.type === 'detection') {
            setDetections(prev => [...prev, data.detection]);
            updateDetectionStats(data.detection);
          } else if (data.type === 'processing_completed') {
            setIsProcessing(false);
            setProgress(100);
          } else if (data.type === 'processing_error') {
            setError(data.error);
            setIsProcessing(false);
          }
        } catch (error) {
          console.error('WebSocket message error:', error);
        }
      };
      
      socket.onclose = () => {
        console.log('WebSocket disconnected');
        setSocketConnected(false);
      };
      
      return () => {
        if (socketRef.current) {
          socketRef.current.close();
        }
      };
    } else {
      // Use the websocket from context
      setSocketConnected(true);
      
      // Set up listeners
      websocket.addMessageListener('processing_progress', (data) => {
        setProgress(data.progress);
      });
      
      websocket.addMessageListener('detection', (data) => {
        setDetections(prev => [...prev, data.detection]);
        updateDetectionStats(data.detection);
      });
      
      websocket.addMessageListener('processing_completed', () => {
        setIsProcessing(false);
        setProgress(100);
      });
      
      websocket.addMessageListener('processing_error', (data) => {
        setError(data.error);
        setIsProcessing(false);
      });
      
      return () => {
        websocket.removeMessageListener('processing_progress');
        websocket.removeMessageListener('detection');
        websocket.removeMessageListener('processing_completed');
        websocket.removeMessageListener('processing_error');
      };
    }
  }, [websocket]);

  // Polling for tracking results if a job is in progress
  useEffect(() => {
    if (jobId && isProcessing) {
      resultsInterval.current = setInterval(async () => {
        try {
          const response = await axios.get(`${API_BASE_URL}/face-tracking/tracking-results/${jobId}`);
          
          if (response.data.status === 'completed') {
            setIsProcessing(false);
            clearInterval(resultsInterval.current);
          }
          
          if (response.data.detections && response.data.detections.length > 0) {
            setDetections(response.data.detections);
            
            // Update stats based on detections
            const newStats = { ...stats };
            newStats.people = response.data.detections.length;
            setStats(newStats);
            
            // Notify parent component
            if (onUpdateStats) {
              onUpdateStats(newStats);
            }
          }
          
          setProgress(response.data.progress);
        } catch (error) {
          console.error('Error fetching tracking results:', error);
        }
      }, 1000);
      
      return () => {
        clearInterval(resultsInterval.current);
      };
    }
  }, [jobId, isProcessing]);

  // Update stats based on detection type
  const updateDetectionStats = (detection) => {
    const newStats = { ...stats };
    
    if (detection.type === 'person' || detection.class_name === 'person') {
      newStats.people += 1;
    } else if (detection.type === 'loitering') {
      newStats.loitering += 1;
    } else if (detection.type === 'theft') {
      newStats.theft += 1;
    } else {
      newStats.objects += 1;
    }
    
    setStats(newStats);
    
    // Notify parent component
    if (onUpdateStats) {
      onUpdateStats(newStats);
    }
  };

  // Start video processing with the selected type
  const startProcessing = async (type) => {
    try {
      setIsProcessing(true);
      setProcessingType(type);
      setDetections([]);
      setProgress(0);
      setError(null);
      
      let response;
      
      // For local video files, we'll need to first ensure the video is accessible to the backend
      // We'll assume the videos are in a directory that's accessible by both frontend and backend
      // In a real deployment, you might need to upload the video first
      
      const videoPath = selectedCamera.videoUrl;
      
      switch (type) {
        case 'loitering':
          response = await axios.post(`${API_BASE_URL}/videos/loitering-detection`, null, {
            params: {
              video_path: videoPath,
              threshold_time: 10 // in seconds, adjust as needed
            }
          });
          break;
          
        case 'theft':
          response = await axios.post(`${API_BASE_URL}/videos/theft-detection`, null, {
            params: {
              video_path: videoPath,
              hand_stay_time_chest: 1.0,
              hand_stay_time_waist: 1.5
            }
          });
          break;
          
        case 'face_tracking':
          // You would need to provide a face image for tracking
          // This implementation would require a UI component for face selection
          alert('Face tracking requires a reference face image. Not implemented in this demo.');
          setIsProcessing(false);
          return;
          
        default:
          throw new Error('Invalid processing type');
      }
      
      console.log(`Started ${type} detection:`, response.data);
      
      // Set the job ID if provided by the response
      if (response.data && response.data.job_id) {
        setJobId(response.data.job_id);
      }
      
    } catch (error) {
      console.error(`Error starting ${type} detection:`, error);
      setError(error.response?.data?.detail || error.message);
      setIsProcessing(false);
    }
  };

  // Toggle fullscreen mode
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  // Create a video player that restarts when the camera changes
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.load(); // Reload the video when source changes
    }
  }, [selectedCamera]);

  // Render detection boxes
  const renderDetections = () => {
    return detections.map((detection, index) => {
      // Calculate position based on bbox if available
      if (!detection.bbox) return null;
      
      const [x1, y1, x2, y2] = detection.bbox;
      
      // Get video dimensions for position calculation
      const videoElement = videoRef.current;
      const videoWidth = videoElement?.videoWidth || 640;
      const videoHeight = videoElement?.videoHeight || 480;
      
      const style = {
        position: 'absolute',
        left: `${(x1 / videoWidth) * 100}%`,
        top: `${(y1 / videoHeight) * 100}%`,
        width: `${((x2 - x1) / videoWidth) * 100}%`,
        height: `${((y2 - y1) / videoHeight) * 100}%`,
        border: detection.type === 'theft' ? '2px solid red' : 
                detection.type === 'loitering' ? '2px solid orange' : '2px solid green',
        boxSizing: 'border-box',
        zIndex: 10
      };
      
      return (
        <div key={index} style={style}>
          <div style={{
            backgroundColor: detection.type === 'theft' ? 'rgba(255, 0, 0, 0.7)' : 
                             detection.type === 'loitering' ? 'rgba(255, 165, 0, 0.7)' : 'rgba(0, 128, 0, 0.7)',
            color: 'white',
            padding: '2px 4px',
            fontSize: '10px',
            position: 'absolute',
            top: '0',
            left: '0',
            transform: 'translateY(-100%)'
          }}>
            {detection.type || detection.class_name || 'Object'} {detection.confidence ? `${Math.round(detection.confidence * 100)}%` : ''}
          </div>
        </div>
      );
    });
  };

  // Render keypoints for pose detection
  const renderKeypoints = () => {
    return detections.map((detection, index) => {
      if (!detection.keypoints) return null;
      
      // Get video dimensions for position calculation
      const videoElement = videoRef.current;
      const videoWidth = videoElement?.videoWidth || 640;
      const videoHeight = videoElement?.videoHeight || 480;
      
      return detection.keypoints.map((keypoint, kpIndex) => {
        if (keypoint[0] <= 0 || keypoint[1] <= 0) return null;
        
        const style = {
          position: 'absolute',
          left: `${(keypoint[0] / videoWidth) * 100}%`,
          top: `${(keypoint[1] / videoHeight) * 100}%`,
          width: '4px',
          height: '4px',
          backgroundColor: 'red',
          borderRadius: '50%',
          zIndex: 11
        };
        
        return <div key={`kp-${index}-${kpIndex}`} style={style}></div>;
      });
    });
  };

  return (
    <div className={`${isFullscreen ? 'fixed inset-0 z-50 bg-black p-4' : 'w-full'} bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex flex-col`}>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold flex items-center">
          <Eye className="mr-2 text-blue-500" /> 
          Live Camera Feed
          {isProcessing && <span className="ml-2 text-sm text-blue-500 flex items-center">
            <RefreshCw className="mr-1 animate-spin" size={16} /> 
            Processing: {processingType}
          </span>}
        </h2>
        
        <div className="flex gap-2">
          {!isProcessing ? (
            <div className="relative">
              <button
                className="px-3 py-1.5 bg-blue-500 text-white rounded-lg flex items-center"
                onClick={() => document.getElementById('detection-dropdown').classList.toggle('hidden')}
              >
                <Zap size={16} className="mr-1" />
                Start Detection
                <ChevronDown size={14} className="ml-1" />
              </button>
              
              <div id="detection-dropdown" className="absolute right-0 mt-1 w-48 bg-white rounded-lg shadow-lg border border-gray-200 hidden z-20">
                <ul className="py-1">
                  <li>
                    <button
                      className="px-4 py-2 text-left w-full hover:bg-blue-50 flex items-center"
                      onClick={() => startProcessing('loitering')}
                    >
                      <UserCheck size={14} className="mr-2 text-blue-500" />
                      Loitering Detection
                    </button>
                  </li>
                  <li>
                    <button
                      className="px-4 py-2 text-left w-full hover:bg-blue-50 flex items-center"
                      onClick={() => startProcessing('theft')}
                    >
                      <ShoppingBag size={14} className="mr-2 text-red-500" />
                      Theft Detection
                    </button>
                  </li>
                  <li>
                    <button
                      className="px-4 py-2 text-left w-full hover:bg-blue-50 flex items-center"
                      onClick={() => startProcessing('face_tracking')}
                    >
                      <User size={14} className="mr-2 text-green-500" />
                      Face Tracking
                    </button>
                  </li>
                </ul>
              </div>
            </div>
          ) : (
            <button
              className="px-3 py-1.5 bg-red-500 text-white rounded-lg flex items-center"
              onClick={() => setIsProcessing(false)}
            >
              <AlertTriangle size={16} className="mr-1" />
              Stop Processing
            </button>
          )}
          
          <button
            onClick={toggleFullscreen}
            className="p-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
          >
            {isFullscreen ? (
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="4 14 10 14 10 20"></polyline>
                <polyline points="20 10 14 10 14 4"></polyline>
                <line x1="14" y1="10" x2="21" y2="3"></line>
                <line x1="3" y1="21" x2="10" y2="14"></line>
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="15 3 21 3 21 9"></polyline>
                <polyline points="9 21 3 21 3 15"></polyline>
                <line x1="21" y1="3" x2="14" y2="10"></line>
                <line x1="3" y1="21" x2="10" y2="14"></line>
              </svg>
            )}
          </button>
        </div>
      </div>
      
      {/* Camera view with detections */}
      <div className="relative flex-grow rounded-lg overflow-hidden bg-gray-900">
        <div className={`relative ${isFullscreen ? 'h-full' : 'aspect-video'}`}>
          {/* Replace GIF with actual video element */}
          <video
            ref={videoRef}
            className="w-full h-full object-cover"
            autoPlay
            loop
            muted
            controls={false}
          >
            <source src={selectedCamera.videoUrl} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
          
          {/* Detection boxes and keypoints */}
          {isProcessing && renderDetections()}
          {isProcessing && processingType === 'theft' && renderKeypoints()}
          
          {/* Status indicators */}
          <div className="absolute top-4 left-4 bg-black bg-opacity-60 text-white px-3 py-1 rounded-lg text-sm flex items-center">
            <Clock size={14} className="mr-1" /> {new Date().toLocaleTimeString()}
          </div>
          
          {selectedCamera.status === 'offline' && (
            <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-70">
              <div className="text-center">
                <ShieldAlert size={50} className="mx-auto mb-2 text-red-500" />
                <p className="text-white font-semibold text-lg">Camera Offline</p>
              </div>
            </div>
          )}
          
          {isProcessing && (
            <div className="absolute bottom-4 left-0 right-0 mx-auto w-3/4">
              <div className="bg-black bg-opacity-75 p-2 rounded-lg">
                <div className="flex justify-between text-xs text-white mb-1">
                  <span>Processing: {processingType}</span>
                  <span>{progress.toFixed(0)}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-1.5">
                  <div 
                    className="bg-blue-500 h-1.5 rounded-full" 
                    style={{width: `${progress}%`}}
                  ></div>
                </div>
              </div>
            </div>
          )}
          
          {error && (
            <div className="absolute bottom-4 left-4 right-4 bg-red-500 bg-opacity-90 text-white p-2 rounded-lg text-sm">
              <p className="font-semibold">Error:</p>
              <p>{error}</p>
            </div>
          )}
          
          {/* Camera info overlay */}
          <div className="absolute bottom-4 left-4 bg-black bg-opacity-70 text-white p-3 rounded-lg max-w-[70%]">
            <h3 className="text-lg font-semibold flex items-center">
              {selectedCamera.name}
              {selectedCamera.status === 'online' && (
                <span className="ml-2 flex items-center text-xs bg-green-500 text-white px-2 py-0.5 rounded-full">
                  <span className="w-2 h-2 bg-white rounded-full mr-1 animate-pulse"></span> LIVE
                </span>
              )}
            </h3>
            
            <div className="grid grid-cols-2 gap-x-6 gap-y-1 mt-2">
              <p className="flex items-center text-sm">
                <User size={14} className="mr-1 text-blue-400" /> 
                People: {stats.people}
              </p>
              <p className="flex items-center text-sm">
                <UserCheck size={14} className="mr-1 text-orange-400" /> 
                Loitering: {stats.loitering}
              </p>
              <p className="flex items-center text-sm">
                <ShoppingBag size={14} className="mr-1 text-red-400" /> 
                Theft: {stats.theft}
              </p>
              <p className="flex items-center text-sm">
                <Eye size={14} className="mr-1 text-purple-400" /> 
                Objects: {stats.objects}
              </p>
            </div>
          </div>
        </div>
      </div>
      
      {/* Detection info panel */}
      {isProcessing && detections.length > 0 && (
        <div className="mt-4 bg-gray-50 rounded-lg p-3 max-h-40 overflow-y-auto">
          <h3 className="font-medium mb-2 flex items-center">
            <ShieldAlert size={16} className="mr-1 text-blue-500" />
            Detection Events
          </h3>
          <ul className="space-y-1 text-sm">
            {detections.slice(-5).map((detection, index) => (
              <li key={index} className="flex items-center border-b border-gray-200 pb-1">
                <span className={`w-2 h-2 rounded-full mr-2 ${
                  detection.type === 'theft' ? 'bg-red-500' : 
                  detection.type === 'loitering' ? 'bg-orange-500' : 'bg-green-500'
                }`}></span>
                <span className="font-medium">{detection.type || detection.class_name || 'Object'}</span>
                <span className="mx-1">•</span>
                <span className="text-gray-600">
                  Confidence: {Math.round((detection.confidence || 0) * 100)}%
                </span>
                {detection.timestamp && (
                  <>
                    <span className="mx-1">•</span>
                    <span className="text-gray-600">
                      {new Date(detection.timestamp).toLocaleTimeString()}
                    </span>
                  </>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default LiveCameraComponent;