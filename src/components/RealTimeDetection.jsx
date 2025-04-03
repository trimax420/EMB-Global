import React, { useState, useEffect, useRef } from 'react';
import { ShieldCheck, Camera, UserCheck, ShoppingBag, AlertCircle, Video } from 'lucide-react';

// Define the base API URL - hardcode for development
const API_BASE_URL = 'http://localhost:8000/api';
// Define WebSocket URLs to try (in order)
const WS_URLS = [
  'ws://localhost:8000/api/ws/inference',  // With API prefix
  'ws://localhost:8000/ws/inference'       // Without API prefix
];
// Define WebRTC URL
const WEBRTC_SIGNAL_URL = 'http://localhost:8000/api/webrtc/signal';

// Mock data for offline development
const MOCK_DETECTION_ENABLED = true; // Set to false when backend is ready
const CONNECTION_RETRY_DELAY = 3000; // ms to wait between connection attempts
const USE_WEBRTC = true; // Enable WebRTC instead of WebSocket if available

const RealTimeDetection = ({ cameraId, onUpdateStats }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [detections, setDetections] = useState([]);
  const [detectionStats, setDetectionStats] = useState({
    people: 0,
    objects: 0,
    loitering: 0,
    theft: 0
  });
  const [isUsingMock, setIsUsingMock] = useState(false);
  const [wsUrlIndex, setWsUrlIndex] = useState(0);
  const [isUsingWebRTC, setIsUsingWebRTC] = useState(false);

  const wsRef = useRef(null);
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const mockIntervalRef = useRef(null);
  const connectionAttemptsRef = useRef(0);
  
  // Set up WebRTC or WebSocket connection on component mount
  useEffect(() => {
    if (cameraId) {
      if (USE_WEBRTC) {
        console.log('Attempting WebRTC connection first');
        setupWebRTC();
      } else {
        connectToWebSocket();
      }
    }
    
    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (peerConnectionRef.current) {
        peerConnectionRef.current.close();
        peerConnectionRef.current = null;
      }
      if (mockIntervalRef.current) {
        clearInterval(mockIntervalRef.current);
        mockIntervalRef.current = null;
      }
    };
  }, [cameraId, wsUrlIndex]);

  // Setup WebRTC connection with simplified error handling
  const setupWebRTC = async () => {
    try {
      console.log('Setting up WebRTC connection to:', WEBRTC_SIGNAL_URL);
      
      // Close any existing connection
      if (peerConnectionRef.current) {
        peerConnectionRef.current.close();
      }
      
      // Create RTCPeerConnection
      const configuration = { 
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' }
        ] 
      };
      
      const peerConnection = new RTCPeerConnection(configuration);
      peerConnectionRef.current = peerConnection;
      
      // Set up data channel for control messages
      const dataChannel = peerConnection.createDataChannel('controls');
      dataChannel.onopen = () => {
        console.log('WebRTC data channel open');
        setIsConnected(true);
        setError(null);
        setIsUsingMock(false);
        setIsUsingWebRTC(true);
        
        // Subscribe to camera
        dataChannel.send(JSON.stringify({
          action: 'subscribe',
          camera_id: cameraId,
          detection_types: ['object', 'theft', 'loitering']
        }));
      };
      
      dataChannel.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebRTC message received:', data.type);
          
          if (data.type === 'detections') {
            handleDetectionUpdate(data);
          } else if (data.type === 'inference_started') {
            setIsStreaming(true);
          } else if (data.type === 'inference_stopped') {
            setIsStreaming(false);
          } else if (data.type === 'error') {
            console.error('WebRTC error:', data.message);
            setError(data.message);
          }
        } catch (error) {
          console.error('Error processing WebRTC message:', error);
        }
      };
      
      dataChannel.onclose = () => {
        console.log('WebRTC data channel closed');
        setIsConnected(false);
        setIsStreaming(false);
        
        // Fall back to WebSocket if WebRTC fails
        if (!isUsingMock) {
          console.log('Falling back to WebSocket');
          setIsUsingWebRTC(false);
          connectToWebSocket();
        }
      };
      
      // Handle ICE candidate events
      peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
          // Send ICE candidate to server
          sendSignalingMessage({
            type: 'ice_candidate',
            candidate: event.candidate
          });
        }
      };
      
      // Handle connection state changes
      peerConnection.onconnectionstatechange = () => {
        console.log('WebRTC connection state:', peerConnection.connectionState);
        
        if (peerConnection.connectionState === 'connected') {
          setIsConnected(true);
          setError(null);
        } else if (peerConnection.connectionState === 'disconnected' || 
                  peerConnection.connectionState === 'failed') {
          setIsConnected(false);
          setIsStreaming(false);
          
          // Fall back to WebSocket or mock mode
          if (!isUsingMock) {
            console.log('WebRTC connection failed, falling back to WebSocket');
            setIsUsingWebRTC(false);
            connectToWebSocket();
          }
        }
      };
      
      // Handle video track from remote peer
      peerConnection.ontrack = (event) => {
        console.log('Received remote track', event.track.kind);
        if (event.track.kind === 'video' && videoRef.current) {
          videoRef.current.srcObject = event.streams[0];
          setIsStreaming(true);
        }
      };
      
      // Create and send offer
      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);
      
      // Send the offer to the server via HTTP
      console.log('Sending WebRTC offer to signaling server');
      const signalResult = await sendSignalingMessage({
        type: 'offer',
        sdp: peerConnection.localDescription,
        camera_id: cameraId
      });
      
      // If we get here without an error, the signaling server is working
      console.log('WebRTC signaling successful, waiting for connection');
      
    } catch (error) {
      console.error('WebRTC setup error:', error);
      setError(`WebRTC not available (${error.message}). Trying WebSocket...`);
      setIsUsingWebRTC(false);
      
      // Fall back to WebSocket immediately
      connectToWebSocket();
    }
  };
  
  // Send signaling messages to the backend with better error handling
  const sendSignalingMessage = async (message) => {
    try {
      console.log(`Sending ${message.type} to ${WEBRTC_SIGNAL_URL}`);
      
      const response = await fetch(WEBRTC_SIGNAL_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(message)
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error (${response.status}): ${errorText}`);
      }
      
      // Handle response if needed
      const data = await response.json();
      console.log('Received signaling response:', data.type);
      
      // If we get an answer, set it as remote description
      if (data.type === 'answer' && peerConnectionRef.current) {
        console.log('Setting remote description from answer');
        await peerConnectionRef.current.setRemoteDescription(new RTCSessionDescription(data.sdp));
        return data;
      } 
      // If we get an ICE candidate, add it
      else if (data.type === 'ice_candidate' && peerConnectionRef.current) {
        console.log('Adding ICE candidate');
        await peerConnectionRef.current.addIceCandidate(new RTCIceCandidate(data.candidate));
        return data;
      }
      
      return data;
    } catch (error) {
      console.error('Signaling error:', error);
      // Rethrow so the caller can handle it
      throw error;
    }
  };

  // Try to connect to WebSocket with the current URL index
  const connectToWebSocket = () => {
    try {
      // Close any existing connection
      if (wsRef.current) {
        wsRef.current.close();
      }
      
      if (wsUrlIndex >= WS_URLS.length) {
        console.log('All WebSocket URLs failed, falling back to mock mode');
        setError('All connection attempts failed. Using mock detection mode.');
        enableMockMode();
        return;
      }
      
      const wsUrl = WS_URLS[wsUrlIndex];
      console.log(`Connecting to WebSocket at ${wsUrl} (attempt ${connectionAttemptsRef.current + 1})`);
      
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected successfully to: ' + wsUrl);
        setIsConnected(true);
        setError(null);
        setIsUsingMock(false);
        connectionAttemptsRef.current = 0;
        
        // Subscribe to camera
        ws.send(JSON.stringify({
          action: 'subscribe',
          camera_id: cameraId,
          detection_types: ['object', 'theft', 'loitering']
        }));
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket message received:', data.type);
          
          if (data.type === 'subscribed') {
            console.log(`Subscribed to camera ${data.camera_id}`);
          }
          else if (data.type === 'live_detection') {
            // Only process if this is our camera
            if (data.camera_id === cameraId) {
              // Process the detections
              handleDetectionUpdate(data);
              
              // Update streaming status
              setIsStreaming(true);
            }
          }
          else if (data.type === 'inference_started') {
            console.log(`Inference started for camera ${data.camera_id}`);
            setIsStreaming(true);
          }
          else if (data.type === 'inference_stopped') {
            console.log(`Inference stopped for camera ${data.camera_id}`);
            setIsStreaming(false);
          }
          else if (data.type === 'error') {
            console.error('WebSocket error:', data.message);
            setError(data.message);
          }
        } catch (error) {
          console.error('Error processing WebSocket message:', error);
        }
      };
      
      ws.onclose = (event) => {
        console.log('WebSocket disconnected, code:', event.code, 'reason:', event.reason);
        setIsConnected(false);
        setIsStreaming(false);
        
        // If we just connected and immediately disconnected, try the next URL
        if (connectionAttemptsRef.current === 0) {
          connectionAttemptsRef.current++;
          setWsUrlIndex(prevIndex => prevIndex + 1);
        } else {
          // If we were connected for a while and then disconnected, try to reconnect to the same URL
          setTimeout(() => {
            if (!isUsingMock) {
              connectToWebSocket();
            }
          }, CONNECTION_RETRY_DELAY);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        // Try the next URL
        setWsUrlIndex(prevIndex => prevIndex + 1);
      };
      
      // Store reference
      wsRef.current = ws;
    } catch (error) {
      console.error('WebSocket connection error:', error);
      setError(`WebSocket error: ${error.message}`);
      
      // Try the next URL
      setWsUrlIndex(prevIndex => prevIndex + 1);
    }
  };

  // Enable mock mode
  const enableMockMode = () => {
    if (!isUsingMock && MOCK_DETECTION_ENABLED) {
      console.log('Enabling mock detection mode');
      setIsUsingMock(true);
      startMockDetection();
    }
  };

  // Generate mock detections for development
  const generateMockDetections = () => {
    // Random number of detections (0-8)
    const numDetections = Math.floor(Math.random() * 8);
    const mockDetections = [];
    
    // Object/person detections
    const numPersons = Math.floor(Math.random() * 4);
    for (let i = 0; i < numPersons; i++) {
      mockDetections.push({
        type: 'object',
        class_name: 'person',
        confidence: Math.random() * 0.2 + 0.7,
        bbox: [
          Math.random() * 400,
          Math.random() * 300,
          Math.random() * 100 + 50,
          Math.random() * 200 + 100
        ]
      });
    }
    
    // Object detections
    const numObjects = Math.floor(Math.random() * 3);
    const objectClasses = ['chair', 'bottle', 'laptop', 'keyboard'];
    for (let i = 0; i < numObjects; i++) {
      const classIndex = Math.floor(Math.random() * objectClasses.length);
      mockDetections.push({
        type: 'object',
        class_name: objectClasses[classIndex],
        confidence: Math.random() * 0.3 + 0.6,
        bbox: [
          Math.random() * 400,
          Math.random() * 300,
          Math.random() * 80 + 20,
          Math.random() * 80 + 20
        ]
      });
    }
    
    // Add occasional threat detections
    if (Math.random() < 0.1) {
      mockDetections.push({
        type: 'theft',
        confidence: Math.random() * 0.2 + 0.7,
        bbox: [
          Math.random() * 400,
          Math.random() * 300,
          Math.random() * 100 + 50,
          Math.random() * 200 + 100
        ],
        zone: Math.random() < 0.5 ? 'chest' : 'waist'
      });
    }
    
    if (Math.random() < 0.05) {
      mockDetections.push({
        type: 'loitering',
        confidence: Math.random() * 0.2 + 0.7,
        bbox: [
          Math.random() * 400,
          Math.random() * 300,
          Math.random() * 100 + 50,
          Math.random() * 200 + 100
        ],
        time_present: Math.random() * 20 + 10
      });
    }
    
    // Create mock data structure
    const mockData = {
      camera_id: cameraId,
      detections: mockDetections,
      frame_number: Math.floor(Math.random() * 1000),
      timestamp: new Date().toISOString()
    };
    
    // Handle the mock detections
    handleDetectionUpdate(mockData);
  };

  // Start mock detection
  const startMockDetection = () => {
    if (mockIntervalRef.current) {
      clearInterval(mockIntervalRef.current);
    }
    
    // Generate an initial detection
    generateMockDetections();
    
    // Set up interval for mock detections
    mockIntervalRef.current = setInterval(generateMockDetections, 2000);
    
    // Update state
    setIsStreaming(true);
    setIsConnected(true);
  };

  // Stop mock detection
  const stopMockDetection = () => {
    if (mockIntervalRef.current) {
      clearInterval(mockIntervalRef.current);
      mockIntervalRef.current = null;
    }
    
    setIsStreaming(false);
  };

  // Handle detection update
  const handleDetectionUpdate = (data) => {
    if (!data.detections || !Array.isArray(data.detections)) {
      return;
    }
    
    // Update detections list
    const newDetections = data.detections.map(detection => ({
      ...detection,
      id: `${detection.type}_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
      receivedAt: new Date()
    }));
    
    // Update state if we have detections (keep last 20 detections)
    if (newDetections.length > 0) {
      setDetections(prevDetections => {
        const combined = [...newDetections, ...prevDetections];
        return combined.slice(0, 20);
      });
    }
    
    // Process the detection image if available
    if (data.frame && canvasRef.current) {
      displayDetectionFrame(data.frame);
    } else if (isUsingMock && canvasRef.current) {
      // If we're in mock mode and there's no frame, draw mock visualization
      drawMockDetections(newDetections);
    }
    
    // Count detection types for statistics
    const stats = {
      people: 0,
      objects: 0,
      loitering: 0,
      theft: 0
    };
    
    data.detections.forEach(detection => {
      if (detection.type === 'object' && detection.class_name === 'person') {
        stats.people++;
      } else if (detection.type === 'object') {
        stats.objects++;
      } else if (detection.type === 'loitering') {
        stats.loitering++;
      } else if (detection.type === 'theft') {
        stats.theft++;
      }
    });
    
    // Update detection stats
    setDetectionStats(prevStats => {
      const newStats = {
        people: Math.max(prevStats.people, stats.people),
        objects: Math.max(prevStats.objects, stats.objects),
        loitering: prevStats.loitering + stats.loitering,
        theft: prevStats.theft + stats.theft
      };
      
      // Call parent update if provided
      if (onUpdateStats) {
        onUpdateStats(newStats);
      }
      
      return newStats;
    });
  };

  // Draw mock detections on canvas
  const drawMockDetections = (detections) => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Set canvas size if not already set
    if (canvas.width === 0) {
      canvas.width = 640;
      canvas.height = 480;
    }
    
    // Clear canvas
    ctx.fillStyle = '#1a1a2e'; // Dark background
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw a grid pattern for visual reference
    ctx.strokeStyle = '#2a2a3e';
    ctx.lineWidth = 1;
    
    // Draw grid lines
    for (let i = 0; i < canvas.width; i += 40) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, canvas.height);
      ctx.stroke();
    }
    
    for (let i = 0; i < canvas.height; i += 40) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(canvas.width, i);
      ctx.stroke();
    }
    
    // Draw timestamp
    ctx.fillStyle = 'white';
    ctx.font = '14px monospace';
    const timestamp = new Date().toLocaleTimeString();
    ctx.fillText(`${timestamp} - Camera ${cameraId}`, 10, 20);
    
    // Draw each detection bbox
    detections.forEach(detection => {
      if (!detection.bbox) return;
      
      // Set color based on detection type
      let color;
      switch (detection.type) {
        case 'theft':
          color = 'rgba(239, 68, 68, 0.7)'; // red
          break;
        case 'loitering':
          color = 'rgba(245, 158, 11, 0.7)'; // amber
          break;
        case 'object':
          if (detection.class_name === 'person') {
            color = 'rgba(59, 130, 246, 0.7)'; // blue
          } else {
            color = 'rgba(16, 185, 129, 0.7)'; // green
          }
          break;
        default:
          color = 'rgba(209, 213, 219, 0.7)'; // gray
      }
      
      // Extract bounding box
      let [x, y, width, height] = detection.bbox;
      
      // If bbox is in [x1, y1, x2, y2] format, convert to [x, y, width, height]
      if (detection.bbox.length === 4 && detection.bbox[2] < canvas.width && detection.bbox[3] < canvas.height) {
        width = detection.bbox[2] - detection.bbox[0];
        height = detection.bbox[3] - detection.bbox[1];
      }
      
      // Draw rectangle
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
      
      // Add transparent fill
      ctx.fillStyle = color.replace('0.7', '0.2');
      ctx.fillRect(x, y, width, height);
      
      // Draw label
      const label = detection.class_name || detection.type;
      const confidence = detection.confidence ? `${Math.round(detection.confidence * 100)}%` : '';
      const displayText = `${label} ${confidence}`;
      
      // Background for text
      const textWidth = ctx.measureText(displayText).width + 10;
      ctx.fillStyle = color.replace('0.7', '0.9');
      ctx.fillRect(x, y - 20, textWidth, 20);
      
      // Text
      ctx.fillStyle = 'white';
      ctx.font = '14px Arial';
      ctx.fillText(displayText, x + 5, y - 5);
    });
  };

  // Display frame with detections
  const displayDetectionFrame = (frameData) => {
    if (!canvasRef.current) return;
    
    try {
      // Convert base64 to image
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        // Set canvas size to match image
        canvas.width = img.width;
        canvas.height = img.height;
        
        // Draw image
        ctx.drawImage(img, 0, 0);
      };
      
      // Set image source (handle both base64 string and binary formats)
      if (typeof frameData === 'string') {
        // Base64 string
        img.src = `data:image/jpeg;base64,${frameData}`;
      } else {
        // Binary data - convert to blob URL
        const blob = new Blob([frameData], { type: 'image/jpeg' });
        const url = URL.createObjectURL(blob);
        img.src = url;
        // Clean up blob URL after loading
        img.onload = () => {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);
          URL.revokeObjectURL(url);
        };
      }
    } catch (error) {
      console.error('Error displaying frame:', error);
    }
  };

  // Start streaming (using appropriate technology)
  const startStreaming = () => {
    if (isUsingMock) {
      startMockDetection();
      return;
    }
    
    if (isUsingWebRTC) {
      startWebRTCStreaming();
      return;
    }
    
    // WebSocket streaming (existing code)
    if (!isConnected || !wsRef.current) {
      // Reset connection attempts and try from the first URL
      connectionAttemptsRef.current = 0;
      setWsUrlIndex(0);
      connectToWebSocket();
      return;
    }
    
    // Send start inference message
    try {
      wsRef.current.send(JSON.stringify({
        action: 'start_inference',
        camera_id: cameraId,
        detection_types: ['object', 'theft', 'loitering']
      }));
      console.log('Sent start_inference request');
    } catch (error) {
      console.error('Error sending start_inference:', error);
      setError(`Failed to start streaming: ${error.message}`);
      // If we can't send messages, try to reconnect
      connectionAttemptsRef.current = 0;
      setWsUrlIndex(0);
      connectToWebSocket();
    }
  };

  // Start WebRTC streaming
  const startWebRTCStreaming = () => {
    if (!peerConnectionRef.current) {
      setupWebRTC();
      return;
    }
    
    // Use data channel to send start command
    const dataChannels = peerConnectionRef.current.getDataChannels?.() || 
                         peerConnectionRef.current.dataChannels || 
                         [];
    const dataChannel = dataChannels[0] || peerConnectionRef.current.dataChannel;
    
    if (dataChannel && dataChannel.readyState === 'open') {
      dataChannel.send(JSON.stringify({
        action: 'start_inference',
        camera_id: cameraId,
        detection_types: ['object', 'theft', 'loitering']
      }));
      console.log('Sent WebRTC start_inference request');
    } else {
      console.error('WebRTC data channel not ready');
      setError('WebRTC connection not ready. Trying to reconnect...');
      setupWebRTC();
    }
  };

  // Stop streaming (using appropriate technology)
  const stopStreaming = () => {
    if (isUsingMock) {
      stopMockDetection();
      return;
    }
    
    if (isUsingWebRTC) {
      stopWebRTCStreaming();
      return;
    }
    
    // WebSocket stop (existing code)
    if (!isConnected || !wsRef.current) return;
    
    // Send stop inference message
    try {
      wsRef.current.send(JSON.stringify({
        action: 'stop_inference',
        camera_id: cameraId
      }));
      console.log('Sent stop_inference request');
    } catch (error) {
      console.error('Error sending stop_inference:', error);
      // If we can't stop properly, at least update the UI
      setIsStreaming(false);
    }
  };

  // Stop WebRTC streaming
  const stopWebRTCStreaming = () => {
    if (!peerConnectionRef.current) return;
    
    // Use data channel to send stop command
    const dataChannels = peerConnectionRef.current.getDataChannels?.() || 
                         peerConnectionRef.current.dataChannels || 
                         [];
    const dataChannel = dataChannels[0] || peerConnectionRef.current.dataChannel;
    
    if (dataChannel && dataChannel.readyState === 'open') {
      dataChannel.send(JSON.stringify({
        action: 'stop_inference',
        camera_id: cameraId
      }));
      console.log('Sent WebRTC stop_inference request');
    } else {
      // If data channel is closed, just update UI
      setIsStreaming(false);
    }
  };

  // Start mock detection if we're in mock mode and not streaming
  useEffect(() => {
    if (isUsingMock && !isStreaming && !mockIntervalRef.current) {
      console.log('Auto-starting mock detection');
      startMockDetection();
    }
  }, [isUsingMock, isStreaming]);

  // Get the icon for a detection type
  const getDetectionIcon = (type) => {
    switch (type) {
      case 'theft':
        return <ShoppingBag className="text-red-500" size={16} />;
      case 'loitering':
        return <UserCheck className="text-amber-500" size={16} />;
      case 'object':
        return <Camera className="text-blue-500" size={16} />;
      default:
        return <ShieldCheck className="text-blue-500" size={16} />;
    }
  };

  // Format relative time
  const getRelativeTime = (date) => {
    const seconds = Math.floor((new Date() - date) / 1000);
    
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  return (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden">
      {/* Video/Canvas for displaying detection frames */}
      <div className="relative aspect-video bg-gray-900">
        {isUsingWebRTC ? (
          <video
            ref={videoRef}
            className="w-full h-full object-contain"
            autoPlay
            playsInline
            muted
          />
        ) : (
          <canvas
            ref={canvasRef}
            className="w-full h-full object-contain"
          />
        )}
        
        {/* Status overlay */}
        {!isStreaming && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-60 text-white text-center">
            <div>
              {error ? (
                <>
                  <AlertCircle size={40} className="mx-auto mb-2" />
                  <p className="font-medium">Error</p>
                  <p className="text-sm">{error}</p>
                </>
              ) : !isConnected ? (
                <>
                  <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-white mx-auto mb-2"></div>
                  <p className="font-medium">Connecting...</p>
                  {!isUsingWebRTC && wsUrlIndex < WS_URLS.length && (
                    <p className="text-xs mt-1">Trying {WS_URLS[wsUrlIndex]}</p>
                  )}
                </>
              ) : (
                <>
                  <Camera size={40} className="mx-auto mb-2" />
                  <p className="font-medium">Ready</p>
                  <p className="text-sm">Click Start to begin detection</p>
                </>
              )}
            </div>
          </div>
        )}
        
        {/* Connection type indicator */}
        {isConnected && (
          <div className="absolute top-2 left-2 bg-green-500 bg-opacity-90 text-white text-xs px-2 py-1 rounded-full flex items-center gap-1">
            {isUsingWebRTC ? <Video size={12} /> : (isUsingMock ? 'Mock' : 'WS')}
            <span>Connected</span>
          </div>
        )}
        
        {/* Live indicator */}
        {isStreaming && (
          <div className="absolute top-2 right-2 bg-black bg-opacity-50 text-white text-xs px-2 py-1 rounded-full flex items-center gap-1">
            <span className="h-2 w-2 bg-red-500 rounded-full animate-pulse"></span>
            Live
          </div>
        )}
      </div>
      
      {/* Controls */}
      <div className="p-4 border-t border-gray-100">
        <div className="flex justify-between items-center mb-4">
          <h3 className="font-semibold">
            {isUsingWebRTC ? 'WebRTC Stream' : (isUsingMock ? 'Mock Detection' : 'WebSocket Stream')}
          </h3>
          <div className="flex gap-2">
            <button
              onClick={startStreaming}
              disabled={isStreaming}
              className={`px-3 py-1 rounded text-sm ${isStreaming ? 'bg-gray-100 text-gray-400' : 'bg-blue-500 text-white hover:bg-blue-600'}`}
            >
              Start
            </button>
            <button
              onClick={stopStreaming}
              disabled={!isStreaming}
              className={`px-3 py-1 rounded text-sm ${!isStreaming ? 'bg-gray-100 text-gray-400' : 'bg-red-500 text-white hover:bg-red-600'}`}
            >
              Stop
            </button>
          </div>
        </div>
        
        {/* Detection Stats */}
        <div className="grid grid-cols-4 gap-4 mb-4 text-center">
          <div>
            <p className="text-xs text-gray-500">People</p>
            <p className="text-xl font-semibold">{detectionStats.people}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Objects</p>
            <p className="text-xl font-semibold">{detectionStats.objects}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Loitering</p>
            <p className={`text-xl font-semibold ${detectionStats.loitering > 0 ? 'text-amber-500' : ''}`}>
              {detectionStats.loitering}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Theft</p>
            <p className={`text-xl font-semibold ${detectionStats.theft > 0 ? 'text-red-500' : ''}`}>
              {detectionStats.theft}
            </p>
          </div>
        </div>
        
        {/* Recent Detections */}
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-2">Recent Detections</h4>
          <div className="max-h-[200px] overflow-y-auto rounded-md bg-gray-50">
            {detections.length === 0 ? (
              <p className="text-center text-gray-500 py-4 text-sm">No detections yet</p>
            ) : (
              <ul className="divide-y divide-gray-100">
                {detections.map(detection => (
                  <li key={detection.id} className="px-3 py-2 flex items-center gap-2 text-sm">
                    {getDetectionIcon(detection.type)}
                    <div className="flex-1">
                      <p className="font-medium">{detection.class_name || detection.type}</p>
                      <p className="text-xs text-gray-500">
                        {detection.confidence ? `${Math.round(detection.confidence * 100)}%` : ''}
                        {detection.time_present ? ` • ${detection.time_present.toFixed(1)}s present` : ''}
                      </p>
                    </div>
                    <span className="text-xs text-gray-400">
                      {getRelativeTime(detection.receivedAt)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RealTimeDetection;
