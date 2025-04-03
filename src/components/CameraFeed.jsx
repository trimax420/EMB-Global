import React, { useEffect, useRef, useState } from 'react';
import { Box, Typography, Paper, CircularProgress, Badge, Chip, Alert, Button, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';
import WarningIcon from '@mui/icons-material/Warning';
import PersonIcon from '@mui/icons-material/Person';
import RefreshIcon from '@mui/icons-material/Refresh';
import VideoLibraryIcon from '@mui/icons-material/VideoLibrary';
import SimulationIcon from '@mui/icons-material/CameraIndoor';
import useWebSocket from '../hooks/useWebSocket';
import './CameraFeed.css';

const CameraFeed = ({ 
  cameraId = '0', 
  detectionType = 'both',
  title = 'Camera Feed',
  height = '360px',
  backendUrl = 'localhost:8000'
}) => {
  const canvasRef = useRef(null);
  const [status, setStatus] = useState('connecting');
  const [alert, setAlert] = useState(null);
  const [isMockFeed, setIsMockFeed] = useState(false);
  const [isVideoFile, setIsVideoFile] = useState(false);
  const [videoInfo, setVideoInfo] = useState(null);
  const [sourceType, setSourceType] = useState('auto'); // 'auto', 'camera', 'video', 'mock'
  const [stats, setStats] = useState({
    fps: 0,
    detections: 0,
    criticalEvents: 0
  });

  // Connect to WebSocket for real-time inference with parameters based on selected source
  const getWebSocketUrl = () => {
    const baseUrl = `ws://${backendUrl}/api/ws/realtime-detection?detection_type=${detectionType}&camera_id=${cameraId}`;
    
    // Set default to video for numbered camera IDs (1-4)
    let defaultSourceType = 'auto';
    if (cameraId && /^[1-4]$/.test(cameraId)) {
      defaultSourceType = 'video';
    }
    
    const useMock = sourceType === 'mock' || (sourceType === 'auto' && defaultSourceType !== 'video');
    const useVideo = sourceType === 'video' || (sourceType === 'auto' && defaultSourceType === 'video');
    
    return `${baseUrl}&use_mock=${useMock}&use_video=${useVideo}`;
  };
  
  const wsUrl = getWebSocketUrl();
  
  const { 
    connected, 
    lastMessage, 
    sendMessage,
    error,
    reconnect
  } = useWebSocket(wsUrl, {
    reconnectInterval: 3000,
    maxReconnectAttempts: 10,
    debug: true
  });

  // Update status based on connection state
  useEffect(() => {
    if (connected) {
      setStatus('connected');
    } else if (error) {
      setStatus('error');
      setAlert(`Connection error: ${error}`);
      console.log(`Camera ${cameraId} WebSocket connection error:`, error);
    } else {
      setStatus('connecting');
    }
  }, [connected, error, cameraId]);

  // Function to manually reconnect
  const handleReconnect = () => {
    setStatus('connecting');
    setAlert('Attempting to reconnect...');
    reconnect();
  };

  // Handle source type change
  const handleSourceChange = (event) => {
    const newSource = event.target.value;
    setSourceType(newSource);
    
    // Force a reconnection with new parameters
    setTimeout(() => {
      reconnect();
    }, 100);
  };

  // Process incoming WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;
    
    try {
      const data = typeof lastMessage === 'string' ? JSON.parse(lastMessage) : lastMessage;
      
      if (data.type === 'inference_result') {
        // Check feed type
        const mockFeed = data.ui_metadata?.is_mock || false;
        const videoFileFeed = data.ui_metadata?.is_video_file || false;
        
        setIsMockFeed(mockFeed);
        setIsVideoFile(videoFileFeed);
        
        // Store video file info if available
        if (videoFileFeed && data.ui_metadata?.video_info) {
          setVideoInfo(data.ui_metadata.video_info);
        }
        
        // Update canvas with the new frame and detection results
        updateCanvas(data);
        
        // Check for detections
        const theftDetected = data.detections?.theft?.detected || false;
        const theftConfidence = data.detections?.theft?.confidence || 0;
        const loiteringDetected = data.detections?.loitering?.detected || false;
        const loiteringDuration = data.detections?.loitering?.duration || 0;
        
        // Count detections for stats
        const totalDetections = (theftDetected ? 1 : 0) + (loiteringDetected ? 1 : 0);
        
        // Update stats
        setStats(prev => ({
          fps: data.ui_metadata?.fps || prev.fps,
          detections: prev.detections + totalDetections,
          criticalEvents: data.ui_metadata?.has_critical_detection 
            ? prev.criticalEvents + 1 
            : prev.criticalEvents
        }));
        
        // Show alerts for detections - improved alerts with higher thresholds for better notifications
        if (theftDetected && theftConfidence > 0.75) {
          setAlert(`⚠️ THEFT DETECTED with ${(theftConfidence * 100).toFixed(1)}% confidence! (${new Date().toLocaleTimeString()})`);
        } 
        else if (loiteringDetected && loiteringDuration > 10) {
          setAlert(`⚠️ LOITERING DETECTED for ${loiteringDuration.toFixed(1)} seconds in restricted area! (${new Date().toLocaleTimeString()})`);
        }
        else if (data.ui_metadata?.has_critical_detection) {
          setAlert(`⚠️ SECURITY ALERT: Potential security issue detected! (${new Date().toLocaleTimeString()})`);
        }
      } 
      else if (data.type === 'error') {
        setAlert(`Error: ${data.message}`);
        setStatus('error');
      }
      else if (data.type === 'connection_established') {
        setStatus('connected');
        setAlert('Connection established');
      }
    } catch (e) {
      console.error('Error processing message:', e);
    }
  }, [lastMessage, cameraId]);

  // Function to update canvas with frame and detection results
  const updateCanvas = (data) => {
    if (!canvasRef.current || !data.frame) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Load the frame image
    const img = new Image();
    img.onload = () => {
      // Resize canvas if needed
      if (canvas.width !== img.width || canvas.height !== img.height) {
        canvas.width = img.width;
        canvas.height = img.height;
      }
      
      // Draw the frame
      ctx.drawImage(img, 0, 0);
      
      // Draw theft detection bounding boxes with enhanced styling
      if (data.detections?.theft?.detected) {
        ctx.strokeStyle = '#ff0000'; // Red for theft
        ctx.lineWidth = 3;
        ctx.font = '18px Arial';
        ctx.fillStyle = '#ff0000';
        
        // Add theft detection alert text at the top of the frame
        const confidence = (data.detections.theft.confidence * 100).toFixed(0);
        ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
        ctx.fillRect(10, 10, 260, 30);
        ctx.fillStyle = 'white';
        ctx.fillText(`⚠️ THEFT DETECTED - ${confidence}%`, 20, 30);
        
        // Draw boxes with enhanced styling
        for (const box of data.detections.theft.bounding_boxes || []) {
          const [x1, y1, x2, y2] = box;
          
          // Draw pulsing effect (based on timestamp)
          const date = new Date();
          const pulseEffect = 0.7 + (0.3 * Math.sin(date.getTime() / 200));
          
          // Draw semi-transparent highlight with pulsing effect
          ctx.fillStyle = `rgba(255, 0, 0, ${0.2 * pulseEffect})`;
          ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
          
          // Draw border with dashed effect
          ctx.strokeStyle = '#ff0000';
          ctx.lineWidth = 3;
          ctx.setLineDash([5, 3]);
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          ctx.setLineDash([]);
          
          // Draw label with better visibility
          ctx.font = '16px Arial';
          ctx.fillStyle = '#ff0000';
          ctx.fillRect(x1, y1 - 25, 140, 20);
          ctx.fillStyle = 'white';
          const confidence = (data.detections.theft.confidence * 100).toFixed(0);
          ctx.fillText(`Theft ${confidence}%`, x1 + 5, y1 - 10);
        }
      }
      
      // Draw loitering detection regions with enhanced styling
      if (data.detections?.loitering?.detected) {
        // Add loitering detection alert text at the top of the frame
        const duration = data.detections.loitering.duration.toFixed(1);
        ctx.fillStyle = 'rgba(0, 0, 255, 0.8)';
        ctx.fillRect(10, 50, 300, 30);
        ctx.fillStyle = 'white';
        ctx.fillText(`⚠️ LOITERING DETECTED - ${duration}s`, 20, 70);
        
        for (const region of data.detections.loitering.regions || []) {
          const {x, y, width, height} = region;
          
          // Draw semi-transparent highlight with pulsing effect
          const date = new Date();
          const pulseEffect = 0.7 + (0.3 * Math.sin(date.getTime() / 300));
          
          ctx.fillStyle = `rgba(0, 0, 255, ${0.2 * pulseEffect})`;
          ctx.fillRect(x, y, width, height);
          
          // Draw border with dashed effect
          ctx.strokeStyle = '#0000ff';
          ctx.lineWidth = 3;
          ctx.setLineDash([5, 3]);
          ctx.strokeRect(x, y, width, height);
          ctx.setLineDash([]);
          
          // Draw label with better visibility
          ctx.font = '16px Arial';
          ctx.fillStyle = '#0000ff';
          ctx.fillRect(x, y - 25, 160, 20);
          ctx.fillStyle = 'white';
          const duration = data.detections.loitering.duration.toFixed(1);
          ctx.fillText(`Loitering ${duration}s`, x + 5, y - 10);
        }
      }
    };
    
    // Set image source from base64 data
    img.src = `data:image/jpeg;base64,${data.frame}`;
  };

  // Draw a default feed in the canvas if no WebSocket connection
  useEffect(() => {
    if (status === 'error' && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      // Set a default size if not already set
      if (canvas.width === 0) {
        canvas.width = 640;
        canvas.height = 480;
      }
      
      // Clear the canvas
      ctx.fillStyle = '#f0f0f0';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw a message
      ctx.font = '20px Arial';
      ctx.fillStyle = '#555';
      ctx.textAlign = 'center';
      ctx.fillText('Connection Error', canvas.width / 2, canvas.height / 2 - 15);
      ctx.font = '16px Arial';
      ctx.fillText('Could not connect to camera feed', canvas.width / 2, canvas.height / 2 + 15);
    }
  }, [status]);

  // Get border color based on feed type
  const getBorderColor = () => {
    if (status === 'error') return 'error.main';
    if (isVideoFile) return 'info.main';
    if (isMockFeed) return 'warning.main';
    return 'success.main';
  };

  // Get background color based on feed type
  const getBackgroundColor = () => {
    if (isVideoFile) return 'rgba(3, 169, 244, 0.05)';
    if (isMockFeed) return 'rgba(255, 152, 0, 0.05)';
    return 'inherit';
  };

  return (
    <Paper 
      elevation={3} 
      className={`camera-feed ${status}`}
      sx={{ 
        position: 'relative',
        height,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        borderColor: getBorderColor(),
        borderWidth: 2
      }}
    >
      {/* Camera Title */}
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between',
        alignItems: 'center', 
        p: 1, 
        borderBottom: '1px solid rgba(0,0,0,0.1)',
        backgroundColor: getBackgroundColor()
      }}>
        <Typography variant="h6" component="div" sx={{ display: 'flex', alignItems: 'center' }}>
          {isVideoFile ? <VideoLibraryIcon sx={{ mr: 1 }} /> : isMockFeed ? <SimulationIcon sx={{ mr: 1 }} /> : <VideocamIcon sx={{ mr: 1 }} />}
          {title}
          {isVideoFile && (
            <Chip
              size="small"
              label="Video File"
              color="info"
              sx={{ ml: 1, height: 20, fontSize: '0.625rem' }}
            />
          )}
          {isMockFeed && (
            <Chip
              size="small"
              label="Mock Feed"
              color="warning"
              sx={{ ml: 1, height: 20, fontSize: '0.625rem' }}
            />
          )}
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 100 }}>
            <InputLabel id="source-select-label">Source</InputLabel>
            <Select
              labelId="source-select-label"
              id="source-select"
              value={sourceType}
              label="Source"
              onChange={handleSourceChange}
              size="small"
            >
              <MenuItem value="auto">Auto</MenuItem>
              <MenuItem value="camera">Camera</MenuItem>
              <MenuItem value="video">Video File</MenuItem>
              <MenuItem value="mock">Mock Data</MenuItem>
            </Select>
          </FormControl>
          
          <Chip 
            size="small"
            label={`${stats.fps} FPS`}
            color={stats.fps > 10 ? "success" : "warning"}
          />
          
          {detectionType.includes('theft') && (
            <Chip 
              size="small"
              icon={<WarningIcon />}
              label="Theft"
              color="error"
              variant={status === 'connected' ? "filled" : "outlined"}
            />
          )}
          
          {detectionType.includes('loitering') && (
            <Chip 
              size="small"
              icon={<PersonIcon />}
              label="Loitering"
              color="primary"
              variant={status === 'connected' ? "filled" : "outlined"}
            />
          )}
        </Box>
      </Box>
      
      {/* Camera Canvas */}
      <Box sx={{ 
        flex: 1, 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        position: 'relative',
        backgroundColor: '#f0f0f0'
      }}>
        {status === 'connecting' && (
          <Box sx={{ position: 'absolute', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <CircularProgress size={40} />
            <Typography variant="body2" sx={{ mt: 1 }}>Connecting to camera...</Typography>
          </Box>
        )}
        
        {status === 'error' && !connected && (
          <Box sx={{ position: 'absolute', textAlign: 'center', p: 2 }}>
            <Typography variant="body1" color="error">Connection Error</Typography>
            <Typography variant="body2">Could not connect to camera feed</Typography>
            <Button 
              startIcon={<RefreshIcon />} 
              size="small" 
              variant="outlined" 
              color="primary"
              onClick={handleReconnect}
              sx={{ mt: 2 }}
            >
              Reconnect
            </Button>
          </Box>
        )}
        
        <canvas 
          ref={canvasRef} 
          className="camera-canvas"
          style={{ maxWidth: '100%', maxHeight: '100%', display: status !== 'connecting' ? 'block' : 'none' }}
        />
      </Box>
      
      {/* Status & Alerts */}
      <Box sx={{ p: 0.5, borderTop: '1px solid rgba(0,0,0,0.1)', backgroundColor: '#f9f9f9' }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="caption" sx={{ display: 'flex', alignItems: 'center' }}>
            Status: 
            <Box 
              component="span" 
              sx={{ 
                width: 8, 
                height: 8, 
                borderRadius: '50%', 
                ml: 0.5, 
                mr: 0.5,
                backgroundColor: status === 'connected' 
                  ? (isVideoFile ? 'blue' : isMockFeed ? 'orange' : 'green') 
                  : status === 'connecting' ? 'orange' : 'red'
              }} 
            />
            <span style={{ marginLeft: '4px' }}>
              {status === 'connected' 
                ? (isVideoFile ? 'Video File' : isMockFeed ? 'Mock Feed' : 'Live Camera') 
                : status === 'connecting' ? 'Connecting' : 'Offline'}
            </span>
            
            {isVideoFile && videoInfo && (
              <span style={{ marginLeft: '8px', fontSize: '0.7rem' }}>
                ({videoInfo.filename})
              </span>
            )}
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Typography variant="caption">
              Detections: {stats.detections}
            </Typography>
            {stats.criticalEvents > 0 && (
              <Typography variant="caption" color="error">
                Critical Events: {stats.criticalEvents}
              </Typography>
            )}
          </Box>
        </Box>
        
        {alert && (
          <Alert 
            severity={
              alert.includes('Error') ? 'error' : 
              alert.includes('mock') ? 'warning' : 
              alert.includes('detected') ? 'warning' : 'info'
            } 
            sx={{ mt: 0.5, py: 0 }}
          >
            <Typography variant="caption">{alert}</Typography>
          </Alert>
        )}
      </Box>
    </Paper>
  );
};

export default CameraFeed;
