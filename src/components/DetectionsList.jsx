import React, { useState, useEffect } from 'react';
import { 
  Box, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemIcon, 
  Typography, 
  Divider,
  Paper,
  Chip,
  IconButton
} from '@mui/material';

// Simple SVG icon components to replace Material UI icons
const WarningIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 2L1 21h22L12 2z"></path>
    <path d="M12 9v4"></path>
    <path d="M12 16h.01"></path>
  </svg>
);

const PersonIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
    <circle cx="12" cy="7" r="4"></circle>
  </svg>
);

const VideocamIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M23 7l-7 5 7 5V7z"></path>
    <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
  </svg>
);

// Replace CircularProgress component with a simple loading indicator
const CircularProgress = () => (
  <div className="loading-spinner" style={{ 
    width: '24px', 
    height: '24px', 
    border: '3px solid #f3f3f3',
    borderTop: '3px solid #3498db',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite'
  }}></div>
);

// Add this CSS somewhere in your component or in a global stylesheet
const spinnerStyles = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const DetectionsList = ({ 
  title = 'Recent Detections', 
  maxItems = 5,
  websocketRef = null,  // Optional websocket reference for real-time updates
  detectionType = 'all',
  backendUrl = 'localhost:8000'
}) => {
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch recent detections from API
  const fetchDetections = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Use backendUrl instead of relative URL to ensure it works in all environments
      const apiUrl = `http://${backendUrl}/api/detections/recent?limit=${maxItems}&detection_type=${detectionType}`;
      
      const response = await fetch(apiUrl);
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      
      const data = await response.json();
      setDetections(data);
    } catch (err) {
      console.error('Error fetching detections:', err);
      setError('Failed to load detections');
      
      // Add dummy data for testing when server is unavailable
      if (process.env.NODE_ENV === 'development') {
        console.log('Using dummy data for development');
        setDetections(generateDummyDetections(maxItems));
      }
    } finally {
      setLoading(false);
    }
  };

  // Generate dummy detection data for testing purposes
  const generateDummyDetections = (count = 5) => {
    return Array(count).fill(null).map((_, i) => ({
      id: `dummy-${i+1}`,
      detection_type: i % 2 === 0 ? 'theft' : 'loitering',
      timestamp: new Date(Date.now() - i * 15 * 60000).toISOString(), // Each 15 minutes back
      confidence: i % 2 === 0 ? Math.random() * 0.5 + 0.5 : Math.floor(Math.random() * 20) + 5,
      camera_id: String(Math.floor(Math.random() * 3)),
      severity: i < 2 ? 'high' : i < 4 ? 'medium' : 'low'
    }));
  };

  // Initial fetch
  useEffect(() => {
    fetchDetections();
    
    // Set up an interval to refresh data regularly
    const interval = setInterval(fetchDetections, 60000);  // Refresh every minute
    
    return () => clearInterval(interval);
  }, [detectionType, maxItems]);

  // Listen for WebSocket updates if provided
  useEffect(() => {
    if (!websocketRef || !websocketRef.current) return;
    
    const handleMessage = (message) => {
      try {
        // Check if message is a string or already parsed
        const data = typeof message === 'string' ? JSON.parse(message) : message;
        
        // Handle new detection messages
        if (data.type === 'new_incident' || data.type === 'theft_detection_completed' || 
            data.type === 'loitering_detection_completed') {
          
          // Add the new detection to the list and maintain max items
          setDetections(prevDetections => {
            const newDetection = {
              id: data.incident?.id || `temp-${Date.now()}`,
              detection_type: data.incident?.type || data.type.split('_')[0],
              timestamp: data.incident?.timestamp || new Date().toISOString(),
              confidence: data.incident?.confidence || 0.8,
              image_path: data.incident?.image_path || '',
              camera_id: data.incident?.camera_id || 'unknown',
              severity: data.incident?.severity || 'medium'
            };
            
            return [newDetection, ...prevDetections].slice(0, maxItems);
          });
        }
        // Handle inference results with critical detections
        else if (data.type === 'inference_result' && data.ui_metadata?.has_critical_detection) {
          // Add the new detection to the list and maintain max items
          setDetections(prevDetections => {
            const detectionType = data.detections?.theft?.detected ? 'theft' : 'loitering';
            const confidence = data.detections?.theft?.confidence || 0;
            const duration = data.detections?.loitering?.duration || 0;
            
            const newDetection = {
              id: `temp-${Date.now()}`,
              detection_type: detectionType,
              timestamp: data.timestamp || new Date().toISOString(),
              confidence: detectionType === 'theft' ? confidence : duration,
              camera_id: data.camera_id || 'unknown',
              severity: confidence > 0.8 || duration > 15 ? 'high' : 'medium'
            };
            
            return [newDetection, ...prevDetections].slice(0, maxItems);
          });
        }
      } catch (err) {
        console.error('Error processing WebSocket message', err);
      }
    };
    
    // Check if websocketRef has lastMessage property (new format)
    if (websocketRef.current.lastMessage) {
      handleMessage(websocketRef.current.lastMessage);
    }

    // Set up an effect to process new messages as they arrive
    const intervalId = setInterval(() => {
      if (websocketRef.current.lastMessage) {
        handleMessage(websocketRef.current.lastMessage);
      }
    }, 1000); // Check for new messages every second
    
    return () => {
      clearInterval(intervalId);
    };
  }, [websocketRef, maxItems]);

  const getSeverityColor = (severity) => {
    switch(severity) {
      case 'high': return '#f44336';
      case 'medium': return '#ff9800';
      case 'low': return '#4caf50';
      default: return '#757575';
    }
  };

  const getIcon = (type) => {
    switch(type) {
      case 'theft': return <WarningIcon />;
      case 'loitering': return <PersonIcon />;
      default: return <VideocamIcon />;
    }
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return 'Unknown';
    const date = new Date(timestamp);
    return isNaN(date.getTime()) 
      ? 'Invalid date' 
      : date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatDate = (timestamp) => {
    if (!timestamp) return 'Unknown';
    const date = new Date(timestamp);
    return isNaN(date.getTime()) 
      ? 'Invalid date' 
      : date.toLocaleDateString();
  };

  return (
    <Paper elevation={3} sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ 
        p: 2, 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        borderBottom: '1px solid rgba(0,0,0,0.1)'
      }}>
        <Typography variant="h6" component="div">
          {title}
        </Typography>
        <IconButton onClick={fetchDetections} disabled={loading} size="small">
          {loading ? <CircularProgress /> : <span>🔄</span>}
        </IconButton>
      </Box>
      
      {error && (
        <Box sx={{ p: 2, color: 'error.main' }}>
          <Typography>{error}</Typography>
        </Box>
      )}
      
      {!error && detections.length === 0 && !loading && (
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            No detections found
          </Typography>
        </Box>
      )}
      
      {loading && detections.length === 0 && (
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <CircularProgress />
        </Box>
      )}
      
      <List sx={{ 
        flex: 1, 
        overflow: 'auto', 
        '& .MuiListItem-root': { py: 1.5 }
      }}>
        {detections.map((detection, index) => (
          <React.Fragment key={detection.id || index}>
            <ListItem 
              alignItems="flex-start"
              secondaryAction={
                <IconButton edge="end" size="small">
                  <span>👁️</span>
                </IconButton>
              }
            >
              <ListItemIcon sx={{ minWidth: 40 }}>
                {getIcon(detection.detection_type)}
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="subtitle2" component="div">
                      {detection.detection_type === 'theft' ? 'Theft Detection' : 'Loitering Detection'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {formatTime(detection.timestamp)}
                    </Typography>
                  </Box>
                }
                secondary={
                  <Box sx={{ mt: 0.5 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="caption" color="text.secondary">
                        {formatDate(detection.timestamp)} • Camera {detection.camera_id || 'Unknown'}
                      </Typography>
                      <Chip
                        label={detection.severity}
                        size="small"
                        sx={{
                          height: 20,
                          fontSize: '0.7rem',
                          color: 'white',
                          bgcolor: getSeverityColor(detection.severity)
                        }}
                      />
                    </Box>
                    <Typography variant="body2" sx={{ mt: 0.5 }}>
                      {detection.detection_type === 'theft' 
                        ? `Potential theft detected with ${Math.round(detection.confidence * 100)}% confidence.` 
                        : `Person loitering for ${Math.round(detection.confidence)}s in restricted area.`}
                    </Typography>
                  </Box>
                }
              />
            </ListItem>
            {index < detections.length - 1 && <Divider variant="inset" component="li" />}
          </React.Fragment>
        ))}
      </List>
    </Paper>
  );
};

export default DetectionsList;
