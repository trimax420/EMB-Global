import React, { useRef, useState, useEffect } from 'react';
import CameraFeed from '../components/CameraFeed';
import DetectionsList from '../components/DetectionsList';
import StatsSummary from '../components/StatsSummary';
import useWebSocket from '../hooks/useWebSocket';
import { Box, Typography, Chip, Alert, Snackbar, Paper, Stack } from '@mui/material';
import WarningIcon from '@mui/icons-material/Warning';
import PersonIcon from '@mui/icons-material/Person';

// Get backend URL from environment or use default
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'localhost:8000';

const Home = () => {
  // Create a WebSocket reference for sharing with child components
  const mainWebSocketUrl = `ws://${BACKEND_URL}/api/ws/realtime-detection?detection_type=both&camera_id=3`;
  const { connected, lastMessage, error, sendMessage } = useWebSocket(mainWebSocketUrl, {
    reconnectInterval: 3000,
    maxReconnectAttempts: 20
  });
  
  // Keep a reference to allow child components to use the same websocket data
  const websocketRef = useRef({ connected, lastMessage, error });
  
  // State for notifications and status
  const [notification, setNotification] = useState(null);
  const [detectionStats, setDetectionStats] = useState({
    theftCount: 0,
    loiteringCount: 0,
    lastDetectionTime: null
  });

  // Update websocket reference when data changes
  useEffect(() => {
    websocketRef.current = { connected, lastMessage, error };
    
    // Show connection status in console for debugging
    if (error) {
      console.log('WebSocket connection error:', error);
    }
  }, [connected, lastMessage, error]);

  // Process incoming WebSocket messages for global state management
  useEffect(() => {
    if (!lastMessage) return;
    
    try {
      const data = typeof lastMessage === 'string' ? JSON.parse(lastMessage) : lastMessage;
      
      if (data.type === 'inference_result') {
        // Handle theft detection
        if (data.detections?.theft?.detected) {
          setDetectionStats(prev => ({
            ...prev,
            theftCount: prev.theftCount + 1,
            lastDetectionTime: new Date()
          }));
          
          setNotification({
            type: 'theft',
            message: `Theft detected with ${(data.detections.theft.confidence * 100).toFixed(1)}% confidence`,
            severity: 'error'
          });
        }
        
        // Handle loitering detection
        if (data.detections?.loitering?.detected) {
          setDetectionStats(prev => ({
            ...prev,
            loiteringCount: prev.loiteringCount + 1,
            lastDetectionTime: new Date()
          }));
          
          setNotification({
            type: 'loitering',
            message: `Loitering detected for ${data.detections.loitering.duration.toFixed(1)} seconds`,
            severity: 'warning'
          });
        }
      } 
      else if (data.type === 'new_incident') {
        // Handle new incidents from the server
        setNotification({
          type: data.incident.type,
          message: `New ${data.incident.type} incident detected with ${data.incident.severity} severity`,
          severity: data.incident.type === 'theft' ? 'error' : 'warning'
        });
      }
      else if (data.type === 'error') {
        setNotification({
          type: 'error',
          message: data.message,
          severity: 'error'
        });
      }
    } catch (e) {
      console.error('Error processing message:', e);
    }
  }, [lastMessage]);

  // Close notification after timeout
  const handleCloseNotification = () => {
    setNotification(null);
  };

  return (
    <Paper elevation={0} sx={{ p: 3, maxWidth: '1400px', mx: 'auto', bgcolor: 'background.default' }}>
      <Stack spacing={3}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h4" component="h1" fontWeight="bold">
            Security Dashboard
          </Typography>
          
          <Stack direction="row" spacing={1.5}>
            <Chip 
              icon={<WarningIcon />}
              label={`Theft Alerts: ${detectionStats.theftCount}`}
              color="error"
              variant={detectionStats.theftCount > 0 ? "filled" : "outlined"}
            />
            <Chip 
              icon={<PersonIcon />}
              label={`Loitering Alerts: ${detectionStats.loiteringCount}`}
              color="warning"
              variant={detectionStats.loiteringCount > 0 ? "filled" : "outlined"}
            />
            <Chip 
              label={connected ? "Backend Connected" : "Disconnected"}
              color={connected ? "success" : "error"}
            />
          </Stack>
        </Box>

        {/* Multi-Camera View CTA */}
        <Paper 
          elevation={0}
          sx={{ 
            p: 2, 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            bgcolor: 'rgba(25, 118, 210, 0.08)', 
            borderRadius: 2,
            border: '1px solid rgba(25, 118, 210, 0.2)'
          }}
        >
          <Box>
            <Typography variant="h6" color="primary">
              New! Multi-Camera Video Files View
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Test our security system with actual video footage. Stream all cameras simultaneously with real-time detection.
            </Typography>
          </Box>
          <a href="/multicamera" style={{ textDecoration: 'none' }}>
            <Box 
              sx={{ 
                bgcolor: 'primary.main',
                color: 'white',
                px: 3,
                py: 1.5,
                borderRadius: 2,
                fontWeight: 'bold',
                '&:hover': {
                  bgcolor: 'primary.dark'
                }
              }}
            >
              View All Cameras
            </Box>
          </a>
        </Paper>

        {/* Main Content Grid */}
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '3fr 1fr' }, gap: 3 }}>
          {/* Camera Feeds Section */}
          <Stack spacing={2}>
            {/* Top Row - Two Cameras */}
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2 }}>
              <CameraFeed 
                cameraId="1" 
                detectionType="theft" 
                title="Cheese Store"
                height="320px"
                backendUrl={BACKEND_URL}
              />
              <CameraFeed 
                cameraId="4" 
                detectionType="loitering" 
                title="Cleaning Section"
                height="320px"
                backendUrl={BACKEND_URL}
              />
            </Box>
            
            {/* Bottom Row - Main Camera */}
            <CameraFeed 
              cameraId="3" 
              detectionType="both" 
              title="Cheese Section"
              height="400px"
              backendUrl={BACKEND_URL}
            />
          </Stack>

          {/* Sidebar - Detections & Stats */}
          <Stack spacing={2}>
            <DetectionsList 
              title="Recent Incidents" 
              maxItems={7}
              websocketRef={websocketRef} 
              backendUrl={BACKEND_URL}
            />
            <StatsSummary 
              websocketRef={websocketRef}
              theftCount={detectionStats.theftCount}
              loiteringCount={detectionStats.loiteringCount}
              lastDetectionTime={detectionStats.lastDetectionTime}
            />
          </Stack>
        </Box>
      </Stack>
      
      {/* Notification Snackbar */}
      <Snackbar 
        open={Boolean(notification)} 
        autoHideDuration={6000} 
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        {notification && (
          <Alert 
            onClose={handleCloseNotification} 
            severity={notification.severity} 
            sx={{ width: '100%' }}
          >
            {notification.message}
          </Alert>
        )}
      </Snackbar>
    </Paper>
  );
};

export default Home;