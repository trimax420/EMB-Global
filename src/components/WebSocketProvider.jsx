import React, { createContext, useContext, useEffect, useState, useRef } from 'react';

// Create a context for the WebSocket
const WebSocketContext = createContext(null);

// Custom hook to use the WebSocket
export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

// WebSocket Provider component
export const WebSocketProvider = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const socket = useRef(null);
  const inferenceConnections = useRef({});
  
  // Function to initialize the WebSocket connection
  const connectWebSocket = () => {
    if (socket.current && socket.current.readyState === WebSocket.OPEN) {
      return; // Already connected
    }
    
    // Get WebSocket URL from current location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const baseUrl = `${protocol}//${window.location.host}`;
    const wsUrl = `${baseUrl}/api/ws`;
    
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      socket.current = ws;
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setLastMessage(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      // Try to reconnect after a delay
      setTimeout(() => {
        connectWebSocket();
      }, 3000);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      ws.close();
    };
    
    socket.current = ws;
  };
  
  // Function to connect to inference WebSocket
  const connectToInferenceWebSocket = (cameraId) => {
    // Check if we already have a connection
    if (
      inferenceConnections.current[cameraId] && 
      inferenceConnections.current[cameraId].readyState === WebSocket.OPEN
    ) {
      return inferenceConnections.current[cameraId];
    }
    
    // Get WebSocket URL from current location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const baseUrl = `${protocol}//${window.location.host}`;
    const wsUrl = `${baseUrl}/api/ws/inference`;
    
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log(`Inference WebSocket connected for camera ${cameraId}`);
      
      // Subscribe to camera
      ws.send(JSON.stringify({
        action: 'subscribe',
        camera_id: cameraId
      }));
    };
    
    ws.onclose = () => {
      console.log(`Inference WebSocket disconnected for camera ${cameraId}`);
      
      // Remove from connections
      delete inferenceConnections.current[cameraId];
      
      // Try to reconnect after a delay
      setTimeout(() => {
        if (cameraId) {
          connectToInferenceWebSocket(cameraId);
        }
      }, 3000);
    };
    
    ws.onerror = (error) => {
      console.error(`Inference WebSocket error for camera ${cameraId}:`, error);
      ws.close();
    };
    
    // Store connection
    inferenceConnections.current[cameraId] = ws;
    
    return ws;
  };
  
  // Function to send a message through the WebSocket
  const sendMessage = (data) => {
    if (socket.current && socket.current.readyState === WebSocket.OPEN) {
      socket.current.send(JSON.stringify(data));
    } else {
      console.error('WebSocket is not connected');
    }
  };
  
  // Connect on component mount
  useEffect(() => {
    connectWebSocket();
    
    // Cleanup on unmount
    return () => {
      if (socket.current) {
        socket.current.close();
      }
      
      // Close all inference connections
      Object.values(inferenceConnections.current).forEach(ws => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.close();
        }
      });
    };
  }, []);
  
  // Create a heartbeat to keep the connection alive
  useEffect(() => {
    const interval = setInterval(() => {
      if (socket.current && socket.current.readyState === WebSocket.OPEN) {
        socket.current.send(JSON.stringify({ type: 'ping' }));
      } else if (!isConnected) {
        connectWebSocket();
      }
    }, 30000); // Send a ping every 30 seconds
    
    return () => clearInterval(interval);
  }, [isConnected]);
  
  // Context value
  const value = {
    isConnected,
    lastMessage,
    sendMessage,
    socket: socket.current,
    connectToInferenceWebSocket
  };
  
  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

export default WebSocketProvider;