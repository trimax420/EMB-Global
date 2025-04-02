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
  
  // Function to initialize the WebSocket connection
  const connectWebSocket = () => {
    if (socket.current && socket.current.readyState === WebSocket.OPEN) {
      return; // Already connected
    }
    
    const ws = new WebSocket('ws://localhost:8000/api/ws');
    
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
    socket: socket.current
  };
  
  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

export default WebSocketProvider;