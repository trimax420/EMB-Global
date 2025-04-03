import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Custom hook for WebSocket connections with auto reconnect
 * @param {string} url - The WebSocket URL to connect to
 * @param {Object} options - Configuration options
 * @param {number} options.reconnectInterval - Time in ms to attempt reconnection (default: 2000)
 * @param {number} options.maxReconnectAttempts - Max number of reconnect attempts (default: 10)
 * @returns {Object} WebSocket connection state and methods
 */
const useWebSocket = (url, options = {}) => {
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [error, setError] = useState(null);
  const socketRef = useRef(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimerRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  const {
    reconnectInterval = 2000,
    maxReconnectAttempts = 10,
    onOpen,
    onMessage,
    onError,
    onClose,
    debug = false
  } = options;

  // Debug logger
  const log = useCallback((message, ...args) => {
    if (debug) {
      console.log(`[WebSocket] ${message}`, ...args);
    }
  }, [debug]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    // Clear any existing connection
    if (socketRef.current) {
      log('Closing existing connection before reconnect');
      socketRef.current.close();
    }

    try {
      log(`Attempting to connect to ${url}`);
      // Create new WebSocket connection
      const ws = new WebSocket(url);
      socketRef.current = ws;

      // Connection opened
      ws.onopen = (event) => {
        log('Connection opened', event);
        setConnected(true);
        setError(null);
        reconnectCountRef.current = 0;
        if (onOpen) onOpen(event);
      };

      // Listen for messages
      ws.onmessage = (event) => {
        setLastMessage(event.data);
        if (onMessage) onMessage(event.data);
      };

      // Connection closed
      ws.onclose = (event) => {
        log('Connection closed', event);
        setConnected(false);
        if (onClose) onClose(event);
        
        // Only attempt reconnect if not a clean closure
        if (!event.wasClean) {
          log('Connection closed uncleanly, scheduling reconnect');
          scheduleReconnect();
        }
      };

      // Connection error
      ws.onerror = (event) => {
        log('Connection error', event);
        setError('WebSocket connection error');
        if (onError) onError(event);
      };
    } catch (err) {
      log('Failed to create WebSocket', err);
      setError(`Failed to create WebSocket: ${err.message}`);
      scheduleReconnect();
    }
  }, [url, onOpen, onMessage, onClose, onError, log]);

  // Schedule a reconnection attempt
  const scheduleReconnect = useCallback(() => {
    if (reconnectCountRef.current >= maxReconnectAttempts) {
      log(`Failed to connect after ${maxReconnectAttempts} attempts`);
      setError(`Failed to connect after ${maxReconnectAttempts} attempts`);
      return;
    }

    // Clear any existing timers
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }

    log(`Scheduling reconnect attempt ${reconnectCountRef.current + 1}/${maxReconnectAttempts} in ${reconnectInterval}ms`);
    reconnectTimerRef.current = setTimeout(() => {
      reconnectCountRef.current += 1;
      connect();
    }, reconnectInterval);
  }, [connect, maxReconnectAttempts, reconnectInterval, log]);

  // Send a message through the WebSocket
  const sendMessage = useCallback((message) => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      log('Cannot send message: WebSocket is not connected');
      setError('Cannot send message: WebSocket is not connected');
      return false;
    }

    try {
      socketRef.current.send(typeof message === 'string' ? message : JSON.stringify(message));
      return true;
    } catch (err) {
      log('Failed to send message', err);
      setError(`Failed to send message: ${err.message}`);
      return false;
    }
  }, [log]);

  // Initial connection and cleanup
  useEffect(() => {
    log('Initializing WebSocket connection to', url);
    connect();

    // Setup a ping interval to keep connection alive
    const pingInterval = setInterval(() => {
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        log('Sending ping to keep connection alive');
        sendMessage(JSON.stringify({ type: 'ping' }));
      }
    }, 30000); // Ping every 30 seconds

    // Cleanup on unmount
    return () => {
      log('Cleaning up WebSocket connection');
      if (socketRef.current) {
        socketRef.current.close();
      }
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      clearInterval(pingInterval);
    };
  }, [connect, url, sendMessage, log]);

  return {
    connected,
    lastMessage,
    error,
    sendMessage,
    reconnect: connect
  };
};

export default useWebSocket;
