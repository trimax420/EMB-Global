// src/services/websocketService.js

/**
 * WebSocket service for real-time communications
 */

let socket = null;
let messageListeners = {};
let isConnected = false;

/**
 * Connect to the WebSocket server
 * @returns {Promise<void>} - Promise that resolves when connected
 */
export const connectWebSocket = () => {
  return new Promise((resolve, reject) => {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/ws';
    socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
      console.log('WebSocket connected');
      isConnected = true;
      resolve();
    };
    
    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const messageType = data.type;
        
        if (messageListeners[messageType]) {
          messageListeners[messageType].forEach(callback => {
            callback(data);
          });
        }
        
        // Also trigger any 'all' listeners
        if (messageListeners['all']) {
          messageListeners['all'].forEach(callback => {
            callback(data);
          });
        }
      } catch (error) {
        console.error('WebSocket message error:', error);
      }
    };
    
    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      reject(error);
    };
    
    socket.onclose = () => {
      console.log('WebSocket disconnected');
      isConnected = false;
    };
  });
};

/**
 * Add a message listener for a specific message type
 * @param {string} messageType - Type of message to listen for
 * @param {Function} callback - Callback function when message received
 */
export const addMessageListener = (messageType, callback) => {
  if (!messageListeners[messageType]) {
    messageListeners[messageType] = [];
  }
  messageListeners[messageType].push(callback);
};

/**
 * Remove a message listener
 * @param {string} messageType - Type of message to stop listening for
 * @param {Function} callback - Callback function to remove (optional)
 */
export const removeMessageListener = (messageType, callback) => {
  if (!messageListeners[messageType]) return;
  
  if (callback) {
    messageListeners[messageType] = messageListeners[messageType].filter(cb => cb !== callback);
  } else {
    delete messageListeners[messageType];
  }
};

/**
 * Send a message to the WebSocket server
 * @param {Object} message - Message to send
 */
export const sendMessage = (message) => {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    throw new Error('WebSocket is not connected');
  }
  
  socket.send(JSON.stringify(message));
};

/**
 * Start real-time video inference
 * @param {Object} params - Inference parameters
 */
export const startRealTimeInference = (params) => {
  const wsUrl = import.meta.env.VITE_WS_INFERENCE_URL || 'ws://localhost:8000/api/ws/inference';
  const inferenceSocket = new WebSocket(wsUrl);
  
  inferenceSocket.onopen = () => {
    inferenceSocket.send(JSON.stringify({
      type: 'start_inference',
      params
    }));
  };
  
  return inferenceSocket;
};

/**
 * Send a video frame for real-time inference
 * @param {WebSocket} inferenceSocket - WebSocket connection for inference
 * @param {ImageData|Blob} frame - Video frame to analyze
 */
export const sendFrameForInference = (inferenceSocket, frame) => {
  if (inferenceSocket.readyState === WebSocket.OPEN) {
    inferenceSocket.send(JSON.stringify({
      type: 'frame',
      frame: frame instanceof Blob ? frame : frame.data
    }));
  }
};

/**
 * Close the WebSocket connection
 */
export const closeConnection = () => {
  if (socket) {
    socket.close();
    socket = null;
  }
};

export default {
  connectWebSocket,
  addMessageListener,
  removeMessageListener,
  sendMessage,
  startRealTimeInference,
  sendFrameForInference,
  closeConnection,
  get isConnected() { return isConnected; }
};