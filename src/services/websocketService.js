// src/services/websocketService.js
/**
 * Service for handling real-time WebSocket communication
 */

let socket = null;
const listeners = new Map();
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY = 3000; // 3 seconds

/**
 * Connect to the WebSocket server
 * @returns {Promise<WebSocket>} - WebSocket connection
 */
export const connectWebSocket = () => {
  return new Promise((resolve, reject) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      resolve(socket);
      return;
    }

    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/ws';
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log('WebSocket connection established');
      reconnectAttempts = 0;
      resolve(socket);
    };

    socket.onclose = (event) => {
      console.log('WebSocket connection closed', event);
      
      // Attempt to reconnect if not closed deliberately
      if (!event.wasClean && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        console.log(`Attempting to reconnect (${reconnectAttempts + 1}/${MAX_RECONNECT_ATTEMPTS})...`);
        reconnectAttempts++;
        setTimeout(() => {
          connectWebSocket()
            .then(newSocket => {
              socket = newSocket;
              // Re-register all listeners
              listeners.forEach((callback, messageType) => {
                addMessageListener(messageType, callback);
              });
            })
            .catch(error => console.error('Reconnection failed:', error));
        }, RECONNECT_DELAY);
      }
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      reject(error);
    };

    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        const messageType = message.type;
        
        if (listeners.has(messageType)) {
          listeners.get(messageType)(message);
        }
        
        // Also trigger 'all' listeners
        if (listeners.has('all')) {
          listeners.get('all')(message);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
  });
};

/**
 * Add a message listener for specific message types
 * @param {string} messageType - Type of message to listen for, or 'all' for all messages
 * @param {Function} callback - Callback function to handle the message
 */
export const addMessageListener = (messageType, callback) => {
  listeners.set(messageType, callback);
};

/**
 * Remove a message listener
 * @param {string} messageType - Type of message to stop listening for
 */
export const removeMessageListener = (messageType) => {
  listeners.delete(messageType);
};

/**
 * Send a message to the WebSocket server
 * @param {Object} message - Message to send
 * @returns {Promise<void>}
 */
export const sendMessage = async (message) => {
  try {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      await connectWebSocket();
    }
    
    socket.send(JSON.stringify(message));
  } catch (error) {
    console.error('Error sending WebSocket message:', error);
    throw error;
  }
};

/**
 * Start camera stream
 * @param {string} cameraId - Camera ID to stream
 * @returns {Promise<void>}
 */
export const startCameraStream = async (cameraId) => {
  return sendMessage({
    type: 'start_stream',
    camera_id: cameraId
  });
};

/**
 * Stop camera stream
 * @param {string} cameraId - Camera ID to stop streaming
 * @returns {Promise<void>}
 */
export const stopCameraStream = async (cameraId) => {
  return sendMessage({
    type: 'stop_stream',
    camera_id: cameraId
  });
};

/**
 * Close WebSocket connection
 */
export const closeConnection = () => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.close();
  }
};

// Auto-reconnect on window focus if connection was lost
if (typeof window !== 'undefined') {
  window.addEventListener('focus', () => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      connectWebSocket().catch(error => {
        console.error('Failed to reconnect on window focus:', error);
      });
    }
  });
}