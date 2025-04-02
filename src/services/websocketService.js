// src/services/websocketService.js

/**
 * Service for WebSocket communication
 */

let socket = null;
let isConnected = false;
const messageListeners = new Map();
const reconnectDelay = 3000; // 3 seconds
let reconnectAttempt = 0;
const maxReconnectAttempts = 5;

/**
 * Connect to WebSocket server
 * @returns {Promise<WebSocket>} WebSocket connection
 */
export const connectWebSocket = () => {
  return new Promise((resolve, reject) => {
    // If already connected, return existing socket
    if (socket && isConnected) {
      resolve(socket);
      return;
    }
    
    // Get WebSocket URL from environment or use default
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/ws';
    
    try {
      // Create new WebSocket
      socket = new WebSocket(wsUrl);
      
      // Set up event handlers
      socket.onopen = () => {
        console.log('WebSocket connected');
        isConnected = true;
        reconnectAttempt = 0;
        resolve(socket);
      };
      
      socket.onclose = (event) => {
        console.log('WebSocket disconnected', event);
        isConnected = false;
        
        // Try to reconnect if not closed deliberately
        if (!event.wasClean && reconnectAttempt < maxReconnectAttempts) {
          reconnectAttempt++;
          console.log(`Reconnecting attempt ${reconnectAttempt}/${maxReconnectAttempts}...`);
          
          setTimeout(() => {
            connectWebSocket()
              .then(newSocket => {
                socket = newSocket;
                isConnected = true;
                console.log('WebSocket reconnected');
              })
              .catch(error => {
                console.error('WebSocket reconnection failed:', error);
              });
          }, reconnectDelay);
        }
      };
      
      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };
      
      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const messageType = data.type;
          
          // Dispatch to specific message listeners
          if (messageListeners.has(messageType)) {
            messageListeners.get(messageType)(data);
          }
          
          // Dispatch to "all" listeners
          if (messageListeners.has('all')) {
            messageListeners.get('all')(data);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      reject(error);
    }
  });
};

/**
 * Add a message listener
 * @param {string} messageType - Type of message to listen for
 * @param {Function} callback - Callback function
 */
export const addMessageListener = (messageType, callback) => {
  messageListeners.set(messageType, callback);
};

/**
 * Remove a message listener
 * @param {string} messageType - Type of message to remove listener for
 */
export const removeMessageListener = (messageType) => {
  messageListeners.delete(messageType);
};

/**
 * Send a message through WebSocket
 * @param {Object} message - Message to send
 * @returns {Promise<void>}
 */
export const sendMessage = async (message) => {
  // Connect if not already connected
  if (!socket || !isConnected) {
    await connectWebSocket();
  }
  
  // Send message
  socket.send(JSON.stringify(message));
};

/**
 * Close WebSocket connection
 */
export const closeConnection = () => {
  if (socket) {
    socket.close();
    socket = null;
    isConnected = false;
  }
};

/**
 * Get connection status
 * @returns {boolean} - Whether WebSocket is connected
 */
export const getConnectionStatus = () => {
  return isConnected;
};

// Setup auto-reconnect when window regains focus
if (typeof window !== 'undefined') {
  window.addEventListener('focus', () => {
    if (!isConnected) {
      connectWebSocket().catch(error => {
        console.error('Failed to reconnect on window focus:', error);
      });
    }
  });
}

export default {
  connectWebSocket,
  addMessageListener,
  removeMessageListener,
  sendMessage,
  closeConnection,
  isConnected: getConnectionStatus
};