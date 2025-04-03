class WebSocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.callbacks = {};
  }

  connect(url) {
    if (this.socket) {
      this.disconnect();
    }

    this.socket = new WebSocket(url);

    this.socket.onopen = () => {
      this.isConnected = true;
      this.triggerCallback('connect');
    };

    this.socket.onclose = () => {
      this.isConnected = false;
      this.triggerCallback('disconnect');
    };

    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.triggerCallback('message', data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.triggerCallback('error', error);
    };
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.isConnected = false;
    }
  }

  send(data) {
    if (this.isConnected) {
      this.socket.send(JSON.stringify(data));
    } else {
      console.error('Cannot send message: WebSocket not connected');
    }
  }

  on(event, callback) {
    if (!this.callbacks[event]) {
      this.callbacks[event] = [];
    }
    this.callbacks[event].push(callback);
  }

  off(event, callback) {
    if (this.callbacks[event]) {
      this.callbacks[event] = this.callbacks[event].filter(cb => cb !== callback);
    }
  }

  triggerCallback(event, data) {
    if (this.callbacks[event]) {
      this.callbacks[event].forEach(callback => callback(data));
    }
  }
}

export const websocketService = new WebSocketService();
export default websocketService;
