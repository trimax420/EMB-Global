// Backend API configuration
export const BACKEND_URL = 'http://localhost:8000';

// WebSocket configuration
export const WS_URL = 'ws://localhost:8000';

// API endpoints
export const API_ENDPOINTS = {
    cameras: '/api/cameras',
    incidents: '/api/incidents',
    systemStatus: '/api/system-status',
    detections: '/api/detections',
    dashboard: '/api/dashboard'
};

// WebSocket endpoints
export const WS_ENDPOINTS = {
    live: '/ws',
    dashboard: '/ws/dashboard'
};

// Update intervals (in milliseconds)
export const UPDATE_INTERVALS = {
    dashboard: 1000,  // 1 second
    detections: 100   // 100 milliseconds
}; 