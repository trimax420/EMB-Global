import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Videos
export const getVideos = () => api.get('/videos');
export const getVideoDetections = (videoId) => api.get(`/videos/${videoId}/detections`);
export const uploadVideo = (formData) => api.post('/videos/upload', formData, {
  headers: { 'Content-Type': 'multipart/form-data' }
});
export const getVideoStatus = (videoId) => api.get(`/videos/${videoId}/status`);

// Incidents
export const getIncidents = (recent = false) => api.get('/incidents', { params: { recent } });

// System Status
export const getSystemStatus = () => api.get('/system-status');

// Billing Activity
export const getBillingActivity = (filter = 'all') => api.get('/billing-activity', { params: { filter } });

// Customer Data
export const getCustomerData = (filters) => api.get('/customer-data', { params: filters });

// Daily Reports
export const getDailyReport = (date) => api.get('/daily-report', { params: { date } });

// Error handler middleware
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    throw error;
  }
);

// Camera API
export const cameraService = {
  getAllCameras: () => api.get('/cameras'),
  getCameraById: (id) => api.get(`/cameras/${id}`),
  getCameraStream: (id) => api.get(`/cameras/${id}/stream`),
  updateCameraStatus: (id, status) => 
    api.put(`/cameras/${id}/status`, { status }),
};

// Alerts API
export const alertService = {
  getAlerts: (filter = 'recent') => 
    api.get('/alerts', { params: { filter } }),
  getMallStructure: () => 
    api.get('/alerts/mall-structure'),
  getHeatmapData: () => 
    api.get('/alerts/heatmap'),
};

// Billing Activity API
export const billingService = {
  getTransactionDetails: (transactionId) => 
    api.get(`/billing/${transactionId}`),
  getSuspiciousActivity: () => 
    api.get('/billing/suspicious'),
};

// Daily Report API
export const reportService = {
  getHourlyBreakdown: (date) => 
    api.get('/reports/hourly', { params: { date } }),
  exportReport: (date, format = 'csv') => 
    api.get('/reports/export', { params: { date, format } }),
};

// Customer Data API
export const customerService = {
  getRegularCustomers: () => 
    api.get('/customers/regular'),
  exportCustomerData: (filters = {}) => 
    api.post('/customers/export', filters),
};

// Dashboard/Home API
export const dashboardService = {
  getStatistics: () => 
    api.get('/dashboard/statistics'),
  getRecentIncidents: () => 
    api.get('/dashboard/incidents'),
  getQuickActions: () => 
    api.get('/dashboard/quick-actions'),
  getCrowdDensity: () => 
    api.get('/dashboard/crowd-density'),
};

export const incidentService = {
  // Get all incidents
  getAllIncidents: async (recentOnly = false) => {
    const url = `/incidents${recentOnly ? '?recent=true' : ''}`;
    const response = await api.get(url);
    if (!response.ok) {
      throw new Error('Failed to fetch incidents');
    }
    return response.data;
  }
}; 