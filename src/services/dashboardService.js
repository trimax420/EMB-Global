// src/services/dashboardService.js
/**
 * Service for dashboard statistics and overview data
 */

import { get } from './api';

/**
 * Get security dashboard summary statistics
 * @param {string} timeframe - Time period (today, week, month)
 * @returns {Promise<Object>} - Dashboard summary data
 */
export const getDashboardSummary = async (timeframe = 'today') => {
  return get(`/dashboard/summary?timeframe=${timeframe}`);
};

/**
 * Get camera status information
 * @returns {Promise<Array>} - List of cameras with status
 */
export const getCameraStatus = async () => {
  return get('/dashboard/cameras');
};

/**
 * Get detection trends over time
 * @param {string} timeframe - Time period (today, week, month)
 * @returns {Promise<Object>} - Detection trend data
 */
export const getDetectionTrends = async (timeframe = 'today') => {
  return get(`/dashboard/detection-trends?timeframe=${timeframe}`);
};

/**
 * Get recent incidents for dashboard
 * @param {number} limit - Number of incidents to return
 * @returns {Promise<Array>} - List of recent incidents
 */
export const getRecentIncidents = async (limit = 5) => {
  return get(`/dashboard/recent-incidents?limit=${limit}`);
};

/**
 * Get system status information
 * @returns {Promise<Object>} - System status data
 */
export const getSystemStatus = async () => {
  return get('/dashboard/system-status');
};

/**
 * Get incident statistics by location
 * @returns {Promise<Array>} - Location-based incident data
 */
export const getLocationStats = async () => {
  return get('/dashboard/location-stats');
};

/**
 * Get security metrics by time of day
 * @returns {Promise<Object>} - Time-based security metrics
 */
export const getTimeOfDayMetrics = async () => {
  return get('/dashboard/time-metrics');
};

/**
 * Get detailed security report
 * @param {string} timeframe - Time period (daily, weekly, monthly)
 * @returns {Promise<Object>} - Detailed security report
 */
export const getSecurityReport = async (timeframe = 'daily') => {
  return get(`/dashboard/report?timeframe=${timeframe}`);
};

/**
 * Export dashboard report
 * @param {string} timeframe - Time period (daily, weekly, monthly)
 * @param {string} format - Export format (pdf, csv)
 * @returns {Promise<Blob>} - Report file
 */
export const exportDashboardReport = async (timeframe = 'daily', format = 'pdf') => {
  // Use raw fetch for blob handling
  const response = await fetch(
    `${import.meta.env.VITE_API_BASE_URL}/dashboard/export?timeframe=${timeframe}&format=${format}`, 
    {
      method: 'GET',
      headers: {
        'Accept': format === 'pdf' ? 'application/pdf' : 'text/csv',
      }
    }
  );
  
  if (!response.ok) {
    throw new Error(`Export failed: ${response.status}`);
  }
  
  return await response.blob();
};