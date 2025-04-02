// src/services/incidentsService.js
/**
 * Service for managing security incidents and alerts
 */

import { get, post, put, del } from './api';

/**
 * Get all incidents with optional filtering
 * @param {Object} filters - Filter parameters (type, severity, recent)
 * @returns {Promise<Array>} - List of incidents
 */
export const getIncidents = async (filters = {}) => {
  const queryParams = new URLSearchParams();
  
  if (filters.type) queryParams.append('type', filters.type);
  if (filters.severity) queryParams.append('severity', filters.severity);
  if (filters.recent) queryParams.append('recent', true);
  
  const endpoint = `/incidents?${queryParams.toString()}`;
  return get(endpoint);
};

/**
 * Get incident details by ID
 * @param {number} id - Incident ID
 * @returns {Promise<Object>} - Incident details
 */
export const getIncidentById = async (id) => {
  return get(`/incidents/${id}`);
};

/**
 * Update incident status
 * @param {number} id - Incident ID
 * @param {string} status - New status
 * @param {string} notes - Optional resolution notes
 * @returns {Promise<Object>} - Updated incident
 */
export const updateIncidentStatus = async (id, status, notes = "") => {
  return put(`/incidents/${id}/status`, { status, notes });
};

/**
 * Add incident notes
 * @param {number} id - Incident ID
 * @param {string} notes - Notes to add
 * @returns {Promise<Object>} - Updated incident
 */
export const addIncidentNotes = async (id, notes) => {
  return post(`/incidents/${id}/notes`, { notes });
};

/**
 * Get incident trends/statistics
 * @param {string} timeframe - Timeframe for trends (daily, weekly, monthly)
 * @returns {Promise<Object>} - Trend data
 */
export const getIncidentTrends = async (timeframe = 'daily') => {
  return get(`/incidents/trends?timeframe=${timeframe}`);
};

/**
 * Get incidents by location
 * @returns {Promise<Object>} - Location-based incident data for heatmap
 */
export const getIncidentsByLocation = async () => {
  return get('/incidents/locations');
};

/**
 * Assign incident to staff member
 * @param {number} incidentId - Incident ID
 * @param {number} staffId - Staff ID
 * @returns {Promise<Object>} - Updated incident
 */
export const assignIncident = async (incidentId, staffId) => {
  return post(`/incidents/${incidentId}/assign`, { staff_id: staffId });
};

/**
 * Get video footage for an incident
 * @param {number} incidentId - Incident ID
 * @returns {Promise<Object>} - Video URL and metadata
 */
export const getIncidentFootage = async (incidentId) => {
  return get(`/incidents/${incidentId}/footage`);
};

/**
 * Export incident report
 * @param {number} id - Incident ID or null for all incidents
 * @param {string} format - Export format (pdf, csv)
 * @returns {Promise<Blob>} - Report file
 */
export const exportIncidentReport = async (id = null, format = 'pdf') => {
  const endpoint = id 
    ? `/incidents/${id}/export?format=${format}` 
    : `/incidents/export?format=${format}`;
  
  // Use raw fetch for blob handling
  const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}${endpoint}`, {
    method: 'GET',
    headers: {
      'Accept': format === 'pdf' ? 'application/pdf' : 'text/csv',
    }
  });
  
  if (!response.ok) {
    throw new Error(`Export failed: ${response.status}`);
  }
  
  return await response.blob();
};