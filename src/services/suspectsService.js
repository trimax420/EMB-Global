// src/services/suspectsService.js
/**
 * Service for managing suspect tracking and information
 */

import { get, post, put, del, uploadFile } from './api';

/**
 * Get all suspects with optional filtering
 * @param {Object} filters - Filter criteria
 * @returns {Promise<Array>} - List of suspects
 */
export const getSuspects = async (filters = {}) => {
  const queryParams = new URLSearchParams();
  
  if (filters.status) queryParams.append('status', filters.status);
  if (filters.risk) queryParams.append('risk', filters.risk);
  if (filters.tags && filters.tags.length) {
    filters.tags.forEach(tag => queryParams.append('tag', tag));
  }
  if (filters.search) queryParams.append('search', filters.search);
  
  const endpoint = `/suspects?${queryParams.toString()}`;
  return get(endpoint);
};

/**
 * Get suspect by ID
 * @param {number} id - Suspect ID
 * @returns {Promise<Object>} - Suspect details
 */
export const getSuspectById = async (id) => {
  return get(`/suspects/${id}`);
};

/**
 * Create new suspect
 * @param {Object} suspectData - Suspect information
 * @returns {Promise<Object>} - Created suspect
 */
export const createSuspect = async (suspectData) => {
  return post('/suspects', suspectData);
};

/**
 * Update suspect
 * @param {number} id - Suspect ID
 * @param {Object} suspectData - Updated suspect information
 * @returns {Promise<Object>} - Updated suspect
 */
export const updateSuspect = async (id, suspectData) => {
  return put(`/suspects/${id}`, suspectData);
};

/**
 * Delete suspect
 * @param {number} id - Suspect ID
 * @returns {Promise<void>}
 */
export const deleteSuspect = async (id) => {
  return del(`/suspects/${id}`);
};

/**
 * Upload suspect image
 * @param {number} id - Suspect ID
 * @param {File} imageFile - Image file
 * @returns {Promise<Object>} - Upload result
 */
export const uploadSuspectImage = async (id, imageFile) => {
  return uploadFile(`/suspects/${id}/image`, imageFile);
};

/**
 * Get suspect detection history
 * @param {number} id - Suspect ID
 * @returns {Promise<Array>} - Detection history
 */
export const getSuspectDetections = async (id) => {
  return get(`/suspects/${id}/detections`);
};

/**
 * Update suspect status (active/inactive)
 * @param {number} id - Suspect ID
 * @param {string} status - New status
 * @returns {Promise<Object>} - Updated suspect
 */
export const updateSuspectStatus = async (id, status) => {
  return put(`/suspects/${id}/status`, { status });
};

/**
 * Track suspect with face recognition
 * @param {File} faceImage - Face image to search for
 * @param {string} videoPath - Video to search in (optional)
 * @returns {Promise<Object>} - Tracking job information
 */
export const trackSuspectByFace = async (faceImage, videoPath = null) => {
  const additionalData = videoPath ? { video_path: videoPath } : {};
  return uploadFile('/face-tracking/track-person', faceImage, additionalData);
};

/**
 * Get tracking results
 * @param {string} jobId - Tracking job ID
 * @returns {Promise<Object>} - Tracking results
 */
export const getTrackingResults = async (jobId) => {
  return get(`/face-tracking/tracking-results/${jobId}`);
};

/**
 * Get suspect activity log
 * @returns {Promise<Array>} - Activity log entries
 */
export const getSuspectActivityLog = async () => {
  return get('/suspects/activity-log');
};

/**
 * Add tags to suspect
 * @param {number} id - Suspect ID
 * @param {Array<string>} tags - Tags to add
 * @returns {Promise<Object>} - Updated suspect
 */
export const addSuspectTags = async (id, tags) => {
  return post(`/suspects/${id}/tags`, { tags });
};

/**
 * Remove tag from suspect
 * @param {number} id - Suspect ID
 * @param {string} tag - Tag to remove
 * @returns {Promise<Object>} - Updated suspect
 */
export const removeSuspectTag = async (id, tag) => {
  return del(`/suspects/${id}/tags/${tag}`);
};