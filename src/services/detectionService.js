// src/services/detectionService.js
/**
 * Service for video processing and object detection features
 */

import { get, post, uploadFile } from './api';

/**
 * Start theft detection processing on a video
 * @param {string} videoPath - Path to video file on server
 * @param {Object} options - Processing options
 * @returns {Promise<Object>} - Processing job information
 */
export const startTheftDetection = async (videoPath, options = {}) => {
  const params = new URLSearchParams();
  params.append('video_path', videoPath);
  
  if (options.handStayTimeChest) {
    params.append('hand_stay_time_chest', options.handStayTimeChest);
  }
  
  if (options.handStayTimeWaist) {
    params.append('hand_stay_time_waist', options.handStayTimeWaist);
  }
  
  return get(`/videos/theft-detection?${params.toString()}`);
};

/**
 * Start loitering detection processing on a video
 * @param {string} videoPath - Path to video file on server
 * @param {number} thresholdTime - Time threshold in seconds
 * @returns {Promise<Object>} - Processing job information
 */
export const startLoiteringDetection = async (videoPath, thresholdTime = 10) => {
  const params = new URLSearchParams();
  params.append('video_path', videoPath);
  params.append('threshold_time', thresholdTime);
  
  return get(`/videos/loitering-detection?${params.toString()}`);
};

/**
 * Extract faces from a video
 * @param {string} videoPath - Path to video file
 * @param {number} confidenceThreshold - Detection confidence threshold
 * @returns {Promise<Object>} - Face extraction results
 */
export const extractFacesFromVideo = async (videoPath, confidenceThreshold = 0.5) => {
  const params = new URLSearchParams();
  params.append('video_path', videoPath);
  params.append('confidence_threshold', confidenceThreshold);
  
  return post(`/videos/face-extraction?${params.toString()}`);
};

/**
 * Upload video for processing
 * @param {File} videoFile - Video file to upload
 * @param {string} detectionType - Type of detection to perform
 * @returns {Promise<Object>} - Upload result
 */
export const uploadVideoForProcessing = async (videoFile, detectionType) => {
  return uploadFile('/videos/upload', videoFile, { detection_type: detectionType });
};

/**
 * Get all processed videos
 * @returns {Promise<Array>} - List of processed videos
 */
export const getProcessedVideos = async () => {
  return get('/videos');
};

/**
 * Get detections for a video
 * @param {number} videoId - Video ID
 * @returns {Promise<Array>} - List of detections
 */
export const getVideoDetections = async (videoId) => {
  return get(`/videos/${videoId}/detections`);
};

/**
 * Get processing status for a job
 * @param {string} jobId - Processing job ID
 * @returns {Promise<Object>} - Processing status
 */
export const getProcessingStatus = async (jobId) => {
  return get(`/processing/status/${jobId}`);
};

/**
 * Track person across cameras by face
 * @param {File} faceImage - Face image to track
 * @param {string} videoPath - Optional video path to limit search
 * @returns {Promise<Object>} - Tracking job information
 */
export const trackPersonByFace = async (faceImage, videoPath = null) => {
  const additionalData = videoPath ? { video_path: videoPath } : {};
  return uploadFile('/face-tracking/track-person', faceImage, additionalData);
};

/**
 * Get results of face tracking
 * @param {string} jobId - Tracking job ID
 * @returns {Promise<Object>} - Tracking results
 */
export const getFaceTrackingResults = async (jobId) => {
  return get(`/face-tracking/tracking-results/${jobId}`);
};

/**
 * Search customer history by face
 * @param {string} faceImagePath - Path to face image on server
 * @returns {Promise<Object>} - Customer history matching the face
 */
export const searchCustomerByFace = async (faceImagePath) => {
  const params = new URLSearchParams();
  params.append('face_image_path', faceImagePath);
  
  return get(`/face-tracking/customer-history?${params.toString()}`);
};

/**
 * Add customer face encoding
 * @param {number} customerId - Customer ID
 * @param {File} faceImage - Customer face image
 * @returns {Promise<Object>} - Result
 */
export const addCustomerFaceEncoding = async (customerId, faceImage) => {
  return uploadFile(`/customers/${customerId}/face-encoding`, faceImage);
};