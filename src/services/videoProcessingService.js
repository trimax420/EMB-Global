// src/services/videoProcessingService.js

/**
 * Service for video processing operations
 */

import { get, post } from './api';

/**
 * Start loitering detection processing on a video
 * @param {string} videoPath - Path to video file
 * @param {number} thresholdTime - Time threshold (seconds) for loitering detection
 * @returns {Promise<Object>} - Processing job information
 */
export const startLoiteringDetection = async (videoPath, thresholdTime = 10) => {
  const params = new URLSearchParams();
  params.append('video_path', videoPath);
  params.append('threshold_time', thresholdTime);
  
  return post(`/videos/loitering-detection?${params.toString()}`);
};

/**
 * Start theft detection processing on a video
 * @param {string} videoPath - Path to video file
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
  
  return post(`/videos/theft-detection?${params.toString()}`);
};

/**
 * Get processing status
 * @param {string} jobId - Processing job ID
 * @returns {Promise<Object>} - Processing status
 */
export const getProcessingStatus = async (jobId) => {
  return get(`/processing/status/${jobId}`);
};

/**
 * Track face across video
 * @param {File} faceImage - Face image file
 * @param {string} videoPath - Path to video file
 * @returns {Promise<Object>} - Tracking results
 */
export const trackFace = async (faceImage, videoPath) => {
  const formData = new FormData();
  formData.append('face_image', faceImage);
  formData.append('video_path', videoPath);
  
  return post('/face-tracking/track', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
};

/**
 * Get face tracking results
 * @param {string} jobId - Tracking job ID
 * @returns {Promise<Object>} - Tracking results
 */
export const getTrackingResults = async (jobId) => {
  return get(`/face-tracking/results/${jobId}`);
};

/**
 * Upload video for processing
 * @param {File} videoFile - Video file
 * @param {string} detectionType - Type of detection to perform
 * @returns {Promise<Object>} - Upload result
 */
export const uploadVideo = async (videoFile, detectionType) => {
  const formData = new FormData();
  formData.append('video', videoFile);
  formData.append('detection_type', detectionType);
  
  return post('/videos/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
};

export default {
  startLoiteringDetection,
  startTheftDetection,
  getProcessingStatus,
  trackFace,
  getTrackingResults,
  uploadVideo
};