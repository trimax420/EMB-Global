// Add these functions to your src/services/detectionService.js file

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
  
  if (options.cameraId) {
    params.append('camera_id', options.cameraId);
  }
  
  try {
    const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/videos/theft-detection?${params.toString()}`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to start theft detection');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error starting theft detection:', error);
    throw error;
  }
};

/**
 * Start loitering detection processing on a video
 * @param {string} videoPath - Path to video file on server
 * @param {number} thresholdTime - Time threshold in seconds
 * @param {number} cameraId - Optional camera ID
 * @returns {Promise<Object>} - Processing job information
 */
export const startLoiteringDetection = async (videoPath, thresholdTime = 10, cameraId = null) => {
  const params = new URLSearchParams();
  params.append('video_path', videoPath);
  params.append('threshold_time', thresholdTime);
  
  if (cameraId) {
    params.append('camera_id', cameraId);
  }
  
  try {
    const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/videos/loitering-detection?${params.toString()}`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to start loitering detection');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error starting loitering detection:', error);
    throw error;
  }
};

/**
 * Get processing status for a job
 * @param {string} jobId - Processing job ID
 * @returns {Promise<Object>} - Processing status
 */
export const getProcessingStatus = async (jobId) => {
  try {
    const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/processing/status/${jobId}`);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to get processing status');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error getting processing status:', error);
    throw error;
  }
};

/**
 * Poll for processing status until complete
 * @param {string} jobId - Processing job ID
 * @param {Function} onProgress - Callback for progress updates
 * @param {Function} onComplete - Callback for completion
 * @param {Function} onError - Callback for errors
 * @param {number} interval - Polling interval in ms
 * @returns {Object} - Controller with stop method
 */
export const pollProcessingStatus = (jobId, onProgress, onComplete, onError, interval = 2000) => {
  let isPolling = true;
  
  const poll = async () => {
    if (!isPolling) return;
    
    try {
      const result = await getProcessingStatus(jobId);
      
      if (result.status === 'completed') {
        onComplete(result);
        isPolling = false;
      } else if (result.status === 'failed') {
        onError(new Error(result.message || 'Processing failed'));
        isPolling = false;
      } else {
        onProgress(result);
        setTimeout(poll, interval);
      }
    } catch (error) {
      onError(error);
      isPolling = false;
    }
  };
  
  // Start polling
  poll();
  
  // Return controller
  return {
    stop: () => {
      isPolling = false;
    }
  };
};