// src/services/api.js
/**
 * Base API service for making HTTP requests to the backend
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

/**
 * Basic fetch wrapper with error handling
 * @param {string} endpoint - API endpoint to call
 * @param {Object} options - Fetch options
 * @returns {Promise<any>} - Response data
 */
export const fetchAPI = async (endpoint, options = {}) => {
  try {
    const url = `${API_BASE_URL}${endpoint}`;
    
    // Default headers
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    const response = await fetch(url, {
      ...options,
      headers,
    });

    // Check if the response is ok (status in the range 200-299)
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API error: ${response.status}`);
    }

    // Check if response is empty
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      return await response.json();
    }
    
    return await response.text();
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
};

/**
 * GET request helper
 * @param {string} endpoint - API endpoint
 * @param {Object} options - Additional fetch options
 * @returns {Promise<any>} - Response data
 */
export const get = (endpoint, options = {}) => {
  return fetchAPI(endpoint, {
    method: 'GET',
    ...options,
  });
};

/**
 * POST request helper
 * @param {string} endpoint - API endpoint
 * @param {Object} data - Data to send
 * @param {Object} options - Additional fetch options
 * @returns {Promise<any>} - Response data
 */
export const post = (endpoint, data, options = {}) => {
  return fetchAPI(endpoint, {
    method: 'POST',
    body: JSON.stringify(data),
    ...options,
  });
};

/**
 * PUT request helper
 * @param {string} endpoint - API endpoint
 * @param {Object} data - Data to send
 * @param {Object} options - Additional fetch options
 * @returns {Promise<any>} - Response data
 */
export const put = (endpoint, data, options = {}) => {
  return fetchAPI(endpoint, {
    method: 'PUT',
    body: JSON.stringify(data),
    ...options,
  });
};

/**
 * DELETE request helper
 * @param {string} endpoint - API endpoint
 * @param {Object} options - Additional fetch options
 * @returns {Promise<any>} - Response data
 */
export const del = (endpoint, options = {}) => {
  return fetchAPI(endpoint, {
    method: 'DELETE',
    ...options,
  });
};

/**
 * Upload file helper
 * @param {string} endpoint - API endpoint
 * @param {File} file - File to upload
 * @param {Object} additionalData - Additional form data
 * @param {Object} options - Additional fetch options
 * @returns {Promise<any>} - Response data
 */
export const uploadFile = (endpoint, file, additionalData = {}, options = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  
  // Add any additional data to form data
  Object.entries(additionalData).forEach(([key, value]) => {
    formData.append(key, value);
  });
  
  return fetchAPI(endpoint, {
    method: 'POST',
    body: formData,
    headers: {}, // Let the browser set the content type for form data
    ...options,
  });
};