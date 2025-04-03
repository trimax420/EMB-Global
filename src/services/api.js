/**
 * API utilities for making HTTP requests
 */

import axios from 'axios';

// Get base URL from environment or use default
const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

// Create axios instance with default config
const api = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for authentication
api.interceptors.request.use(
  (config) => {
    // Get token from storage if available
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    // Handle specific error codes
    if (error.response) {
      // Authentication errors
      if (error.response.status === 401) {
        localStorage.removeItem('token');
        // Redirect to login if needed
        // window.location.href = '/login';
      }
      
      // Format error message
      const errorMessage = error.response.data?.detail || 'An error occurred';
      error.message = errorMessage;
    }
    
    return Promise.reject(error);
  }
);

/**
 * Make a GET request
 * @param {string} url - URL path
 * @param {Object} options - Axios request options
 * @returns {Promise<any>} - Response data
 */
export const get = (url, options = {}) => api.get(url, options);

/**
 * Make a POST request
 * @param {string} url - URL path
 * @param {Object} data - Request body
 * @param {Object} options - Axios request options
 * @returns {Promise<any>} - Response data
 */
export const post = (url, data = {}, options = {}) => api.post(url, data, options);

/**
 * Make a PUT request
 * @param {string} url - URL path
 * @param {Object} data - Request body
 * @param {Object} options - Axios request options
 * @returns {Promise<any>} - Response data
 */
export const put = (url, data = {}, options = {}) => api.put(url, data, options);

/**
 * Make a DELETE request
 * @param {string} url - URL path
 * @param {Object} options - Axios request options
 * @returns {Promise<any>} - Response data
 */
export const del = (url, options = {}) => api.delete(url, options);

/**
 * Upload a file with form data
 * @param {string} url - URL path
 * @param {File} file - File to upload
 * @param {Object} additionalData - Additional form data
 * @param {Object} options - Axios request options
 * @returns {Promise<any>} - Response data
 */
export const uploadFile = (url, file, additionalData = {}, options = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  
  // Add any additional data to form
  Object.entries(additionalData).forEach(([key, value]) => {
    formData.append(key, value);
  });
  
  return api.post(url, formData, {
    ...options,
    headers: {
      ...options.headers,
      'Content-Type': 'multipart/form-data',
    },
  });
};

export default {
  get,
  post,
  put,
  del,
  uploadFile,
  baseURL
};