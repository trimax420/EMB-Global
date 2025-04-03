// src/contexts/ApiContext.jsx
/**
 * Context provider for API services
 * Makes all API services available throughout the application
 */

import React, { createContext, useContext, useState, useEffect } from 'react';
import * as incidentsService from '../services/incidentsService';
import * as suspectsService from '../services/suspectsService';
import * as dashboardService from '../services/dashboardService';
import * as detectionService from '../services/detectionService';

// Create context
const ApiContext = createContext();

export const ApiProvider = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Wrap API calls with loading and error handling
  const withLoadingAndErrorHandling = async (apiCall) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const result = await apiCall();
      return result;
    } catch (err) {
      setError(err.message || 'An error occurred');
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  // Create a wrapped version of all services
  const wrappedServices = {
    incidents: Object.fromEntries(
      Object.entries(incidentsService).map(([key, func]) => [
        key,
        (...args) => withLoadingAndErrorHandling(() => func(...args)),
      ])
    ),
    suspects: Object.fromEntries(
      Object.entries(suspectsService).map(([key, func]) => [
        key,
        (...args) => withLoadingAndErrorHandling(() => func(...args)),
      ])
    ),
    dashboard: Object.fromEntries(
      Object.entries(dashboardService).map(([key, func]) => [
        key,
        (...args) => withLoadingAndErrorHandling(() => func(...args)),
      ])
    ),
    detection: Object.fromEntries(
      Object.entries(detectionService).map(([key, func]) => [
        key,
        (...args) => withLoadingAndErrorHandling(() => func(...args)),
      ])
    ),
  };

  // Context value
  const contextValue = {
    ...wrappedServices,
    isLoading,
    error,
    clearError: () => setError(null)
  };

  return (
    <ApiContext.Provider value={contextValue}>
      {children}
    </ApiContext.Provider>
  );
};

// Custom hook to use the API context
export const useApi = () => {
  const context = useContext(ApiContext);
  if (context === undefined) {
    throw new Error('useApi must be used within an ApiProvider');
  }
  return context;
};

export default ApiContext;