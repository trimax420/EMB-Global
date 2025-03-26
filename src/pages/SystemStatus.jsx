import React, { useState, useEffect } from 'react';
import { getSystemStatus } from '../services/api';

const SystemStatusPage = () => {
  const [systemData, setSystemData] = useState({
    cameras: [],
    modelPerformance: {
      is_working: false,
      accuracy: '0%',
      true_positives: 0,
      false_positives: 0
    },
    frameSkipping: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        setLoading(true);
        const response = await getSystemStatus();
        console.log('System status response:', response); // Debug log
        
        // Ensure we have all required data with defaults
        setSystemData({
          cameras: response.cameras || [],
          modelPerformance: {
            is_working: response.model_performance?.is_working || false,
            accuracy: response.model_performance?.accuracy || '0%',
            true_positives: response.model_performance?.true_positives || 0,
            false_positives: response.model_performance?.false_positives || 0
          },
          frameSkipping: response.frame_skipping || []
        });
      } catch (err) {
        console.error('Error fetching system status:', err);
        setError(err.message || 'Failed to fetch system status');
      } finally {
        setLoading(false);
      }
    };

    fetchSystemStatus();

    // Set up polling interval
    const interval = setInterval(fetchSystemStatus, 30000); // Poll every 30 seconds

    // Cleanup interval on component unmount
    return () => clearInterval(interval);
  }, []);

  // Helper function to determine status color
  const getStatusColor = (status) => {
    return status?.toLowerCase() === 'online' ? 'bg-green-500' : 'bg-red-500';
  };

  // Helper function to get camera name by id
  const getCameraNameById = (id) => {
    const camera = systemData.cameras.find((camera) => camera.id === id);
    return camera ? camera.name : 'Unknown Camera';
  };

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center">
        <div className="text-lg text-gray-600">Loading system status...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <p className="font-bold">Error</p>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Header Section */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800">System Status</h1>
      </div>

      {/* Camera Activities Section */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Camera Activities</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white border border-gray-300">
            <thead className='text-left'>
              <tr className="bg-gray-100">
                <th className="py-2 px-4 border-b">Camera Name</th>
                <th className="py-2 px-4 border-b">Status</th>
                <th className="py-2 px-4 border-b">FPS</th>
              </tr>
            </thead>
            <tbody>
              {systemData.cameras.length > 0 ? (
                systemData.cameras.map((camera) => (
                  <tr key={camera.id} className="hover:bg-gray-50">
                    <td className="py-2 px-4 border-b">{camera.name}</td>
                    <td className="py-2 px-4 border-b">
                      <span
                        className={`inline-block px-2 py-1 rounded text-white ${getStatusColor(
                          camera.status
                        )}`}
                      >
                        {camera.status}
                      </span>
                    </td>
                    <td className="py-2 px-4 border-b">{camera.fps}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="3" className="py-4 text-center text-gray-500">
                    No cameras available
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model Performance Section */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Model Performance</h2>
        <div className="grid grid-cols-2 gap-4 p-6 bg-white border border-gray-300 rounded shadow">
          <div>
            <p className="text-gray-600">True Positives:</p>
            <p className="font-bold">{systemData.modelPerformance.true_positives}</p>
          </div>
          <div>
            <p className="text-gray-600">False Positives:</p>
            <p className="font-bold">{systemData.modelPerformance.false_positives}</p>
          </div>
          <div>
            <p className="text-gray-600">Accuracy:</p>
            <p className="font-bold">{systemData.modelPerformance.accuracy}</p>
          </div>
          <div>
            <p className="text-gray-600">Model Status:</p>
            <span
              className={`inline-block px-2 py-1 rounded text-white ${
                systemData.modelPerformance.is_working ? 'bg-green-500' : 'bg-red-500'
              }`}
            >
              {systemData.modelPerformance.is_working ? 'Working Properly' : 'Not Working'}
            </span>
          </div>
        </div>
      </div>

      {/* Frame Skipping Section */}
      <div>
        <h2 className="text-xl font-semibold mb-4">Frame Skipping</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white border border-gray-300">
            <thead>
              <tr className="bg-gray-100 text-left">
                <th className="py-2 px-4 border-b">Camera Name</th>
                <th className="py-2 px-4 border-b">Skipped Frames</th>
                <th className="py-2 px-4 border-b">Details</th>
              </tr>
            </thead>
            <tbody>
              {systemData.frameSkipping.length > 0 ? (
                systemData.frameSkipping.map((frame) => (
                  <tr key={frame.id} className="hover:bg-gray-50">
                    <td className="py-2 px-4 border-b">{getCameraNameById(frame.camera_id)}</td>
                    <td className="py-2 px-4 border-b">{frame.skipped_frames}</td>
                    <td className="py-2 px-4 border-b">
                      <span className="text-gray-500">
                        Possible cause: Network instability or camera overload.
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="3" className="py-4 text-center text-gray-500">
                    No frame skipping data available
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default SystemStatusPage;
