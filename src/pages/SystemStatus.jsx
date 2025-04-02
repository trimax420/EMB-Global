import React from 'react';

const SystemStatusPage = () => {
  // Dummy data for cameras
  const cameras = [
    { id: 1, name: 'Camera A', status: 'Online', fps: 30, lastActive: '2023-10-10T10:00:00' },
    { id: 2, name: 'Camera B', status: 'Offline', fps: 0, lastActive: '2023-10-09T15:45:00' },
    { id: 3, name: 'Camera C', status: 'Online', fps: 25, lastActive: '2023-10-10T11:30:00' }
  ];

  // Dummy data for model performance
  const modelPerformance = {
    isWorking: true,
    truePositives: 1200,
    falsePositives: 15,
    missedDetections: 5,
    accuracy: '98.5%'
  };

  // Dummy data for frame skipping with camera details
  const frameSkipping = [
    { id: 1, cameraId: 1, timestamp: '2023-10-10T09:00:00', skippedFrames: 5 },
    { id: 2, cameraId: 2, timestamp: '2023-10-10T10:15:00', skippedFrames: 3 },
    { id: 3, cameraId: 3, timestamp: '2023-10-10T12:00:00', skippedFrames: 2 }
  ];

  // Helper function to determine status color
  const getStatusColor = (status) => {
    return status === 'Online' ? 'bg-green-500' : 'bg-red-500';
  };

  // Helper function to get camera name by id
  const getCameraNameById = (id) => {
    const camera = cameras.find((camera) => camera.id === id);
    return camera ? camera.name : 'Unknown Camera';
  };

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
                <th className="py-2 px-4 border-b">Last Active</th>
              </tr>
            </thead>
            <tbody>
              {cameras.map((camera) => (
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
                  <td className="py-2 px-4 border-b">
                    {new Date(camera.lastActive).toLocaleString()}
                  </td>
                </tr>
              ))}
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
            <p className="font-bold">{modelPerformance.truePositives}</p>
          </div>
          <div>
            <p className="text-gray-600">False Positives:</p>
            <p className="font-bold">{modelPerformance.falsePositives}</p>
          </div>
          <div>
            <p className="text-gray-600">Missed Detections:</p>
            <p className="font-bold">{modelPerformance.missedDetections}</p>
          </div>
          <div>
            <p className="text-gray-600">Accuracy:</p>
            <p className="font-bold">{modelPerformance.accuracy}</p>
          </div>
          <div colSpan="2">
            <p className="text-gray-600">Model Status:</p>
            <span
              className={`inline-block px-2 py-1 rounded text-white ${modelPerformance.isWorking ? 'bg-green-500' : 'bg-red-500'}`}
            >
              {modelPerformance.isWorking ? 'Working Properly' : 'Not Working'}
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
                <th className="py-2 px-4 border-b">Timestamp</th>
                <th className="py-2 px-4 border-b">Skipped Frames</th>
                <th className="py-2 px-4 border-b">Details</th>
              </tr>
            </thead>
            <tbody>
              {frameSkipping.map((frame) => (
                <tr key={frame.id} className="hover:bg-gray-50">
                  <td className="py-2 px-4 border-b">{getCameraNameById(frame.cameraId)}</td>
                  <td className="py-2 px-4 border-b">
                    {new Date(frame.timestamp).toLocaleString()}
                  </td>
                  <td className="py-2 px-4 border-b">{frame.skippedFrames}</td>
                  <td className="py-2 px-4 border-b">
                    <span className="text-gray-500">Possible cause: Network instability or camera overload.</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default SystemStatusPage;
