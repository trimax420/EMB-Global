import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import RealTimeDetection from "../components/RealTimeDetection";
import { getProcessedVideos, getCameraDetections, getDetectionStats, startCameraInference, stopCameraInference } from "../services/detectionService";
import axios from "axios";

const Cameras = () => {
  const [filter, setFilter] = useState("all");
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [detectionStats, setDetectionStats] = useState({});
  const navigate = useNavigate();

  // Fetch cameras on component mount
  useEffect(() => {
    fetchCameras();
  }, []);

  // Fetch cameras from API
  const fetchCameras = async () => {
    try {
      setLoading(true);
      
      // In a real app, fetch from API
      const response = await axios.get('/api/cameras');
      setCameras(response.data || dummyCameraData);
      
      setLoading(false);
    } catch (error) {
      console.error('Error fetching cameras:', error);
      setCameras(dummyCameraData);
      setError('Failed to load cameras from API. Using dummy data.');
      setLoading(false);
    }
  };

  // Dummy camera data as fallback
  const dummyCameraData = [
    { id: 1, name: "Store Entrance", status: "online", streamUrl: "/videos/store_entrance.mp4", location: "North Wing" },
    { id: 2, name: "electronics", status: "online", streamUrl: "/videos/electronics_section.mp4", location: "Main Floor" },
    { id: 3, name: "CheckOut", status: "online", streamUrl: "/videos/checkout.mp4", location: "East Wing" },
    { id: 4, name: "Parking Lot", status: "offline", streamUrl: "", location: "Exterior" },
  ];

  // Update detection stats for a camera
  const handleUpdateDetectionStats = (cameraId, stats) => {
    setDetectionStats(prevStats => ({
      ...prevStats,
      [cameraId]: {
        ...stats,
        lastUpdated: new Date()
      }
    }));
  };

  // Filter cameras based on status
  const filteredCameras =
    filter === "all"
      ? cameras
      : cameras.filter((camera) => camera.status === filter);

  // Select a camera for viewing
  const selectCamera = (camera) => {
    setSelectedCamera(camera);
  };

  // Back to camera grid
  const backToGrid = () => {
    setSelectedCamera(null);
  };

  return (
    <div className="p-6">
      {/* Header Section */}
      <h1 className="text-3xl font-bold mb-6">Camera Management</h1>

      {/* Filters Section */}
      <div className="mb-6">
        <button
          onClick={() => setFilter("all")}
          className={`mr-3 px-4 py-2 rounded ${
            filter === "all" ? "bg-blue-500 text-white" : "bg-gray-200"
          }`}
        >
          All
        </button>
        <button
          onClick={() => setFilter("online")}
          className={`mr-3 px-4 py-2 rounded ${
            filter === "online" ? "bg-green-500 text-white" : "bg-gray-200"
          }`}
        >
          Online
        </button>
        <button
          onClick={() => setFilter("offline")}
          className={`px-4 py-2 rounded ${
            filter === "offline" ? "bg-red-500 text-white" : "bg-gray-200"
          }`}
        >
          Offline
        </button>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      )}
      
      {/* Error message */}
      {error && !loading && (
        <div className="mb-4 p-3 bg-red-100 text-red-700 rounded">
          {error}
        </div>
      )}

      {/* Selected Camera View */}
      {selectedCamera && (
        <div className="mb-4">
          <button
            onClick={backToGrid}
            className="mb-4 px-4 py-2 bg-gray-200 rounded hover:bg-gray-300"
          >
            ← Back to All Cameras
          </button>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <RealTimeDetection 
                cameraId={selectedCamera.id} 
                onUpdateStats={(stats) => handleUpdateDetectionStats(selectedCamera.id, stats)}
              />
            </div>
            
            <div className="bg-white rounded-lg shadow p-4">
              <h2 className="text-xl font-semibold mb-3">{selectedCamera.name}</h2>
              <p className="text-gray-600 mb-4">{selectedCamera.location}</p>
              
              <h3 className="font-medium mb-2">Camera Details</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Status:</span>
                  <span className={`${selectedCamera.status === 'online' ? 'text-green-600' : 'text-red-600'}`}>
                    {selectedCamera.status}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">ID:</span>
                  <span>{selectedCamera.id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Location:</span>
                  <span>{selectedCamera.location}</span>
                </div>
              </div>
              
              <h3 className="font-medium mt-6 mb-2">Detection Statistics</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 p-3 rounded">
                  <div className="text-lg font-semibold">{detectionStats[selectedCamera.id]?.people || 0}</div>
                  <div className="text-sm text-gray-600">People</div>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <div className="text-lg font-semibold">{detectionStats[selectedCamera.id]?.objects || 0}</div>
                  <div className="text-sm text-gray-600">Objects</div>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <div className="text-lg font-semibold text-amber-500">{detectionStats[selectedCamera.id]?.loitering || 0}</div>
                  <div className="text-sm text-gray-600">Loitering</div>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <div className="text-lg font-semibold text-red-500">{detectionStats[selectedCamera.id]?.theft || 0}</div>
                  <div className="text-sm text-gray-600">Theft</div>
                </div>
              </div>
              
              <div className="mt-6">
                <button 
                  onClick={() => navigate(`/alerts?camera=${selectedCamera.id}`)}
                  className="w-full py-2 px-4 bg-blue-500 text-white rounded hover:bg-blue-600"
                >
                  View Alerts History
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Camera Grid */}
      {!selectedCamera && !loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filteredCameras.map((camera) => (
            <div
              key={camera.id}
              className={`relative bg-white p-4 rounded-lg shadow-md overflow-hidden ${
                camera.status === "online" ? "border-l-4 border-green-500" : "border-l-4 border-red-500"
              }`}
            >
              <div
                className="w-full h-48 object-cover rounded-lg cursor-pointer bg-gray-100 flex items-center justify-center overflow-hidden mb-3"
                onClick={() => camera.status === "online" && selectCamera(camera)}
              >
                {camera.status === "online" ? (
                  camera.streamUrl ? (
                    <video
                      src={camera.streamUrl}
                      alt={camera.name}
                      className="w-full h-full object-cover"
                      muted
                      loop
                      autoPlay
                      playsInline
                    />
                  ) : (
                    <div className="text-gray-400">No preview available</div>
                  )
                ) : (
                  <div className="text-gray-400">Camera Offline</div>
                )}
                
                {/* Status indicator */}
                <div className={`absolute top-2 right-2 px-2 py-1 rounded-full text-xs text-white ${
                  camera.status === "online" ? "bg-green-500" : "bg-red-500"
                }`}>
                  {camera.status}
                </div>
              </div>
              
              <h2 className="font-semibold text-lg">{camera.name}</h2>
              <p className="text-sm text-gray-600">{camera.location || "No location"}</p>
              
              {/* Stats summary if camera is online */}
              {camera.status === "online" && detectionStats[camera.id] && (
                <div className="mt-2 grid grid-cols-4 gap-1 text-center text-xs">
                  <div>
                    <div className="font-medium">{detectionStats[camera.id]?.people || 0}</div>
                    <div className="text-gray-500">People</div>
                  </div>
                  <div>
                    <div className="font-medium">{detectionStats[camera.id]?.objects || 0}</div>
                    <div className="text-gray-500">Objects</div>
                  </div>
                  <div>
                    <div className="font-medium text-amber-500">{detectionStats[camera.id]?.loitering || 0}</div>
                    <div className="text-gray-500">Loiter</div>
                  </div>
                  <div>
                    <div className="font-medium text-red-500">{detectionStats[camera.id]?.theft || 0}</div>
                    <div className="text-gray-500">Theft</div>
                  </div>
                </div>
              )}
              
              {/* Actions */}
              <div className="mt-3 flex justify-end">
                <button
                  onClick={() => selectCamera(camera)}
                  disabled={camera.status !== "online"}
                  className={`px-3 py-1 rounded text-sm ${
                    camera.status === "online"
                      ? "bg-blue-500 text-white hover:bg-blue-600"
                      : "bg-gray-200 text-gray-500 cursor-not-allowed"
                  }`}
                >
                  {camera.status === "online" ? "View" : "Unavailable"}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Cameras;