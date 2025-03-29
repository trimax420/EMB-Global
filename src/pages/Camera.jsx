import React, { useState, useEffect, useRef } from "react";
import { cameraService } from "../services/api";
import { BACKEND_URL, WS_URL, WS_ENDPOINTS } from '../config';

const Cameras = () => {
  const [filter, setFilter] = useState("all");
  const [fullscreenCamera, setFullscreenCamera] = useState(null);
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentFrames, setCurrentFrames] = useState({});
  const [currentDetections, setCurrentDetections] = useState({});
  const wsRefs = useRef({});

  useEffect(() => {
    const fetchCameras = async () => {
      try {
        setLoading(true);
        const data = await cameraService.getAllCameras();
        setCameras(data);
        
        // Initialize WebSocket connections for each camera
        data.forEach(camera => {
          if (!wsRefs.current[camera.id]) {
            const ws = new WebSocket(`${WS_URL}${WS_ENDPOINTS.live}`);
            
            ws.onopen = () => {
              console.log(`WebSocket connected for camera ${camera.id}`);
              // Start streaming for this camera
              ws.send(JSON.stringify({
                type: 'start_stream',
                camera_id: camera.id
              }));
            };

            ws.onmessage = (event) => {
              try {
                const data = JSON.parse(event.data);
                if (data.type === 'live_detection' && data.camera_id === camera.id) {
                  // Update frame
                  setCurrentFrames(prev => ({
                    ...prev,
                    [camera.id]: data.frame
                  }));
                  
                  // Update detections
                  if (data.detections) {
                    setCurrentDetections(prev => ({
                      ...prev,
                      [camera.id]: data.detections
                    }));
                  }
                }
              } catch (error) {
                console.error(`Error processing message for camera ${camera.id}:`, error);
              }
            };

            ws.onerror = (error) => {
              console.error(`WebSocket error for camera ${camera.id}:`, error);
            };

            ws.onclose = () => {
              console.log(`WebSocket closed for camera ${camera.id}`);
              // Attempt to reconnect after a delay
              setTimeout(() => {
                if (wsRefs.current[camera.id]) {
                  wsRefs.current[camera.id].close();
                  delete wsRefs.current[camera.id];
                }
              }, 5000);
            };

            wsRefs.current[camera.id] = ws;
          }
        });
      } catch (err) {
        setError(err.message);
        console.error('Error fetching cameras:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchCameras();

    // Cleanup function
    return () => {
      // Close all WebSocket connections
      Object.values(wsRefs.current).forEach(ws => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.close();
        }
      });
    };
  }, []);

  // Filter cameras based on status
  const filteredCameras =
    filter === "all"
      ? cameras
      : cameras.filter((camera) => camera.status === filter);

  // Exit fullscreen mode
  const exitFullscreen = () => {
    setFullscreenCamera(null);
  };

  // Render camera feed with detections
  const renderCameraFeed = (camera) => {
    const frame = currentFrames[camera.id];
    const detections = currentDetections[camera.id] || [];

    return (
      <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
        {frame ? (
          <div className="relative w-full h-full">
            <img
              src={`data:image/jpeg;base64,${frame}`}
              alt={camera.name}
              className="w-full h-full object-contain"
            />
            {/* Display detections */}
            {detections.map((detection, index) => (
              <div
                key={index}
                className="absolute px-2 py-1 bg-black bg-opacity-50 text-white rounded text-sm"
                style={{
                  left: `${detection.bbox ? detection.bbox[0] : 0}px`,
                  top: `${detection.bbox ? detection.bbox[1] : 0}px`,
                }}
              >
                {detection.class_name}: {(detection.confidence * 100).toFixed(1)}%
              </div>
            ))}
          </div>
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-white">
            Loading feed...
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return <div className="p-6">Loading cameras...</div>;
  }

  if (error) {
    return <div className="p-6 text-red-500">Error: {error}</div>;
  }

  return (
    <div className="p-6">
      {/* Header Section */}
      <h1 className="text-3xl font-bold mb-6">Live Camera Feeds</h1>

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

      {/* Fullscreen Camera View */}
      {fullscreenCamera && (
        <div className="fixed inset-0 bg-black z-50 flex items-center justify-center">
          <div className="relative w-full h-full">
            {renderCameraFeed(fullscreenCamera)}
            <button
              onClick={exitFullscreen}
              className="absolute top-4 right-4 bg-gray-800 text-white px-4 py-2 rounded hover:bg-gray-700"
            >
              Exit Fullscreen
            </button>
          </div>
        </div>
      )}

      {/* Camera Grid */}
      {!fullscreenCamera && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filteredCameras.map((camera) => (
            <div
              key={camera.id}
              className={`relative bg-gray-50 p-4 rounded-lg shadow-md cursor-pointer ${
                camera.status === "online" ? "border-4 border-green-500" : "border-4 border-red-500"
              }`}
              onClick={() => setFullscreenCamera(camera)}
            >
              {renderCameraFeed(camera)}
              <h2 className="mt-2 font-semibold text-lg">{camera.name}</h2>
              <p className="text-sm">
                Status:{" "}
                <span
                  className={`uppercase ${
                    camera.status === "online" ? "text-green-600" : "text-red-600"
                  }`}
                >
                  {camera.status}
                </span>
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Cameras;