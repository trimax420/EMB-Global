import React, { useState, useEffect } from 'react';
import WebRTCVideoStream from './WebRTCVideoStream';

const LiveCameraComponent = ({ 
  selectedCamera, 
  processingType = 'all', 
  onUpdateStats,
  apiUrl = 'http://localhost:8000/api' 
}) => {
  const [useWebRTC, setUseWebRTC] = useState(true);
  const [stats, setStats] = useState({});
  
  // Handle WebRTC stats updates
  const handleWebRTCStats = (rtcStats) => {
    setStats(rtcStats);
    if (onUpdateStats) {
      onUpdateStats(rtcStats);
    }
  };
  
  if (!selectedCamera) {
    return <div className="flex items-center justify-center aspect-video bg-gray-200 rounded-lg">
      <p className="text-gray-500">Select a camera to view</p>
    </div>;
  }
  
  // Debug info about camera and video path
  console.log('LiveCameraComponent with:', {
    cameraId: selectedCamera.id, 
    videoPath: selectedCamera.videoUrl,
    detectionType: processingType
  });
  
  return (
    <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
      {useWebRTC ? (
        <WebRTCVideoStream
          videoPath={selectedCamera.videoUrl}
          cameraId={selectedCamera.id}
          detectionType={processingType}
          onUpdateStats={handleWebRTCStats}
          className="aspect-video w-full"
          controls={false}
          autoPlay={true}
          preferredResolution="720p"
        />
      ) : (
        <div className="flex items-center justify-center h-full">
          <div>
            <p className="text-white text-center mb-4">WebRTC is disabled</p>
            <button 
              onClick={() => setUseWebRTC(true)}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg"
            >
              Enable WebRTC
            </button>
          </div>
        </div>
      )}
      
      {/* Info overlay */}
      <div className="absolute bottom-2 left-2 bg-black bg-opacity-60 rounded px-2 py-1 text-white text-xs z-10">
        <div>{selectedCamera.name}</div>
        <div>Mode: {processingType}</div>
        {stats.people !== undefined && (
          <div>People: {stats.people}</div>
        )}
      </div>
    </div>
  );
};

export default LiveCameraComponent;