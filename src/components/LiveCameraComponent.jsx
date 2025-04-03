import React, { useEffect, useRef, useState } from 'react';
import { FaExpand, FaCompress, FaUserNinja, FaUserClock } from "react-icons/fa";

const LiveCameraComponent = ({ selectedCamera, onUpdateStats }) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [detections, setDetections] = useState([]);
  const videoRef = useRef(null);
  
  // Update video source when selected camera changes
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.load();
      
      // For demo purposes: simulate random detection events
      const timer = setInterval(() => {
        simulateDetections();
      }, 5000);
      
      return () => clearInterval(timer);
    }
  }, [selectedCamera]);
  
  // Handle fullscreen toggle
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };
  
  // Simulate detection events for demo purposes
  const simulateDetections = () => {
    // Only simulate detections if the video is playing and camera is online
    if (videoRef.current && videoRef.current.paused === false && selectedCamera.status === 'online') {
      const capabilities = selectedCamera.capabilities || [];
      
      // Generate random counts based on camera capabilities
      const newStats = {
        people: Math.floor(Math.random() * 10),
        objects: Math.floor(Math.random() * 15),
        loitering: 0,
        theft: 0
      };
      
      // Only add loitering if camera has that capability
      if (capabilities.includes('loitering_detection')) {
        newStats.loitering = Math.random() > 0.7 ? Math.floor(Math.random() * 3) : 0;
      }
      
      // Only add theft if camera has that capability
      if (capabilities.includes('theft_detection')) {
        newStats.theft = Math.random() > 0.85 ? Math.floor(Math.random() * 2) : 0;
      }
      
      // Report stats back to parent component
      if (onUpdateStats) {
        onUpdateStats(newStats);
      }
      
      // Create visual detections for display
      const newDetections = [];
      
      // People detections
      for (let i = 0; i < newStats.people; i++) {
        newDetections.push({
          id: `person-${Date.now()}-${i}`,
          type: 'person',
          x: Math.random() * 80 + 10, // 10-90% of width
          y: Math.random() * 80 + 10, // 10-90% of height
          width: 60,
          height: 120,
          confidence: Math.random() * 0.3 + 0.7, // 0.7-1.0
          color: 'rgba(59, 130, 246, 0.5)' // blue
        });
      }
      
      // Loitering detections
      for (let i = 0; i < newStats.loitering; i++) {
        newDetections.push({
          id: `loitering-${Date.now()}-${i}`,
          type: 'loitering',
          x: Math.random() * 80 + 10,
          y: Math.random() * 80 + 10,
          width: 70,
          height: 130,
          confidence: Math.random() * 0.2 + 0.6, // 0.6-0.8
          color: 'rgba(245, 158, 11, 0.5)' // orange/amber
        });
      }
      
      // Theft detections
      for (let i = 0; i < newStats.theft; i++) {
        newDetections.push({
          id: `theft-${Date.now()}-${i}`,
          type: 'theft',
          x: Math.random() * 80 + 10,
          y: Math.random() * 80 + 10,
          width: 65,
          height: 125,
          confidence: Math.random() * 0.2 + 0.7, // 0.7-0.9
          color: 'rgba(239, 68, 68, 0.5)' // red
        });
      }
      
      setDetections(newDetections);
    }
  };

  return (
    <div className={`bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden ${isFullscreen ? 'fixed top-0 left-0 w-full h-full z-50' : ''}`}>
      <div className='p-4 border-b border-gray-100 flex justify-between items-center'>
        <div>
          <h1 className='font-bold text-gray-800'>{selectedCamera.name}</h1>
          <p className='text-gray-500 text-sm'>{selectedCamera.location}</p>
        </div>
        <button
          onClick={toggleFullscreen}
          className='p-2 text-gray-500 hover:bg-gray-100 rounded-lg transition-all'
        >
          {isFullscreen ? <FaCompress /> : <FaExpand />}
        </button>
      </div>
      
      <div className='relative bg-gray-900'>
        <div className='aspect-video max-h-[600px] relative'>
          {selectedCamera.status === 'online' ? (
            <>
              <video
                ref={videoRef}
                className='w-full h-full object-contain'
                autoPlay
                muted
                loop
                playsInline
                controls
              >
                <source src={selectedCamera.videoUrl} type='video/mp4' />
                Your browser does not support the video tag.
              </video>
              
              {/* Overlay detection boxes */}
              {detections.map((detection) => (
                <div
                  key={detection.id}
                  style={{
                    position: 'absolute',
                    left: `${detection.x}%`,
                    top: `${detection.y}%`,
                    width: `${detection.width}px`,
                    height: `${detection.height}px`,
                    border: `2px solid ${detection.color.replace('0.5', '1')}`,
                    backgroundColor: detection.color,
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'flex-start',
                    padding: '2px',
                    boxSizing: 'border-box',
                    fontSize: '12px',
                    color: 'white',
                    textShadow: '1px 1px 1px rgba(0,0,0,0.5)'
                  }}
                >
                  <div className="flex items-center gap-1">
                    {detection.type === 'loitering' ? (
                      <FaUserClock size={10} />
                    ) : detection.type === 'theft' ? (
                      <FaUserNinja size={10} />
                    ) : null}
                    <span>
                      {detection.type === 'person' ? 'Person' : 
                       detection.type === 'loitering' ? 'Loitering' : 
                       detection.type === 'theft' ? 'Theft' : detection.type}
                    </span>
                  </div>
                  <span>{Math.round(detection.confidence * 100)}%</span>
                </div>
              ))}
            </>
          ) : (
            <div className='w-full h-full flex flex-col items-center justify-center text-gray-400'>
              <svg className="w-16 h-16 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className='font-medium'>Camera Offline</p>
              <p className='text-sm'>This camera is currently unavailable</p>
            </div>
          )}
        </div>
        
        {/* Live Status Indicator */}
        {selectedCamera.status === 'online' && (
          <div className='absolute top-3 right-3 bg-black bg-opacity-50 text-white text-xs px-2 py-1 rounded flex items-center gap-1'>
            <span className='h-2 w-2 bg-red-500 rounded-full animate-pulse'></span>
            Live
          </div>
        )}
      </div>
      
      {/* Camera Details */}
      <div className='p-4'>
        <div className='grid grid-cols-2 md:grid-cols-4 gap-4'>
          <div className='text-center'>
            <p className='text-sm text-gray-500'>People</p>
            <p className='text-xl font-bold'>{selectedCamera.details.people || 0}</p>
          </div>
          <div className='text-center'>
            <p className='text-sm text-gray-500'>Objects</p>
            <p className='text-xl font-bold'>{selectedCamera.details.objects || 0}</p>
          </div>
          <div className='text-center'>
            <p className='text-sm text-gray-500'>Loitering</p>
            <p className={`text-xl font-bold ${selectedCamera.details.loitering > 0 ? 'text-orange-500' : ''}`}>
              {selectedCamera.details.loitering || 0}
            </p>
          </div>
          <div className='text-center'>
            <p className='text-sm text-gray-500'>Theft</p>
            <p className={`text-xl font-bold ${selectedCamera.details.theft > 0 ? 'text-red-500' : ''}`}>
              {selectedCamera.details.theft || 0}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveCameraComponent;