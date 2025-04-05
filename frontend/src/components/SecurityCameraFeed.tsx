// src/components/SecurityCameraFeed.tsx
"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, SkipForward, Camera } from 'lucide-react';

interface CameraOption {
  id: string;
  name: string;
}

const SecurityCameraFeed: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState('cam_001');
  
  const cameras: CameraOption[] = [
    { id: 'cam_001', name: 'Perfume Section Main' },
    { id: 'cam_002', name: 'Perfume Section North' },
    { id: 'cam_003', name: 'Entrance' },
    { id: 'cam_004', name: 'Checkout Area' }
  ];
  
  useEffect(() => {
    // In a real implementation, this would connect to a streaming service
    // For demo purposes, we're simulating a video feed
    if (videoRef.current) {
      videoRef.current.src = `/api/video-feed/${selectedCamera}`;
    }
  }, [selectedCamera]);
  
  const togglePlayback = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };
  
  const handleCameraChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedCamera(event.target.value);
  };
  
  return (
    <div className="bg-white rounded-lg shadow h-full flex flex-col">
      <div className="px-4 py-3 border-b flex justify-between items-center">
        <h2 className="font-semibold">Live Camera Feed</h2>
        <div className="flex items-center text-sm">
          <Camera className="h-4 w-4 mr-1" />
          <select 
            value={selectedCamera}
            onChange={handleCameraChange}
            className="border rounded p-1"
          >
            {cameras.map(camera => (
              <option key={camera.id} value={camera.id}>
                {camera.name}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="flex-1 p-4 bg-black relative">
        {/* Simulated video feed with detection overlay */}
        <div className="h-full bg-gray-800 relative">
          <video 
            ref={videoRef}
            className="w-full h-full object-contain"
            muted
          />
          
          {/* Example detection boxes */}
          <div className="absolute inset-0 pointer-events-none">
            {/* Person detection box */}
            <div className="absolute border-2 border-blue-500" style={{ top: '40%', left: '30%', width: '10%', height: '30%' }}>
              <div className="absolute -top-5 left-0 bg-blue-500 text-white text-xs px-1">
                Person (95%)
              </div>
            </div>
            
            {/* Staff detection box */}
            <div className="absolute border-2 border-green-500" style={{ top: '50%', left: '60%', width: '12%', height: '28%' }}>
              <div className="absolute -top-5 left-0 bg-green-500 text-white text-xs px-1">
                Staff (92%)
              </div>
            </div>
            
            {/* Suspicious activity detection */}
            <div className="absolute border-2 border-red-500" style={{ top: '45%', left: '20%', width: '15%', height: '30%' }}>
              <div className="absolute -top-5 left-0 bg-red-500 text-white text-xs px-1">
                Suspicious (89%)
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="p-4 flex justify-center space-x-4">
        <button 
          onClick={togglePlayback}
          className="bg-blue-500 text-white rounded-full p-2"
        >
          {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
        </button>
        <button className="bg-gray-200 rounded-full p-2">
          <SkipForward className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
};

export default SecurityCameraFeed;