import React, { useState, useEffect, useRef } from 'react';
import { 
  Eye, 
  Clock, 
  ShieldAlert, 
  UserCheck, 
  ShoppingBag,
  User,
  Pause,
  Play
} from 'lucide-react';

const LiveCameraComponent = ({ selectedCamera, onUpdateStats }) => {
  const [isVideoPlaying, setIsVideoPlaying] = useState(true);
  const [videoError, setVideoError] = useState(false);
  const videoRef = useRef(null);

  // Toggle video playback
  const toggleVideo = () => {
    if (videoRef.current) {
      if (isVideoPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsVideoPlaying(!isVideoPlaying);
    }
  };

  // Reset video when camera changes
  useEffect(() => {
    setVideoError(false);
    
    // Handle autoplay for new camera selection
    if (videoRef.current && selectedCamera) {
      videoRef.current.load();
      const playPromise = videoRef.current.play();
      
      if (playPromise !== undefined) {
        playPromise
          .then(() => setIsVideoPlaying(true))
          .catch(error => {
            console.error("Video autoplay failed:", error);
            setIsVideoPlaying(false);
          });
      }
    }
  }, [selectedCamera]);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold flex items-center">
          <Eye className="mr-2 text-blue-500" /> 
          Live Camera Feed
        </h2>
        
        <div className="flex gap-2">
          <button
            onClick={toggleVideo}
            className="p-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
            title={isVideoPlaying ? "Pause" : "Play"}
            disabled={!selectedCamera || videoError}
          >
            {isVideoPlaying ? <Pause size={16} /> : <Play size={16} />}
          </button>
        </div>
      </div>
      
      {/* Camera view */}
      <div className="relative flex-grow rounded-lg overflow-hidden bg-gray-900">
        <div className="relative aspect-video">
          {/* Camera feed */}
          {selectedCamera ? (
            <video
              ref={videoRef}
              className="w-full h-full object-cover"
              autoPlay
              loop
              muted
              playsInline
              onError={() => setVideoError(true)}
              poster="https://via.placeholder.com/640x360?text=Loading+Camera+Feed"
            >
              <source src={selectedCamera.videoUrl} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gray-800 text-gray-400">
              No camera selected
            </div>
          )}
          
          {/* Status indicators */}
          <div className="absolute top-4 left-4 bg-black bg-opacity-60 text-white px-3 py-1 rounded-lg text-sm flex items-center">
            <Clock size={14} className="mr-1" /> {new Date().toLocaleTimeString()}
          </div>
          
          {videoError && (
            <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-70">
              <div className="text-center">
                <ShieldAlert size={50} className="mx-auto mb-2 text-red-500" />
                <p className="text-white font-semibold text-lg">Video Error</p>
                <p className="text-white text-sm">Unable to load camera feed</p>
              </div>
            </div>
          )}
          
          {selectedCamera && selectedCamera.status === 'offline' && (
            <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-70">
              <div className="text-center">
                <ShieldAlert size={50} className="mx-auto mb-2 text-red-500" />
                <p className="text-white font-semibold text-lg">Camera Offline</p>
              </div>
            </div>
          )}
          
          {/* Camera info overlay */}
          {selectedCamera && (
            <div className="absolute bottom-4 left-4 bg-black bg-opacity-70 text-white p-3 rounded-lg max-w-[70%]">
              <h3 className="text-lg font-semibold flex items-center">
                {selectedCamera.name}
                {selectedCamera.status === 'online' && (
                  <span className="ml-2 flex items-center text-xs bg-green-500 text-white px-2 py-0.5 rounded-full">
                    <span className="w-2 h-2 bg-white rounded-full mr-1 animate-pulse"></span> LIVE
                  </span>
                )}
              </h3>
              
              <div className="grid grid-cols-2 gap-x-6 gap-y-1 mt-2">
                <p className="flex items-center text-sm">
                  <User size={14} className="mr-1 text-blue-400" /> 
                  People: {selectedCamera.details?.people || 0}
                </p>
                <p className="flex items-center text-sm">
                  <UserCheck size={14} className="mr-1 text-orange-400" /> 
                  Loitering: {0}
                </p>
                <p className="flex items-center text-sm">
                  <ShoppingBag size={14} className="mr-1 text-red-400" /> 
                  Theft: {0}
                </p>
                <p className="flex items-center text-sm">
                  <Eye size={14} className="mr-1 text-purple-400" /> 
                  Objects: {selectedCamera.details?.objects || 0}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LiveCameraComponent;