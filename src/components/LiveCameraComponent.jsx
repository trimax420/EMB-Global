import React, { useState, useEffect, useRef } from 'react';
import { 
  Eye, 
  Clock, 
  ShieldAlert, 
  UserCheck, 
  ShoppingBag,
  User,
  Pause,
  Play,
  RefreshCw
} from 'lucide-react';

const LiveCameraComponent = ({ selectedCamera, onUpdateStats }) => {
  const [isVideoPlaying, setIsVideoPlaying] = useState(true);
  const [videoError, setVideoError] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [useFallback, setUseFallback] = useState(false);
  const videoRef = useRef(null);

  // Reliable fallback video URL
  const fallbackVideoUrl = "https://user-images.githubusercontent.com/11428131/137016574-0d180d9b-fb9a-42a9-94b7-fbc0dbc18560.gif";

  // Get the current video URL to use (original or fallback)
  const currentVideoUrl = useFallback ? fallbackVideoUrl : selectedCamera?.videoUrl;

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

  // Retry loading the video
  const retryVideoLoad = () => {
    if (videoRef.current && selectedCamera) {
      setVideoError(false);
      setRetryCount(prev => prev + 1);
      
      if (retryCount >= 2 && !useFallback) {
        // After two failed attempts, switch to fallback video
        console.log('Switching to fallback video after multiple failures');
        setUseFallback(true);
        
        // Reset the video element with the fallback URL
        setTimeout(() => {
          videoRef.current.src = fallbackVideoUrl;
          videoRef.current.load();
          
          const playPromise = videoRef.current.play();
          if (playPromise !== undefined) {
            playPromise
              .then(() => {
                setIsVideoPlaying(true);
                console.log('Fallback video playback started successfully');
              })
              .catch(error => {
                console.error("Fallback video play failed:", error);
                setIsVideoPlaying(false);
                setVideoError(true);
              });
          }
        }, 100);
        
        return;
      }
      
      // Create a completely new URL with timestamp to avoid caching
      let videoUrl = selectedCamera.videoUrl;
      
      // For S3 URLs, add proper parameters
      if (videoUrl.includes('s3.') || videoUrl.includes('cloudfront.net')) {
        // Parse the existing URL and remove any existing timestamp
        const urlParts = videoUrl.split('?');
        const baseUrl = urlParts[0];
        
        // Create query parameters
        const params = new URLSearchParams();
        params.append('t', new Date().getTime());
        
        // Rebuild the URL
        videoUrl = `${baseUrl}?${params.toString()}`;
        console.log('Retrying with new URL:', videoUrl);
      } else {
        // For non-S3 URLs, just add a timestamp
        videoUrl = videoUrl.includes('?') 
          ? `${videoUrl.split('?')[0]}?t=${new Date().getTime()}`
          : `${videoUrl}?t=${new Date().getTime()}`;
      }
      
      // Set the updated URL directly on the video element
      videoRef.current.src = videoUrl;
      
      // Add a small delay before reloading to ensure the DOM is updated
      setTimeout(() => {
        videoRef.current.load();
        
        // Attempt to play after a short delay
        setTimeout(() => {
          const playPromise = videoRef.current.play();
          if (playPromise !== undefined) {
            playPromise
              .then(() => {
                setIsVideoPlaying(true);
                console.log('Video playback resumed successfully');
              })
              .catch(error => {
                console.error("Video retry play failed:", error);
                setIsVideoPlaying(false);
                setVideoError(true);
              });
          }
        }, 300);
      }, 100);
    }
  };

  // Reset video when camera changes
  useEffect(() => {
    setVideoError(false);
    setRetryCount(0);
    setUseFallback(false);
    
    // Handle autoplay for new camera selection
    if (videoRef.current && selectedCamera) {
      // Delay loading to ensure proper cleanup
      const timeoutId = setTimeout(() => {
        console.log('Loading video:', selectedCamera.videoUrl);
        videoRef.current.load();
        
        // Add better error handling for S3 URLs
        if ((selectedCamera.videoUrl && selectedCamera.videoUrl.includes('s3.')) || 
            selectedCamera.videoUrl.includes('cloudfront.net')) {
          console.log('Loading cloud video:', selectedCamera.videoUrl);
        }
        
        const playPromise = videoRef.current.play();
        
        if (playPromise !== undefined) {
          playPromise
            .then(() => setIsVideoPlaying(true))
            .catch(error => {
              console.error("Video autoplay failed:", error);
              setIsVideoPlaying(false);
            });
        }
      }, 100); // Small delay to prevent race conditions
      
      return () => clearTimeout(timeoutId);
    }
  }, [selectedCamera]);

  // Add this new useEffect to log errors and set up better error handling
  useEffect(() => {
    const videoElement = videoRef.current;
    
    const handleError = (e) => {
      console.error("Video error detected:", e);
      console.error("Error code:", videoElement?.error?.code);
      console.error("Error message:", videoElement?.error?.message);
      setVideoError(true);
    };
    
    if (videoElement) {
      videoElement.addEventListener('error', handleError);
      return () => videoElement.removeEventListener('error', handleError);
    }
  }, []);

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
              src={currentVideoUrl}
              autoPlay
              loop
              muted
              playsInline
              preload="auto"
              crossOrigin="anonymous"
              onError={(e) => {
                console.error("Video error event:", e);
                console.error("Video error details:", videoRef.current?.error);
                setVideoError(true);
              }}
              poster="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQwIiBoZWlnaHQ9IjM2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iNjQwIiBoZWlnaHQ9IjM2MCIgZmlsbD0iIzEyMjAyQyIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMjQiIGZpbGw9IiNmZmZmZmYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGRvbWluYW50LWJhc2VsaW5lPSJtaWRkbGUiPkxvYWRpbmcgQ2FtZXJhIEZlZWQ8L3RleHQ+PC9zdmc+"
            />
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
                <button 
                  onClick={retryVideoLoad}
                  className="mt-4 bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded-lg flex items-center mx-auto"
                >
                  <RefreshCw size={16} className="mr-2" /> Retry Loading
                </button>
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