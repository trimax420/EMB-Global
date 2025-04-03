import React, { useEffect, useRef, useState } from 'react';
import PropTypes from 'prop-types';

const WebRTCVideoStream = ({ 
  cameraId = null, 
  videoPath = null, 
  detectionType = 'all',
  onUpdateStats = null,
  className = '',
  autoPlay = true,
  controls = false,
  apiBaseUrl = 'http://localhost:8000/api',
  preferredResolution = '720p'
}) => {
  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentDetectionType, setCurrentDetectionType] = useState(detectionType);
  const [stats, setStats] = useState({
    people: 0,
    loitering: 0,
    theft: 0,
    objects: 0,
    resolution: { width: 0, height: 0 },
    connected: false
  });

  // Function to start WebRTC connection
  const startWebRTC = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Close existing connections
      if (pcRef.current) {
        console.log('Closing existing connection before creating a new one');
        pcRef.current.close();
      }

      // Use the direct path if available
      let serverVideoPath = videoPath;
      
      // Log the video path for debugging
      console.log(`Attempting WebRTC connection with video path: ${serverVideoPath}, resolution: ${preferredResolution}`);
      
      // Create a new RTCPeerConnection
      const pc = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'stun:stun1.l.google.com:19302' },
          { urls: 'stun:stun2.l.google.com:19302' },
          { urls: 'stun:stun3.l.google.com:19302' }
        ]
      });
      pcRef.current = pc;

      // Handle ICE candidates
      pc.onicecandidate = (event) => {
        if (event.candidate) {
          console.log('ICE candidate', event.candidate.candidate);
        }
      };

      // Add more detailed ICE gathering state logging
      pc.onicegatheringstatechange = () => {
        console.log('ICE gathering state changed to:', pc.iceGatheringState);
      };

      // Handle connection state changes with better logging
      pc.onconnectionstatechange = () => {
        console.log('Connection state changed to:', pc.connectionState);
        if (pc.connectionState === 'connected') {
          console.log('WebRTC connection established successfully');
          setIsConnected(true);
          setIsLoading(false);
          setStats(prev => ({ ...prev, connected: true }));
        } else if (pc.connectionState === 'disconnected' || 
                   pc.connectionState === 'failed' || 
                   pc.connectionState === 'closed') {
          console.log('WebRTC connection lost:', pc.connectionState);
          setIsConnected(false);
          setStats(prev => ({ ...prev, connected: false }));
          if (pc.connectionState === 'failed') {
            // Set error state to trigger reconnection
            setError('Connection failed. Will retry in 5 seconds...');
          }
        }
      };

      // Handle ICE connection state changes
      pc.oniceconnectionstatechange = () => {
        console.log('ICE connection state changed to:', pc.iceConnectionState);
        if (pc.iceConnectionState === 'failed') {
          console.error('ICE connection failed - STUN server might be unreachable');
          setError('ICE connection failed. Network issue detected. Will retry in 5 seconds...');
        }
      };

      // Handle track events (receiving video)
      pc.ontrack = (event) => {
        console.log('Received track', event.track.kind);
        if (event.track.kind === 'video' && videoRef.current) {
          console.log('Setting video source from incoming stream');
          videoRef.current.srcObject = event.streams[0];
          
          // Update stats with video dimensions when loaded
          videoRef.current.onloadedmetadata = () => {
            const videoWidth = videoRef.current.videoWidth;
            const videoHeight = videoRef.current.videoHeight;
            console.log(`Video dimensions: ${videoWidth}x${videoHeight}`);
            setStats(prev => ({
              ...prev,
              resolution: { width: videoWidth, height: videoHeight }
            }));
          };
          
          // Set timeouts for connection monitoring
          let noFramesTimeout = setTimeout(() => {
            if (isConnected) {
              console.warn('No video frames received in 10 seconds, checking connection');
              if (videoRef.current && (!videoRef.current.videoWidth || !videoRef.current.videoHeight)) {
                setError('Video stream connected but no frames received. Will retry.');
              }
            }
          }, 10000);

          // Clear timeout when component unmounts
          return () => clearTimeout(noFramesTimeout);
        }
      };

      // Create a timeout for connection establishment
      const connectionTimeout = setTimeout(() => {
        if (!isConnected && !error) {
          console.error('Connection timeout - 15 seconds without successful connection');
          setError('Connection timeout. Server might be overloaded. Retrying...');
        }
      }, 15000);

      // Create an offer
      console.log('Creating WebRTC offer...');
      const offer = await pc.createOffer({
        offerToReceiveVideo: true,
        offerToReceiveAudio: false,
      });
      await pc.setLocalDescription(offer);
      console.log('Local description set:', offer.type);

      // Use WebSocket for signaling if available, otherwise fallback to REST API
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        console.log('Using WebSocket for signaling');
        // Send offer via WebSocket
        wsRef.current.send(JSON.stringify({
          type: 'webrtc_offer',
          offer: {
            sdp: pc.localDescription.sdp,
            type: pc.localDescription.type
          },
          camera_id: cameraId,
          video_path: serverVideoPath,
          detection_type: currentDetectionType,
          resolution: preferredResolution
        }));
      } else {
        console.log('Using REST API for signaling');
        // Make sure the video path is properly encoded
        const encodedVideoPath = serverVideoPath ? encodeURIComponent(serverVideoPath) : '';
        const endpoint = `${apiBaseUrl}/webrtc/offer?camera_id=${cameraId || ''}&video_path=${encodedVideoPath}&detection_type=${currentDetectionType}&resolution=${preferredResolution}`;
        console.log(`Sending offer to: ${endpoint}`);
        
        // Fallback to REST API - use encodeURIComponent to handle special characters in path
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            sdp: pc.localDescription.sdp,
            type: pc.localDescription.type
          })
        });

        if (!response.ok) {
          const errorText = await response.text();
          console.error(`Server error (${response.status}):`, errorText);
          throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const answer = await response.json();
        console.log('Received answer from server:', answer);
        await pc.setRemoteDescription(new RTCSessionDescription(answer));
        console.log('Remote description set successfully');
      }
      
      console.log(`Started WebRTC stream with detection type: ${currentDetectionType} for video: ${serverVideoPath || 'camera feed'} at resolution: ${preferredResolution}`);
      
      // Clear the connection timeout since we've successfully initiated the connection attempt
      clearTimeout(connectionTimeout);
    } catch (err) {
      console.error('WebRTC connection failed:', err);
      setError(`WebRTC error: ${err.message}`);
      setIsLoading(false);
    }
  };

  // Effect to update detection type when prop changes
  useEffect(() => {
    // If the detection type has changed and we're already connected, 
    // we need to restart the connection with the new detection type
    if (detectionType !== currentDetectionType && isConnected) {
      setCurrentDetectionType(detectionType);
      // Force restart connection with new detection type
      startWebRTC();
    } else if (detectionType !== currentDetectionType) {
      setCurrentDetectionType(detectionType);
    }
  }, [detectionType]);

  // Connect to WebSocket for signaling
  useEffect(() => {
    // Connect to WebSocket for signaling
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/api/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = async (event) => {
      try {
        const message = JSON.parse(event.data);
        
        if (message.type === 'webrtc_answer' && pcRef.current) {
          console.log('Received WebRTC answer via WebSocket');
          await pcRef.current.setRemoteDescription(
            new RTCSessionDescription(message.answer)
          );
        } else if (message.type === 'detection_stats') {
          // Handle real-time detection statistics from server
          if (message.stats) {
            const newStats = {
              ...stats,
              ...message.stats
            };
            setStats(newStats);
            // Forward stats to parent component if needed
            if (onUpdateStats) {
              onUpdateStats(newStats);
            }
          }
        }
      } catch (err) {
        console.error('WebSocket message error:', err);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };

    // Initial WebRTC connection
    startWebRTC();

    // Cleanup
    return () => {
      if (pcRef.current) {
        pcRef.current.close();
      }
      ws.close();
    };
  }, []);

  // Handle video metrics for stats
  useEffect(() => {
    if (!onUpdateStats || !videoRef.current || !isConnected) return;

    // Create a periodic stats collector
    const statsInterval = setInterval(async () => {
      if (!pcRef.current) return;
      
      try {
        // Get connection stats
        const rtcStats = await pcRef.current.getStats();
        let videoBytesReceived = 0;
        let frameRate = 0;
        
        rtcStats.forEach(report => {
          if (report.type === 'inbound-rtp' && report.kind === 'video') {
            videoBytesReceived = report.bytesReceived;
            frameRate = report.framesPerSecond;
          }
        });
        
        // Analyze the video frame for detections
        // This is a basic implementation to detect motion
        if (videoRef.current) {
          const videoHeight = videoRef.current.videoHeight;
          const videoWidth = videoRef.current.videoWidth;
          
          // Update stats
          const newStats = {
            ...stats,
            resolution: { width: videoWidth, height: videoHeight },
            connected: isConnected,
            bytesReceived: videoBytesReceived,
            frameRate: frameRate,
            detectionType: currentDetectionType
          };
          
          setStats(newStats);
          
          // Pass stats to parent component
          if (onUpdateStats) {
            onUpdateStats(newStats);
          }
        }
      } catch (err) {
        console.error('Error collecting video stats:', err);
      }
    }, 1000);

    return () => clearInterval(statsInterval);
  }, [isConnected, onUpdateStats]);

  // Reconnect if connection fails
  useEffect(() => {
    if (error) {
      const reconnectTimer = setTimeout(() => {
        console.log('Attempting to reconnect WebRTC...');
        startWebRTC();
      }, 5000);
      
      return () => clearTimeout(reconnectTimer);
    }
  }, [error]);

  // Change detection type
  const changeDetectionType = (newType) => {
    if (newType !== currentDetectionType) {
      setCurrentDetectionType(newType);
      startWebRTC();
    }
  };

  return (
    <div className={`webrtc-video-container relative ${className}`}>
      <video
        ref={videoRef}
        autoPlay={autoPlay}
        playsInline
        controls={controls}
        className="w-full h-full object-cover rounded-lg"
      />
      
      {/* Fallback image for preview - shown briefly until WebRTC loads */}
      {isLoading && videoPath && (
        <div className="absolute inset-0 bg-black z-10">
          <img 
            src={videoPath} 
            alt="Video preview" 
            className="w-full h-full object-cover opacity-50"
            onError={(e) => e.target.style.display = 'none'}
          />
        </div>
      )}
      
      {/* Detection type selector overlay - always visible when connected */}
      {isConnected && (
        <div className="absolute top-2 right-2 bg-black bg-opacity-60 rounded-lg p-2 z-20">
          <div className="flex flex-col gap-1 text-xs">
            <button 
              className={`px-2 py-1 rounded ${currentDetectionType === 'all' ? 'bg-blue-500 text-white' : 'bg-gray-700 text-gray-300'}`}
              onClick={() => changeDetectionType('all')}
            >
              All Detections
            </button>
            <button 
              className={`px-2 py-1 rounded ${currentDetectionType === 'loitering' ? 'bg-orange-500 text-white' : 'bg-gray-700 text-gray-300'}`}
              onClick={() => changeDetectionType('loitering')}
            >
              Loitering Only
            </button>
            <button 
              className={`px-2 py-1 rounded ${currentDetectionType === 'theft' ? 'bg-red-500 text-white' : 'bg-gray-700 text-gray-300'}`}
              onClick={() => changeDetectionType('theft')}
            >
              Theft Only
            </button>
          </div>
        </div>
      )}
      
      {/* Stats overlay - always visible when connected */}
      {isConnected && (
        <div className="absolute bottom-2 left-2 bg-black bg-opacity-60 rounded-lg p-2 text-white text-xs z-20">
          <div className="grid grid-cols-2 gap-x-3 gap-y-1">
            <div>Resolution: {stats.resolution.width}x{stats.resolution.height}</div>
            <div>FPS: {stats.frameRate || 'N/A'}</div>
            <div>Type: {currentDetectionType.toUpperCase()}</div>
            <div>Status: {isConnected ? 'Connected' : 'Disconnected'}</div>
          </div>
        </div>
      )}
      
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded-lg">
          <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full"></div>
        </div>
      )}
      
      {error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black bg-opacity-70 rounded-lg text-white p-4">
          <div className="text-red-500 mb-2">Error: {error}</div>
          <button 
            onClick={startWebRTC}
            className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-md transition-colors"
          >
            Retry Connection
          </button>
        </div>
      )}
      
      {!isConnected && !isLoading && !error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded-lg">
          <button 
            onClick={startWebRTC}
            className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-md transition-colors text-white"
          >
            Connect Stream
          </button>
        </div>
      )}
    </div>
  );
};

WebRTCVideoStream.propTypes = {
  cameraId: PropTypes.number,
  videoPath: PropTypes.string,
  detectionType: PropTypes.string,
  onUpdateStats: PropTypes.func,
  className: PropTypes.string,
  autoPlay: PropTypes.bool,
  controls: PropTypes.bool,
  apiBaseUrl: PropTypes.string,
  preferredResolution: PropTypes.string
};

export default WebRTCVideoStream; 