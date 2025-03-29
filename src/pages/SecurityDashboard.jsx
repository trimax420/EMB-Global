import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement } from 'chart.js';
import { Pie, Bar } from 'react-chartjs-2';
import { dashboardService, alertService, securityService } from '../services/api';

// Register Chart.js components
ChartJS.register(ArcElement, CategoryScale, LinearScale, BarElement, Tooltip, Legend);

const SecurityDashboard = () => {
  // State for security data
  const [securityStats, setSecurityStats] = useState({
    loitering: { count: 0, recent: [] },
    theft: { count: 0, recent: [] },
    damage: { count: 0, recent: [] },
    total: 0
  });
  
  // State for customer demographics
  const [demographics, setDemographics] = useState({
    gender: { male: 0, female: 0, unknown: 0 },
    ageGroups: { child: 0, young: 0, adult: 0, senior: 0 }
  });
  
  // State for active camera stream
  const [activeCamera, setActiveCamera] = useState(1);
  const [cameraList, setCameraList] = useState([]);
  const [streamUrl, setStreamUrl] = useState('');
  const [incidents, setIncidents] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [websocket, setWebsocket] = useState(null);
  
  // WebSocket connection for real-time updates
  useEffect(() => {
    // Create WebSocket connection
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
      console.log('WebSocket connection established');
      // Start camera stream
      ws.send(JSON.stringify({ 
        action: 'start_stream', 
        camera_id: activeCamera 
      }));
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Received message type:', data.type);
        
        if (data.type === 'frame') {
          // Update video frame
          const imageElement = document.getElementById('live-feed');
          if (imageElement && data.frame) {
            imageElement.src = `data:image/jpeg;base64,${data.frame}`;
          }
          
          // Process batch of detections if available
          if (data.detections && Array.isArray(data.detections)) {
            updateSecurityStats(data.detections);
          }
          
          // Update demographics if included
          if (data.demographics) {
            setDemographics(prev => ({
              gender: {
                male: data.demographics.male || prev.gender.male,
                female: data.demographics.female || prev.gender.female,
                unknown: data.demographics.unknown || prev.gender.unknown
              },
              ageGroups: {
                child: data.demographics.age_groups?.child || prev.ageGroups.child,
                young: data.demographics.age_groups?.young || prev.ageGroups.young,
                adult: data.demographics.age_groups?.adult || prev.ageGroups.adult,
                senior: data.demographics.age_groups?.senior || prev.ageGroups.senior
              }
            }));
          }
        } else if (data.type === 'stream_status') {
          console.log('Stream status:', data);
          // Update stream status display
          const statusElement = document.getElementById('stream-status');
          if (statusElement) {
            statusElement.textContent = `Status: ${data.status} | FPS: ${data.fps?.toFixed(1) || 0}`;
            statusElement.className = `absolute bottom-2 right-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs ${
              data.status === 'connected' ? 'text-green-400' : 'text-yellow-400'
            }`;
          }
        } else if (data.type === 'error') {
          console.error('Stream error:', data.error);
          setError(`Stream error: ${data.error}`);
        }
      } catch (error) {
        console.error('Error processing WebSocket message:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Failed to connect to real-time feed');
    };
    
    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };
    
    setWebsocket(ws);
    
    // Cleanup function
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [activeCamera]);
  
  // Fetch initial dashboard data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        
        // Fetch security statistics
        const stats = await securityService.getSecurityStats();
        
        // Fetch recent incidents of each type
        const [loiteringIncidents, theftIncidents, damageIncidents, allIncidents] = await Promise.all([
          securityService.getIncidentsByType('loitering', 3),
          securityService.getIncidentsByType('theft', 3),
          securityService.getIncidentsByType('damage', 3),
          securityService.getIncidentsByType('all', 5)
        ]);
        
        // Fetch camera list
        const cameras = await axios.get('http://localhost:8000/api/cameras');
        
        // Process security statistics
        const securityData = {
          loitering: { 
            count: stats.loitering_count || 0, 
            recent: loiteringIncidents || []
          },
          theft: { 
            count: stats.theft_count || 0, 
            recent: theftIncidents || []
          },
          damage: { 
            count: stats.damage_count || 0, 
            recent: damageIncidents || []
          },
          total: stats.total_incidents || 0
        };
        
        // Process demographics data
        const demographicsData = {
          gender: {
            male: stats.demographics?.male_count || 0,
            female: stats.demographics?.female_count || 0,
            unknown: stats.demographics?.unknown_gender_count || 0
          },
          ageGroups: {
            child: stats.demographics?.child_count || 0,
            young: stats.demographics?.young_count || 0,
            adult: stats.demographics?.adult_count || 0,
            senior: stats.demographics?.senior_count || 0
          }
        };
        
        setSecurityStats(securityData);
        setDemographics(demographicsData);
        setIncidents(allIncidents);
        setCameraList(cameras.data);
        setIsLoading(false);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data');
        setIsLoading(false);
      }
    };
    
    fetchData();
    
    // Set up polling for updates every 30 seconds
    const intervalId = setInterval(fetchData, 30000);
    
    // Cleanup
    return () => clearInterval(intervalId);
  }, []);
  
  // Update security stats with new detections
  const updateSecurityStats = (detections) => {
    setSecurityStats(prev => {
      const updated = {...prev};
      
      // Process each detection in the array
      detections.forEach(detection => {
        // Determine detection type
        if (detection.type === 'loitering') {
          updated.loitering.count += 1;
          // Only add to recent if it has location info
          if (detection.bbox) {
            const newDetection = {
              timestamp: new Date().toISOString(),
              location: 'Main Entrance',
              image_path: `/static/uploads/detections/frames/loitering_${Date.now()}.jpg`,
              ...detection
            };
            updated.loitering.recent = [newDetection, ...updated.loitering.recent.slice(0, 2)];
          }
        } else if (detection.type === 'theft') {
          updated.theft.count += 1;
          if (detection.bbox) {
            const newDetection = {
              timestamp: new Date().toISOString(),
              location: 'Electronics Section',
              image_path: `/static/uploads/detections/frames/theft_${Date.now()}.jpg`,
              ...detection
            };
            updated.theft.recent = [newDetection, ...updated.theft.recent.slice(0, 2)];
          }
        } else if (detection.type === 'damage') {
          updated.damage.count += 1;
          if (detection.bbox) {
            const newDetection = {
              timestamp: new Date().toISOString(),
              location: 'Fragile Items Section',
              image_path: `/static/uploads/detections/frames/damage_${Date.now()}.jpg`,
              ...detection
            };
            updated.damage.recent = [newDetection, ...updated.damage.recent.slice(0, 2)];
          }
        }
      });
      
      // Update total count
      updated.total = updated.loitering.count + updated.theft.count + updated.damage.count;
      return updated;
    });
  };
  
  // Handle camera change
  const handleCameraChange = (cameraId) => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      console.log(`Changing camera to ${cameraId}`);
      
      // Stop current stream
      websocket.send(JSON.stringify({ 
        action: 'stop_stream',
        camera_id: activeCamera
      }));
      
      // Start new camera stream with configuration
      websocket.send(JSON.stringify({ 
        action: 'start_stream',
        camera_id: cameraId,
        config: {
          batch_size: 4,
          process_every_n_frames: 1,
          detection_types: ['face', 'loitering', 'theft', 'damage'],
          output_format: 'jpg'
        }
      }));
      
      setActiveCamera(cameraId);
    }
  };
  
  // Chart data for gender distribution
  const genderData = {
    labels: ['Male', 'Female', 'Unknown'],
    datasets: [
      {
        data: [demographics.gender.male, demographics.gender.female, demographics.gender.unknown],
        backgroundColor: ['#3b82f6', '#ec4899', '#9ca3af'],
        borderWidth: 1,
      },
    ],
  };
  
  // Chart data for age groups
  const ageData = {
    labels: ['Child', 'Young Adult', 'Adult', 'Senior'],
    datasets: [
      {
        label: 'Customer Age Distribution',
        data: [
          demographics.ageGroups.child,
          demographics.ageGroups.young,
          demographics.ageGroups.adult,
          demographics.ageGroups.senior
        ],
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',
          'rgba(54, 162, 235, 0.5)',
          'rgba(255, 206, 86, 0.5)',
          'rgba(75, 192, 192, 0.5)',
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };
  
  if (isLoading) {
    return (
      <div className="p-6 flex justify-center items-center min-h-screen">
        <div className="text-xl">Loading dashboard data...</div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="p-6 text-red-500">
        <h2 className="text-2xl font-bold mb-4">Error</h2>
        <p>{error}</p>
      </div>
    );
  }
  
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-800 mb-6">Security Dashboard</h1>
      
      {/* Security Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-2">Total Incidents</h2>
          <p className="text-3xl font-bold">{securityStats.total}</p>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-2 text-yellow-600">Loitering</h2>
          <p className="text-3xl font-bold">{securityStats.loitering.count}</p>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-2 text-red-600">Theft Attempts</h2>
          <p className="text-3xl font-bold">{securityStats.theft.count}</p>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-2 text-orange-600">Product Damage</h2>
          <p className="text-3xl font-bold">{securityStats.damage.count}</p>
        </div>
      </div>
      
      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Live feed section */}
        <div className="lg:col-span-2">
          <div className="bg-white p-4 rounded-lg shadow mb-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold">Live Security Feed</h2>
              <div className="flex space-x-2">
                {cameraList.map(camera => (
                  <button
                    key={camera.id}
                    onClick={() => handleCameraChange(camera.id)}
                    className={`px-3 py-1 rounded text-sm ${
                      activeCamera === camera.id 
                        ? 'bg-blue-500 text-white' 
                        : 'bg-gray-200 text-gray-700'
                    }`}
                  >
                    {camera.name}
                  </button>
                ))}
              </div>
            </div>
            <div className="relative aspect-video bg-black rounded overflow-hidden">
              <img 
                id="live-feed" 
                src="/images/placeholder-camera.jpg" 
                alt="Live Camera Feed"
                className="w-full h-full object-contain"
              />
              <div className="absolute top-2 left-2 bg-red-500 text-white px-2 py-1 rounded text-xs opacity-75">
                LIVE
              </div>
              <div 
                id="stream-status" 
                className="absolute bottom-2 right-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs"
              >
                Status: Processing | Buffer: 0 frames | Connected: Yes
              </div>
            </div>
          </div>
          
          {/* Recent alerts section */}
          <div className="bg-white p-4 rounded-lg shadow">
            <h2 className="text-lg font-semibold mb-4">Recent Security Alerts</h2>
            <div className="overflow-hidden">
              <table className="min-w-full bg-white">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="py-2 px-4 text-left">Time</th>
                    <th className="py-2 px-4 text-left">Type</th>
                    <th className="py-2 px-4 text-left">Location</th>
                    <th className="py-2 px-4 text-left">Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {incidents.length > 0 ? (
                    incidents.slice(0, 5).map((incident, index) => (
                      <tr key={index} className="hover:bg-gray-50 border-b">
                        <td className="py-2 px-4">{new Date(incident.timestamp).toLocaleTimeString()}</td>
                        <td className="py-2 px-4">
                          <span className={`px-2 py-1 rounded text-xs ${
                            incident.type === 'theft' ? 'bg-red-100 text-red-800' :
                            incident.type === 'loitering' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-orange-100 text-orange-800'
                          }`}>
                            {incident.type.charAt(0).toUpperCase() + incident.type.slice(1)}
                          </span>
                        </td>
                        <td className="py-2 px-4">{incident.location}</td>
                        <td className="py-2 px-4">
                          <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div 
                              className="bg-blue-600 h-2.5 rounded-full" 
                              style={{ width: `${(incident.confidence || 0.75) * 100}%` }}
                            ></div>
                          </div>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="4" className="py-4 text-center">No recent incidents</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        
        {/* Demographics section */}
        <div className="lg:col-span-1">
          <div className="bg-white p-4 rounded-lg shadow mb-6">
            <h2 className="text-lg font-semibold mb-4">Customer Demographics</h2>
            
            <div className="mb-6">
              <h3 className="text-md font-medium mb-2">Gender Distribution</h3>
              <div style={{ height: '200px' }}>
                <Pie data={genderData} options={{ maintainAspectRatio: false }} />
              </div>
            </div>
            
            <div>
              <h3 className="text-md font-medium mb-2">Age Distribution</h3>
              <div style={{ height: '200px' }}>
                <Bar 
                  data={ageData} 
                  options={{ 
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        display: false
                      }
                    }
                  }} 
                />
              </div>
            </div>
          </div>
          
          {/* Security type breakdown */}
          <div className="bg-white p-4 rounded-lg shadow">
            <h2 className="text-lg font-semibold mb-4">Security Breakdown</h2>
            
            <div className="space-y-4">
              <div>
                <h3 className="text-md font-medium mb-2 text-yellow-600">Recent Loitering</h3>
                {securityStats.loitering.recent.length > 0 ? (
                  securityStats.loitering.recent.map((item, index) => (
                    <div key={index} className="flex items-center p-2 bg-yellow-50 rounded mb-2">
                      {item.image_path && (
                        <img src={item.image_path} alt="Loitering" className="w-10 h-10 rounded object-cover mr-3" />
                      )}
                      <div>
                        <p className="text-sm">{new Date(item.timestamp).toLocaleTimeString()}</p>
                        <p className="text-xs text-gray-600">{item.location || 'Main Entrance'}</p>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-gray-500">No recent loitering detected</p>
                )}
              </div>
              
              <div>
                <h3 className="text-md font-medium mb-2 text-red-600">Recent Theft Attempts</h3>
                {securityStats.theft.recent.length > 0 ? (
                  securityStats.theft.recent.map((item, index) => (
                    <div key={index} className="flex items-center p-2 bg-red-50 rounded mb-2">
                      {item.image_path && (
                        <img src={item.image_path} alt="Theft" className="w-10 h-10 rounded object-cover mr-3" />
                      )}
                      <div>
                        <p className="text-sm">{new Date(item.timestamp).toLocaleTimeString()}</p>
                        <p className="text-xs text-gray-600">{item.location || 'Electronics Section'}</p>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-gray-500">No recent theft attempts detected</p>
                )}
              </div>
              
              <div>
                <h3 className="text-md font-medium mb-2 text-orange-600">Recent Product Damage</h3>
                {securityStats.damage.recent.length > 0 ? (
                  securityStats.damage.recent.map((item, index) => (
                    <div key={index} className="flex items-center p-2 bg-orange-50 rounded mb-2">
                      {item.image_path && (
                        <img src={item.image_path} alt="Damage" className="w-10 h-10 rounded object-cover mr-3" />
                      )}
                      <div>
                        <p className="text-sm">{new Date(item.timestamp).toLocaleTimeString()}</p>
                        <p className="text-xs text-gray-600">{item.location || 'Fragile Items Section'}</p>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-gray-500">No recent product damage detected</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SecurityDashboard; 