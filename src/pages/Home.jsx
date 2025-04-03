import React, { useState, useEffect } from 'react';
import { 
  FaCamera, 
  FaUsers, 
  FaBell, 
  FaChartBar, 
  FaExclamationTriangle, 
  FaEye, 
  FaHistory, 
  FaTimes,
  FaUserClock,
  FaUserNinja,
  FaListAlt,
  FaExclamation
} from "react-icons/fa";
import { ImNotification } from "react-icons/im";
import { TbActivityHeartbeat } from "react-icons/tb";
import { Bar, Line, Pie } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, Title, Tooltip, Legend } from 'chart.js';
import { useNavigate } from 'react-router-dom';
import LiveCameraComponent from '../components/LiveCameraComponent';
import axios from 'axios';
import DetectionControls from '../components/DetectionControls';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, Title, Tooltip, Legend, ArcElement);

// API Base URL
const API_BASE_URL = 'http://localhost:8000/api';

// Enhanced camera data with more detailed properties and video feeds from your API
const cameras = [
  {
    id: 1,
    name: "Front Entrance",
    videoUrl: "https://intelliod.s3.ap-south-1.amazonaws.com/cheese-1.mp4",
    details: { people: 4, vehicles: 1, alerts: 4, objects: 5 },
    status: "online",
    resolution: { width: 640, height: 480 },
    location: "North Wing",
    capabilities: ["face_detection", "theft_detection", "loitering_detection"]
  },
  {
    id: 2,
    name: "Parking Lot",
    videoUrl: "https://drive.google.com/uc?export=download&id=1sMVrW01Y32gYd6C0FMPf0VO8YdBh7mvT",
    details: { people: 2, vehicles: 3, alerts: 1, objects: 2 },
    status: "online",
    resolution: { width: 640, height: 480 },
    location: "Outdoor",
    capabilities: ["loitering_detection"]
  },
  {
    id: 3,
    name: "Electronics Department",
    videoUrl: "https://drive.google.com/uc?export=download&id=1XNxfvpgVKZvKPsDhJ5XWykLakPCxNTeR",
    details: { people: 0, vehicles: 0, alerts: 0, objects: 1 },
    status: "online",
    resolution: { width: 640, height: 480 },
    location: "West Wing",
    capabilities: ["theft_detection", "face_detection"]
  },
  {
    id: 4,
    name: "Storage Room",
    videoUrl: "https://user-images.githubusercontent.com/11428131/137016574-0d180d9b-fb9a-42a9-94b7-fbc0dbc18560.gif",
    details: { people: 1, vehicles: 0, alerts: 2, objects: 3 },
    status: "offline",
    resolution: { width: 640, height: 480 },
    location: "East Wing",
    capabilities: ["theft_detection"]
  }
];

// Sample time filter data - using a simplified version since we don't need all of this
const timeFilterData = {
  today: {
    stats: [
      { icon: <FaCamera />, title: "Total Cameras", value: "5", description: "4 offline, 8 online", trend: "neutral" },
      { icon: <FaUsers />, title: "Active Detections", value: "25", description: "+15% from last hour", trend: "up" },
      { icon: <ImNotification />, title: "Current Alerts", value: "7", description: "+2 new alerts", trend: "up" },
      { icon: <TbActivityHeartbeat />, title: "System Status", value: "Optimal", description: "All systems operational", trend: "neutral" },
    ],
    incidents: [
      { title: "Unauthorized Access", location: "Front Entrance", time: "14:35", severity: "high", isNew: true },
      { title: "Vehicle Stopped", location: "Parking Lot", time: "10:35", severity: "low", isNew: false },
      { title: "Person Detected", location: "Restricted Area", time: "14:35", severity: "medium", isNew: true },
      { title: "Motion Detected", location: "Storage Room", time: "14:35", severity: "low", isNew: false },
    ],
    detectionTrendData: {
      labels: ['8:00', '9:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00'],
      datasets: [
        {
          label: 'People',
          data: [4, 3, 7, 12, 9, 8, 17, 25],
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.5)',
          fill: false,
        }
      ],
    },
    incidentData: {
      labels: ['Morning', 'Afternoon', 'Evening', 'Night'],
      datasets: [
        {
          label: 'Incidents Today',
          data: [3, 5, 2, 1],
          backgroundColor: 'rgba(59, 130, 246, 0.5)',
          borderColor: 'rgb(59, 130, 246)',
          borderWidth: 1,
        },
      ],
    },
  },
  week: {
    stats: [/* Week stats here */],
    incidents: [/* Week incidents here */],
    detectionTrendData: {/* Week trend data here */},
    incidentData: {/* Week incident data here */}
  },
  month: {
    stats: [/* Month stats here */],
    incidents: [/* Month incidents here */],
    detectionTrendData: {/* Month trend data here */},
    incidentData: {/* Month incident data here */}
  }
};

const Home = () => {
  const [selectedCamera, setSelectedCamera] = useState(cameras[0]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [timeFilter, setTimeFilter] = useState('today');
  const [showReport, setShowReport] = useState(false);
  const [liveStats, setLiveStats] = useState({
    people: 0,
    loitering: 0,
    theft: 0,
    objects: 0
  });
  const [availableCameras, setAvailableCameras] = useState(cameras);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  // Effect to fetch cameras from API on component mount
  useEffect(() => {
    const fetchCameras = async () => {
      try {
        setIsLoading(true);
        // In a real application, you would fetch camera data from your API
        // const response = await axios.get(`${API_BASE_URL}/cameras`);
        // setAvailableCameras(response.data);
        
        // For now, we'll use the static data
        setAvailableCameras(cameras);
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching cameras:', error);
        setError('Failed to load cameras. Please try again later.');
        setIsLoading(false);
      }
    };

    fetchCameras();
  }, []);

  // Get data based on current time filter
  const currentData = timeFilterData[timeFilter];

  // Handle camera selection
  const handleCameraSelect = (camera) => {
    setSelectedCamera(camera);
    // Reset stats when changing camera
    setLiveStats({
      people: camera.details.people || 0,
      loitering: 0,
      theft: 0,
      objects: camera.details.objects || 0
    });
  };

  // Handle stats updates from LiveCameraComponent
  const handleUpdateStats = (newStats) => {
    setLiveStats(prevStats => ({
      ...prevStats,
      ...newStats
    }));
    
    // Also update the camera details in the cameras array
    setAvailableCameras(prevCameras => 
      prevCameras.map(cam => 
        cam.id === selectedCamera.id 
          ? {
              ...cam,
              details: {
                ...cam.details,
                people: newStats.people || cam.details.people,
                objects: newStats.objects || cam.details.objects,
                alerts: (newStats.loitering || 0) + (newStats.theft || 0) || cam.details.alerts
              }
            }
          : cam
      )
    );
  };

  // Toggle report modal
  const toggleReport = () => {
    setShowReport(!showReport);
  };

  // Navigate to the "View All Cameras" page
  const handleViewAll = () => {
    navigate("/Live-Feed");
  };

  // Toggle fullscreen for the selected camera
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  // Filter incidents based on time
  const filteredIncidents = currentData.incidents;

  // Count incidents by type for display in the stats section
  const incidentCounts = {
    total: currentData.incidents.length,
    high: currentData.incidents.filter(i => i.severity === 'high').length,
    new: currentData.incidents.filter(i => i.isNew).length
  };

  return (
    <div className='bg-gray-50 p-4 md:p-6 min-h-screen'>
      {/* Header Section */}
      <div className='flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4'>
        <div>
          <h1 className='text-2xl font-bold text-gray-800'>Security Dashboard</h1>
          <p className='text-gray-500'>Monitor your security system status and activities</p>
        </div>
        <div className='flex gap-3 self-end md:self-auto'>
          <div className='flex rounded-lg overflow-hidden border border-gray-200 shadow-sm'>
            <button 
              className={`px-4 py-2 ${timeFilter === 'today' ? 'bg-blue-500 text-white' : 'bg-white text-gray-700 hover:bg-gray-100'}`}
              onClick={() => setTimeFilter('today')}
            >
              Today
            </button>
            <button 
              className={`px-4 py-2 ${timeFilter === 'week' ? 'bg-blue-500 text-white' : 'bg-white text-gray-700 hover:bg-gray-100'}`}
              onClick={() => setTimeFilter('week')}
            >
              Week
            </button>
            <button 
              className={`px-4 py-2 ${timeFilter === 'month' ? 'bg-blue-500 text-white' : 'bg-white text-gray-700 hover:bg-gray-100'}`}
              onClick={() => setTimeFilter('month')}
            >
              Month
            </button>
          </div>
          <button 
            className='px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-all duration-200 shadow-sm flex items-center gap-2'
            onClick={toggleReport}
          >
            <FaHistory className="text-sm" />
            View Report
          </button>
        </div>
      </div>

      {/* Stats Section - Using simple version for brevity */}
      <div className='grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6'>
        {currentData.stats.map(({ icon, title, value, description, trend }) => (
          <div
            className='bg-white p-5 rounded-xl shadow-sm hover:shadow-md transition-all duration-200 border border-gray-100 flex flex-col'
            key={title}
          >
            <div className='flex justify-between items-start mb-2'>
              <div className='text-2xl text-blue-500 p-2 bg-blue-50 rounded-lg'>{icon}</div>
              {trend === 'up' && <span className='text-green-500 text-xs font-medium bg-green-50 px-2 py-1 rounded-full'>↑ Increasing</span>}
              {trend === 'down' && <span className='text-red-500 text-xs font-medium bg-red-50 px-2 py-1 rounded-full'>↓ Decreasing</span>}
            </div>
            <h2 className='text-sm font-medium text-gray-500 mt-1'>{title}</h2>
            <p className='text-2xl font-bold text-gray-900 mt-1'>{value}</p>
            <p className='text-xs text-gray-500 mt-1'>{description}</p>
          </div>
        ))}
      </div>

      {/* Main Content Section */}
      <div className='lg:flex gap-6'>
        {/* Live Security Feed */}
        <div className="lg:w-[70%] mb-6 lg:mb-0">
          <LiveCameraComponent 
            selectedCamera={selectedCamera} 
            onUpdateStats={handleUpdateStats}
          />
          {/* Detection Controls - Added for theft and loitering detection */}
            <div className="mt-6">
              <DetectionControls selectedCamera={selectedCamera} />
            </div>
          {/* Available Cameras */}
          <div className='mt-6'>
            <div className='flex justify-between items-center mb-4'>
              <h1 className='text-lg font-semibold text-gray-800'>Available Cameras</h1>
              <button
                className='text-blue-500 hover:text-blue-600 transition-all duration-200 flex items-center gap-1 text-sm'
                onClick={handleViewAll}
              >
                <FaEye className='text-xs' /> View All
              </button>
            </div>
            
            {isLoading ? (
              <div className="flex justify-center items-center h-40">
                <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
              </div>
            ) : error ? (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
                {error}
              </div>
            ) : (
              <div className='grid grid-cols-2 md:grid-cols-4 gap-4'>
                {availableCameras.map((camera) => (
                  <div
                    key={camera.id}
                    className={`relative group bg-gray-900 rounded-lg overflow-hidden hover:ring-2 hover:ring-blue-400 transition-all duration-200 cursor-pointer ${
                      selectedCamera.id === camera.id ? 'ring-2 ring-blue-500' : ''
                    }`}
                    onClick={() => handleCameraSelect(camera)}
                  >
                    <div className='aspect-video'>
                      <img 
                        src={camera.videoUrl} 
                        alt={camera.name}
                        className='w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity'
                      />
                      {camera.status === 'offline' && (
                        <div className='absolute inset-0 bg-black bg-opacity-60 flex items-center justify-center'>
                          <span className='text-xs font-medium text-white bg-red-500 px-2 py-1 rounded-full'>Offline</span>
                        </div>
                      )}
                    </div>
                    <div className='absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-2'>
                      <p className='text-white text-sm font-medium'>{camera.name}</p>
                      <div className='flex justify-between text-xs text-gray-300'>
                        <span>{camera.details.people} people</span>
                        <span>{camera.details.alerts > 0 && (
                          <span className='text-red-400'>{camera.details.alerts} alerts</span>
                        )}</span>
                      </div>
                    </div>
                    {/* Display camera capabilities */}
                    {camera.capabilities && camera.capabilities.length > 0 && (
                      <div className="absolute top-2 right-2 flex flex-col gap-1">
                        {camera.capabilities.includes('theft_detection') && (
                          <span className="text-xs bg-red-500 text-white px-1.5 py-0.5 rounded">Theft</span>
                        )}
                        {camera.capabilities.includes('loitering_detection') && (
                          <span className="text-xs bg-orange-500 text-white px-1.5 py-0.5 rounded">Loitering</span>
                        )}
                        {camera.capabilities.includes('face_detection') && (
                          <span className="text-xs bg-green-500 text-white px-1.5 py-0.5 rounded">Face</span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right Sidebar - Simplified for brevity */}
        <div className='lg:w-[30%] space-y-6'>
          {/* Detection Trends */}
          <div className='bg-white rounded-xl shadow-sm border border-gray-100 p-4'>
            <h1 className='text-lg font-bold text-gray-800'>Detection Trends</h1>
            <p className='text-gray-500 text-sm mb-4'>{timeFilter === 'today' ? "Today's activity" : timeFilter === 'week' ? "This week's activity" : "This month's activity"}</p>
            <div className='bg-gray-50 p-4 rounded-lg'>
              <Bar 
                data={currentData.detectionTrendData} 
                options={{
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      beginAtZero: true,
                      title: {
                        display: true,
                        text: 'Count'
                      }
                    },
                    x: {
                      title: {
                        display: true,
                        text: timeFilter === 'today' ? 'Time' : timeFilter === 'week' ? 'Day' : 'Week'
                      }
                    }
                  }
                }} 
                height={180}
              />
            </div>
          </div>

          {/* Recent Incidents */}
          <div className='bg-white rounded-xl shadow-sm border border-gray-100 p-4'>
            <h1 className='text-lg font-bold text-gray-800 mb-1'>Recent Incidents</h1>
            <p className='text-gray-500 text-sm mb-4'>{timeFilter === 'today' ? "Last 24 hours" : timeFilter === 'week' ? "This week" : "This month"}</p>
            <div className='space-y-3 max-h-[300px] overflow-y-auto pr-1'>
              {filteredIncidents.map(({ title, location, time, severity, isNew }, index) => (
                <div 
                  className='flex justify-between items-center p-3 rounded-lg hover:bg-gray-50 transition-colors border border-gray-100' 
                  key={index}
                >
                  <div className='flex gap-3 items-start'>
                    <div className={`p-2 rounded-lg ${
                      severity === 'high' 
                        ? 'bg-red-100 text-red-500' 
                        : severity === 'medium'
                        ? 'bg-yellow-100 text-yellow-500'
                        : 'bg-green-100 text-green-500'
                    }`}>
                      <FaExclamationTriangle />
                    </div>
                    <div>
                      <div className='flex items-center'>
                        <h2 className='font-medium text-gray-800'>{title}</h2>
                        {isNew && <span className='ml-2 text-xs bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full'>New</span>}
                      </div>
                      <p className='text-sm text-gray-500'>{location} • {time}</p>
                    </div>
                  </div>
                  <span
                    className={`px-3 py-1 text-xs rounded-full font-medium ${
                      severity === 'high'
                        ? 'bg-red-100 text-red-600'
                        : severity === 'medium'
                        ? 'bg-yellow-100 text-yellow-600'
                        : 'bg-green-100 text-green-600'
                    }`}
                  >
                    {severity}
                  </span>
                </div>
              ))}
            </div>
            <button className='w-full text-center text-blue-500 hover:text-blue-600 mt-4 transition-all duration-200 text-sm font-medium'>
              View All Incidents
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Component to be rendered when we add WebSocket support to check server connection
const ConnectionStatus = ({ connected }) => (
  <div className={`fixed bottom-4 right-4 z-40 px-3 py-1.5 rounded-full flex items-center gap-1 text-sm ${
    connected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
  }`}>
    <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`}></span>
    {connected ? 'Connected' : 'Disconnected'}
  </div>
);

export default Home;