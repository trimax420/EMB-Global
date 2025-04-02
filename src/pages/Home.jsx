import React, { useState, useEffect } from 'react';
import { CiCamera } from "react-icons/ci";
import { ImNotification } from "react-icons/im";
import { TbActivityHeartbeat } from "react-icons/tb";
import { FaUserGroup } from "react-icons/fa6";
import { FaExclamationTriangle, FaEye, FaExpand, FaHistory, FaTimes, FaDownload, FaFilter } from "react-icons/fa";
import { Bar, Line, Pie } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, Title, Tooltip, Legend } from 'chart.js';
import { useNavigate } from 'react-router-dom';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, Title, Tooltip, Legend);

// Sample data for different time filters
const timeFilterData = {
  today: {
    stats: [
      { icon: <CiCamera />, title: "Total Cameras", value: "5", description: "4 offline, 8 online", trend: "neutral" },
      { icon: <FaUserGroup />, title: "Active Detections", value: "25", description: "+15% from last hour", trend: "up" },
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
    stats: [
      { icon: <CiCamera />, title: "Total Cameras", value: "12", description: "2 offline, 10 online", trend: "up" },
      { icon: <FaUserGroup />, title: "Active Detections", value: "187", description: "+22% from last week", trend: "up" },
      { icon: <ImNotification />, title: "Current Alerts", value: "35", description: "+5 from yesterday", trend: "up" },
      { icon: <TbActivityHeartbeat />, title: "System Status", value: "Moderate", description: "Minor issues detected", trend: "down" },
    ],
    incidents: [
      { title: "Unauthorized Access", location: "Front Entrance", time: "Yesterday", severity: "high", isNew: false },
      { title: "Vehicle Stopped", location: "Parking Lot", time: "3 days ago", severity: "low", isNew: false },
      { title: "Person Detected", location: "Restricted Area", time: "Today", severity: "medium", isNew: true },
      { title: "Motion Detected", location: "Storage Room", time: "4 days ago", severity: "low", isNew: false },
      { title: "Multiple People", location: "Back Door", time: "2 days ago", severity: "high", isNew: false },
      { title: "Perimeter Breach", location: "East Fence", time: "Today", severity: "high", isNew: true },
    ],
    detectionTrendData: {
      labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
      datasets: [
        {
          label: 'People',
          data: [24, 33, 27, 42, 39, 18, 47],
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.5)',
          fill: false,
        }
      ],
    },
    incidentData: {
      labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
      datasets: [
        {
          label: 'Incidents This Week',
          data: [5, 7, 4, 8, 12, 6, 9],
          backgroundColor: 'rgba(59, 130, 246, 0.5)',
          borderColor: 'rgb(59, 130, 246)',
          borderWidth: 1,
        },
      ],
    },
  },
  month: {
    stats: [
      { icon: <CiCamera />, title: "Total Cameras", value: "15", description: "3 new installations", trend: "up" },
      { icon: <FaUserGroup />, title: "Active Detections", value: "843", description: "+18% month-over-month", trend: "up" },
      { icon: <ImNotification />, title: "Total Alerts", value: "156", description: "-5% from last month", trend: "down" },
      { icon: <TbActivityHeartbeat />, title: "System Status", value: "Excellent", description: "98.7% uptime", trend: "up" },
    ],
    incidents: [
      { title: "Unauthorized Access", location: "Multiple Locations", time: "This Month", severity: "high", isNew: false },
      { title: "Vehicle Incidents", location: "Parking Areas", time: "This Month", severity: "medium", isNew: false },
      { title: "Person Detected", location: "Restricted Zones", time: "This Month", severity: "medium", isNew: false },
      { title: "System Outages", location: "Network Infrastructure", time: "This Month", severity: "high", isNew: false },
      { title: "Maintenance Events", location: "System-wide", time: "Scheduled", severity: "low", isNew: false },
    ],
    detectionTrendData: {
      labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
      datasets: [
        {
          label: 'People',
          data: [145, 187, 203, 252],
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.5)',
          fill: false,
        }
      ],
    },
    incidentData: {
      labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
      datasets: [
        {
          label: 'Incidents This Month',
          data: [23, 35, 29, 42],
          backgroundColor: 'rgba(59, 130, 246, 0.5)',
          borderColor: 'rgb(59, 130, 246)',
          borderWidth: 1,
        },
      ],
    },
  }
};

const systemStatus = [
  { title: "Video Processing", status: "Operational", statusClass: "bg-green-100 text-green-500" },
  { title: "Object Detection", status: "Operational", statusClass: "bg-green-100 text-green-500" },
  { title: "Facial Recognition", status: "Degraded", statusClass: "bg-yellow-100 text-yellow-600" },
  { title: "License Plate Reader", status: "Offline", statusClass: "bg-red-100 text-red-500" },
];

// Camera data with video URLs
const cameras = [
  {
    id: 1,
    name: "Front Entrance",
    videoUrl: "https://developer-blogs.nvidia.com/wp-content/uploads/2022/12/Figure8-output_blurred-compressed.gif",
    details: { people: 4, vehicles: 1, alerts: 4, objects: 5 },
    status: "online"
  },
  {
    id: 2,
    name: "Parking Lot",
    videoUrl: "https://user-images.githubusercontent.com/11428131/139924111-58637f2e-f2f6-42d8-8812-ab42fece92b4.gif",
    details: { people: 2, vehicles: 3, alerts: 1, objects: 2 },
    status: "online"
  },
  {
    id: 3,
    name: "Restricted Area",
    videoUrl: "https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/gif-people-in-store-bounding-boxes.gif",
    details: { people: 0, vehicles: 0, alerts: 0, objects: 1 },
    status: "online"
  },
  {
    id: 4,
    name: "Storage Room",
    videoUrl: "https://user-images.githubusercontent.com/11428131/137016574-0d180d9b-fb9a-42a9-94b7-fbc0dbc18560.gif",
    details: { people: 1, vehicles: 0, alerts: 2, objects: 3 },
    status: "offline"
  },
];

function Home() {
  const [selectedCamera, setSelectedCamera] = useState(cameras[0]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [timeFilter, setTimeFilter] = useState('today');
  const [showReport, setShowReport] = useState(false);
  const navigate = useNavigate();

  // Get data based on current time filter
  const currentData = timeFilterData[timeFilter];

  // Handle camera selection
  const handleCameraSelect = (camera) => {
    setSelectedCamera(camera);
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

  // Generate period label based on time filter
  const getPeriodLabel = () => {
    const now = new Date();
    switch(timeFilter) {
      case 'today':
        return now.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
      case 'week':
        const weekStart = new Date(now);
        weekStart.setDate(now.getDate() - now.getDay());
        const weekEnd = new Date(weekStart);
        weekEnd.setDate(weekStart.getDate() + 6);
        return `${weekStart.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${weekEnd.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}`;
      case 'month':
        return now.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
      default:
        return '';
    }
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

      {/* Stats Section */}
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
        <div className={`${isFullscreen ? 'fixed inset-0 z-50 bg-black p-4' : 'lg:w-[70%]'} bg-white rounded-xl shadow-sm border border-gray-100 p-4 mb-6 lg:mb-0`}>
          <div className='flex justify-between items-center mb-4'>
            <h1 className='text-xl font-bold text-gray-800'>Live Security Feed</h1>
            <div className='flex gap-2'>
              <button 
                className={`px-3 py-1.5 ${selectedCamera.status === 'online' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'} rounded-lg text-sm font-medium`}
              >
                {selectedCamera.status === 'online' ? 'Live' : 'Offline'}
              </button>
              <button 
                className='p-2 text-gray-600 hover:text-blue-500 transition-all duration-200'
                onClick={toggleFullscreen}
              >
                <FaExpand />
              </button>
            </div>
          </div>
          <div className='relative rounded-xl overflow-hidden bg-gray-900'>
            <div className='aspect-video relative overflow-hidden'>
              <img
                src={selectedCamera.videoUrl}
                alt="Live Security Feed"
                className='w-full h-full object-cover'
              />
              {selectedCamera.status === 'offline' && (
                <div className='absolute inset-0 flex items-center justify-center bg-black bg-opacity-70'>
                  <div className='text-center'>
                    <FaExclamationTriangle className='text-4xl text-yellow-500 mx-auto mb-2' />
                    <p className='text-white font-bold'>Camera Offline</p>
                    <p className='text-gray-300 text-sm'>Connection lost</p>
                  </div>
                </div>
              )}
              <div className='absolute top-4 left-4 bg-black bg-opacity-50 text-white p-2 rounded-lg text-sm'>
                {new Date().toLocaleTimeString()}
              </div>
              <div className='absolute bottom-4 left-4 bg-black bg-opacity-70 text-white p-3 rounded-lg max-w-[70%]'>
                <h2 className='text-lg font-semibold flex items-center'>
                  {selectedCamera.name}
                  {selectedCamera.status === 'online' && (
                    <span className='ml-2 flex items-center text-xs bg-green-500 text-white px-2 py-0.5 rounded-full'>
                      <span className='w-2 h-2 bg-white rounded-full mr-1 animate-pulse'></span> LIVE
                    </span>
                  )}
                </h2>
                <div className='grid grid-cols-2 gap-x-6 gap-y-1 mt-2'>
                  <p className='flex items-center text-sm'><FaUserGroup className='mr-2 text-blue-400' /> People: {selectedCamera.details.people}</p>
                  <p className='flex items-center text-sm'><ImNotification className='mr-2 text-orange-400' /> Alerts: {selectedCamera.details.alerts}</p>
                  <p className='flex items-center text-sm'><CiCamera className='mr-2 text-green-400' /> Vehicles: {selectedCamera.details.vehicles}</p>
                  <p className='flex items-center text-sm'><TbActivityHeartbeat className='mr-2 text-purple-400' /> Objects: {selectedCamera.details.objects}</p>
                </div>
              </div>
            </div>
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
            <div className='grid grid-cols-2 md:grid-cols-4 gap-4'>
              {cameras.map((camera) => (
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
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Sidebar */}
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

          {/* System Status */}
          <div className='bg-white rounded-xl shadow-sm border border-gray-100 p-4'>
            <h1 className='text-lg font-bold text-gray-800 mb-1'>System Status</h1>
            <p className='text-gray-500 text-sm mb-4'>All systems operational</p>
            <div className='space-y-3'>
              {systemStatus.map(({ title, status, statusClass }, index) => (
                <div className='flex justify-between items-center p-2' key={index}>
                  <p className='text-sm text-gray-700 font-medium'>{title}</p>
                  <span className={`px-3 py-1 text-xs rounded-full font-medium ${statusClass}`}>
                    {status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Weekly Incident Report */}
      <div className='mt-6'>
        <div className='bg-white rounded-xl shadow-sm border border-gray-100 p-4'>
          <div className='flex justify-between items-center mb-4'>
            <div>
              <h1 className='text-lg font-bold text-gray-800'>{timeFilter === 'today' ? 'Daily' : timeFilter === 'week' ? 'Weekly' : 'Monthly'} Incident Report</h1>
              <p className='text-gray-500 text-sm'>{getPeriodLabel()}</p>
            </div>
            <button className='text-blue-500 hover:text-blue-600 transition-all duration-200 text-sm'>
              Export Data
            </button>
          </div>
          <div className='h-64'>
            <Bar data={currentData.incidentData} options={{
              responsive: true,
              plugins: {
                legend: {
                  position: 'top',
                },
                title: {
                  display: true,
                  text: `${timeFilter === 'today' ? 'Today\'s' : timeFilter === 'week' ? 'Weekly' : 'Monthly'} Incident Report`,
                },
              },
            }} />
          </div>
        </div>
      </div>

      {/* Report Modal */}
      {showReport && (
        <div className='fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4'>
          <div className='bg-white rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto'>
            <div className='p-6 border-b border-gray-200'>
              <div className='flex justify-between items-center'>
                <h1 className='text-2xl font-bold text-gray-800'>Security Report</h1>
                <button 
                  onClick={toggleReport}
                  className='text-gray-500 hover:text-gray-700'
                >
                  <FaTimes className='text-xl' />
                </button>
              </div>
            </div>
            <div className='p-6'>
              <div className='flex justify-between items-center mb-6'>
                <h2 className='text-xl font-semibold text-gray-700'>{timeFilter === 'today' ? 'Daily' : timeFilter === 'week' ? 'Weekly' : 'Monthly'} Report Summary</h2>
                <p className='text-gray-500'>{getPeriodLabel()}</p>
              </div>

              {/* Report Filters */}
              <div className='bg-gray-50 p-4 rounded-xl mb-6 flex flex-wrap gap-3 items-center'>
                <div className='font-medium text-gray-700 flex items-center'>
                  <FaFilter className='mr-2 text-blue-500' /> Filters:
                </div>
                <div className='flex flex-wrap gap-2'>
                  <button className='px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium flex items-center'>
                    All Cameras <FaTimes className='ml-2 text-xs' />
                  </button>
                  <button className='px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium flex items-center'>
                    High Priority <FaTimes className='ml-2 text-xs' />
                  </button>
                  <button className='px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm font-medium hover:bg-blue-50'>
                    + Add Filter
                  </button>
                </div>
              </div>

              {/* Stats Grid */}
              <div className='grid grid-cols-1 md:grid-cols-2 gap-4 mb-6'>
                <div className='bg-white border border-gray-200 rounded-xl p-5 shadow-sm'>
                  <h3 className='text-lg font-semibold text-gray-800 mb-2'>Incident Summary</h3>
                  <div className='grid grid-cols-2 gap-4'>
                    <div className='bg-gray-50 p-3 rounded-lg'>
                      <p className='text-sm text-gray-500'>Total Incidents</p>
                      <p className='text-3xl font-bold text-gray-900'>{timeFilter === 'today' ? '11' : timeFilter === 'week' ? '51' : '129'}</p>
                      <p className='text-xs text-green-500 mt-1'>
                        {timeFilter === 'today' ? '↓ 15% from yesterday' : timeFilter === 'week' ? '↓ 8% from last week' : '↓ 5% from last month'}
                      </p>
                    </div>
                    <div className='bg-gray-50 p-3 rounded-lg'>
                      <p className='text-sm text-gray-500'>High Priority</p>
                      <p className='text-3xl font-bold text-red-500'>{timeFilter === 'today' ? '3' : timeFilter === 'week' ? '12' : '35'}</p>
                      <p className='text-xs text-red-500 mt-1'>
                        {timeFilter === 'today' ? '↑ 2 more than yesterday' : timeFilter === 'week' ? '↑ 20% from last week' : '↑ 12% from last month'}
                      </p>
                    </div>
                    <div className='bg-gray-50 p-3 rounded-lg'>
                      <p className='text-sm text-gray-500'>Resolved</p>
                      <p className='text-3xl font-bold text-green-500'>{timeFilter === 'today' ? '8' : timeFilter === 'week' ? '39' : '94'}</p>
                      <p className='text-xs text-gray-500 mt-1'>
                        {timeFilter === 'today' ? '72% resolution rate' : timeFilter === 'week' ? '76% resolution rate' : '73% resolution rate'}
                      </p>
                    </div>
                    <div className='bg-gray-50 p-3 rounded-lg'>
                      <p className='text-sm text-gray-500'>Avg Response Time</p>
                      <p className='text-3xl font-bold text-blue-500'>{timeFilter === 'today' ? '5m' : timeFilter === 'week' ? '8m' : '11m'}</p>
                      <p className='text-xs text-green-500 mt-1'>
                        {timeFilter === 'today' ? '↓ 2m faster than target' : timeFilter === 'week' ? '↑ 1m above target' : '↑ 4m above target'}
                      </p>
                    </div>
                  </div>
                </div>
                <div className='bg-white border border-gray-200 rounded-xl p-5 shadow-sm'>
                  <h3 className='text-lg font-semibold text-gray-800 mb-3'>Incident Types</h3>
                  <div className='h-64'>
                    <Pie 
                      data={{
                        labels: ['Unauthorized Access', 'Motion Detection', 'Person Detected', 'Vehicle Detection', 'System Alert'],
                        datasets: [
                          {
                            data: timeFilter === 'today' ? [3, 4, 2, 1, 1] : timeFilter === 'week' ? [18, 14, 9, 6, 4] : [42, 35, 27, 15, 10],
                            backgroundColor: [
                              'rgba(220, 38, 38, 0.7)',  // Red
                              'rgba(59, 130, 246, 0.7)', // Blue
                              'rgba(245, 158, 11, 0.7)', // Amber
                              'rgba(16, 185, 129, 0.7)', // Green
                              'rgba(139, 92, 246, 0.7)'  // Purple
                            ],
                            borderWidth: 1,
                          },
                        ],
                      }}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Detection Trend Chart */}
              <div className='bg-white border border-gray-200 rounded-xl p-5 shadow-sm mb-6'>
                <h3 className='text-lg font-semibold text-gray-800 mb-2'>Detection Trends</h3>
                <p className='text-sm text-gray-500 mb-4'>Person detection frequency over time</p>
                <div className='h-64'>
                  <Line 
                    data={{
                      labels: timeFilter === 'today' 
                        ? ['6:00', '8:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00'] 
                        : timeFilter === 'week'
                        ? ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                        : ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                      datasets: [
                        {
                          label: 'People Detected',
                          data: timeFilter === 'today' 
                            ? [4, 8, 15, 12, 25, 22, 18, 10, 5] 
                            : timeFilter === 'week'
                            ? [42, 48, 37, 53, 61, 43, 38]
                            : [145, 187, 203, 252],
                          borderColor: 'rgb(59, 130, 246)',
                          backgroundColor: 'rgba(59, 130, 246, 0.1)',
                          fill: true,
                          tension: 0.4,
                        },
                        {
                          label: 'Alerts Generated',
                          data: timeFilter === 'today' 
                            ? [1, 2, 4, 3, 7, 5, 4, 2, 1] 
                            : timeFilter === 'week'
                            ? [11, 15, 9, 17, 21, 12, 10]
                            : [32, 45, 38, 51],
                          borderColor: 'rgb(239, 68, 68)',
                          backgroundColor: 'rgba(239, 68, 68, 0.1)',
                          fill: true,
                          tension: 0.4,
                        }
                      ],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      scales: {
                        y: {
                          beginAtZero: true
                        }
                      }
                    }}
                  />
                </div>
              </div>

              {/* Recent Incidents Table */}
              <div className='bg-white border border-gray-200 rounded-xl p-5 shadow-sm mb-6'>
                <div className='flex justify-between items-center mb-4'>
                  <h3 className='text-lg font-semibold text-gray-800'>Recent Incidents</h3>
                  <div className='flex gap-2'>
                    <button className='px-3 py-1.5 border border-gray-200 rounded-lg text-sm text-gray-700 hover:bg-gray-50'>
                      Filter
                    </button>
                    <button className='px-3 py-1.5 bg-blue-500 text-white rounded-lg text-sm hover:bg-blue-600 flex items-center gap-1'>
                      <FaDownload className='text-xs' /> Export
                    </button>
                  </div>
                </div>
                <div className='overflow-x-auto'>
                  <table className='min-w-full divide-y divide-gray-200'>
                    <thead>
                      <tr>
                        <th className='px-3 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>ID</th>
                        <th className='px-3 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>Type</th>
                        <th className='px-3 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>Location</th>
                        <th className='px-3 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>Time</th>
                        <th className='px-3 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>Severity</th>
                        <th className='px-3 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider'>Status</th>
                      </tr>
                    </thead>
                    <tbody className='bg-white divide-y divide-gray-200'>
                      {[
                        { id: 'INC-1089', type: 'Unauthorized Access', location: 'Front Entrance', time: '14:35', severity: 'high', status: 'Open' },
                        { id: 'INC-1088', type: 'Person Detected', location: 'Restricted Area', time: '13:22', severity: 'medium', status: 'In Progress' },
                        { id: 'INC-1087', type: 'Motion Detected', location: 'Storage Room', time: '12:15', severity: 'low', status: 'Resolved' },
                        { id: 'INC-1086', type: 'Vehicle Stopped', location: 'Parking Lot', time: '10:35', severity: 'low', status: 'Closed' },
                        { id: 'INC-1085', type: 'System Alert', location: 'Server Room', time: '09:12', severity: 'medium', status: 'Resolved' },
                      ].map((incident, index) => (
                        <tr key={index} className='hover:bg-gray-50'>
                          <td className='px-3 py-4 whitespace-nowrap text-sm font-medium text-blue-600'>{incident.id}</td>
                          <td className='px-3 py-4 whitespace-nowrap text-sm text-gray-700'>{incident.type}</td>
                          <td className='px-3 py-4 whitespace-nowrap text-sm text-gray-700'>{incident.location}</td>
                          <td className='px-3 py-4 whitespace-nowrap text-sm text-gray-700'>{incident.time}</td>
                          <td className='px-3 py-4 whitespace-nowrap'>
                            <span className={`px-2 py-1 text-xs rounded-full font-medium ${
                              incident.severity === 'high'
                                ? 'bg-red-100 text-red-600'
                                : incident.severity === 'medium'
                                ? 'bg-yellow-100 text-yellow-600'
                                : 'bg-green-100 text-green-600'
                            }`}>
                              {incident.severity}
                            </span>
                          </td>
                          <td className='px-3 py-4 whitespace-nowrap'>
                            <span className={`px-2 py-1 text-xs rounded-full font-medium ${
                              incident.status === 'Open'
                                ? 'bg-blue-100 text-blue-600'
                                : incident.status === 'In Progress'
                                ? 'bg-purple-100 text-purple-600'
                                : incident.status === 'Resolved'
                                ? 'bg-green-100 text-green-600'
                                : 'bg-gray-100 text-gray-600'
                            }`}>
                              {incident.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className='mt-4 flex justify-between items-center'>
                  <p className='text-sm text-gray-500'>Showing 5 of {timeFilter === 'today' ? '11' : timeFilter === 'week' ? '51' : '129'} incidents</p>
                  <div className='flex gap-2'>
                    <button className='px-3 py-1 border border-gray-200 rounded-lg text-sm'>Previous</button>
                    <button className='px-3 py-1 bg-blue-50 text-blue-600 border border-blue-200 rounded-lg text-sm'>1</button>
                    <button className='px-3 py-1 border border-gray-200 rounded-lg text-sm'>2</button>
                    <button className='px-3 py-1 border border-gray-200 rounded-lg text-sm'>Next</button>
                  </div>
                </div>
              </div>

              {/* Camera Summary */}
              <div className='bg-white border border-gray-200 rounded-xl p-5 shadow-sm'>
                <h3 className='text-lg font-semibold text-gray-800 mb-4'>Camera Performance Summary</h3>
                <div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4'>
                  {cameras.map(camera => (
                    <div key={camera.id} className='border border-gray-200 rounded-lg p-3 hover:shadow-sm transition-shadow'>
                      <div className='flex justify-between items-center mb-2'>
                        <h4 className='font-medium text-gray-800'>{camera.name}</h4>
                        <span className={`px-2 py-0.5 text-xs rounded-full ${
                          camera.status === 'online' ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'
                        }`}>
                          {camera.status}
                        </span>
                      </div>
                      <div className='aspect-video rounded overflow-hidden bg-gray-100 mb-2'>
                        <img src={camera.videoUrl} alt={camera.name} className='w-full h-full object-cover' />
                      </div>
                      <div className='grid grid-cols-2 gap-2 text-xs'>
                        <div className='flex items-center'>
                          <FaUserGroup className='mr-1 text-blue-500' /> 
                          <span>{camera.details.people} people</span>
                        </div>
                        <div className='flex items-center'>
                          <ImNotification className='mr-1 text-red-500' /> 
                          <span>{camera.details.alerts} alerts</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className='border-t border-gray-200 p-4 flex justify-between'>
              <button 
                className='px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50'
                onClick={toggleReport}
              >
                Close
              </button>
              <button className='px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600'>
                Download Full Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Home;