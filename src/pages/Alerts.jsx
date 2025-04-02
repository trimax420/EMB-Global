import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Rectangle, Popup, Circle } from 'react-leaflet';
import { FaExclamationTriangle, FaBell, FaHistory, FaMapMarkedAlt, FaVideo, FaUserClock, FaUserNinja, FaListAlt } from 'react-icons/fa';
import { ImNotification } from 'react-icons/im';
import { FiFilter } from 'react-icons/fi';
import { TbActivityHeartbeat } from 'react-icons/tb';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  BarElement
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// Dummy incident data with additional loitering and theft incidents
const dummyIncidents = [
  {
    id: 1,
    timestamp: '2025-03-31T10:15:00',
    location: 'Entrance A',
    type: 'Customer Conflict',
    severity: 'medium',
    description: 'Argument between two customers over parking space.',
    image: 'https://c.ndtvimg.com/2024-02/9aqh0su_ghaziabad-fight_625x300_26_February_24.jpg?im=FeatureCrop,algorithm=dnn,width=545,height=307',
    videoUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
    isNew: false,
    coordinates: [40.748817, -73.979428]
  },
  {
    id: 2,
    timestamp: '2025-03-31T12:30:00',
    location: 'Food Court',
    type: 'Staff Misconduct',
    severity: 'low',
    description: 'Staff member rude to customer during busy period.',
    image: 'https://i.dailymail.co.uk/i/pix/2013/04/12/article-2307865-193FCF86000005DC-541_634x419.jpg',
    videoUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
    isNew: false,
    coordinates: [40.738817, -73.969428]
  },
  {
    id: 3,
    timestamp: '2025-03-30T15:45:00',
    location: 'Restroom B',
    type: 'Vandalism',
    severity: 'medium',
    description: 'Graffiti found on restroom walls requiring cleanup.',
    image: 'https://www.icecleaning.co.uk/media/images/uploaded/inline/01-Graffiti-in-bathrooms.1613384066.jpg',
    videoUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
    isNew: false,
    coordinates: [40.728817, -73.959428]
  },
  {
    id: 4,
    timestamp: new Date().toISOString(),
    location: 'Main Hall',
    type: 'Suspicious Activity',
    severity: 'medium',
    description: 'Person acting suspiciously near the main entrance.',
    image: 'https://as2.ftcdn.net/v2/jpg/09/33/19/41/1000_F_933194117_nog0ll5439LvRIxEPZrvwgfOK3vNRcON.jpg',
    videoUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
    isNew: true,
    coordinates: [40.718817, -73.949428]
  },
  {
    id: 5,
    timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(), // 30 minutes ago
    location: 'Electronics Department',
    type: 'Theft',
    severity: 'high',
    description: 'Individual caught putting merchandise in bag without paying.',
    image: 'https://qph.cf2.quoracdn.net/main-qimg-b6fa006d2e3a5ce54f95c24cee8541bd-lq',
    videoUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
    isNew: true,
    coordinates: [40.743817, -73.975428]
  },
  {
    id: 6,
    timestamp: new Date(Date.now() - 1000 * 60 * 45).toISOString(), // 45 minutes ago
    location: 'Parking Lot Corner',
    type: 'Loitering',
    severity: 'medium',
    description: 'Group of teenagers loitering in parking area for over 60 minutes.',
    image: 'https://s.hdnux.com/photos/01/34/01/61/24096458/3/1200x0.jpg',
    videoUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
    isNew: true,
    coordinates: [40.723817, -73.950428]
  },
  {
    id: 7,
    timestamp: new Date(Date.now() - 1000 * 60 * 120).toISOString(), // 2 hours ago
    location: 'Storage Area',
    type: 'Theft',
    severity: 'high',
    description: 'Employee detected removing inventory without authorization.',
    image: 'https://c8.alamy.com/comp/D7HHAT/employee-theft-concept-with-words-and-documents-D7HHAT.jpg',
    videoUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
    isNew: true,
    coordinates: [40.733817, -73.960428]
  },
  {
    id: 8,
    timestamp: new Date(Date.now() - 1000 * 60 * 180).toISOString(), // 3 hours ago
    location: 'Front Entrance',
    type: 'Loitering',
    severity: 'low',
    description: 'Person loitering outside entrance for 45+ minutes.',
    image: 'https://d207ibygpg2z1x.cloudfront.net/image/upload/v1571085554/articles_upload/content/zjnlakox6mtjbkxly1o5.jpg',
    videoUrl: 'https://www.youtube.com/embed/dQw4w9WgXcQ',
    isNew: false,
    coordinates: [40.748817, -73.979428]
  }
];

// JSON-based location structure for heatmap
const locationStructure = [
  { id: 1, name: 'Entrance A', crowdDensity: 85, bounds: [[40.749817, -73.986428], [40.747817, -73.974428]], fillColor: '#FF4D4D', incidents: 2 },
  { id: 2, name: 'Entrance B', crowdDensity: 60, bounds: [[40.749817, -73.974428], [40.737817, -73.976428]], fillColor: '#FFC107', incidents: 1 },
  { id: 3, name: 'Food Court', crowdDensity: 95, bounds: [[40.739817, -73.974428], [40.737817, -73.964428]], fillColor: '#FF4D4D', incidents: 1 },
  { id: 4, name: 'Restroom A', crowdDensity: 40, bounds: [[40.739817, -73.964428], [40.727817, -73.966428]], fillColor: '#28A745', incidents: 0 },
  { id: 5, name: 'Restroom B', crowdDensity: 70, bounds: [[40.729817, -73.964428], [40.727817, -73.954428]], fillColor: '#FFC107', incidents: 1 },
  { id: 6, name: 'Parking Lot', crowdDensity: 50, bounds: [[40.729817, -73.954428], [40.717817, -73.956428]], fillColor: '#28A745', incidents: 1 },
  { id: 7, name: 'Main Hall', crowdDensity: 80, bounds: [[40.719817, -73.954428], [40.717817, -73.944428]], fillColor: '#FF4D4D', incidents: 1 },
  { id: 8, name: 'Electronics Department', crowdDensity: 75, bounds: [[40.744817, -73.976428], [40.742817, -73.974428]], fillColor: '#FFC107', incidents: 1 },
  { id: 9, name: 'Storage Area', crowdDensity: 30, bounds: [[40.734817, -73.961428], [40.732817, -73.959428]], fillColor: '#28A745', incidents: 1 }
];

// Incident trend data
const incidentTrendData = {
  labels: ['6 AM', '8 AM', '10 AM', '12 PM', '2 PM', '4 PM', '6 PM', '8 PM', '10 PM'],
  datasets: [
    {
      label: 'All Incidents',
      data: [1, 3, 4, 7, 5, 8, 10, 6, 3],
      borderColor: 'rgb(59, 130, 246)',
      backgroundColor: 'rgba(59, 130, 246, 0.5)',
      tension: 0.3,
    },
    {
      label: 'Theft',
      data: [0, 1, 1, 2, 1, 3, 4, 2, 1],
      borderColor: 'rgb(220, 38, 38)',
      backgroundColor: 'rgba(220, 38, 38, 0.5)',
      tension: 0.3,
    },
    {
      label: 'Loitering',
      data: [1, 2, 1, 3, 2, 3, 2, 1, 1],
      borderColor: 'rgb(245, 158, 11)',
      backgroundColor: 'rgba(245, 158, 11, 0.5)',
      tension: 0.3,
    }
  ]
};

// Chart options
const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      position: 'top',
    },
    title: {
      display: true,
      text: 'Incident Trend (Today)',
    },
  },
  scales: {
    y: {
      beginAtZero: true,
      title: {
        display: true,
        text: 'Number of Incidents'
      }
    }
  }
};

const AlertsPage = () => {
  const [activeTab, setActiveTab] = useState('recent');
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [typeFilter, setTypeFilter] = useState('all');
  const [severityFilter, setSeverityFilter] = useState('all');
  const [showFilters, setShowFilters] = useState(false);

  // Filter incidents based on all active filters
  const getFilteredIncidents = () => {
    const now = new Date();
    const recentThreshold = new Date(now.getTime() - 24 * 60 * 60 * 1000); // Last 24 hours
    
    let filtered = [...dummyIncidents];
    
    // Apply time filter
    if (activeTab === 'recent') {
      filtered = filtered.filter((incident) => new Date(incident.timestamp) >= recentThreshold);
    }
    
    // Apply type filter
    if (typeFilter !== 'all') {
      filtered = filtered.filter((incident) => incident.type === typeFilter);
    }
    
    // Apply severity filter
    if (severityFilter !== 'all') {
      filtered = filtered.filter((incident) => incident.severity === severityFilter);
    }
    
    // Sort by timestamp (newest first)
    return filtered.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  };

  const filteredIncidents = getFilteredIncidents();

  // Close video modal
  const closeVideoModal = () => {
    setSelectedVideo(null);
  };

  // Close incident detail modal
  const closeIncidentDetail = () => {
    setSelectedIncident(null);
  };

  // View incident details
  const viewIncidentDetails = (incident) => {
    setSelectedIncident(incident);
  };

  // Count incidents by type
  const incidentCounts = {
    total: dummyIncidents.length,
    theft: dummyIncidents.filter(i => i.type === 'Theft').length,
    loitering: dummyIncidents.filter(i => i.type === 'Loitering').length,
    other: dummyIncidents.filter(i => i.type !== 'Theft' && i.type !== 'Loitering').length,
    high: dummyIncidents.filter(i => i.severity === 'high').length,
    new: dummyIncidents.filter(i => i.isNew).length
  };

  // Toggle filters display
  const toggleFilters = () => {
    setShowFilters(!showFilters);
  };

  return (
    <div className="p-4 md:p-6 bg-gray-50 min-h-screen">
      {/* Header Section */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Security Alerts</h1>
          <p className="text-gray-500">Monitoring suspicious activities and incidents</p>
        </div>
        
        <div className="flex gap-2 self-end md:self-auto">
          <button 
            className="px-4 py-2 flex items-center gap-1 border border-gray-300 rounded-lg hover:bg-gray-100 transition-all"
            onClick={toggleFilters}
          >
            <FiFilter /> Filters
          </button>
          <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-all duration-200 shadow-sm flex items-center gap-1">
            <FaHistory className="text-sm" /> Export Report
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      
<div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
  <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
    <div className="flex justify-between items-start">
      <div className="text-xl p-2 bg-blue-50 text-blue-500 rounded-lg">
        <ImNotification />
      </div>
      <span className="text-xs font-medium bg-blue-100 text-blue-600 px-2 py-1 rounded-full">All</span>
    </div>
    <p className="text-sm text-gray-500 mt-2">Total Alerts</p>
    <p className="text-2xl font-bold">{incidentCounts.total}</p>
  </div>
  
  <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
    <div className="flex justify-between items-start">
      <div className="text-xl p-2 bg-red-50 text-red-500 rounded-lg">
        <FaUserNinja />
      </div>
      <span className="text-xs font-medium bg-red-100 text-red-600 px-2 py-1 rounded-full">Critical</span>
    </div>
    <p className="text-sm text-gray-500 mt-2">Theft Alerts</p>
    <p className="text-2xl font-bold">{incidentCounts.theft}</p>
  </div>
  
  <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
    <div className="flex justify-between items-start">
      <div className="text-xl p-2 bg-yellow-50 text-yellow-500 rounded-lg">
        <FaUserClock />
      </div>
      <span className="text-xs font-medium bg-yellow-100 text-yellow-600 px-2 py-1 rounded-full">Warning</span>
    </div>
    <p className="text-sm text-gray-500 mt-2">Loitering</p>
    <p className="text-2xl font-bold">{incidentCounts.loitering}</p>
  </div>

        
        {/* <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
          <div className="flex justify-between items-start">
            <div className="text-xl p-2 bg-purple-50 text-purple-500 rounded-lg">
              <FaBell />
            </div>
          </div>
          <p className="text-sm text-gray-500 mt-2">Other Incidents</p>
          <p className="text-2xl font-bold">{incidentCounts.other}</p>
        </div>
        
        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
          <div className="flex justify-between items-start">
            <div className="text-xl p-2 bg-orange-50 text-orange-500 rounded-lg">
              <FaExclamationTriangle />
            </div>
          </div>
          <p className="text-sm text-gray-500 mt-2">High Priority</p>
          <p className="text-2xl font-bold">{incidentCounts.high}</p>
        </div>
        
        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100">
          <div className="flex justify-between items-start">
            <div className="text-xl p-2 bg-green-50 text-green-500 rounded-lg">
              <TbActivityHeartbeat />
            </div>
            <span className={`text-xs font-medium bg-green-100 text-green-600 px-2 py-1 rounded-full ${incidentCounts.new > 0 ? 'animate-pulse' : ''}`}>New</span>
          </div>
          <p className="text-sm text-gray-500 mt-2">New Alerts</p>
          <p className="text-2xl font-bold">{incidentCounts.new}</p>
        </div> */}
      </div>
      
      {/* Filter Controls - Conditional */}
      {showFilters && (
        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-100 mb-6">
          <h2 className="text-lg font-semibold mb-3">Filter Alerts</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Incident Type</label>
              <select 
                className="w-full border border-gray-300 rounded-lg p-2"
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
              >
                <option value="all">All Types</option>
                <option value="Theft">Theft</option>
                <option value="Loitering">Loitering</option>
                <option value="Customer Conflict">Customer Conflict</option>
                <option value="Staff Misconduct">Staff Misconduct</option>
                <option value="Vandalism">Vandalism</option>
                <option value="Suspicious Activity">Suspicious Activity</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Severity</label>
              <select 
                className="w-full border border-gray-300 rounded-lg p-2"
                value={severityFilter}
                onChange={(e) => setSeverityFilter(e.target.value)}
              >
                <option value="all">All Severities</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Time Period</label>
              <div className="flex rounded-lg overflow-hidden border border-gray-300">
                <button 
                  className={`flex-1 py-2 ${activeTab === 'recent' ? 'bg-blue-500 text-white' : 'bg-white text-gray-700'}`}
                  onClick={() => setActiveTab('recent')}
                >
                  Recent (24h)
                </button>
                <button 
                  className={`flex-1 py-2 ${activeTab === 'all' ? 'bg-blue-500 text-white' : 'bg-white text-gray-700'}`}
                  onClick={() => setActiveTab('all')}
                >
                  All Time
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex flex-wrap gap-2 mb-6">
        <button
          onClick={() => setActiveTab('recent')}
          className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
            activeTab === 'recent' ? 'bg-blue-500 text-white shadow-sm' : 'bg-white text-gray-700 border border-gray-200'
          }`}
        >
          <FaListAlt /> Recent Alerts
        </button>
        <button
          onClick={() => setActiveTab('all')}
          className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
            activeTab === 'all' ? 'bg-blue-500 text-white shadow-sm' : 'bg-white text-gray-700 border border-gray-200'
          }`}
        >
          <FaHistory /> All Alerts
        </button>
        <button
          onClick={() => setActiveTab('trends')}
          className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
            activeTab === 'trends' ? 'bg-blue-500 text-white shadow-sm' : 'bg-white text-gray-700 border border-gray-200'
          }`}
        >
          <TbActivityHeartbeat /> Incident Trends
        </button>
        <button
          onClick={() => setActiveTab('heatmap')}
          className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
            activeTab === 'heatmap' ? 'bg-blue-500 text-white shadow-sm' : 'bg-white text-gray-700 border border-gray-200'
          }`}
        >
          <FaMapMarkedAlt /> Location Heatmap
        </button>
      </div>

      {/* Content Based on Active Tab */}
      {activeTab === 'heatmap' ? (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
        <h2 className="text-xl font-semibold mb-4">Security Incident Map</h2>
        <div className="relative" style={{ height: '600px', width: '100%' }}>
          <MapContainer 
            center={[40.738817, -73.965428]} 
            zoom={14} 
            style={{ 
              height: '100%', 
              width: '100%', 
              position: 'absolute',
              top: 0,
              left: 0,
              borderRadius: '0.5rem',
              overflow: 'hidden'
            }}
          >
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            
            {/* Rest of your map content remains exactly the same */}
            {locationStructure.map((zone) => (
              <Rectangle
                key={zone.id}
                bounds={zone.bounds}
                pathOptions={{ 
                  color: zone.fillColor,
                  weight: 1,
                  fillOpacity: 0.4
                }}
              >
                <Popup>
                  <div className="p-1">
                    <h3 className="font-semibold">{zone.name}</h3>
                    <p className="text-sm">Crowd Density: {zone.crowdDensity}%</p>
                    <p className="text-sm">Incidents: {zone.incidents}</p>
                  </div>
                </Popup>
              </Rectangle>
            ))}
            
            {dummyIncidents.map((incident) => (
              <Circle
                key={incident.id}
                center={incident.coordinates}
                radius={50}
                pathOptions={{
                  color: incident.severity === 'high' ? '#DC2626' : 
                         incident.severity === 'medium' ? '#F59E0B' : '#10B981',
                  fillColor: incident.severity === 'high' ? '#DC2626' : 
                            incident.severity === 'medium' ? '#F59E0B' : '#10B981',
                  fillOpacity: 0.7
                }}
              >
                <Popup>
                  <div className="p-1">
                    <h3 className="font-semibold">{incident.type}</h3>
                    <p className="text-sm">{incident.location}</p>
                    <p className="text-sm">{incident.description}</p>
                    <p className="text-xs text-gray-500">
                      {new Date(incident.timestamp).toLocaleString()}
                    </p>
                    <button 
                      className="mt-2 px-2 py-1 bg-blue-500 text-white text-xs rounded"
                      onClick={() => viewIncidentDetails(incident)}
                    >
                      View Details
                    </button>
                  </div>
                </Popup>
              </Circle>
            ))}
          </MapContainer>
        </div>
      </div>
      ) : activeTab === 'trends' ? (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
          <h2 className="text-xl font-semibold mb-4">Incident Trends</h2>
          <div className="h-[400px] w-full mb-4">
            <Line data={incidentTrendData} options={chartOptions} />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
            <div className="p-4 border border-gray-200 rounded-lg">
              <h3 className="font-semibold text-red-600 flex items-center gap-2">
                <FaUserNinja /> Theft Incidents
              </h3>
              <p className="text-3xl font-bold mt-2">{incidentCounts.theft}</p>
              <p className="text-sm text-gray-500 mt-1">+2 from last week</p>
            </div>
            
            <div className="p-4 border border-gray-200 rounded-lg">
              <h3 className="font-semibold text-yellow-600 flex items-center gap-2">
                <FaUserClock /> Loitering Incidents
              </h3>
              <p className="text-3xl font-bold mt-2">{incidentCounts.loitering}</p>
              <p className="text-sm text-gray-500 mt-1">-1 from last week</p>
            </div>
            
            <div className="p-4 border border-gray-200 rounded-lg">
              <h3 className="font-semibold text-blue-600 flex items-center gap-2">
                <ImNotification /> Peak Alert Time
              </h3>
              <p className="text-3xl font-bold mt-2">6:00 PM</p>
              <p className="text-sm text-gray-500 mt-1">High traffic period</p>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">
              {activeTab === 'recent' ? 'Recent Alerts' : 'All Alerts'}
            </h2>
            <span className="text-sm text-gray-500">
              Showing {filteredIncidents.length} of {dummyIncidents.length} alerts
            </span>
          </div>
          
          {/* Card-based incident list for better mobile experience */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredIncidents.length > 0 ? (
              filteredIncidents.map((incident) => (
                <div 
                  key={incident.id} 
                  className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-all duration-200"
                >
                  <div className="relative">
                    <img
                      src={incident.image}
                      alt={incident.type}
                      className="w-full h-40 object-cover cursor-pointer"
                      onClick={() => setSelectedVideo(incident.videoUrl)}
                    />
                    <div className="absolute top-2 right-2">
                      <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                        incident.severity === 'high' 
                          ? 'bg-red-100 text-red-600' 
                          : incident.severity === 'medium'
                          ? 'bg-yellow-100 text-yellow-600'
                          : 'bg-green-100 text-green-600'
                      }`}>
                        {incident.severity}
                      </span>
                    </div>
                    {incident.isNew && (
                      <div className="absolute top-2 left-2">
                        <span className="text-xs font-medium bg-blue-100 text-blue-600 px-2 py-1 rounded-full animate-pulse">
                          New
                        </span>
                      </div>
                    )}
                    {incident.type === 'Loitering' && (
                      <div className="absolute bottom-2 left-2 bg-black bg-opacity-70 text-white px-2 py-1 rounded flex items-center gap-1 text-xs">
                        <FaUserClock /> Loitering Detection
                      </div>
                    )}
                    {incident.type === 'Theft' && (
                      <div className="absolute bottom-2 left-2 bg-black bg-opacity-70 text-white px-2 py-1 rounded flex items-center gap-1 text-xs">
                        <FaUserNinja /> Theft Detection
                      </div>
                    )}
                  </div>
                  
                  <div className="p-4">
                    <div className="flex justify-between items-start mb-2">
                      <h3 className="font-semibold">{incident.type}</h3>
                      <span className="text-xs text-gray-500">
                        {new Date(incident.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700 mb-2">{incident.location}</p>
                    <p className="text-sm text-gray-600 mb-3 line-clamp-2">{incident.description}</p>
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-gray-500">
                        {new Date(incident.timestamp).toLocaleDateString()}
                      </span>
                      <button 
                        className="px-3 py-1 bg-blue-500 text-white text-xs rounded-lg hover:bg-blue-600 transition-all"
                        onClick={() => viewIncidentDetails(incident)}
                      >
                        View Details
                      </button>
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="col-span-3 py-8 text-center">
                <FaBell className="mx-auto text-4xl text-gray-300 mb-3" />
                <p className="text-gray-500">No incidents match your current filters</p>
                <button 
                  className="mt-2 px-4 py-2 bg-blue-500 text-white text-sm rounded-lg"
                  onClick={() => {
                    setTypeFilter('all');
                    setSeverityFilter('all');
                  }}
                >
                  Reset Filters
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Video Modal */}
      {selectedVideo && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
          <div className="bg-white p-6 rounded-lg w-full max-w-3xl relative">
            <button 
              onClick={closeVideoModal} 
              className="absolute top-3 right-3 w-8 h-8 flex items-center justify-center text-gray-600 hover:text-gray-800 bg-gray-100 hover:bg-gray-200 rounded-full transition-all"
            >
              ✕
            </button>
            <h2 className="text-xl font-semibold mb-4">Incident Footage</h2>
            <div className="relative pt-[56.25%]"> {/* 16:9 aspect ratio */}
              <iframe
                className="absolute top-0 left-0 w-full h-full rounded-lg"
                src={selectedVideo}
                title="Incident Video"
                frameBorder="0"
                allowFullScreen
              ></iframe>
            </div>
            <div className="mt-4 text-sm text-gray-500">
              <p>Video evidence related to the security incident. Reviewing this footage may help identify individuals involved and understand the situation better.</p>
            </div>
          </div>
        </div>
      )}

      {/* Incident Detail Modal */}
      {selectedIncident && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
          <div className="bg-white p-6 rounded-lg w-full max-w-3xl relative max-h-[90vh] overflow-y-auto">
            <button 
              onClick={closeIncidentDetail} 
              className="absolute top-3 right-3 w-8 h-8 flex items-center justify-center text-gray-600 hover:text-gray-800 bg-gray-100 hover:bg-gray-200 rounded-full transition-all"
            >
              ✕
            </button>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h2 className="text-xl font-semibold mb-1">{selectedIncident.type}</h2>
                <p className="text-gray-500 mb-4">ID: #{selectedIncident.id}</p>
                
                <div className="mb-4">
                  <img
                    src={selectedIncident.image}
                    alt={selectedIncident.type}
                    className="w-full h-auto rounded-lg"
                  />
                  <button 
                    className="mt-2 w-full py-2 bg-blue-500 text-white rounded-lg flex items-center justify-center gap-2"
                    onClick={() => setSelectedVideo(selectedIncident.videoUrl)}
                  >
                    <FaVideo /> View Video Evidence
                  </button>
                </div>
              </div>
              
              <div>
                <div className="mb-4">
                  <h3 className="text-sm font-medium text-gray-500">Incident Location</h3>
                  <p className="text-lg">{selectedIncident.location}</p>
                </div>
                
                <div className="mb-4">
                  <h3 className="text-sm font-medium text-gray-500">Timestamp</h3>
                  <p className="text-lg">{new Date(selectedIncident.timestamp).toLocaleString()}</p>
                </div>
                
                <div className="mb-4">
                  <h3 className="text-sm font-medium text-gray-500">Severity</h3>
                  <span className={`inline-block px-3 py-1 text-sm rounded-full ${
                    selectedIncident.severity === 'high' 
                      ? 'bg-red-100 text-red-600' 
                      : selectedIncident.severity === 'medium'
                      ? 'bg-yellow-100 text-yellow-600'
                      : 'bg-green-100 text-green-600'
                  }`}>
                    {selectedIncident.severity.charAt(0).toUpperCase() + selectedIncident.severity.slice(1)}
                  </span>
                </div>
                
                <div className="mb-4">
                  <h3 className="text-sm font-medium text-gray-500">Description</h3>
                  <p className="text-base">{selectedIncident.description}</p>
                </div>
                
                <div className="border-t border-gray-200 pt-4">
                  <h3 className="text-sm font-medium text-gray-500 mb-2">Actions</h3>
                  <div className="flex flex-wrap gap-2">
                    <button className="px-4 py-2 bg-blue-100 text-blue-600 rounded-lg hover:bg-blue-200 transition-all">
                      Create Report
                    </button>
                    <button className="px-4 py-2 bg-green-100 text-green-600 rounded-lg hover:bg-green-200 transition-all">
                      Mark Resolved
                    </button>
                    <button className="px-4 py-2 bg-purple-100 text-purple-600 rounded-lg hover:bg-purple-200 transition-all">
                      Assign to Team
                    </button>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="mt-6 border-t border-gray-200 pt-4">
              <h3 className="text-lg font-semibold mb-3">Activity Log</h3>
              <div className="space-y-3">
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 rounded-full bg-blue-500 mt-2"></div>
                  <div>
                    <p className="text-sm font-medium">Alert Generated</p>
                    <p className="text-xs text-gray-500">{new Date(selectedIncident.timestamp).toLocaleString()}</p>
                  </div>
                </div>
                
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 rounded-full bg-blue-500 mt-2"></div>
                  <div>
                    <p className="text-sm font-medium">Security Team Notified</p>
                    <p className="text-xs text-gray-500">{new Date(new Date(selectedIncident.timestamp).getTime() + 60000).toLocaleString()}</p>
                  </div>
                </div>
                
                {selectedIncident.type === 'Theft' && (
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 rounded-full bg-red-500 mt-2"></div>
                    <div>
                      <p className="text-sm font-medium">AI Identified Potential Theft</p>
                      <p className="text-xs text-gray-500">{new Date(new Date(selectedIncident.timestamp).getTime() + 30000).toLocaleString()}</p>
                      <p className="text-xs text-gray-600 mt-1">Confidence level: 92%</p>
                    </div>
                  </div>
                )}
                
                {selectedIncident.type === 'Loitering' && (
                  <div className="flex items-start gap-3">
                    <div className="w-2 h-2 rounded-full bg-yellow-500 mt-2"></div>
                    <div>
                      <p className="text-sm font-medium">Person detected in area for extended period</p>
                      <p className="text-xs text-gray-500">{new Date(new Date(selectedIncident.timestamp).getTime() - 3600000).toLocaleString()}</p>
                      <p className="text-xs text-gray-600 mt-1">Duration: 60+ minutes</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AlertsPage;