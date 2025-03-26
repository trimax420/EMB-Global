import React, { useState, useEffect } from 'react';
import { alertService } from '../services/api';

const AlertsPage = () => {
  const [activeTab, setActiveTab] = useState('recent');
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [incidents, setIncidents] = useState([]);
  const [mallStructure, setMallStructure] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [alertsData, mallData] = await Promise.all([
          alertService.getAlerts(activeTab),
          alertService.getMallStructure()
        ]);
        setIncidents(alertsData);
        setMallStructure(mallData);
      } catch (err) {
        setError(err.message);
        console.error('Error fetching alerts:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [activeTab]);

  // Close video modal
  const closeVideoModal = () => {
    setSelectedVideo(null);
  };

  if (loading) {
    return <div className="p-6">Loading...</div>;
  }

  if (error) {
    return <div className="p-6 text-red-500">Error: {error}</div>;
  }

  return (
    <div className="p-6">
      {/* Header Section */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800">Incident Alerts</h1>
      </div>

      {/* Tabs */}
      <div className="flex space-x-4 mb-6">
        <button
          onClick={() => setActiveTab('recent')}
          className={`px-4 py-2 rounded ${
            activeTab === 'recent' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'
          }`}
        >
          Recent
        </button>
        <button
          onClick={() => setActiveTab('all')}
          className={`px-4 py-2 rounded ${
            activeTab === 'all' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'
          }`}
        >
          All Time
        </button>
        <button
          onClick={() => setActiveTab('heatmap')}
          className={`px-4 py-2 rounded ${
            activeTab === 'heatmap' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'
          }`}
        >
          Heatmap
        </button>
      </div>

      {/* Content Based on Active Tab */}
      {activeTab === 'heatmap' ? (
        <div>
          <h2 className="text-xl font-semibold mb-4">Mall Crowd Heatmap</h2>
          <div className="relative rounded-xl overflow-hidden">
            {mallStructure.map((zone) => (
              <div
                key={zone.id}
                className="absolute"
                style={{
                  left: `${zone.bounds[0][1]}%`,
                  top: `${zone.bounds[0][0]}%`,
                  width: `${zone.bounds[1][1] - zone.bounds[0][1]}%`,
                  height: `${zone.bounds[1][0] - zone.bounds[0][0]}%`,
                  backgroundColor: zone.fillColor,
                  opacity: 0.7,
                }}
                onClick={() => alert(`Zone: ${zone.name}, Crowd Density: ${zone.crowdDensity}%`)}
              >
                <div className="p-2 text-white text-sm">{zone.name}</div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div>
          <h2 className="text-xl font-semibold mb-4">
            {activeTab === 'recent' ? 'Recent Incidents' : 'All Time Incidents'}
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white border border-gray-300">
              <thead>
                <tr className="bg-gray-100">
                  <th className="py-2 px-4 border-b">Image</th>
                  <th className="py-2 px-4 border-b">Timestamp</th>
                  <th className="py-2 px-4 border-b">Location</th>
                  <th className="py-2 px-4 border-b">Type</th>
                  <th className="py-2 px-4 border-b">Description</th>
                </tr>
              </thead>
              <tbody>
                {incidents.length > 0 ? (
                  incidents.map((incident) => (
                    <tr key={incident.id} className="hover:bg-gray-50">
                      <td className="py-2 px-4 border-b">
                        <img
                          src={incident.image}
                          alt="Incident"
                          className="w-12 h-12 object-cover cursor-pointer rounded"
                          onClick={() => setSelectedVideo(incident.videoUrl)}
                        />
                      </td>
                      <td className="py-2 px-4 border-b">
                        {new Date(incident.timestamp).toLocaleString()}
                      </td>
                      <td className="py-2 px-4 border-b">{incident.location}</td>
                      <td className="py-2 px-4 border-b">{incident.type}</td>
                      <td className="py-2 px-4 border-b">{incident.description}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="5" className="py-4 text-center">
                      No incidents to display
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Video Modal */}
      {selectedVideo && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg w-3/4 max-w-2xl relative">
            <button
              onClick={closeVideoModal}
              className="absolute top-2 right-2 text-gray-600 hover:text-gray-800"
            >
              &times;
            </button>
            <iframe
              width="100%"
              height="400"
              src={selectedVideo}
              title="Incident Video"
              frameBorder="0"
              allowFullScreen
            ></iframe>
          </div>
        </div>
      )}
    </div>
  );
};

export default AlertsPage;
