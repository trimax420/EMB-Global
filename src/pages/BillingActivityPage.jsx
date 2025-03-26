import React, { useState, useEffect } from 'react';
import { getBillingActivity } from '../services/api';

const BillingActivityPage = () => {
  const [filter, setFilter] = useState('all');
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [billingData, setBillingData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchBillingData = async () => {
      try {
        setLoading(true);
        const data = await getBillingActivity(filter);
        setBillingData(data);
      } catch (err) {
        setError(err.message);
        console.error('Error fetching billing data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchBillingData();
  }, [filter]);

  // Close video modal
  const closeVideoModal = () => {
    setSelectedVideo(null);
  };

  if (loading) {
    return <div className="p-6">Loading billing data...</div>;
  }

  if (error) {
    return <div className="p-6 text-red-500">Error: {error}</div>;
  }

  return (
    <div className="p-6">
      {/* Header Section */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800">Billing Activity</h1>
      </div>

      {/* Filters Section */}
      <div className="mb-6">
        <label htmlFor="filter" className="mr-4 font-medium text-gray-700">
          Filter By:
        </label>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded"
        >
          <option value="all">All</option>
          <option value="suspicious">Suspicious Activity</option>
          <option value="skipped">Skipped Items</option>
        </select>
      </div>

      {/* Billing Data Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white border border-gray-300">
          <thead>
            <tr className="bg-gray-100">
              <th className="py-2 px-4 border-b">Transaction ID</th>
              <th className="py-2 px-4 border-b">Customer ID</th>
              <th className="py-2 px-4 border-b">Products</th>
              <th className="py-2 px-4 border-b">Total Amount</th>
              <th className="py-2 px-4 border-b">Status</th>
              <th className="py-2 px-4 border-b">Skipped Items</th>
              <th className="py-2 px-4 border-b">Suspicious</th>
            </tr>
          </thead>
          <tbody>
            {billingData.length > 0 ? (
              billingData.map((entry) => (
                <tr key={entry.id} className="hover:bg-gray-50">
                  <td className="py-2 px-4 border-b">{entry.transaction_id}</td>
                  <td className="py-2 px-4 border-b">{entry.customer_id}</td>
                  <td className="py-2 px-4 border-b">
                    {entry.products.map((product) => 
                      `${product.name} (${product.quantity})`
                    ).join(', ')}
                  </td>
                  <td className="py-2 px-4 border-b">${entry.total_amount}</td>
                  <td className="py-2 px-4 border-b">
                    <span
                      className={`inline-block px-2 py-1 rounded text-white ${
                        entry.status === 'completed'
                          ? 'bg-green-500'
                          : entry.status === 'suspicious'
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                      }`}
                    >
                      {entry.status}
                    </span>
                  </td>
                  <td className="py-2 px-4 border-b">
                    {entry.skipped_items?.length > 0 ? entry.skipped_items.join(', ') : '-'}
                  </td>
                  <td className="py-2 px-4 border-b">
                    <span
                      className={`inline-block px-2 py-1 rounded ${
                        entry.suspicious
                          ? 'bg-red-100 text-red-600'
                          : 'bg-green-100 text-green-600'
                      }`}
                    >
                      {entry.suspicious ? 'Yes' : 'No'}
                    </span>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="7" className="py-4 text-center">
                  No data available
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

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
              title="Transaction Video"
              frameBorder="0"
              allowFullScreen
            ></iframe>
          </div>
        </div>
      )}
    </div>
  );
};

export default BillingActivityPage;