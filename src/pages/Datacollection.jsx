import React, { useState, useEffect } from 'react';
import * as XLSX from 'xlsx';
import { getCustomerData } from '../services/api';

const Datacollection = () => {
  const [filters, setFilters] = useState({ gender: 'all', date: '', timePeriod: 'all' });
  const [customerData, setCustomerData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Time period options
  const timePeriodOptions = [
    'All',
    'Morning (6 AM–12 PM)',
    'Afternoon (12 PM–6 PM)',
    'Evening (6 PM–9 PM)',
    'Night (9 PM–6 AM)'
  ];

  useEffect(() => {
    const fetchCustomerData = async () => {
      try {
        setLoading(true);
        const data = await getCustomerData(filters);
        setCustomerData(data);
      } catch (err) {
        setError(err.message);
        console.error('Error fetching customer data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchCustomerData();
  }, [filters]);

  // Export data as Excel
  const handleExport = () => {
    if (!customerData.length) return;

    const worksheet = XLSX.utils.json_to_sheet(customerData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Customer Data');
    XLSX.writeFile(workbook, 'customer_data.xlsx');
  };

  if (loading) {
    return <div className="p-6">Loading customer data...</div>;
  }

  if (error) {
    return <div className="p-6 text-red-500">Error: {error}</div>;
  }

  return (
    <div className="p-6">
      {/* Header Section */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800">Customer Entry Data Collection</h1>
        <button
          onClick={handleExport}
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
          disabled={!customerData.length}
        >
          Export Data
        </button>
      </div>

      {/* Filters Section */}
      <div className="flex space-x-4 mb-6">
        <select
          value={filters.gender}
          onChange={(e) => setFilters({ ...filters, gender: e.target.value })}
          className="px-4 py-2 border border-gray-300 rounded"
        >
          <option value="all">All</option>
          <option value="male">Male</option>
          <option value="female">Female</option>
        </select>

        <input
          type="date"
          value={filters.date}
          onChange={(e) => setFilters({ ...filters, date: e.target.value })}
          className="px-4 py-2 border border-gray-300 rounded"
        />

        <select
          value={filters.timePeriod}
          onChange={(e) => setFilters({ ...filters, timePeriod: e.target.value })}
          className="px-4 py-2 border border-gray-300 rounded"
        >
          {timePeriodOptions.map((period) => (
            <option key={period} value={period}>
              {period}
            </option>
          ))}
        </select>
      </div>

      {/* Table Section */}
      <div className="overflow-x-auto">
        <table className="min-w-full text-left bg-white border border-gray-300">
          <thead>
            <tr className="bg-gray-100">
              <th className="py-2 px-4 border-b">Image</th>
              <th className="py-2 px-4 border-b">Gender</th>
              <th className="py-2 px-4 border-b">Time</th>
              <th className="py-2 px-4 border-b">Date</th>
              <th className="py-2 px-4 border-b">Age Group</th>
              <th className="py-2 px-4 border-b">Clothing Color</th>
              <th className="py-2 px-4 border-b">Notes</th>
            </tr>
          </thead>
          <tbody>
            {customerData.length > 0 ? (
              customerData.map((entry) => (
                <tr key={entry.id} className="hover:bg-gray-50">
                  <td className="py-2 px-4 border-b">
                    <img
                      src={entry.image_url}
                      alt="Customer"
                      className="w-12 h-12 object-cover rounded"
                    />
                  </td>
                  <td className="py-2 px-4 border-b">{entry.gender}</td>
                  <td className="py-2 px-4 border-b">{entry.entry_time}</td>
                  <td className="py-2 px-4 border-b">{entry.entry_date}</td>
                  <td className="py-2 px-4 border-b">{entry.age_group}</td>
                  <td className="py-2 px-4 border-b">{entry.clothing_color}</td>
                  <td className="py-2 px-4 border-b">{entry.notes || '-'}</td>
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
    </div>
  );
};

export default Datacollection;
