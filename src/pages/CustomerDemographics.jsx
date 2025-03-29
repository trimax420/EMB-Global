import React, { useState, useEffect } from 'react';
import { customerService } from '../services/api';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title } from 'chart.js';
import { Pie, Bar } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(ArcElement, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const CustomerDemographics = () => {
  const [customerData, setCustomerData] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [filters, setFilters] = useState({
    gender: 'all',
    ageGroup: 'all',
    date: '',
    timePeriod: 'all',
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [demographicsSummary, setDemographicsSummary] = useState({
    gender: { male: 0, female: 0, unknown: 0 },
    age: { child: 0, young: 0, adult: 0, senior: 0 },
    colors: {},
    hourly: Array(24).fill(0),
  });

  // Fetch customer data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        
        // Fetch customer data and demographics summary in parallel
        const [data, summary] = await Promise.all([
          customerService.getCustomerData(),
          customerService.getDemographicsSummary()
        ]);
        
        setCustomerData(data);
        setFilteredData(data);
        
        // Set demographics summary from API response
        setDemographicsSummary({
          gender: summary.gender || { male: 0, female: 0, unknown: 0 },
          age: summary.age_groups || { child: 0, young: 0, adult: 0, senior: 0 },
          colors: summary.popular_colors || {},
          hourly: summary.hourly_distribution || Array(24).fill(0),
        });
        
        setIsLoading(false);
      } catch (err) {
        console.error('Error fetching customer data:', err);
        setError('Failed to load customer demographic data');
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  // Apply filters when they change
  useEffect(() => {
    applyFilters();
  }, [filters, customerData]);

  // Apply selected filters to customer data
  const applyFilters = () => {
    if (!customerData.length) return;
    
    let filtered = [...customerData];

    // Apply gender filter
    if (filters.gender !== 'all') {
      filtered = filtered.filter(customer => customer.gender === filters.gender);
    }

    // Apply age group filter
    if (filters.ageGroup !== 'all') {
      filtered = filtered.filter(customer => customer.age_group === filters.ageGroup);
    }

    // Apply date filter
    if (filters.date) {
      const selectedDate = new Date(filters.date).toISOString().split('T')[0];
      filtered = filtered.filter(customer => {
        const customerDate = new Date(customer.entry_time).toISOString().split('T')[0];
        return customerDate === selectedDate;
      });
    }

    // Apply time period filter
    if (filters.timePeriod !== 'all') {
      filtered = filtered.filter(customer => {
        const hour = new Date(customer.entry_time).getHours();
        if (filters.timePeriod === 'morning') return hour >= 6 && hour < 12;
        if (filters.timePeriod === 'afternoon') return hour >= 12 && hour < 18;
        if (filters.timePeriod === 'evening') return hour >= 18 && hour < 22;
        if (filters.timePeriod === 'night') return hour >= 22 || hour < 6;
        return true;
      });
    }

    setFilteredData(filtered);
  };

  // Handle filter changes
  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Reset all filters
  const resetFilters = () => {
    setFilters({
      gender: 'all',
      ageGroup: 'all',
      date: '',
      timePeriod: 'all',
    });
  };

  // Export data as CSV
  const exportData = async () => {
    try {
      const response = await customerService.exportDemographicsData(filters);
      if (response && response.download_url) {
        // Create a temporary anchor element to trigger download
        const link = document.createElement('a');
        link.href = response.download_url;
        link.setAttribute('download', response.filename);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } else {
        throw new Error('No download URL received');
      }
    } catch (err) {
      console.error('Error exporting data:', err);
      alert('Failed to export data');
    }
  };

  // Chart data for gender distribution
  const genderData = {
    labels: ['Male', 'Female', 'Unknown'],
    datasets: [
      {
        data: [
          demographicsSummary.gender.male,
          demographicsSummary.gender.female,
          demographicsSummary.gender.unknown
        ],
        backgroundColor: ['#3b82f6', '#ec4899', '#9ca3af'],
        borderWidth: 1,
      },
    ],
  };

  // Chart data for age distribution
  const ageData = {
    labels: ['Child', 'Young Adult', 'Adult', 'Senior'],
    datasets: [
      {
        data: [
          demographicsSummary.age.child,
          demographicsSummary.age.young,
          demographicsSummary.age.adult,
          demographicsSummary.age.senior
        ],
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',
          'rgba(54, 162, 235, 0.5)',
          'rgba(255, 206, 86, 0.5)',
          'rgba(75, 192, 192, 0.5)',
        ],
        borderWidth: 1,
      },
    ],
  };

  // Chart data for hourly distribution
  const hourlyData = {
    labels: Array.from({ length: 24 }, (_, i) => `${i}:00`),
    datasets: [
      {
        label: 'Hourly Customer Count',
        data: demographicsSummary.hourly,
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      },
    ],
  };

  // Get top 5 clothing colors
  const topColors = Object.entries(demographicsSummary.colors)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  // Chart data for clothing colors
  const colorData = {
    labels: topColors.map(([color]) => color.charAt(0).toUpperCase() + color.slice(1)),
    datasets: [
      {
        data: topColors.map(([_, count]) => count),
        backgroundColor: [
          '#ef4444', // red
          '#3b82f6', // blue
          '#10b981', // green
          '#f59e0b', // yellow
          '#8b5cf6', // purple
        ],
        borderWidth: 1,
      },
    ],
  };

  if (isLoading) {
    return (
      <div className="p-6 flex justify-center items-center min-h-screen">
        <div className="text-xl">Loading demographic data...</div>
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
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800">Customer Demographics Analysis</h1>
        <button
          onClick={exportData}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Export Data
        </button>
      </div>

      {/* Filters */}
      <div className="bg-white p-4 rounded-lg shadow mb-6">
        <h2 className="text-lg font-semibold mb-4">Filter Demographics</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Gender</label>
            <select
              name="gender"
              value={filters.gender}
              onChange={handleFilterChange}
              className="w-full p-2 border border-gray-300 rounded"
            >
              <option value="all">All Genders</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="unknown">Unknown</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Age Group</label>
            <select
              name="ageGroup"
              value={filters.ageGroup}
              onChange={handleFilterChange}
              className="w-full p-2 border border-gray-300 rounded"
            >
              <option value="all">All Age Groups</option>
              <option value="child">Child</option>
              <option value="young">Young Adult</option>
              <option value="adult">Adult</option>
              <option value="senior">Senior</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Date</label>
            <input
              type="date"
              name="date"
              value={filters.date}
              onChange={handleFilterChange}
              className="w-full p-2 border border-gray-300 rounded"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Time Period</label>
            <select
              name="timePeriod"
              value={filters.timePeriod}
              onChange={handleFilterChange}
              className="w-full p-2 border border-gray-300 rounded"
            >
              <option value="all">All Day</option>
              <option value="morning">Morning (6AM-12PM)</option>
              <option value="afternoon">Afternoon (12PM-6PM)</option>
              <option value="evening">Evening (6PM-10PM)</option>
              <option value="night">Night (10PM-6AM)</option>
            </select>
          </div>
        </div>

        <div className="mt-4 flex justify-end">
          <button
            onClick={resetFilters}
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
          >
            Reset Filters
          </button>
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-2">Total Customers</h2>
          <p className="text-3xl font-bold">{filteredData.length}</p>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-2">Gender Ratio</h2>
          <p className="text-sm">
            <span className="mr-2">Male: {((demographicsSummary.gender.male / (demographicsSummary.gender.male + demographicsSummary.gender.female + demographicsSummary.gender.unknown)) * 100 || 0).toFixed(1)}%</span>
            <span className="mr-2">Female: {((demographicsSummary.gender.female / (demographicsSummary.gender.male + demographicsSummary.gender.female + demographicsSummary.gender.unknown)) * 100 || 0).toFixed(1)}%</span>
          </p>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-2">Dominant Age Group</h2>
          <p className="text-3xl font-bold">
            {Object.entries(demographicsSummary.age)
              .sort((a, b) => b[1] - a[1])[0]?.[0]
              .charAt(0).toUpperCase() + 
              Object.entries(demographicsSummary.age)
                .sort((a, b) => b[1] - a[1])[0]?.[0]
                .slice(1) || 'None'}
          </p>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-2">Peak Time</h2>
          <p className="text-3xl font-bold">
            {demographicsSummary.hourly.indexOf(Math.max(...demographicsSummary.hourly))}:00
          </p>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-4">Gender Distribution</h2>
          <div style={{ height: '300px' }}>
            <Pie data={genderData} options={{ maintainAspectRatio: false }} />
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-4">Age Group Distribution</h2>
          <div style={{ height: '300px' }}>
            <Pie data={ageData} options={{ maintainAspectRatio: false }} />
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-4">Hourly Customer Flow</h2>
          <div style={{ height: '300px' }}>
            <Bar 
              data={hourlyData} 
              options={{ 
                maintainAspectRatio: false,
                scales: {
                  y: {
                    beginAtZero: true,
                  }
                }
              }} 
            />
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-4">Popular Clothing Colors</h2>
          <div style={{ height: '300px' }}>
            <Pie data={colorData} options={{ maintainAspectRatio: false }} />
          </div>
        </div>
      </div>

      {/* Customer List */}
      <div className="bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-4">Customer Records ({filteredData.length})</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white">
            <thead className="bg-gray-100">
              <tr>
                <th className="py-2 px-4 border-b">Image</th>
                <th className="py-2 px-4 border-b">Gender</th>
                <th className="py-2 px-4 border-b">Age Group</th>
                <th className="py-2 px-4 border-b">Entry Time</th>
                <th className="py-2 px-4 border-b">Clothing Color</th>
                <th className="py-2 px-4 border-b">Repeat Customer</th>
              </tr>
            </thead>
            <tbody>
              {filteredData.length > 0 ? (
                filteredData.slice(0, 10).map((customer, index) => (
                  <tr key={index} className="hover:bg-gray-50 border-b">
                    <td className="py-2 px-4">
                      {customer.image_url && (
                        <img 
                          src={customer.image_url} 
                          alt="Customer" 
                          className="w-10 h-10 rounded-full object-cover"
                        />
                      )}
                    </td>
                    <td className="py-2 px-4">
                      {customer.gender.charAt(0).toUpperCase() + customer.gender.slice(1)}
                    </td>
                    <td className="py-2 px-4">
                      {customer.age_group.charAt(0).toUpperCase() + customer.age_group.slice(1)}
                    </td>
                    <td className="py-2 px-4">
                      {new Date(customer.entry_time).toLocaleString()}
                    </td>
                    <td className="py-2 px-4">
                      <span className="px-2 py-1 rounded" style={{ 
                        backgroundColor: `${customer.clothing_color.toLowerCase()}20`,
                        color: customer.clothing_color.toLowerCase()
                      }}>
                        {customer.clothing_color.charAt(0).toUpperCase() + customer.clothing_color.slice(1)}
                      </span>
                    </td>
                    <td className="py-2 px-4">
                      {customer.is_repeat_customer ? 'Yes' : 'No'}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="6" className="py-4 text-center">
                    No customer records found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
          {filteredData.length > 10 && (
            <div className="text-right text-sm text-gray-500 mt-2">
              Showing 10 of {filteredData.length} records
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CustomerDemographics; 