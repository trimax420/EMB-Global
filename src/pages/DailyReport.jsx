import React, { useState, useEffect } from 'react';
import * as XLSX from 'xlsx';
import { getDailyReport } from '../services/api';

const DailyReport = () => {
  const [selectedDate, setSelectedDate] = useState('');
  const [reportData, setReportData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Get today's date in YYYY-MM-DD format
  const getTodayDate = () => {
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0');
    const day = String(today.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  };

  // Load report data
  useEffect(() => {
    const fetchReport = async () => {
      try {
        setLoading(true);
        const today = getTodayDate();
        setSelectedDate(today);
        const data = await getDailyReport(today);
        setReportData(data);
      } catch (err) {
        setError(err.message);
        console.error('Error fetching report:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchReport();
  }, []);

  // Handle date selection
  const handleDateChange = async (e) => {
    const date = e.target.value;
    setSelectedDate(date);
    try {
      setLoading(true);
      const data = await getDailyReport(date);
      setReportData(data);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching report:', err);
    } finally {
      setLoading(false);
    }
  };

  // Export report as Excel
  const handleExport = () => {
    if (!reportData) return;

    const exportData = [
      {
        Metric: 'Total Entries',
        Value: reportData.total_entries
      },
      {
        Metric: 'Total Purchases',
        Value: reportData.total_purchases
      },
      {
        Metric: 'No Purchase',
        Value: reportData.no_purchase
      },
      {
        Metric: 'Peak Hour',
        Value: reportData.peak_hour
      },
      {
        Metric: 'Average Time Spent',
        Value: reportData.average_time_spent
      },
      {}, // Empty row for separation
      { Hour: 'Hour', Entries: 'Entries', Purchases: 'Purchases' },
      ...reportData.hourly_breakdown.map((entry) => ({
        Hour: entry.hour,
        Entries: entry.entries,
        Purchases: entry.purchases
      }))
    ];

    const worksheet = XLSX.utils.json_to_sheet(exportData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Daily Report');
    XLSX.writeFile(workbook, `daily_report_${selectedDate}.xlsx`);
  };

  if (loading) {
    return <div className="p-6">Loading report data...</div>;
  }

  if (error) {
    return <div className="p-6 text-red-500">Error: {error}</div>;
  }

  return (
    <div className="p-6">
      {/* Header Section */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800">Daily Customer Report</h1>
        <button
          onClick={handleExport}
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
          disabled={!reportData}
        >
          Export Report
        </button>
      </div>

      {/* Date Filter */}
      <div className="mb-6">
        <label htmlFor="date" className="mr-4 font-medium text-gray-700">
          Select Date:
        </label>
        <input
          type="date"
          value={selectedDate}
          onChange={handleDateChange}
          className="px-4 py-2 border border-gray-300 rounded"
        />
      </div>

      {reportData && (
        <>
          {/* Metrics Summary */}
          <div className="mb-6 p-6 bg-white border border-gray-300 rounded shadow">
            <h2 className="text-xl font-semibold mb-4">Summary for {selectedDate}</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-gray-600">Total Entries:</p>
                <p className="font-bold">{reportData.total_entries}</p>
              </div>
              <div>
                <p className="text-gray-600">Total Purchases:</p>
                <p className="font-bold">{reportData.total_purchases}</p>
              </div>
              <div>
                <p className="text-gray-600">No Purchase:</p>
                <p className="font-bold">{reportData.no_purchase}</p>
              </div>
              <div>
                <p className="text-gray-600">Peak Hour:</p>
                <p className="font-bold">{reportData.peak_hour}</p>
              </div>
              <div>
                <p className="text-gray-600">Average Time Spent:</p>
                <p className="font-bold">{reportData.average_time_spent}</p>
              </div>
            </div>
          </div>

          {/* Hourly Breakdown Table */}
          <div className="overflow-x-auto">
            <h2 className="text-xl font-semibold mb-4">Hourly Breakdown</h2>
            <table className="min-w-full text-left bg-white border border-gray-300">
              <thead>
                <tr className="bg-gray-100">
                  <th className="py-2 px-4 border-b">Hour</th>
                  <th className="py-2 px-4 border-b">Entries</th>
                  <th className="py-2 px-4 border-b">Purchases</th>
                </tr>
              </thead>
              <tbody>
                {reportData.hourly_breakdown.map((entry, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="py-2 px-4 border-b">{entry.hour}</td>
                    <td className="py-2 px-4 border-b">{entry.entries}</td>
                    <td className="py-2 px-4 border-b">{entry.purchases}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
};

export default DailyReport;