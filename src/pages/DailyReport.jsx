import React, { useState, useEffect } from 'react';
import * as XLSX from 'xlsx';

const DailyReport = () => {
  // Dummy data for daily report
  const dummyData = [
    {
      date: '2025-03-18',
      totalEntries: 500,
      totalPurchases: 200,
      noPurchase: 300,
      peakHour: '12 PM - 1 PM',
      averageTimeSpent: '25 minutes'
    },
    {
      date: '2023-10-09',
      totalEntries: 450,
      totalPurchases: 180,
      noPurchase: 270,
      peakHour: '3 PM - 4 PM',
      averageTimeSpent: '20 minutes'
    }
  ];

  // Hourly breakdown data for the selected date
  const hourlyData = [
    { hour: '6 AM', entries: 20, purchases: 10 },
    { hour: '7 AM', entries: 30, purchases: 15 },
    { hour: '8 AM', entries: 40, purchases: 20 },
    { hour: '9 AM', entries: 50, purchases: 25 },
    { hour: '10 AM', entries: 60, purchases: 30 },
    { hour: '11 AM', entries: 70, purchases: 35 },
    { hour: '12 PM', entries: 80, purchases: 40 },
    { hour: '1 PM', entries: 90, purchases: 45 },
    { hour: '2 PM', entries: 50, purchases: 25 },
    { hour: '3 PM', entries: 40, purchases: 20 },
    { hour: '4 PM', entries: 30, purchases: 15 },
    { hour: '5 PM', entries: 20, purchases: 10 }
  ];

  const [selectedDate, setSelectedDate] = useState('');
  const [reportData, setReportData] = useState(null);

  // Get today's date in YYYY-MM-DD format
  const getTodayDate = () => {
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0');
    const day = String(today.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  };

  // Load today's report on page load
  useEffect(() => {
    const today = getTodayDate();
    setSelectedDate(today);
    const foundData = dummyData.find((item) => item.date === today);
    setReportData(foundData);
  }, []);

  // Handle date selection
  const handleDateChange = (e) => {
    const date = e.target.value;
    setSelectedDate(date);
    const foundData = dummyData.find((item) => item.date === date);
    setReportData(foundData);
  };

  // Export report as CSV
  const handleExport = () => {
    const exportData = [
      {
        Metric: 'Total Entries',
        Value: reportData?.totalEntries || '-'
      },
      {
        Metric: 'Total Purchases',
        Value: reportData?.totalPurchases || '-'
      },
      {
        Metric: 'No Purchase',
        Value: reportData?.noPurchase || '-'
      },
      {
        Metric: 'Peak Hour',
        Value: reportData?.peakHour || '-'
      },
      {
        Metric: 'Average Time Spent',
        Value: reportData?.averageTimeSpent || '-'
      },
      {}, // Empty row for separation
      { Hour: 'Hour', Entries: 'Entries', Purchases: 'Purchases' },
      ...hourlyData.map((entry) => ({
        Hour: entry.hour,
        Entries: entry.entries,
        Purchases: entry.purchases
      }))
    ];

    const worksheet = XLSX.utils.json_to_sheet(exportData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Daily Report');
    XLSX.writeFile(workbook, `daily_report_${selectedDate}.csv`);
  };

  return (
    <div className="p-6">
      {/* Header Section */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800">Daily Customer Report</h1>
        <button
          onClick={handleExport}
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
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

      {/* Metrics Summary */}
      {reportData && (
        <div className="mb-6 p-6 bg-white border border-gray-300 rounded shadow">
          <h2 className="text-xl font-semibold mb-4">Summary for {reportData.date}</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-gray-600">Total Entries:</p>
              <p className="font-bold">{reportData.totalEntries}</p>
            </div>
            <div>
              <p className="text-gray-600">Total Purchases:</p>
              <p className="font-bold">{reportData.totalPurchases}</p>
            </div>
            <div>
              <p className="text-gray-600">No Purchase:</p>
              <p className="font-bold">{reportData.noPurchase}</p>
            </div>
            <div>
              <p className="text-gray-600">Peak Hour:</p>
              <p className="font-bold">{reportData.peakHour}</p>
            </div>
            <div>
              <p className="text-gray-600">Average Time Spent:</p>
              <p className="font-bold">{reportData.averageTimeSpent}</p>
            </div>
          </div>
        </div>
      )}

      {/* Hourly Breakdown Table */}
      {reportData && (
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
              {hourlyData.map((entry, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="py-2 px-4 border-b">{entry.hour}</td>
                  <td className="py-2 px-4 border-b">{entry.entries}</td>
                  <td className="py-2 px-4 border-b">{entry.purchases}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default DailyReport;