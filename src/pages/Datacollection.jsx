// import React, { useState } from 'react';
// import * as XLSX from 'xlsx';

// const Datacollection = () => {
//   // Dummy customer data
//   const dummyData = [
//     {
//       id: 1,
//       image_url: 'https://via.placeholder.com/100',
//       gender: 'Male',
//       entry_time: '10:15 AM',
//       entry_date: '2023-10-10',
//       age_group: '25-34',
//       clothing_color: 'Blue',
//       notes: 'Carrying bag'
//     },
//     {
//       id: 2,
//       image_url: 'https://via.placeholder.com/100',
//       gender: 'Female',
//       entry_time: '03:17 PM',
//       entry_date: '2023-10-10',
//       age_group: '18-24',
//       clothing_color: 'Red',
//       notes: 'Wearing hat'
//     },
//     {
//       id: 3,
//       image_url: 'https://via.placeholder.com/100',
//       gender: 'Male',
//       entry_time: '08:20 PM',
//       entry_date: '2023-10-10',
//       age_group: '35-44',
//       clothing_color: 'Black',
//       notes: 'Pushing cart'
//     }
//   ];

//   const [data, setData] = useState(dummyData);
//   const [filters, setFilters] = useState({ gender: 'all', date: '', timePeriod: 'all' });

//   // Time period options
//   const timePeriodOptions = ['All', 'Morning (6 AM–12 PM)', 'Afternoon (12 PM–6 PM)', 'Evening (6 PM–9 PM)', 'Night (9 PM–6 AM)'];

//   // Helper function to determine the time period of an entry
//   const getTimePeriod = (time) => {
//     const [hour, minute, modifier] = time.match(/(\d+):(\d+)\s*(AM|PM)/i).slice(1);
//     let hourIn24Format = parseInt(hour, 10);

//     if (modifier.toUpperCase() === 'PM' && hourIn24Format !== 12) {
//       hourIn24Format += 12;
//     }
//     if (modifier.toUpperCase() === 'AM' && hourIn24Format === 12) {
//       hourIn24Format = 0;
//     }

//     if (hourIn24Format >= 6 && hourIn24Format < 12) return 'Morning (6 AM–12 PM)';
//     if (hourIn24Format >= 12 && hourIn24Format < 18) return 'Afternoon (12 PM–6 PM)';
//     if (hourIn24Format >= 18 && hourIn24Format < 21) return 'Evening (6 PM–9 PM)';
//     return 'Night (9 PM–6 AM)';
//   };

//   // Apply filters to the dummy data
//   const filteredData = data.filter((entry) => {
//     if (filters.gender !== 'all' && entry.gender.toLowerCase() !== filters.gender.toLowerCase()) return false;
//     if (filters.date && entry.entry_date !== filters.date) return false;
//     if (filters.timePeriod !== 'all' && getTimePeriod(entry.entry_time) !== filters.timePeriod) return false;
//     return true;
//   });

//   // Export data as CSV
//   const handleExport = () => {
//     const worksheet = XLSX.utils.json_to_sheet(filteredData);
//     const workbook = XLSX.utils.book_new();
//     XLSX.utils.book_append_sheet(workbook, worksheet, 'Customer Data');
//     XLSX.writeFile(workbook, 'customer_data.csv');
//   };

//   return (
//     <div className="p-6">
//       {/* Header Section */}
//       <div className="flex justify-between items-center mb-6">
//         <h1 className="text-2xl font-bold text-gray-800">Customer Entry Data Collection</h1>
//         <button
//           onClick={handleExport}
//           className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
//         >
//           Export Data
//         </button>
//       </div>

//       {/* Filters Section */}
//       <div className="flex space-x-4 mb-6">
//         <select
//           value={filters.gender}
//           onChange={(e) => setFilters({ ...filters, gender: e.target.value })}
//           className="px-4 py-2 border border-gray-300 rounded"
//         >
//           <option value="all">All</option>
//           <option value="male">Male</option>
//           <option value="female">Female</option>
//         </select>

//         <input
//           type="date"
//           value={filters.date}
//           onChange={(e) => setFilters({ ...filters, date: e.target.value })}
//           className="px-4 py-2 border border-gray-300 rounded"
//         />

//         <select
//           value={filters.timePeriod}
//           onChange={(e) => setFilters({ ...filters, timePeriod: e.target.value })}
//           className="px-4 py-2 border border-gray-300 rounded"
//         >
//           {timePeriodOptions.map((period) => (
//             <option key={period} value={period}>
//               {period}
//             </option>
//           ))}
//         </select>
//       </div>

//       {/* Table Section */}
//       <div className="overflow-x-auto">
//         <table className="min-w-full text-left bg-white border border-gray-300">
//           <thead>
//             <tr className="bg-gray-100">
//               <th className="py-2 px-4 border-b">Image</th>
//               <th className="py-2 px-4 border-b">Gender</th>
//               <th className="py-2 px-4 border-b">Time</th>
//               <th className="py-2 px-4 border-b">Date</th>
//               <th className="py-2 px-4 border-b">Age Group</th>
//               <th className="py-2 px-4 border-b">Clothing Color</th>
//               <th className="py-2 px-4 border-b">Notes</th>
//             </tr>
//           </thead>
//           <tbody>
//             {filteredData.length > 0 ? (
//               filteredData.map((entry) => (
//                 <tr key={entry.id} className="hover:bg-gray-50">
//                   <td className="py-2 px-4 border-b">
//                     <img
//                       src={entry.image_url}
//                       alt="Customer"
//                       className="w-12 h-12 object-cover rounded"
//                     />
//                   </td>
//                   <td className="py-2 px-4 border-b">{entry.gender}</td>
//                   <td className="py-2 px-4 border-b">{entry.entry_time}</td>
//                   <td className="py-2 px-4 border-b">{entry.entry_date}</td>
//                   <td className="py-2 px-4 border-b">{entry.age_group}</td>
//                   <td className="py-2 px-4 border-b">{entry.clothing_color}</td>
//                   <td className="py-2 px-4 border-b">{entry.notes || '-'}</td>
//                 </tr>
//               ))
//             ) : (
//               <tr>
//                 <td colSpan="7" className="py-4 text-center">
//                   No data available
//                 </td>
//               </tr>
//             )}
//           </tbody>
//         </table>
//       </div>
//     </div>
//   );
// };

// export default Datacollection;

import React, { useState } from 'react';
import * as XLSX from 'xlsx';

const Datacollection = () => {
  // Dummy customer data
  const dummyData = [
    {
      id: 1,
      image_url: 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScV579wLUmyQ8AuD5IMUX68PswFX1WOFNBcA&s',
      gender: 'Male',
      entry_time: '10:15 AM',
      entry_date: '2023-10-10',
      age_group: '25-34',
      clothing_color: 'Blue',
      notes: 'Carrying bag'
    },
    {
      id: 2,
      image_url: 'https://m.media-amazon.com/images/I/41ImMQIW+6L._AC_UY1100_.jpg',
      gender: 'Female',
      entry_time: '03:17 PM',
      entry_date: '2023-10-10',
      age_group: '18-24',
      clothing_color: 'Red',
      notes: 'Wearing hat'
    },
    {
      id: 3,
      image_url: 'https://img.freepik.com/premium-photo/smiling-man-pushing-shopping-cart-photo-with-copy-space_252847-13079.jpg',
      gender: 'Male',
      entry_time: '08:20 PM',
      entry_date: '2023-10-10',
      age_group: '35-44',
      clothing_color: 'Black',
      notes: 'Pushing cart'
    }
  ];

  const [data, setData] = useState(dummyData);
  const [filters, setFilters] = useState({ gender: 'all', date: '', timePeriod: 'all' });

  // Time period options
  const timePeriodOptions = ['All', 'Morning (6 AM–12 PM)', 'Afternoon (12 PM–6 PM)', 'Evening (6 PM–9 PM)', 'Night (9 PM–6 AM)'];

  // Helper function to determine the time period of an entry
  const getTimePeriod = (time) => {
    const [hour, minute, modifier] = time.match(/(\d+):(\d+)\s*(AM|PM)/i).slice(1);
    let hourIn24Format = parseInt(hour, 10);

    if (modifier.toUpperCase() === 'PM' && hourIn24Format !== 12) {
      hourIn24Format += 12;
    }
    if (modifier.toUpperCase() === 'AM' && hourIn24Format === 12) {
      hourIn24Format = 0;
    }

    if (hourIn24Format >= 6 && hourIn24Format < 12) return 'Morning (6 AM–12 PM)';
    if (hourIn24Format >= 12 && hourIn24Format < 18) return 'Afternoon (12 PM–6 PM)';
    if (hourIn24Format >= 18 && hourIn24Format < 21) return 'Evening (6 PM–9 PM)';
    return 'Night (9 PM–6 AM)';
  };

  // Apply filters to the dummy data
  const filteredData = data.filter((entry) => {
    // Filter by gender
    if (filters.gender !== 'all' && entry.gender.toLowerCase() !== filters.gender.toLowerCase()) return false;

    // Filter by date
    if (filters.date && entry.entry_date !== filters.date) return false;

    // Filter by time period (only if it's not 'All')
    if (filters.timePeriod !== 'all' && getTimePeriod(entry.entry_time) !== filters.timePeriod) return false;

    return true;
  });

  // Export data as CSV
  const handleExport = () => {
    const worksheet = XLSX.utils.json_to_sheet(filteredData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Customer Data');
    XLSX.writeFile(workbook, 'customer_data.csv');
  };

  return (
    <div className="p-6">
      {/* Header Section */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800">Customer Entry Data Collection</h1>
        <button
          onClick={handleExport}
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
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
            {filteredData.length > 0 ? (
              filteredData.map((entry) => (
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
