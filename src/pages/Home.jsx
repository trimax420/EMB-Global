// import React from 'react';
// import { CiCamera } from "react-icons/ci";
// import { ImNotification } from "react-icons/im";
// import { TbActivityHeartbeat } from "react-icons/tb";
// import { FaUserGroup } from "react-icons/fa6";
// import { Bar } from 'react-chartjs-2';
// import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

// // Register Chart.js components
// ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// const stats = [
//   { icon: <CiCamera />, title: "Total Cameras", value: "5", description: "4 offline, 8 online" },
//   { icon: <FaUserGroup />, title: "Active Detections", value: "25", description: "+15% from last hour" },
//   { icon: <ImNotification />, title: "Current Alerts", value: "7", description: "+2 new alerts" },
//   { icon: <TbActivityHeartbeat />, title: "System Status", value: "Optimal", description: "All systems operational" },
// ];

// const incidents = [
//   { title: "Unauthorized Access", location: "Front Entrance", time: "14:35", severity: "high" },
//   { title: "Vehicle Stopped", location: "Parking Lot", time: "10:35", severity: "low" },
//   { title: "Person Detected", location: "Restricted Area", time: "14:35", severity: "medium" },
//   { title: "Motion Detected", location: "Storage Room", time: "14:35", severity: "low" },
// ];

// const systemStatus = [
//   { title: "Video Processing", status: "Operational", statusClass: "bg-green-100 text-green-500" },
//   { title: "Object Detection", status: "Operational", statusClass: "bg-green-100 text-green-500" },
//   { title: "Facial Recognition", status: "Degraded", statusClass: "bg-yellow-100/50" },
//   { title: "License Plate Reader", status: "Offline", statusClass: "bg-red-100/50 text-red-500" },
// ];

// // Data for the Weekly Incident Report Bar Graph
// const incidentData = {
//   labels: ['March 1', 'March 2', 'March 3', 'March 4', 'March 5', 'March 6', 'March 7'],
//   datasets: [
//     {
//       label: 'Incidents Reported',
//       data: [3, 5, 2, 6, 8, 4, 7],  // Example data for incidents count each day
//       backgroundColor: 'rgba(75, 192, 192, 0.2)', // Bar color (with transparency)
//       borderColor: 'rgb(75, 192, 192)', // Border color of bars
//       borderWidth: 1,
//     },
//   ],
// };

// function Home() {
//   return (
//     <div className='be-white p-5'>
//       <div>
//         <div className='flex justify-between'>
//           <div>
//             <h1 className='text-lg font-medium'>Overview</h1>
//             <p className='text-gray-500'>Monitor your security system status and activities</p>
//           </div>
//           <div className='flex gap-5'>
//             <button className='px-6 h-10 border shadow-sm hover:shadow-md hover:bg-gray-200/50 rounded-lg'>Today</button>
//             <button className='px-6 h-10 border shadow-sm hover:shadow-md hover:bg-gray-200/50 rounded-lg'>View Report</button>
//           </div>
//         </div>

//         <div className='grid md:grid-cols-2 lg:grid-cols-4 gap-5 mt-5'>
//           {stats.map(({ icon, title, value, description }) => (
//             <div className='bg-white p-5 rounded-xl shadow flex flex-col items-center gap-1 border' key={title}>
//               <h1 className='text-xl'>{icon}</h1>
//               <h2 className='font-medium'>{title}</h2>
//               <p className='text-3xl font-medium'>{value}</p>
//               <p className='text-gray-500 text-sm'>{description}</p>
//             </div>
//           ))}
//         </div>

//         <div className='mt-5  lg:flex justify-between gap-5'>
//           <div className='p-5 bg-white rounded-xl shadow text-center border lg:w-[75%]'>
//             <h1 className='text-left font-semibold text-xl'>Live Security Feed</h1>
//             <button className='text-start mb-5'>Live View</button>
//             <div className='bg-gray-200 p-5 rounded-xl shadow text-center border w-[99%] h-[720px]'>
//               <div className='w-[250px] h-[200px] bg-gray-300 rounded-lg p-4 font-medium'>
//                 <h1 className='text-lg mb-3'>Front Entrance . LIVE</h1>
//                 <div className='grid md:grid-cols-2 justify-between'>
//                   <p>People: 4</p>
//                   <p>Vehicles: 1</p>
//                   <p>Alerts: 4</p>
//                   <p>Objects: 5</p>
//                 </div>
//               </div>
//             </div>

//             <div className='mt-5 '>
//               <div className='flex justify-between gap-5 mb-4'>
//                 <h1 className='text-md font-medium'>Available Cameras</h1>
//                 <h1 className='text-md font-medium cursor-pointer'>View All</h1>
//               </div>
//               <div className='grid md:grid-cols-2 lg:grid-cols-4 gap-5'>
//                 {[...Array(4)].map((_, index) => (
//                   <div className='flex flex-col justify-between border border-gray-300 bg-blue-900 rounded-lg text-white font-medium h-[70px] gap-1 hover:border hover:border-black' key={index}>
//                     <div className='text-xs bg-white text-black h-[70%] px-2 py-1 text-center rounded-md'>
//                       <h1>220</h1>
//                       <h1>180</h1>
//                     </div>
//                     <h1>Front Entrance</h1>
//                   </div>
//                 ))}
//               </div>
//             </div>
//           </div>

//           <div className='lg:w-[25%]'>
//             <div className='mb-5 p-5 rounded-xl shadow text-start border bg-white'>
//               <h1 className='text-xl font-medium'>Detection trends</h1>
//               <p className='text-gray-500'>Today's activity</p>
//             </div>
//             <div className='p-5 rounded-xl text-start border shadow bg-white'>
//               <h1 className='text-xl font-medium'>Recent Incidents</h1>
//               <p className='text-gray-500 mb-3'>last 24 hours</p>
//               {incidents.map(({ title, location, time, severity }) => (
//                 <div className='flex gap-5 justify-between mb-3' key={title}>
//                   <div>
//                     <h1 className='font-medium'>{title}</h1>
//                     <p className='text-sm text-gray-500'>{location}</p>
//                     <p className='text-sm text-gray-500'>{time}</p>
//                   </div>
//                   <button className={`px-3 py-1 text-xs h-7 rounded-lg font-medium ${severity === 'high' ? 'bg-red-500 text-white' : ''}`}>{severity}</button>
//                 </div>
//               ))}
//               <div className='border-t mt-5'>
//                 <h1 className='text-gray-500 font-medium cursor-pointer mt-5 text-center'>View All incidents</h1>
//               </div>
//             </div>

//             <div className='p-5 rounded-xl text-start shadow border mt-5 bg-white'>
//               <h1 className='text-xl font-medium'>System Status</h1>
//               <p className='text-gray-500 mb-5'>All systems operational</p>
//               {systemStatus.map(({ title, status, statusClass }) => (
//                 <div className='flex gap-5 justify-between mb-5' key={title}>
//                   <div>
//                     <h1 className='text-sm'>{title}</h1>
//                   </div>
//                   <button className={`px-3 py-1 text-xs rounded-lg font-medium ${statusClass}`}>{status}</button>
//                 </div>
//               ))}
//             </div>
//           </div>
//         </div>

//         <div className='mt-5'>
//           <h1 className='text-xl font-medium'>Weekly Incident Report</h1>
//           <p className='text-gray-500 mb-3'>March 1 - March 7, 2025</p>
//           <div className='bg-white p-5 rounded-xl shadow'>
//             <Bar data={incidentData} />
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// }

// export default Home;

// import React from 'react';
// import { CiCamera } from "react-icons/ci";
// import { ImNotification } from "react-icons/im";
// import { TbActivityHeartbeat } from "react-icons/tb";
// import { FaUserGroup } from "react-icons/fa6";
// import { Bar } from 'react-chartjs-2';
// import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

// // Register Chart.js components
// ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// const stats = [
//   { icon: <CiCamera />, title: "Total Cameras", value: "5", description: "4 offline, 8 online" },
//   { icon: <FaUserGroup />, title: "Active Detections", value: "25", description: "+15% from last hour" },
//   { icon: <ImNotification />, title: "Current Alerts", value: "7", description: "+2 new alerts" },
//   { icon: <TbActivityHeartbeat />, title: "System Status", value: "Optimal", description: "All systems operational" },
// ];

// const incidents = [
//   { title: "Unauthorized Access", location: "Front Entrance", time: "14:35", severity: "high" },
//   { title: "Vehicle Stopped", location: "Parking Lot", time: "10:35", severity: "low" },
//   { title: "Person Detected", location: "Restricted Area", time: "14:35", severity: "medium" },
//   { title: "Motion Detected", location: "Storage Room", time: "14:35", severity: "low" },
// ];

// const systemStatus = [
//   { title: "Video Processing", status: "Operational", statusClass: "bg-green-100 text-green-500" },
//   { title: "Object Detection", status: "Operational", statusClass: "bg-green-100 text-green-500" },
//   { title: "Facial Recognition", status: "Degraded", statusClass: "bg-yellow-100/50" },
//   { title: "License Plate Reader", status: "Offline", statusClass: "bg-red-100/50 text-red-500" },
// ];

// // Data for the Weekly Incident Report Bar Graph
// const incidentData = {
//   labels: ['March 1', 'March 2', 'March 3', 'March 4', 'March 5', 'March 6', 'March 7'],
//   datasets: [
//     {
//       label: 'Incidents Reported',
//       data: [3, 5, 2, 6, 8, 4, 7],  // Example data for incidents count each day
//       backgroundColor: 'rgba(75, 192, 192, 0.2)', // Bar color (with transparency)
//       borderColor: 'rgb(75, 192, 192)', // Border color of bars
//       borderWidth: 1,
//     },
//   ],
// };

// function Home() {
//   return (
//     <div className='bg-gray-50 p-8'>
//       <div className='max-w-7xl mx-auto'>
//         <div className='flex justify-between'>
//           <div>
//             <h1 className='text-2xl font-semibold'>Overview</h1>
//             <p className='text-gray-500 mt-2'>Monitor your security system status and activities in real time</p>
//           </div>
//           <div className='flex gap-5'>
//             <button className='px-6 py-2 border rounded-lg shadow-sm hover:shadow-md hover:bg-gray-100'>Today</button>
//             <button className='px-6 py-2 border rounded-lg shadow-sm hover:shadow-md hover:bg-gray-100'>View Report</button>
//           </div>
//         </div>

//         <div className='grid md:grid-cols-2 lg:grid-cols-4 gap-8 mt-8'>
//           {stats.map(({ icon, title, value, description }) => (
//             <div className='bg-white p-6 rounded-xl shadow-lg flex flex-col items-center gap-3 hover:shadow-xl transition duration-200' key={title}>
//               <div className='text-4xl text-blue-500'>{icon}</div>
//               <h2 className='font-semibold text-lg'>{title}</h2>
//               <p className='text-3xl font-medium text-gray-800'>{value}</p>
//               <p className='text-sm text-gray-500'>{description}</p>
//             </div>
//           ))}
//         </div>

//         <div className='lg:flex justify-between gap-8 mt-10'>
//           {/* Left Side: Live Feed & Available Cameras */}
//           <div className='lg:w-3/4'>
//             <div className='p-6 bg-white rounded-xl shadow-lg text-center border'>
//               <h2 className='text-xl font-semibold mb-4'>Live Security Feed</h2>
//               <button className='mb-5 text-blue-500 hover:text-blue-700'>Live View</button>
//               <div className='bg-gray-200 rounded-xl p-6'>
//                 <img src="https://developer-blogs.nvidia.com/wp-content/uploads/2022/12/Figure8-output_blurred-compressed.gif" alt="Live Security Feed" className="w-full h-96 object-cover rounded-lg" />
//               </div>

//               <div className='mt-6'>
//                 <div className='flex justify-between mb-4'>
//                   <h2 className='text-lg font-semibold'>Available Cameras</h2>
//                   <h2 className='text-sm font-medium text-blue-500 cursor-pointer'>View All</h2>
//                 </div>
//                 <div className='grid md:grid-cols-2 lg:grid-cols-4 gap-8'>
//                   {[...Array(4)].map((_, index) => (
//                     <div className='flex flex-col justify-between bg-blue-900 text-white rounded-lg h-[150px] hover:shadow-xl transition duration-300' key={index}>
//                       <div className='text-center p-4 bg-gray-300 rounded-md'>
//                         <h2 className='text-2xl font-semibold'>Camera {index + 1}</h2>
//                       </div>
//                       <div className='flex justify-between px-4 py-2'>
//                         <p>People: 4</p>
//                         <p>Vehicles: 1</p>
//                       </div>
//                     </div>
//                   ))}
//                 </div>
//               </div>
//             </div>
//           </div>

//           {/* Right Side: Incidents, System Status, and Report */}
//           <div className='lg:w-1/4'>
//             <div className='p-6 bg-white rounded-xl shadow-lg mb-6'>
//               <h2 className='text-xl font-semibold'>Recent Incidents</h2>
//               <p className='text-gray-500 mb-4'>Last 24 hours</p>
//               {incidents.map(({ title, location, time, severity }) => (
//                 <div className='flex justify-between mb-4' key={title}>
//                   <div>
//                     <h3 className='font-semibold'>{title}</h3>
//                     <p className='text-sm text-gray-500'>{location}</p>
//                     <p className='text-sm text-gray-500'>{time}</p>
//                   </div>
//                   <button className={`px-4 py-1 text-xs rounded-lg font-medium ${severity === 'high' ? 'bg-red-500 text-white' : 'bg-yellow-300 text-black'}`}>{severity}</button>
//                 </div>
//               ))}
//               <div className='text-center'>
//                 <h2 className='text-gray-500 font-medium cursor-pointer'>View All Incidents</h2>
//               </div>
//             </div>

//             <div className='p-6 bg-white rounded-xl shadow-lg'>
//               <h2 className='text-xl font-semibold mb-4'>System Status</h2>
//               <p className='text-gray-500 mb-4'>All systems operational</p>
//               {systemStatus.map(({ title, status, statusClass }) => (
//                 <div className='flex justify-between mb-4' key={title}>
//                   <div>
//                     <h3 className='text-sm'>{title}</h3>
//                   </div>
//                   <button className={`px-4 py-1 text-xs rounded-lg font-medium ${statusClass}`}>{status}</button>
//                 </div>
//               ))}
//             </div>
//           </div>
//         </div>

//         <div className='mt-10'>
//           <h2 className='text-xl font-semibold'>Weekly Incident Report</h2>
//           <p className='text-gray-500 mb-4'>March 1 - March 7, 2025</p>
//           <div className='bg-white p-6 rounded-xl shadow-lg'>
//             <Bar data={incidentData} />
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// }

// export default Home;


// import React from 'react';
// import { CiCamera } from "react-icons/ci";
// import { ImNotification } from "react-icons/im";
// import { TbActivityHeartbeat } from "react-icons/tb";
// import { FaUserGroup } from "react-icons/fa6";
// import { Bar } from 'react-chartjs-2';
// import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

// // Register Chart.js components
// ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// const stats = [
//   { icon: <CiCamera />, title: "Total Cameras", value: "5", description: "4 offline, 8 online" },
//   { icon: <FaUserGroup />, title: "Active Detections", value: "25", description: "+15% from last hour" },
//   { icon: <ImNotification />, title: "Current Alerts", value: "7", description: "+2 new alerts" },
//   { icon: <TbActivityHeartbeat />, title: "System Status", value: "Optimal", description: "All systems operational" },
// ];

// const incidents = [
//   { title: "Unauthorized Access", location: "Front Entrance", time: "14:35", severity: "high" },
//   { title: "Vehicle Stopped", location: "Parking Lot", time: "10:35", severity: "low" },
//   { title: "Person Detected", location: "Restricted Area", time: "14:35", severity: "medium" },
//   { title: "Motion Detected", location: "Storage Room", time: "14:35", severity: "low" },
// ];

// const systemStatus = [
//   { title: "Video Processing", status: "Operational", statusClass: "bg-green-100 text-green-500" },
//   { title: "Object Detection", status: "Operational", statusClass: "bg-green-100 text-green-500" },
//   { title: "Facial Recognition", status: "Degraded", statusClass: "bg-yellow-100/50" },
//   { title: "License Plate Reader", status: "Offline", statusClass: "bg-red-100/50 text-red-500" },
// ];

// // Data for the Weekly Incident Report Bar Graph
// const incidentData = {
//   labels: ['March 1', 'March 2', 'March 3', 'March 4', 'March 5', 'March 6', 'March 7'],
//   datasets: [
//     {
//       label: 'Incidents Reported',
//       data: [3, 5, 2, 6, 8, 4, 7],
//       backgroundColor: 'rgba(75, 192, 192, 0.2)',
//       borderColor: 'rgb(75, 192, 192)',
//       borderWidth: 1,
//     },
//   ],
// };

// function Home() {
//   return (
//     <div className='bg-gray-50 p-5 min-h-screen'>
//       {/* Header Section */}
//       <div className='flex justify-between items-center mb-8'>
//         <div>
//           <h1 className='text-2xl font-bold text-gray-800'>Security Dashboard</h1>
//           <p className='text-gray-500'>Monitor your security system status and activities</p>
//         </div>
//         <div className='flex gap-4'>
//           <button className='px-6 h-10 border border-gray-300 shadow-sm hover:shadow-md hover:bg-gray-100 rounded-lg transition-all duration-200'>
//             Today
//           </button>
//           <button className='px-6 h-10 bg-blue-500 text-white shadow-sm hover:shadow-md hover:bg-blue-600 rounded-lg transition-all duration-200'>
//             View Report
//           </button>
//         </div>
//       </div>

//       {/* Stats Section */}
//       <div className='grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8'>
//         {stats.map(({ icon, title, value, description }) => (
//           <div
//             className='bg-white p-6 rounded-xl shadow-md hover:shadow-lg transition-all duration-200 border border-gray-100'
//             key={title}
//           >
//             <div className='text-3xl text-blue-500 mb-4'>{icon}</div>
//             <h2 className='text-lg font-semibold text-gray-700'>{title}</h2>
//             <p className='text-2xl font-bold text-gray-900'>{value}</p>
//             <p className='text-sm text-gray-500'>{description}</p>
//           </div>
//         ))}
//       </div>

//       {/* Main Content Section */}
//       <div className='lg:flex gap-6'>
//         {/* Live Security Feed */}
//         <div className='lg:w-[70%] bg-white rounded-xl shadow-md border border-gray-100 p-6 mb-6 lg:mb-0'>
//           <div className='flex justify-between items-center mb-6'>
//             <h1 className='text-xl font-bold text-gray-800'>Live Security Feed</h1>
//             <button className='px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-all duration-200'>
//               Live View
//             </button>
//           </div>
//           <div className='relative rounded-xl overflow-hidden'>
//             <img
//               src="https://developer-blogs.nvidia.com/wp-content/uploads/2022/12/Figure8-output_blurred-compressed.gif"
//               alt="Live Security Feed"
//               className='w-full h-auto rounded-lg'
//             />
//             <div className='absolute bottom-4 left-4 bg-black bg-opacity-50 text-white p-3 rounded-lg'>
//               <h2 className='text-lg font-semibold'>Front Entrance • LIVE</h2>
//               <div className='grid grid-cols-2 gap-2 mt-2'>
//                 <p>People: 4</p>
//                 <p>Vehicles: 1</p>
//                 <p>Alerts: 4</p>
//                 <p>Objects: 5</p>
//               </div>
//             </div>
//           </div>

//           {/* Available Cameras */}
//           <div className='mt-6'>
//             <div className='flex justify-between items-center mb-4'>
//               <h1 className='text-lg font-semibold text-gray-800'>Available Cameras</h1>
//               <button className='text-blue-500 hover:text-blue-600 transition-all duration-200'>
//                 View All
//               </button>
//             </div>
//             <div className='grid md:grid-cols-2 lg:grid-cols-4 gap-4'>
//               {[...Array(4)].map((_, index) => (
//                 <div
//                   className='bg-blue-900 rounded-lg p-3 text-white hover:bg-blue-800 transition-all duration-200'
//                   key={index}
//                 >
//                   <div className='text-xs bg-white text-black p-1 rounded-md text-center'>
//                     <p>220</p>
//                     <p>180</p>
//                   </div>
//                   <p className='mt-2 text-sm'>Front Entrance</p>
//                 </div>
//               ))}
//             </div>
//           </div>
//         </div>

//         {/* Right Sidebar */}
//         <div className='lg:w-[30%]'>
//           {/* Detection Trends */}
//           <div className='bg-white rounded-xl shadow-md border border-gray-100 p-6 mb-6'>
//             <h1 className='text-xl font-bold text-gray-800'>Detection Trends</h1>
//             <p className='text-gray-500 mb-4'>Today's activity</p>
//             {/* Placeholder for a small chart or visual */}
//             <div className='bg-gray-100 h-32 rounded-lg flex items-center justify-center text-gray-500'>
//               Chart Placeholder
//             </div>
//           </div>

//           {/* Recent Incidents */}
//           <div className='bg-white rounded-xl shadow-md border border-gray-100 p-6 mb-6'>
//             <h1 className='text-xl font-bold text-gray-800'>Recent Incidents</h1>
//             <p className='text-gray-500 mb-4'>Last 24 hours</p>
//             {incidents.map(({ title, location, time, severity }) => (
//               <div className='flex justify-between items-center mb-3' key={title}>
//                 <div>
//                   <h2 className='font-medium text-gray-700'>{title}</h2>
//                   <p className='text-sm text-gray-500'>{location} • {time}</p>
//                 </div>
//                 <span
//                   className={`px-3 py-1 text-xs rounded-full font-medium ${
//                     severity === 'high'
//                       ? 'bg-red-100 text-red-600'
//                       : severity === 'medium'
//                       ? 'bg-yellow-100 text-yellow-600'
//                       : 'bg-green-100 text-green-600'
//                   }`}
//                 >
//                   {severity}
//                 </span>
//               </div>
//             ))}
//             <button className='w-full text-center text-blue-500 hover:text-blue-600 mt-4 transition-all duration-200'>
//               View All Incidents
//             </button>
//           </div>

//           {/* System Status */}
//           <div className='bg-white rounded-xl shadow-md border border-gray-100 p-6'>
//             <h1 className='text-xl font-bold text-gray-800'>System Status</h1>
//             <p className='text-gray-500 mb-4'>All systems operational</p>
//             {systemStatus.map(({ title, status, statusClass }) => (
//               <div className='flex justify-between items-center mb-3' key={title}>
//                 <p className='text-sm text-gray-700'>{title}</p>
//                 <span className={`px-3 py-1 text-xs rounded-full ${statusClass}`}>
//                   {status}
//                 </span>
//               </div>
//             ))}
//           </div>
//         </div>
//       </div>

//       {/* Weekly Incident Report */}
//       <div className='mt-8'>
//         <h1 className='text-xl font-bold text-gray-800'>Weekly Incident Report</h1>
//         <p className='text-gray-500 mb-4'>March 1 - March 7, 2025</p>
//         <div className='bg-white rounded-xl shadow-md border border-gray-100 p-6'>
//           <Bar data={incidentData} />
//         </div>
//       </div>
//     </div>
//   );
// }

// export default Home;

import React, { useState } from 'react';
import { CiCamera } from "react-icons/ci";
import { ImNotification } from "react-icons/im";
import { TbActivityHeartbeat } from "react-icons/tb";
import { FaUserGroup } from "react-icons/fa6";
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { useNavigate } from 'react-router-dom';
// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const stats = [
  { icon: <CiCamera />, title: "Total Cameras", value: "5", description: "4 offline, 8 online" },
  { icon: <FaUserGroup />, title: "Active Detections", value: "25", description: "+15% from last hour" },
  { icon: <ImNotification />, title: "Current Alerts", value: "7", description: "+2 new alerts" },
  { icon: <TbActivityHeartbeat />, title: "System Status", value: "Optimal", description: "All systems operational" },
];

const incidents = [
  { title: "Unauthorized Access", location: "Front Entrance", time: "14:35", severity: "high" },
  { title: "Vehicle Stopped", location: "Parking Lot", time: "10:35", severity: "low" },
  { title: "Person Detected", location: "Restricted Area", time: "14:35", severity: "medium" },
  { title: "Motion Detected", location: "Storage Room", time: "14:35", severity: "low" },
];

const systemStatus = [
  { title: "Video Processing", status: "Operational", statusClass: "bg-green-100 text-green-500" },
  { title: "Object Detection", status: "Operational", statusClass: "bg-green-100 text-green-500" },
  { title: "Facial Recognition", status: "Degraded", statusClass: "bg-yellow-100/50" },
  { title: "License Plate Reader", status: "Offline", statusClass: "bg-red-100/50 text-red-500" },
];

// Data for the Weekly Incident Report Bar Graph
const incidentData = {
  labels: ['March 1', 'March 2', 'March 3', 'March 4', 'March 5', 'March 6', 'March 7'],
  datasets: [
    {
      label: 'Incidents Reported',
      data: [3, 5, 2, 6, 8, 4, 7],
      backgroundColor: 'rgba(75, 192, 192, 0.2)',
      borderColor: 'rgb(75, 192, 192)',
      borderWidth: 1,
    },
  ],
};

// Camera data with video URLs
const cameras = [
  {
    id: 1,
    name: "Front Entrance",
    videoUrl: "https://developer-blogs.nvidia.com/wp-content/uploads/2022/12/Figure8-output_blurred-compressed.gif",
    details: { people: 4, vehicles: 1, alerts: 4, objects: 5 },
  },
  {
    id: 2,
    name: "Parking Lot",
    videoUrl: "https://user-images.githubusercontent.com/11428131/139924111-58637f2e-f2f6-42d8-8812-ab42fece92b4.gif",
    details: { people: 2, vehicles: 3, alerts: 1, objects: 2 },
  },
  {
    id: 3,
    name: "Restricted Area",
    videoUrl: "https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/gif-people-in-store-bounding-boxes.gif",
    details: { people: 0, vehicles: 0, alerts: 0, objects: 1 },
  },
  {
    id: 4,
    name: "Storage Room",
    videoUrl: "https://user-images.githubusercontent.com/11428131/137016574-0d180d9b-fb9a-42a9-94b7-fbc0dbc18560.gif",
    details: { people: 1, vehicles: 0, alerts: 2, objects: 3 },
  },
];

function Home() {
  const [selectedCamera, setSelectedCamera] = useState(cameras[0]); // Default to Front Entrance
  const navigate = useNavigate();
  // Handle camera selection
  const handleCameraSelect = (camera) => {
    setSelectedCamera(camera);
  };

  // Simulate navigation to the "View All Cameras" page
  const handleViewAll = () => {
    // alert("Navigating to the 'View All Cameras' page...");
    // You can replace this with actual navigation logic (e.g., using React Router)
    navigate("/Live-Feed")
  };

  return (
    <div className='bg-gray-50 p-5 min-h-screen'>
      {/* Header Section */}
      <div className='flex justify-between items-center mb-8'>
        <div>
          <h1 className='text-2xl font-bold text-gray-800'>Security Dashboard</h1>
          <p className='text-gray-500'>Monitor your security system status and activities</p>
        </div>
        <div className='flex gap-4'>
          <button className='px-6 h-10 border border-gray-300 shadow-sm hover:shadow-md hover:bg-gray-100 rounded-lg transition-all duration-200'>
            Today
          </button>
          <button className='px-6 h-10 bg-blue-500 text-white shadow-sm hover:shadow-md hover:bg-blue-600 rounded-lg transition-all duration-200'>
            View Report
          </button>
        </div>
      </div>

      {/* Stats Section */}
      <div className='grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8'>
        {stats.map(({ icon, title, value, description }) => (
          <div
            className='bg-white p-6 rounded-xl shadow-md hover:shadow-lg transition-all duration-200 border border-gray-100'
            key={title}
          >
            <div className='text-3xl text-blue-500 mb-4'>{icon}</div>
            <h2 className='text-lg font-semibold text-gray-700'>{title}</h2>
            <p className='text-2xl font-bold text-gray-900'>{value}</p>
            <p className='text-sm text-gray-500'>{description}</p>
          </div>
        ))}
      </div>

      {/* Main Content Section */}
      <div className='lg:flex gap-6'>
        {/* Live Security Feed */}
        <div className='lg:w-[70%] bg-white rounded-xl shadow-md border border-gray-100 p-6 mb-6 lg:mb-0'>
          <div className='flex justify-between items-center mb-6'>
            <h1 className='text-xl font-bold text-gray-800'>Live Security Feed</h1>
            <button className='px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-all duration-200'>
              Live View
            </button>
          </div>
          <div className='relative rounded-xl overflow-hidden'>
            <img
              src={selectedCamera.videoUrl}
              alt="Live Security Feed"
              className='w-full h-auto rounded-lg'
            />
            <div className='absolute bottom-4 left-4 bg-black bg-opacity-50 text-white p-3 rounded-lg'>
              <h2 className='text-lg font-semibold'>{selectedCamera.name} • LIVE</h2>
              <div className='grid grid-cols-2 gap-2 mt-2'>
                <p>People: {selectedCamera.details.people}</p>
                <p>Vehicles: {selectedCamera.details.vehicles}</p>
                <p>Alerts: {selectedCamera.details.alerts}</p>
                <p>Objects: {selectedCamera.details.objects}</p>
              </div>
            </div>
          </div>

          {/* Available Cameras */}
          <div className='mt-6'>
            <div className='flex justify-between items-center mb-4'>
              <h1 className='text-lg font-semibold text-gray-800'>Available Cameras</h1>
              <button
                className='text-blue-500 hover:text-blue-600 transition-all duration-200'
                onClick={handleViewAll}
              >
                View All
              </button>
            </div>
            <div className='grid md:grid-cols-2 lg:grid-cols-4 gap-4'>
              {cameras.map((camera) => (
                <div
                  key={camera.id}
                  className={`bg-blue-900 rounded-lg p-3 text-white hover:bg-blue-800 transition-all duration-200 cursor-pointer ${
                    selectedCamera.id === camera.id ? 'ring-2 ring-blue-500' : ''
                  }`}
                  onClick={() => handleCameraSelect(camera)}
                >
                  <div className='text-xs bg-white text-black p-1 rounded-md text-center'>
                    <p>220</p>
                    <p>180</p>
                  </div>
                  <p className='mt-2 text-sm'>{camera.name}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Sidebar */}
        <div className='lg:w-[30%]'>
          {/* Detection Trends */}
          <div className='bg-white rounded-xl shadow-md border border-gray-100 p-6 mb-6'>
            <h1 className='text-xl font-bold text-gray-800'>Detection Trends</h1>
            <p className='text-gray-500 mb-4'>Today's activity</p>
            <div className='bg-gray-100 h-32 rounded-lg flex items-center justify-center text-gray-500'>
              Chart Placeholder
            </div>
          </div>

          {/* Recent Incidents */}
          <div className='bg-white rounded-xl shadow-md border border-gray-100 p-6 mb-6'>
            <h1 className='text-xl font-bold text-gray-800'>Recent Incidents</h1>
            <p className='text-gray-500 mb-4'>Last 24 hours</p>
            {incidents.map(({ title, location, time, severity }) => (
              <div className='flex justify-between items-center mb-3' key={title}>
                <div>
                  <h2 className='font-medium text-gray-700'>{title}</h2>
                  <p className='text-sm text-gray-500'>{location} • {time}</p>
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
            <button className='w-full text-center text-blue-500 hover:text-blue-600 mt-4 transition-all duration-200'>
              View All Incidents
            </button>
          </div>

          {/* System Status */}
          <div className='bg-white rounded-xl shadow-md border border-gray-100 p-6'>
            <h1 className='text-xl font-bold text-gray-800'>System Status</h1>
            <p className='text-gray-500 mb-4'>All systems operational</p>
            {systemStatus.map(({ title, status, statusClass }) => (
              <div className='flex justify-between items-center mb-3' key={title}>
                <p className='text-sm text-gray-700'>{title}</p>
                <span className={`px-3 py-1 text-xs rounded-full ${statusClass}`}>
                  {status}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Weekly Incident Report */}
      <div className='mt-8'>
        <h1 className='text-xl font-bold text-gray-800'>Weekly Incident Report</h1>
        <p className='text-gray-500 mb-4'>March 1 - March 7, 2025</p>
        <div className='bg-white rounded-xl shadow-md border border-gray-100 p-6'>
          <Bar data={incidentData} />
        </div>
      </div>
    </div>
  );
}

export default Home;