// import React from 'react'
// import Sidebar from './Sidebar'
// import Header from './Header'
// import {Outlet} from 'react-router-dom'

// function Layout() {
//   return (
//     <div>
//         <div className='flex'>
//             <Sidebar />
//             <div className='w-full ml-16 md:ml-56'>
//                 <Header />
//                 <Outlet />
//             </div>
//         </div>
//     </div>
//   )
// }

// export default Layout

import React, { useState } from 'react';
import Sidebar from './Sidebar';
import Header from './Header';
import { Outlet } from 'react-router-dom';

function Layout() {
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(true); // Controls sidebar width

  // Function to toggle sidebar state
  const toggleSidebar = () => {
    setIsSidebarExpanded((prev) => !prev);
  };

  return (
    <div className='flex h-screen'>
      {/* Sidebar */}
      <Sidebar isSidebarExpanded={isSidebarExpanded} toggleSidebar={toggleSidebar} />

      {/* Main Content */}
      <div
        className={`w-full transition-all duration-300 ease-in-out ${
          isSidebarExpanded ? 'ml-[190px]' : 'ml-[76px]' // Adjust margins based on sidebar width
        }`}
      >
        <Header />
        <div className='p-4 overflow-y-auto h-[calc(100vh-64px)]'> {/* Adjust height for scrolling */}
          <Outlet />
        </div>
      </div>
    </div>
  );
}

export default Layout;