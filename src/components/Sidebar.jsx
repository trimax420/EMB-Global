import React from 'react';
import { LuBox, LuUser, LuMessageSquare, LuCalendar, LuLogOut } from 'react-icons/lu';
import { Link, useLocation } from 'react-router-dom';
import { FiShield } from 'react-icons/fi';
import { GoSidebarCollapse } from "react-icons/go";
import { GoSidebarExpand } from "react-icons/go";
import { MdOutlineDashboard } from "react-icons/md";
import { PiCameraBold } from 'react-icons/pi';         // Live Feed icon (Camera)
import { PiClipboardTextBold } from 'react-icons/pi';   // Data Collection icon
import { PiChartBarBold } from 'react-icons/pi';        // Daily Report icon
import { PiBellBold } from 'react-icons/pi';            // Alerts icon
import { MdSettings,MdAttachMoney  } from 'react-icons/md';

function Sidebar({ isSidebarExpanded, toggleSidebar }) {
  const location = useLocation(); // Get the current location

  const SIDEBAR_LINKS = [
    { id: 1, path: '/', name: 'Dashboard', icon: MdOutlineDashboard },
    { id: 2, path: '/Live-Feed', name: 'Live Feed', icon: PiCameraBold },
    { id: 3, path: '/datacollection', name: 'Data Collection', icon: PiClipboardTextBold },
    { id: 4, path: '/dailyreport', name: 'Daily Report', icon: PiChartBarBold },
    { id: 5, path: '/alerts', name: 'Alerts', icon: PiBellBold },
    { id: 6, path: '/System-Status', name: 'System Status', icon: MdSettings },
    { id: 6, path: '/BillingActivityPage', name: 'POS', icon: MdAttachMoney },
  ];

  // Mock user data
  const user = {
    name: 'John Doe',
    email: 'johndoe@example.com',
  };

  return (
    <div
      className={`w-[${isSidebarExpanded ? '224px' : '64px'}] fixed left-0 top-0 z-10 h-screen border-r pt-8 px-4 bg-white transition-all duration-500 ease-in-out`}
    >
      {/* Toggle Button */}
      <button
        onClick={toggleSidebar}
        className='absolute right-[-30px] top-[5%] transform -translate-y-1/2 text-2xl text-black flex items-center justify-center cursor-pointer'
      >
        {isSidebarExpanded ? <GoSidebarExpand /> : <GoSidebarCollapse/>}
      </button>

      {/* Logo Section */}
      <div className='md:mb-6 flex items-center'>
        <FiShield className={`text-2xl font-semibold ml-3 ${isSidebarExpanded ? 'flex' : 'ml-4'}`} />
        <h1 className={`text-2xl font-semibold ml-3 ${isSidebarExpanded ? 'flex' : 'hidden'}`}>
          SecureView
        </h1>
      </div>

      {/* Sidebar Links */}
      <ul className='mt-6 space-y-4'>
        {SIDEBAR_LINKS.map((link, index) => (
          <li
            key={index}
            className={`font-medium rounded-md px-5 hover:bg-gray-200 hover:text-indigo-500 ${location.pathname === link.path ? 'bg-gray-200 text-indigo-500' : ''}`}
          >
            <Link
              to={link.path}
              className='flex items-center md:space-x-5 justify-center md:justify-start py-3'
            >
              <span>{link.icon()}</span>
              <span className={`text-sm text-gray-500 ${isSidebarExpanded ? 'flex' : 'hidden'}`}>
                {link.name}
              </span>
            </Link>
          </li>
        ))}
      </ul>

      {/* User Info */}
      <div className='absolute bottom-5 left-5 cursor-pointer text-center'>
        <div className='flex items-center space-x-2 text-md text-gray-800 hover:bg-gray-100 hover:text-indigo-500 py-2 px-4 rounded-md justify-center md:justify-start'>
          <LuUser className='flex' />
          <div className={`flex-col ${isSidebarExpanded ? 'flex' : 'hidden'}`}>
            <h1 className='text-gray-700 font-medium'>{user.name}</h1>
            <h1 className='text-gray-500 text-xs'>{user.email}</h1>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Sidebar;
