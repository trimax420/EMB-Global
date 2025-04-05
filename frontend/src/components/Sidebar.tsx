// src/components/Sidebar.tsx


import React from 'react';
import Link from 'next/link';
import { Shield, BarChart2, Camera, UserCheck, Map, Settings, LogOut } from 'lucide-react';

const Sidebar: React.FC = () => {
  return (
    <div className="bg-gray-900 text-white w-16 md:w-64 flex flex-col">
      {/* Logo */}
      <div className="flex items-center justify-center md:justify-start px-4 py-6">
        <Shield className="h-8 w-8 text-blue-400" />
        <span className="ml-2 text-xl font-bold hidden md:block">SecureWatch</span>
      </div>
      
      {/* Navigation */}
      <nav className="flex-1 px-2 py-4">
        <Link href="/" className="flex items-center px-2 py-3 text-gray-300 rounded hover:bg-gray-800 mb-1">
          <BarChart2 className="h-5 w-5" />
          <span className="ml-3 hidden md:block">Dashboard</span>
        </Link>
        
        <Link href="/cameras" className="flex items-center px-2 py-3 text-gray-300 rounded hover:bg-gray-800 mb-1">
          <Camera className="h-5 w-5" />
          <span className="ml-3 hidden md:block">Cameras</span>
        </Link>
        
        <Link href="/personnel" className="flex items-center px-2 py-3 text-gray-300 rounded hover:bg-gray-800 mb-1">
          <UserCheck className="h-5 w-5" />
          <span className="ml-3 hidden md:block">Personnel</span>
        </Link>
        
        <Link href="/map" className="flex items-center px-2 py-3 text-gray-300 rounded hover:bg-gray-800 mb-1">
          <Map className="h-5 w-5" />
          <span className="ml-3 hidden md:block">Store Map</span>
        </Link>
      </nav>
      
      {/* Settings & Logout */}
      <div className="px-2 py-4 border-t border-gray-800">
        <Link href="/settings" className="flex items-center px-2 py-3 text-gray-300 rounded hover:bg-gray-800 mb-1">
          <Settings className="h-5 w-5" />
          <span className="ml-3 hidden md:block">Settings</span>
        </Link>
        
        <button className="w-full flex items-center px-2 py-3 text-gray-300 rounded hover:bg-gray-800">
          <LogOut className="h-5 w-5" />
          <span className="ml-3 hidden md:block">Logout</span>
        </button>
      </div>
    </div>
  );
};

export default Sidebar;