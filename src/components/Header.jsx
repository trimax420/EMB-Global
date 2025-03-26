// import React from 'react'
// import {GoBell} from 'react-icons/go'

// function Header() {
//   return (
//     <div className='flex justify-between items-center p-4'>
//       <div className='ml-10'>
//         <h1 className='text-xs'>Welcome Back!</h1>
//         <p className='text-lg font-semibold'>Alexia</p>
//       </div>
//       <div className='flex items-center space-x-5'>
//         <div className='hidden md:flex'>
//         <input type='text' 
//         placeholder='search'
//         className='bg-indigo-100/30 px-4 py-2 rounded-lg focus:outline focus:ring-2 focus:ring-indigo-600'
//         />
//         </div>
        
//       </div>

//     </div>
//   )
// }

// export default Header


import React from 'react';
import { GoBell } from 'react-icons/go';
import { FaSearch } from 'react-icons/fa';

function Header() {
  return (
    <div className="flex justify-between items-center p-4 bg-white rounded-xl">
      {/* Left Section */}
      <div className="ml-10">
        <h1 className="text-sm text-gray-500">Welcome Back!</h1>
        <p className="text-xl font-bold text-gray-800">Security Dashboard</p>
      </div>

      {/* Right Section */}
      <div className="flex items-center space-x-6">
        {/* Notification Icon */}
        <div className="relative">
          <GoBell className="text-3xl text-indigo-600 hover:text-indigo-800 transition duration-300" />
          <div className="absolute top-0 right-0 bg-red-500 text-white text-xs font-semibold rounded-full w-4 h-4 flex items-center justify-center">
            3
          </div>
        </div>

        {/* Search Input (Hidden on smaller screens) */}
        <div className="hidden md:flex items-center relative">
          <input
            type="text"
            placeholder="Search"
            className="bg-indigo-100 pl-10 pr-4 py-2 rounded-full focus:outline-none focus:ring-2 focus:ring-indigo-600 placeholder-gray-500 text-sm w-60"
          />
          <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500" />
        </div>
      </div>
    </div>
  );
}

export default Header;
