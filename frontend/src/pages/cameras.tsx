// src/pages/cameras.tsx
import React from 'react';
import DashboardLayout from '../components/DashboardLayout';
import SecurityCameraFeed from '../components/SecurityCameraFeed';

const CamerasPage: React.FC = () => {
  return (
    <DashboardLayout>
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Camera Feeds</h1>
        <p className="text-gray-600">Monitor all security cameras in real-time.</p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SecurityCameraFeed />
          <SecurityCameraFeed />
        </div>
      </div>
    </DashboardLayout>
  );
};

export default CamerasPage;