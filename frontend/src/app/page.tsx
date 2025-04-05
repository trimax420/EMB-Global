"use client"

import React from 'react';
import DashboardLayout from '../components/DashboardLayout';
import SecurityMetrics from '../components/SecurityMetrics';
import SecurityCameraFeed from '../components/SecurityCameraFeed';
import AlertsPanel from '../components/AlertsPanel';
import TrafficHeatmap from '../components/TrafficHeatmap';
import StoreMap from '../components/StoreMap';

export default function Dashboard() {
  return (
    <DashboardLayout>
      <div className="grid grid-cols-12 gap-4">
        {/* Main video feed */}
        <div className="col-span-8 lg:col-span-6 row-span-2">
          <SecurityCameraFeed />
        </div>
        
        {/* Real-time metrics */}
        <div className="col-span-4 lg:col-span-6">
          <SecurityMetrics />
        </div>
        
        {/* Alerts panel */}
        <div className="col-span-4 lg:col-span-6">
          <AlertsPanel />
        </div>
        
        {/* Heat map */}
        <div className="col-span-6 lg:col-span-4">
          <TrafficHeatmap />
        </div>
        
        {/* Store map */}
        <div className="col-span-6 lg:col-span-8">
          <StoreMap />
        </div>
      </div>
    </DashboardLayout>
  );
}