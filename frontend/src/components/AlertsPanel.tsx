// src/components/AlertsPanel.tsx
"use client";

import React, { useState, useEffect } from 'react';
import { AlertTriangle, Check, ChevronRight } from 'lucide-react';

interface Alert {
  id: number;
  timestamp: string;
  alert_type: string;
  severity: number;
  zone_id: number;
  zone_name?: string;
  description: string;
  acknowledged: boolean;
  snapshot_path?: string;
}

const AlertsPanel: React.FC = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  
  useEffect(() => {
    // In a real implementation, fetch alerts from API
    // For now, using static data for demonstration
    const mockAlerts: Alert[] = [
      {
        id: 1,
        timestamp: new Date().toISOString(),
        alert_type: 'suspicious_pose',
        severity: 4,
        zone_id: 1,
        zone_name: 'Perfume Section',
        description: 'Possible shoplifting posture detected',
        acknowledged: false
      },
      {
        id: 2,
        timestamp: new Date(Date.now() - 1000 * 60 * 5).toISOString(), // 5 minutes ago
        alert_type: 'restricted_area',
        severity: 3,
        zone_id: 3,
        zone_name: 'Storage Room',
        description: 'Unauthorized access to restricted area',
        acknowledged: false
      },
      {
        id: 3,
        timestamp: new Date(Date.now() - 1000 * 60 * 15).toISOString(), // 15 minutes ago
        alert_type: 'loitering',
        severity: 2,
        zone_id: 1,
        zone_name: 'Perfume Section',
        description: 'Person loitering for over 5 minutes',
        acknowledged: false
      }
    ];
    
    setAlerts(mockAlerts);
    
    // In a real app, set up WebSocket or polling
    const intervalId = setInterval(() => {
      // This would be replaced with an actual API call
      console.log("Checking for new alerts...");
    }, 30000);
    
    return () => {
      clearInterval(intervalId);
    };
  }, []);
  
  const acknowledgeAlert = (alertId: number) => {
    // In a real implementation, send to API
    setAlerts(alerts.filter(alert => alert.id !== alertId));
  };
  
  const getSeverityColor = (severity: number) => {
    switch (severity) {
      case 5: return 'bg-red-600';
      case 4: return 'bg-red-500';
      case 3: return 'bg-orange-500';
      case 2: return 'bg-yellow-500';
      default: return 'bg-blue-500';
    }
  };
  
  const getAlertTypeIcon = (type: string) => {
    switch (type) {
      case 'suspicious_pose':
        return <AlertTriangle className="h-5 w-5 text-orange-500" />;
      case 'loitering':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />;
      case 'restricted_area':
        return <AlertTriangle className="h-5 w-5 text-red-500" />;
      default:
        return <AlertTriangle className="h-5 w-5 text-blue-500" />;
    }
  };
  
  const formatAlertTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };
  
  return (
    <div className="bg-white rounded-lg shadow h-full flex flex-col">
      <div className="px-4 py-3 border-b flex justify-between items-center">
        <h2 className="font-semibold">Active Alerts</h2>
        <span className="text-sm font-normal bg-red-100 text-red-800 rounded-full px-2 py-1">
          {alerts.length} unacknowledged
        </span>
      </div>
      <div className="flex-grow p-4 overflow-y-auto">
        {alerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <Check className="h-12 w-12 mb-2" />
            <p>No active alerts</p>
          </div>
        ) : (
          <div className="space-y-3">
            {alerts.map(alert => (
              <div key={alert.id} className="flex items-center bg-white border rounded-lg p-3">
                <div className={`${getSeverityColor(alert.severity)} h-full w-1 rounded-full mr-3`}></div>
                <div className="flex-1">
                  <div className="flex items-center">
                    {getAlertTypeIcon(alert.alert_type)}
                    <span className="ml-2 font-medium">
                      {alert.alert_type.replace('_', ' ')}
                    </span>
                    <span className="ml-auto text-xs text-gray-500">
                      {formatAlertTime(alert.timestamp)}
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 mt-1">{alert.description}</p>
                  <p className="text-xs text-gray-500">Zone: {alert.zone_name}</p>
                </div>
                <button 
                  className="ml-2 text-green-500 hover:text-green-700"
                  onClick={() => acknowledgeAlert(alert.id)}
                >
                  <Check className="h-5 w-5" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="px-4 py-2 border-t flex justify-center">
        <button className="text-sm text-blue-500 flex items-center">
          View all alerts
          <ChevronRight className="h-4 w-4 ml-1" />
        </button>
      </div>
    </div>
  );
};

export default AlertsPanel;