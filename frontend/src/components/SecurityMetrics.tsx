import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { UserIcon, ShieldCheckIcon, MapPinIcon, AlertCircleIcon, Users, UserX } from 'lucide-react';

interface Metrics {
  totalPeople: number;
  staffCount: number;
  customerCount: number;
  maleCount: number;
  femaleCount: number;
  alertsToday: number;
  restrictedAreaCount: number;
}

const SecurityMetrics: React.FC = () => {
  const [metrics, setMetrics] = useState<Metrics>({
    totalPeople: 0,
    staffCount: 0,
    customerCount: 0,
    maleCount: 0,
    femaleCount: 0,
    alertsToday: 0,
    restrictedAreaCount: 0
  });
  
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await axios.get('/api/analytics/current');
        setMetrics(response.data);
      } catch (error) {
        console.error('Error fetching metrics:', error);
      }
    };
    
    fetchMetrics();
    
    // Poll for updates every 10 seconds
    const interval = setInterval(fetchMetrics, 10000);
    return () => clearInterval(interval);
  }, []);
  
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>Real-Time Security Metrics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="flex flex-col items-center p-3 bg-white rounded-lg shadow">
            <Users className="h-8 w-8 text-blue-500 mb-2" />
            <h3 className="text-xl font-bold">{metrics.totalPeople}</h3>
            <p className="text-gray-500 text-sm">Total People</p>
          </div>
          
          <div className="flex flex-col items-center p-3 bg-white rounded-lg shadow">
            <ShieldCheckIcon className="h-8 w-8 text-green-500 mb-2" />
            <h3 className="text-xl font-bold">{metrics.staffCount}</h3>
            <p className="text-gray-500 text-sm">Staff</p>
          </div>
          
          <div className="flex flex-col items-center p-3 bg-white rounded-lg shadow">
            <UserIcon className="h-8 w-8 text-purple-500 mb-2" />
            <h3 className="text-xl font-bold">{metrics.customerCount}</h3>
            <p className="text-gray-500 text-sm">Customers</p>
          </div>
          
          <div className="flex flex-col items-center p-3 bg-white rounded-lg shadow">
            <AlertCircleIcon className="h-8 w-8 text-red-500 mb-2" />
            <h3 className="text-xl font-bold">{metrics.alertsToday}</h3>
            <p className="text-gray-500 text-sm">Alerts Today</p>
          </div>
        </div>
        
        <div className="mt-4 grid grid-cols-2 gap-4">
          <div className="bg-white p-3 rounded-lg shadow">
            <h3 className="font-medium text-gray-700 mb-2">Gender Distribution</h3>
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full bg-blue-500 mr-2"></div>
                <span className="text-sm">Male: {metrics.maleCount}</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full bg-pink-500 mr-2"></div>
                <span className="text-sm">Female: {metrics.femaleCount}</span>
              </div>
            </div>
            <div className="w-full h-2 bg-gray-200 rounded-full mt-2">
              <div 
                className="h-full bg-blue-500 rounded-full" 
                style={{ 
                  width: `${metrics.totalPeople ? (metrics.maleCount / metrics.totalPeople) * 100 : 0}%` 
                }}
              ></div>
            </div>
          </div>
          
          <div className="bg-white p-3 rounded-lg shadow">
            <h3 className="font-medium text-gray-700 mb-2">Restricted Areas</h3>
            <div className="flex items-center">
              <MapPinIcon className="h-5 w-5 text-red-500 mr-2" />
              <span>
                <span className="font-bold">{metrics.restrictedAreaCount}</span> unauthorized presence
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default SecurityMetrics;