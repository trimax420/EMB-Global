// src/components/TrafficHeatmap.tsx
"use client";

import React, { useState, useEffect } from 'react';
import { Calendar } from 'lucide-react';

interface HeatmapData {
  zone_id: number;
  heat_level: number;
}

const TrafficHeatmap: React.FC = () => {
  const [heatmapData, setHeatmapData] = useState<HeatmapData[]>([]);
  const [selectedDate, setSelectedDate] = useState<string>(
    new Date().toISOString().split('T')[0]
  );
  const [selectedHour, setSelectedHour] = useState<number | null>(
    new Date().getHours()
  );
  
  useEffect(() => {
    // In a real implementation, fetch data from API
    // For now, using static data for demonstration
    const mockHeatmapData: HeatmapData[] = [
      { zone_id: 1, heat_level: 8 },
      { zone_id: 2, heat_level: 4 },
      { zone_id: 3, heat_level: 2 }
    ];
    
    setHeatmapData(mockHeatmapData);
  }, [selectedDate, selectedHour]);
  
  const handleDateChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedDate(event.target.value);
  };
  
  const handleHourChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const hour = event.target.value === 'all' 
      ? null 
      : parseInt(event.target.value, 10);
    setSelectedHour(hour);
  };
  
  // Generate hours for dropdown
  const hours = Array.from({ length: 24 }, (_, i) => {
    const hour = i;
    const period = hour >= 12 ? 'PM' : 'AM';
    const displayHour = hour === 0 ? 12 : hour > 12 ? hour - 12 : hour;
    return { 
      value: hour, 
      label: `${displayHour}:00 ${period}`
    };
  });
  
  // Helper to get color for heat level
  const getHeatColor = (level: number) => {
    // Scale from light blue to dark blue
    const intensity = Math.min(255, Math.round((level / 10) * 255));
    return `rgb(0, ${255 - intensity}, ${255 - Math.round(intensity * 0.5)})`;
  };
  
  return (
    <div className="bg-white rounded-lg shadow h-full flex flex-col">
      <div className="px-4 py-3 border-b flex justify-between items-center">
        <h2 className="font-semibold">Traffic Heatmap</h2>
        <div className="flex space-x-2">
          <div className="relative">
            <Calendar className="h-4 w-4 absolute left-2 top-2.5 text-gray-500" />
            <input
              type="date"
              value={selectedDate}
              onChange={handleDateChange}
              className="pl-8 text-sm border rounded p-1"
            />
          </div>
          <select
            value={selectedHour === null ? 'all' : selectedHour}
            onChange={handleHourChange}
            className="text-sm border rounded p-1"
          >
            <option value="all">All Hours</option>
            {hours.map(hour => (
              <option key={hour.value} value={hour.value}>
                {hour.label}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="flex-1 p-4">
        <div className="h-full bg-gray-50 rounded-md p-4 relative">
          {/* Store layout with heatmap overlay */}
          <div className="absolute inset-0 m-4">
            {/* Example store sections */}
            <div 
              className="absolute border border-gray-300" 
              style={{ 
                top: '20%', 
                left: '30%', 
                width: '40%', 
                height: '50%',
                backgroundColor: getHeatColor(heatmapData.find(d => d.zone_id === 1)?.heat_level || 0),
                opacity: 0.7
              }}
            >
              <div className="absolute inset-0 flex items-center justify-center text-xs font-medium">
                Perfume Section
              </div>
            </div>
            
            <div 
              className="absolute border border-gray-300" 
              style={{ 
                top: '20%', 
                right: '10%', 
                width: '15%', 
                height: '20%',
                backgroundColor: getHeatColor(heatmapData.find(d => d.zone_id === 2)?.heat_level || 0),
                opacity: 0.7
              }}
            >
              <div className="absolute inset-0 flex items-center justify-center text-xs font-medium">
                Checkout
              </div>
            </div>
            
            <div 
              className="absolute border border-gray-300" 
              style={{ 
                bottom: '10%', 
                left: '10%', 
                width: '20%', 
                height: '20%',
                backgroundColor: getHeatColor(heatmapData.find(d => d.zone_id === 3)?.heat_level || 0),
                opacity: 0.7
              }}
            >
              <div className="absolute inset-0 flex items-center justify-center text-xs font-medium">
                Storage
              </div>
            </div>
          </div>
          
          {/* Legend */}
          <div className="absolute bottom-2 right-2 bg-white rounded shadow p-2 text-xs z-10">
            <div className="font-medium mb-1">Traffic Density</div>
            <div className="flex items-center mb-1">
              <div className="w-3 h-3 rounded-full mr-1" style={{ backgroundColor: getHeatColor(9) }}></div>
              <span>High</span>
            </div>
            <div className="flex items-center mb-1">
              <div className="w-3 h-3 rounded-full mr-1" style={{ backgroundColor: getHeatColor(5) }}></div>
              <span>Medium</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 rounded-full mr-1" style={{ backgroundColor: getHeatColor(2) }}></div>
              <span>Low</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrafficHeatmap;