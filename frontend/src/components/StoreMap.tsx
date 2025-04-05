// src/components/StoreMap.tsx


import React, { useState, useEffect } from 'react';
import { UserIcon, Users } from 'lucide-react';

interface PersonTrack {
  track_id: string;
  current_zone_id: number;
  is_staff: boolean;
  gender: string;
  position: { x: number, y: number };
}

interface ZoneData {
  id: number;
  name: string;
  coordinates: number[][];
  is_restricted: boolean;
  person_count: number;
}

const StoreMap: React.FC = () => {
  const [personTracks, setPersonTracks] = useState<PersonTrack[]>([]);
  const [zones, setZones] = useState<ZoneData[]>([]);
  
  useEffect(() => {
    // In a real implementation, fetch from API
    // For now, using static data for demonstration
    const mockZones: ZoneData[] = [
      {
        id: 1,
        name: 'Perfume Section',
        coordinates: [[100, 100], [300, 100], [300, 300], [100, 300]],
        is_restricted: false,
        person_count: 3
      },
      {
        id: 2,
        name: 'Checkout',
        coordinates: [[350, 100], [450, 100], [450, 200], [350, 200]],
        is_restricted: false,
        person_count: 1
      },
      {
        id: 3,
        name: 'Storage Room',
        coordinates: [[100, 350], [250, 350], [250, 450], [100, 450]],
        is_restricted: true,
        person_count: 1
      }
    ];
    
    const mockPersonTracks: PersonTrack[] = [
      {
        track_id: '1',
        current_zone_id: 1,
        is_staff: false,
        gender: 'male',
        position: { x: 150, y: 200 }
      },
      {
        track_id: '2',
        current_zone_id: 1,
        is_staff: false,
        gender: 'female',
        position: { x: 250, y: 150 }
      },
      {
        track_id: '3',
        current_zone_id: 2,
        is_staff: true,
        gender: 'male',
        position: { x: 400, y: 150 }
      },
      {
        track_id: '4',
        current_zone_id: 3,
        is_staff: false,
        gender: 'male',
        position: { x: 175, y: 400 }
      }
    ];
    
    setZones(mockZones);
    setPersonTracks(mockPersonTracks);
    
    // In a real app, set up polling or WebSocket for updates
    const intervalId = setInterval(() => {
      // This would be replaced with an actual API call
      console.log("Updating person tracks...");
    }, 2000);
    
    return () => {
      clearInterval(intervalId);
    };
  }, []);
  
  const getZoneColor = (zone: ZoneData) => {
    if (zone.is_restricted) {
      return 'rgba(239, 68, 68, 0.2)'; // Red for restricted
    }
    
    // Color based on person count
    const occupancyRatio = zone.person_count / 10; // Assuming max capacity of 10 per zone
    if (occupancyRatio > 0.7) {
      return 'rgba(239, 68, 68, 0.1)'; // Heavy traffic - light red
    } else if (occupancyRatio > 0.4) {
      return 'rgba(245, 158, 11, 0.1)'; // Medium traffic - light orange
    } else {
      return 'rgba(16, 185, 129, 0.1)'; // Light traffic - light green
    }
  };
  
  return (
    <div className="bg-white rounded-lg shadow h-full flex flex-col">
      <div className="px-4 py-3 border-b">
        <h2 className="font-semibold">Store Layout & Tracking</h2>
      </div>
      <div className="flex-1 p-4 relative">
        <div className="h-full bg-gray-50 rounded-md overflow-hidden relative">
          {/* Store map */}
          <svg width="100%" height="100%" viewBox="0 0 500 500">
            {/* Draw zones */}
            {zones.map(zone => (
              <g key={zone.id}>
                <polygon
                  points={zone.coordinates.map(point => point.join(',')).join(' ')}
                  fill={getZoneColor(zone)}
                  stroke={zone.is_restricted ? "rgb(239, 68, 68)" : "rgb(107, 114, 128)"}
                  strokeWidth="2"
                />
                <text
                  x={zone.coordinates.reduce((sum, point) => sum + point[0], 0) / zone.coordinates.length}
                  y={zone.coordinates.reduce((sum, point) => sum + point[1], 0) / zone.coordinates.length}
                  fontSize="10"
                  textAnchor="middle"
                  fill="gray"
                >
                  {zone.name}
                </text>
              </g>
            ))}
            
            {/* Draw person markers */}
            {personTracks.map(person => (
              <g key={person.track_id} transform={`translate(${person.position.x}, ${person.position.y})`}>
                <circle
                  r="6"
                  fill={person.is_staff ? "rgb(37, 99, 235)" : 
                        person.current_zone_id === 3 ? "rgb(239, 68, 68)" : "rgb(147, 51, 234)"}
                />
                <text
                  y="-8"
                  fontSize="8"
                  textAnchor="middle"
                  fill="gray"
                >
                  {person.is_staff ? "Staff" : "Customer"}
                </text>
              </g>
            ))}
          </svg>
          
          {/* Legend */}
          <div className="absolute bottom-2 right-2 bg-white rounded-md shadow p-2 text-xs">
            <div className="flex items-center mb-1">
              <div className="w-3 h-3 rounded-full bg-purple-500 mr-1"></div>
              <span>Customer</span>
            </div>
            <div className="flex items-center mb-1">
              <div className="w-3 h-3 rounded-full bg-blue-600 mr-1"></div>
              <span>Staff</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 rounded-full bg-red-500 mr-1"></div>
              <span>Restricted Area</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StoreMap;