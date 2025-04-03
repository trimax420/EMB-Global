import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Divider,
} from '@mui/material';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  ResponsiveContainer, 
  Tooltip 
} from 'recharts';

const StatsSummary = ({ 
  websocketRef, 
  theftCount = 0, 
  loiteringCount = 0, 
  lastDetectionTime = null 
}) => {
  const [stats, setStats] = useState({
    totalThefts: theftCount,
    totalLoitering: loiteringCount,
    hourlyData: Array(24).fill().map((_, i) => ({
      hour: i,
      thefts: 0,
      loitering: 0
    }))
  });
  
  // Update stats when props change
  useEffect(() => {
    const hour = lastDetectionTime ? lastDetectionTime.getHours() : new Date().getHours();
    
    setStats(prevStats => {
      // Only update if the counts have changed
      if (prevStats.totalThefts === theftCount && prevStats.totalLoitering === loiteringCount) {
        return prevStats;
      }
      
      // Create a copy of the hourly data
      const newHourlyData = [...prevStats.hourlyData];
      
      // Update the hourly data for the current hour
      if (theftCount > prevStats.totalThefts) {
        newHourlyData[hour] = {
          ...newHourlyData[hour],
          thefts: newHourlyData[hour].thefts + (theftCount - prevStats.totalThefts)
        };
      }
      
      if (loiteringCount > prevStats.totalLoitering) {
        newHourlyData[hour] = {
          ...newHourlyData[hour],
          loitering: newHourlyData[hour].loitering + (loiteringCount - prevStats.totalLoitering)
        };
      }
      
      return {
        totalThefts: theftCount,
        totalLoitering: loiteringCount,
        hourlyData: newHourlyData
      };
    });
  }, [theftCount, loiteringCount, lastDetectionTime]);
  
  // Listen for WebSocket updates to keep stats current
  useEffect(() => {
    if (!websocketRef || !websocketRef.current) return;
    
    const handleMessage = (message) => {
      try {
        // Check if message is already parsed
        const data = typeof message === 'string' ? JSON.parse(message) : message;
        
        // Handle new incidents for stats
        if (data.type === 'new_incident') {
          const incidentType = data.incident?.type;
          const timestamp = data.incident?.timestamp 
            ? new Date(data.incident.timestamp) 
            : new Date();
          const hour = timestamp.getHours();
          
          setStats(prevStats => {
            // Create a copy of the hourly data
            const newHourlyData = [...prevStats.hourlyData];
            
            // Update the appropriate hour's data
            if (incidentType === 'theft') {
              newHourlyData[hour] = {
                ...newHourlyData[hour],
                thefts: newHourlyData[hour].thefts + 1
              };
              
              return {
                ...prevStats,
                hourlyData: newHourlyData
              };
            } 
            else if (incidentType === 'loitering') {
              newHourlyData[hour] = {
                ...newHourlyData[hour],
                loitering: newHourlyData[hour].loitering + 1
              };
              
              return {
                ...prevStats,
                hourlyData: newHourlyData
              };
            }
            
            return prevStats;
          });
        }
      } catch (err) {
        console.error('Error processing WebSocket message', err);
      }
    };
    
    // Check if lastMessage exists in the ref
    if (websocketRef.current.lastMessage) {
      handleMessage(websocketRef.current.lastMessage);
    }
    
    return () => {
      // No cleanup needed as we're not adding event listeners directly
    };
  }, [websocketRef]);

  // Format hour labels for the chart
  const formatHour = (hour) => {
    if (hour === 0) return '12am';
    if (hour === 12) return '12pm';
    return hour < 12 ? `${hour}am` : `${hour-12}pm`;
  };
  
  // Get current hour's data for progress bars
  const currentHour = new Date().getHours();
  const currentHourData = stats.hourlyData[currentHour];
  
  // Format last detection time
  const formattedLastDetection = lastDetectionTime 
    ? lastDetectionTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : 'None';

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Detection Stats
      </Typography>
      
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Total Theft Events
              </Typography>
              <Typography variant="h5" component="div" color="error">
                {stats.totalThefts}
              </Typography>
              <Box sx={{ mt: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  Current Hour
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={Math.min(currentHourData.thefts * 10, 100)} 
                  color="error"
                  sx={{ height: 8, borderRadius: 1 }}
                />
                <Typography variant="caption" color="text.secondary">
                  {currentHourData.thefts} detected
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Total Loitering Events
              </Typography>
              <Typography variant="h5" component="div" color="primary">
                {stats.totalLoitering}
              </Typography>
              <Box sx={{ mt: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  Current Hour
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={Math.min(currentHourData.loitering * 10, 100)} 
                  color="primary"
                  sx={{ height: 8, borderRadius: 1 }}
                />
                <Typography variant="caption" color="text.secondary">
                  {currentHourData.loitering} detected
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      {/* Last detection time */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="caption" color="text.secondary">
          Last Detection: {formattedLastDetection}
        </Typography>
      </Box>
      
      <Divider sx={{ mb: 2 }} />
      
      {/* Hourly detection chart */}
      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        Hourly Detection Trends
      </Typography>
      <ResponsiveContainer width="100%" height={150}>
        <BarChart
          data={stats.hourlyData.slice(Math.max(0, currentHour - 11), currentHour + 1)}
          margin={{ top: 5, right: 5, left: -20, bottom: 5 }}
        >
          <XAxis 
            dataKey="hour" 
            tickFormatter={formatHour}
            tick={{ fontSize: 10 }}
          />
          <YAxis hide={true} />
          <Tooltip 
            formatter={(value, name) => [value, name === 'thefts' ? 'Theft Events' : 'Loitering Events']}
            labelFormatter={(hour) => `Hour: ${formatHour(hour)}`}
          />
          <Bar dataKey="thefts" fill="#f44336" name="Theft" />
          <Bar dataKey="loitering" fill="#2196f3" name="Loitering" />
        </BarChart>
      </ResponsiveContainer>
    </Paper>
  );
};

export default StatsSummary;
