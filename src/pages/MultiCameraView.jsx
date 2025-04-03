import React from 'react';
import { Grid, Box, Typography, Paper, Container, Divider, Stack } from '@mui/material';
import CameraFeed from '../components/CameraFeed';

const MultiCameraView = () => {
  const backendUrl = import.meta.env.VITE_BACKEND_URL || 'localhost:8000';
  
  // Video sources mapping to the provided files
  const videoSources = [
    {
      id: '1',
      title: 'Cheese Store',
      detectionType: 'both'
    },
    {
      id: '2',
      title: 'Cheese Section 1',
      detectionType: 'theft'
    },
    {
      id: '3',
      title: 'Cheese Section 2',
      detectionType: 'both'
    },
    {
      id: '4',
      title: 'Cleaning Section',
      detectionType: 'loitering'
    }
  ];

  return (
    <Box>
      {/* Page title - this will appear under the main app header */}
      <Typography variant="h4" gutterBottom fontWeight="bold" sx={{ mb: 3 }}>
        Multi-Camera Security Dashboard
      </Typography>
      
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Stack spacing={2}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h5" fontWeight="bold">Video Camera Feeds</Typography>
            <Typography variant="body2" color="text.secondary">
              Monitoring {videoSources.length} video sources with real-time detection
            </Typography>
          </Box>
          
          <Divider />
          
          <Grid container spacing={3}>
            {videoSources.map((source) => (
              <Grid item xs={12} md={6} key={source.id}>
                <CameraFeed 
                  cameraId={source.id}
                  detectionType={source.detectionType}
                  title={source.title}
                  height="320px"
                  backendUrl={backendUrl}
                />
              </Grid>
            ))}
          </Grid>
        </Stack>
      </Paper>
      
      <Paper elevation={1} sx={{ p: 2, mb: 4, bgcolor: 'background.paper' }}>
        <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
          Video Sources:
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Box component="ul" sx={{ pl: 2, m: 0 }}>
              <li>
                <Typography variant="caption" color="text.secondary">
                  Camera 1: cheese_store_NVR_2_NVR_2_20250214160950_20250214161659_844965939.mp4
                </Typography>
              </li>
              <li>
                <Typography variant="caption" color="text.secondary">
                  Camera 2: cheese-1.mp4
                </Typography>
              </li>
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box component="ul" sx={{ pl: 2, m: 0 }}>
              <li>
                <Typography variant="caption" color="text.secondary">
                  Camera 3: cheese-2.mp4
                </Typography>
              </li>
              <li>
                <Typography variant="caption" color="text.secondary">
                  Camera 4: Cleaning_acc_section_NVR_2_NVR_2_20250221220849_20250221221420_846094331.mp4
                </Typography>
              </li>
            </Box>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default MultiCameraView; 