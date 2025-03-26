import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Paper,
  LinearProgress,
  Stack,
  Card,
  CardContent,
  Grid
} from '@mui/material';
import { CloudUpload as CloudUploadIcon } from '@mui/icons-material';
import { videoService } from '../services/api';

const VideoUpload = () => {
  const [file, setFile] = useState(null);
  const [detectionType, setDetectionType] = useState('theft');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);

  // Check video processing status
  useEffect(() => {
    const checkStatus = async () => {
      if (uploadedVideo && uploadedVideo.video_id) {
        try {
          const data = await videoService.getVideoStatus(uploadedVideo.video_id);
          setProcessingStatus(data.status);
          
          // If processing is complete, stop checking
          if (data.status === 'completed' || data.status === 'failed') {
            setUploadedVideo(null);
          }
        } catch (err) {
          console.error('Error checking video status:', err);
        }
      }
    };

    // Check status every 5 seconds if there's a video being processed
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, [uploadedVideo]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type.startsWith('video/')) {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('Please select a valid video file');
      setFile(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a video file');
      return;
    }

    setUploading(true);
    setError(null);
    setSuccess(false);
    setProcessingStatus(null);

    try {
      const data = await videoService.uploadVideo(file, detectionType);
      setSuccess(true);
      setUploadedVideo(data);
      setProcessingStatus('pending');
      setFile(null);
    } catch (err) {
      setError(err.message || 'Failed to upload video. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Video Upload
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Stack spacing={3}>
              <Typography variant="h6">
                Upload Video for Analysis
              </Typography>
              
              {error && (
                <Alert severity="error">
                  {error}
                </Alert>
              )}
              
              {success && (
                <Alert severity="success">
                  Video uploaded successfully! Processing will begin shortly.
                </Alert>
              )}

              {processingStatus && (
                <Alert severity="info">
                  Video Status: {processingStatus.toUpperCase()}
                  {processingStatus === 'completed' && ' - Check the dashboard for results'}
                </Alert>
              )}

              <FormControl fullWidth>
                <InputLabel>Detection Type</InputLabel>
                <Select
                  value={detectionType}
                  label="Detection Type"
                  onChange={(e) => setDetectionType(e.target.value)}
                >
                  <MenuItem value="theft">Theft Detection</MenuItem>
                  <MenuItem value="loitering">Loitering Detection</MenuItem>
                  <MenuItem value="face_detection">Face Detection</MenuItem>
                </Select>
              </FormControl>

              <Box
                sx={{
                  border: '2px dashed #ccc',
                  borderRadius: 1,
                  p: 3,
                  textAlign: 'center',
                  cursor: 'pointer',
                  '&:hover': {
                    borderColor: 'primary.main',
                  },
                }}
                onClick={() => document.getElementById('video-upload').click()}
              >
                <input
                  type="file"
                  id="video-upload"
                  accept="video/*"
                  style={{ display: 'none' }}
                  onChange={handleFileChange}
                />
                <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
                <Typography>
                  {file ? file.name : 'Click to select video file'}
                </Typography>
              </Box>

              {uploading && <LinearProgress />}

              <Button
                variant="contained"
                color="primary"
                onClick={handleUpload}
                disabled={!file || uploading}
                startIcon={<CloudUploadIcon />}
              >
                {uploading ? 'Uploading...' : 'Upload Video'}
              </Button>
            </Stack>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detection Types
              </Typography>
              
              <Typography variant="subtitle1" gutterBottom>
                Theft Detection
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Detects suspicious behavior related to theft, including unusual interactions with items and suspicious movements.
              </Typography>

              <Typography variant="subtitle1" gutterBottom>
                Loitering Detection
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Identifies individuals who remain in one area for an extended period, which may indicate suspicious behavior.
              </Typography>

              <Typography variant="subtitle1" gutterBottom>
                Face Detection
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Detects and tracks faces in the video, useful for identifying individuals and monitoring crowd behavior.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default VideoUpload; 