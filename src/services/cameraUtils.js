// src/utils/cameraUtils.js

/**
 * Utility functions for working with camera feeds and videos
 */

/**
 * Get the file extension from a URL or file path
 * @param {string} url - URL or file path
 * @returns {string} - File extension
 */
export const getFileExtension = (url) => {
    if (!url) return '';
    return url.split('.').pop().toLowerCase();
  };
  
  /**
   * Determine video MIME type based on file extension
   * @param {string} url - URL or file path
   * @returns {string} - MIME type
   */
  export const getVideoMimeType = (url) => {
    const extension = getFileExtension(url);
    
    switch (extension) {
      case 'mp4':
        return 'video/mp4';
      case 'webm':
        return 'video/webm';
      case 'ogg':
        return 'video/ogg';
      case 'mov':
        return 'video/quicktime';
      case 'avi':
        return 'video/x-msvideo';
      case 'mkv':
        return 'video/x-matroska';
      default:
        return 'video/mp4'; // Default to MP4
    }
  };
  
  /**
   * Check if a file is a valid video
   * @param {string} url - URL or file path
   * @returns {boolean} - Whether the file is a valid video
   */
  export const isVideoFile = (url) => {
    const videoExtensions = ['mp4', 'webm', 'ogg', 'mov', 'avi', 'mkv'];
    const extension = getFileExtension(url);
    return videoExtensions.includes(extension);
  };
  
  /**
   * Get the absolute path to a video file
   * @param {string} relativePath - Relative path to video file
   * @returns {string} - Absolute path
   */
  export const getVideoPath = (relativePath) => {
    // If it's already an absolute URL, return it
    if (relativePath.startsWith('http://') || relativePath.startsWith('https://')) {
      return relativePath;
    }
    
    // If it's a relative path, prepend the base URL
    const baseUrl = import.meta.env.BASE_URL || '/';
    return `${baseUrl}${relativePath.startsWith('/') ? relativePath.slice(1) : relativePath}`;
  };
  
  /**
   * Convert a video to a poster image (thumbnail)
   * Note: This requires a server-side implementation or a video processing API
   * @param {string} videoUrl - URL of the video
   * @returns {Promise<string>} - URL of the generated poster image
   */
  export const generateVideoPoster = async (videoUrl) => {
    // In a real implementation, you would call a server endpoint
    // that extracts a frame from the video and returns it as an image
    
    // For simplicity, we'll return a placeholder
    return `https://via.placeholder.com/640x360/333333/FFFFFF?text=Video+Thumbnail`;
  };
  
  /**
   * Get cameras with local video file paths
   * @returns {Array} - Array of camera objects with local video paths
   */
  export const getLocalVideoCameras = () => {
    return [
      {
        id: 1,
        name: "Store Entrance",
        videoUrl: "E:\code\EMB Global\public\videos\store_entrance.mp4",
        details: { people: 4, vehicles: 0, alerts: 2, objects: 5 },
        status: "online",
        resolution: { width: 640, height: 480 },
        location: "North Wing",
        capabilities: ["face_detection", "theft_detection", "loitering_detection"]
      },
      {
        id: 2,
        name: "Electronics Section",
        videoUrl: "E:\code\EMB Global\public\videos\electronics_section.mp4",
        details: { people: 2, vehicles: 0, alerts: 1, objects: 8 },
        status: "online",
        resolution: { width: 640, height: 480 },
        location: "West Wing", 
        capabilities: ["theft_detection"]
      },
      {
        id: 3,
        name: "Checkout Area",
        videoUrl: "E:\code\EMB Global\public\videos\checkout.mp4",
        details: { people: 3, vehicles: 0, alerts: 0, objects: 4 },
        status: "online",
        resolution: { width: 640, height: 480 },
        location: "South Wing",
        capabilities: ["theft_detection", "face_detection"]
      },
      {
        id: 4,
        name: "Parking Lot",
        videoUrl: "/videos/parking_lot.mp4",
        details: { people: 1, vehicles: 3, alerts: 0, objects: 2 },
        status: "offline",
        resolution: { width: 640, height: 480 },
        location: "Outdoor",
        capabilities: ["loitering_detection"]
      }
    ];
  };
  
  export default {
    getFileExtension,
    getVideoMimeType,
    isVideoFile,
    getVideoPath,
    generateVideoPoster,
    getLocalVideoCameras
  };