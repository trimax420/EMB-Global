// ...existing code...

// Update the video component to properly handle S3 video
const VideoComponent = ({ videoUrl }) => {
  return (
    <div className="video-container">
      <video 
        controls 
        autoPlay 
        muted
        playsInline
        className="video-player"
        crossOrigin="anonymous"
      >
        <source src={videoUrl} type="video/mp4" />
        Your browser does not support the video tag.
      </video>
    </div>
  );
};

// In your main component where you use the video:
// ...existing code...
<VideoComponent videoUrl="https://your-s3-bucket-url/your-video.mp4" />
// ...existing code...

// Add this CSS to ensure the video is visible
const styles = {
  videoContainer: {
    width: "100%",
    maxWidth: "800px",
    margin: "0 auto",
  },
  videoPlayer: {
    width: "100%",
    height: "auto",
  }
};
// ...existing code...
