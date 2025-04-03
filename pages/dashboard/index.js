// ...existing code...

import RealTimeDetection from '../../components/RealTimeDetection';

// ...existing code...

const Dashboard = () => {
  // ...existing code...
  
  const [realTimeDetection, setRealTimeDetection] = useState({
    isActive: false,
    type: null // 'theft' or 'loitering'
  });

  // Function to toggle real-time detection
  const toggleRealTimeDetection = (detectionType) => {
    if (realTimeDetection.isActive && realTimeDetection.type === detectionType) {
      // Turn off if the same type is clicked again
      setRealTimeDetection({ isActive: false, type: null });
    } else {
      // Turn on with the selected type
      setRealTimeDetection({ isActive: true, type: detectionType });
    }
  };

  // ...existing code...

  // Add this to your render section where you display theft detection or loitering options
  return (
    <div className="dashboard-container">
      {/* ...existing code... */}
      
      <div className="detection-controls">
        <div className="detection-option">
          <h3>Theft Detection</h3>
          <button 
            className={`toggle-button ${realTimeDetection.isActive && realTimeDetection.type === 'theft' ? 'active' : ''}`} 
            onClick={() => toggleRealTimeDetection('theft')}
          >
            {realTimeDetection.isActive && realTimeDetection.type === 'theft' ? 'Stop Real-time Detection' : 'Start Real-time Detection'}
          </button>
        </div>
        
        <div className="detection-option">
          <h3>Loitering Detection</h3>
          <button 
            className={`toggle-button ${realTimeDetection.isActive && realTimeDetection.type === 'loitering' ? 'active' : ''}`} 
            onClick={() => toggleRealTimeDetection('loitering')}
          >
            {realTimeDetection.isActive && realTimeDetection.type === 'loitering' ? 'Stop Real-time Detection' : 'Start Real-time Detection'}
          </button>
        </div>
      </div>
      
      {/* Real-time detection component */}
      {realTimeDetection.isActive && (
        <div className="real-time-detection-container">
          <RealTimeDetection 
            detectionType={realTimeDetection.type}
            isActive={realTimeDetection.isActive}
          />
        </div>
      )}
      
      {/* ...existing code... */}
    </div>
  );
};

// ...existing code...
