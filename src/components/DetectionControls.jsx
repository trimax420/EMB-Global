import React, { useState } from 'react';
import { startTheftDetection, startLoiteringDetection, pollProcessingStatus } from '../services/detectionService';
import { 
  UserCheck, 
  ShoppingBag, 
  PlayCircle, 
  AlertCircle,
  Clock,
  CheckCircle,
  Loader
} from 'lucide-react';

const DetectionControls = ({ selectedCamera }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingType, setProcessingType] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const startTheftDetectionHandler = async () => {
    if (!selectedCamera || !selectedCamera.videoUrl) {
      setError('No camera selected');
      return;
    }

    try {
      setError(null);
      setIsProcessing(true);
      setProcessingType('theft');
      setProgress(0);
      setResult(null);

      const response = await startTheftDetection(selectedCamera.videoUrl, {
        handStayTimeChest: 1.0,
        handStayTimeWaist: 1.5,
        cameraId: selectedCamera.id
      });

      setJobId(response.job_id);

      // Start polling for status
      pollProcessingStatus(
        response.job_id,
        (statusResult) => {
          setProgress(statusResult.progress || 0);
        },
        (completedResult) => {
          setResult(completedResult);
          setIsProcessing(false);
        },
        (err) => {
          setError(err.message);
          setIsProcessing(false);
        }
      );
    } catch (err) {
      setError(err.message);
      setIsProcessing(false);
    }
  };

  const startLoiteringDetectionHandler = async () => {
    if (!selectedCamera || !selectedCamera.videoUrl) {
      setError('No camera selected');
      return;
    }

    try {
      setError(null);
      setIsProcessing(true);
      setProcessingType('loitering');
      setProgress(0);
      setResult(null);

      const response = await startLoiteringDetection(
        selectedCamera.videoUrl, 
        10, // threshold in seconds
        selectedCamera.id
      );

      setJobId(response.job_id);

      // Start polling for status
      pollProcessingStatus(
        response.job_id,
        (statusResult) => {
          setProgress(statusResult.progress || 0);
        },
        (completedResult) => {
          setResult(completedResult);
          setIsProcessing(false);
        },
        (err) => {
          setError(err.message);
          setIsProcessing(false);
        }
      );
    } catch (err) {
      setError(err.message);
      setIsProcessing(false);
    }
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
      <h2 className="text-lg font-semibold mb-4">Detection Controls</h2>
      
      {error && (
        <div className="mb-4 bg-red-100 text-red-800 p-3 rounded-lg flex items-center">
          <AlertCircle size={18} className="mr-2" />
          {error}
        </div>
      )}
      
      {result && (
        <div className="mb-4 bg-green-100 text-green-800 p-3 rounded-lg">
          <h3 className="font-medium flex items-center">
            <CheckCircle size={18} className="mr-2" />
            Processing Complete
          </h3>
          <p className="mt-2">
            {processingType === 'theft' 
              ? `Detected ${result.total_theft_incidents || 0} theft incidents` 
              : `Detected ${result.loitering_count || 0} loitering cases`}
          </p>
          {result.output_path && (
            <a 
              href={result.output_path.startsWith('http') 
                ? result.output_path 
                : `/processed/${result.output_path.split('/').pop()}`} 
              target="_blank" 
              rel="noopener noreferrer"
              className="mt-2 text-blue-600 hover:underline flex items-center"
            >
              View Processed Video
            </a>
          )}
        </div>
      )}
      
      {isProcessing && (
        <div className="mb-4">
          <div className="flex justify-between items-center text-sm text-gray-700 mb-1">
            <div className="flex items-center">
              <Loader size={16} className="mr-2 animate-spin" />
              Processing {processingType === 'theft' ? 'Theft' : 'Loitering'} Detection
            </div>
            <div>{Math.round(progress)}%</div>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>
      )}
      
      <div className="grid grid-cols-2 gap-3">
        <button
          onClick={startTheftDetectionHandler}
          disabled={isProcessing}
          className={`py-2 px-4 rounded-lg flex items-center justify-center gap-2 ${
            isProcessing
              ? 'bg-gray-100 text-gray-500 cursor-not-allowed'
              : 'bg-red-100 text-red-700 hover:bg-red-200'
          }`}
        >
          <ShoppingBag size={18} />
          Theft Detection
        </button>
        
        <button
          onClick={startLoiteringDetectionHandler}
          disabled={isProcessing}
          className={`py-2 px-4 rounded-lg flex items-center justify-center gap-2 ${
            isProcessing
              ? 'bg-gray-100 text-gray-500 cursor-not-allowed'
              : 'bg-orange-100 text-orange-700 hover:bg-orange-200'
          }`}
        >
          <UserCheck size={18} />
          Loitering Detection
        </button>
      </div>
      
      {selectedCamera && (
        <div className="mt-4 text-sm text-gray-500 flex items-center">
          <Clock size={14} className="mr-1" />
          <span>Detection will be performed on: {selectedCamera.name}</span>
        </div>
      )}
    </div>
  );
};

export default DetectionControls;