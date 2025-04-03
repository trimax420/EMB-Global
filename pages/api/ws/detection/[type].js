import { Server } from 'socket.io';
import AWS from 'aws-sdk';

// Initialize AWS services
const kinesis = new AWS.Kinesis({
  region: process.env.AWS_REGION,
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
});

export default function handler(req, res) {
  // This ensures we only upgrade for WebSocket requests
  if (res.socket.server.io) {
    console.log('Socket is already running');
    res.end();
    return;
  }

  // Get the detection type from the URL
  const detectionType = req.query.type;
  
  if (detectionType !== 'theft' && detectionType !== 'loitering') {
    res.status(400).json({ error: 'Invalid detection type' });
    return;
  }

  console.log(`Setting up socket for ${detectionType} detection`);
  
  const io = new Server(res.socket.server);
  res.socket.server.io = io;

  io.on('connection', (socket) => {
    console.log(`Client connected for ${detectionType} detection`);
    
    // Start streaming data specific to the detection type
    const streamName = detectionType === 'theft' ? 
      process.env.THEFT_DETECTION_STREAM : 
      process.env.LOITERING_DETECTION_STREAM;
    
    // Initialize stream consumer
    const params = {
      StreamName: streamName,
      ShardId: 'shardId-000000000000', // Adjust as needed for your Kinesis stream
      ShardIteratorType: 'LATEST'
    };
    
    let shardIterator;
    let streamInterval;
    
    // Get a shard iterator
    kinesis.getShardIterator(params, (err, data) => {
      if (err) {
        console.error('Error getting shard iterator:', err);
        return;
      }
      
      shardIterator = data.ShardIterator;
      
      // Start polling for records
      streamInterval = setInterval(() => {
        kinesis.getRecords({
          ShardIterator: shardIterator,
          Limit: 10
        }, (err, data) => {
          if (err) {
            console.error('Error getting records:', err);
            return;
          }
          
          // Update the shard iterator for next call
          shardIterator = data.NextShardIterator;
          
          // Process any received records
          if (data.Records && data.Records.length > 0) {
            data.Records.forEach(record => {
              try {
                const payload = JSON.parse(Buffer.from(record.Data).toString());
                // Send the detection data to the client
                socket.emit('detection', {
                  type: 'detection',
                  detections: payload.detections,
                  timestamp: payload.timestamp
                });
                
                // If there's image data, also send that
                if (payload.imageData) {
                  socket.emit('frame', {
                    type: 'frame',
                    imageData: payload.imageData
                  });
                }
              } catch (e) {
                console.error('Error processing record:', e);
              }
            });
          }
        });
      }, 500); // Poll every 500ms
    });
    
    // Clean up when socket disconnects
    socket.on('disconnect', () => {
      console.log(`Client disconnected from ${detectionType} detection`);
      if (streamInterval) {
        clearInterval(streamInterval);
      }
    });
  });
  
  res.end();
}
