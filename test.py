import asyncio
import json
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import VideoFrame
from aiohttp import web

# Dummy ML Model (Replace this with your actual ML model)
class DummyModel:
    def process(self, frame):
        # Simulate ML processing by drawing a rectangle on the frame
        cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)
        return frame


# MLVideoStream Class
class MLVideoStream(VideoStreamTrack):
    def __init__(self, model, video_path):
        super().__init__()
        self.model = model  # Load your YOLO/ML model here
        self.video_capture = cv2.VideoCapture(video_path)  # Open the local video file
        if not self.video_capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        self.current_frame = None

    async def recv(self):
        # Read the next frame from the video file
        ret, frame = self.video_capture.read()
        if not ret:
            # If no more frames are available, raise an exception or handle accordingly
            self.video_capture.release()
            raise StopAsyncIteration("End of video file reached.")

        # Process the frame with the ML model
        results = self.model.process(frame)  # Run inference on the frame

        # Convert the processed frame to an aiortc.VideoFrame
        return VideoFrame.from_ndarray(results, format="bgr24")

    def stop(self):
        # Release the video capture resource when the stream is stopped
        if self.video_capture.isOpened():
            self.video_capture.release()
        super().stop()


# Peer Connections
pcs = set()

# Handle Offer
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    # Add video track to the peer connection
    video_path = r"C:\Users\Intelliod Master\Downloads\cheese_store_NVR_2_NVR_2_20250214160950_20250214161659_844965939.mp4" # Replace with your video file path
    model = DummyModel()
    video_stream = MLVideoStream(model=model, video_path=video_path)
    pc.addTrack(video_stream)

    # Set remote description
    await pc.setRemoteDescription(offer)

    # Create an answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


# Handle Answer
async def answer(request):
    params = await request.json()
    answer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Find the peer connection and set the remote description
    pc = pcs.pop()
    await pc.setRemoteDescription(answer)

    return web.Response(status=200)


# Graceful Shutdown
async def on_shutdown(app):
    # Close all peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


# Main Function
if __name__ == "__main__":
    app = web.Application()
    app.router.add_post("/offer", offer)
    app.router.add_post("/answer", answer)
    app.on_shutdown.append(on_shutdown)
    web.run_app(app, port=8080)