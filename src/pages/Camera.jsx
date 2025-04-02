import React, { useState } from "react";

const Cameras = () => {
  const [filter, setFilter] = useState("all");
  const [fullscreenCamera, setFullscreenCamera] = useState(null);

  // Dummy camera data
  const cameraList = [
    { id: 1, name: "Front Entrance", status: "online", streamUrl: "https://developer-blogs.nvidia.com/wp-content/uploads/2022/12/Figure8-output_blurred-compressed.gif" },
    { id: 2, name: "Main Hall", status: "online", streamUrl: "https://user-images.githubusercontent.com/11428131/139924111-58637f2e-f2f6-42d8-8812-ab42fece92b4.gif" },
    { id: 3, name: "Zone 2", status: "online", streamUrl: "https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/gif-people-in-store-bounding-boxes.gif" },
    { id: 4, name: "Parking Lot", status: "offline", streamUrl: "" },
  ];

  // Filter cameras based on status
  const filteredCameras =
    filter === "all"
      ? cameraList
      : cameraList.filter((camera) => camera.status === filter);

  // Exit fullscreen mode
  const exitFullscreen = () => {
    setFullscreenCamera(null);
  };

  return (
    <div className="p-6">
      {/* Header Section */}
      <h1 className="text-3xl font-bold mb-6">Live Camera Feeds</h1>

      {/* Filters Section */}
      <div className="mb-6">
        <button
          onClick={() => setFilter("all")}
          className={`mr-3 px-4 py-2 rounded ${
            filter === "all" ? "bg-blue-500 text-white" : "bg-gray-200"
          }`}
        >
          All
        </button>
        <button
          onClick={() => setFilter("online")}
          className={`mr-3 px-4 py-2 rounded ${
            filter === "online" ? "bg-green-500 text-white" : "bg-gray-200"
          }`}
        >
          Online
        </button>
        <button
          onClick={() => setFilter("offline")}
          className={`px-4 py-2 rounded ${
            filter === "offline" ? "bg-red-500 text-white" : "bg-gray-200"
          }`}
        >
          Offline
        </button>
      </div>

      {/* Fullscreen Camera View */}
      {fullscreenCamera && (
        <div className="fixed inset-0 bg-black z-50 flex items-center justify-center">
          <div className="relative w-full h-full">
            <img
              src={fullscreenCamera.streamUrl}
              alt={fullscreenCamera.name}
              className="w-full h-full object-cover"
            />
            <button
              onClick={exitFullscreen}
              className="absolute top-4 right-4 bg-gray-800 text-white px-4 py-2 rounded hover:bg-gray-700"
            >
              Exit Fullscreen
            </button>
          </div>
        </div>
      )}

      {/* Camera Grid */}
      {!fullscreenCamera && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filteredCameras.map((camera) => (
            <div
              key={camera.id}
              className={`relative bg-gray-50 p-4 rounded-lg shadow-md ${
                camera.status === "online" ? "border-4 border-green-500" : "border-4 border-red-500"
              }`}
            >
              <img
                src={camera.streamUrl}
                alt={camera.name}
                className="w-full h-48 object-cover rounded-lg cursor-pointer"
                onClick={() => setFullscreenCamera(camera)}
              />
              <h2 className="mt-2 font-semibold text-lg">{camera.name}</h2>
              <p className="text-sm">
                Status:{" "}
                <span
                  className={`uppercase ${
                    camera.status === "online" ? "text-green-600" : "text-red-600"
                  }`}
                >
                  {camera.status}
                </span>
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Cameras;