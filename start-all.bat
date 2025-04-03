@echo off
echo Starting Security Dashboard Application...

echo Starting FastAPI Backend...
start cmd /k "cd backend && python main.py"

echo Starting React Frontend...
start cmd /k "npm run dev"

echo Both services have been started.
echo Frontend URL: http://localhost:5173
echo Backend API: http://localhost:8000
echo API Documentation: http://localhost:8000/docs 