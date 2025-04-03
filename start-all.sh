#!/bin/bash
echo "Starting Security Dashboard Application..."

echo "Starting FastAPI Backend..."
gnome-terminal -- bash -c "cd backend && python main.py" || \
xterm -e "cd backend && python main.py" || \
open -a Terminal.app ./backend && cd backend && python main.py

echo "Starting React Frontend..."
gnome-terminal -- bash -c "npm run dev" || \
xterm -e "npm run dev" || \
open -a Terminal.app . && npm run dev

echo "Both services have been started."
echo "Frontend URL: http://localhost:5173 or http://localhost:5174"
echo "Backend API: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs" 