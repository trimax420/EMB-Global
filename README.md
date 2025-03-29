# SecureView - AI-Powered Security Dashboard

SecureView is a proof-of-concept security dashboard that demonstrates the capabilities of computer vision models for retail security monitoring. The system detects security incidents such as loitering, theft attempts, and product damage in real-time, while also collecting customer demographics data.

## Features

- **Real-time Security Monitoring**: Detect and alert on loitering, theft attempts, and product damage events
- **Customer Demographics Analysis**: Collect and analyze customer gender, age groups, clothing colors, and repeat visit patterns
- **Live Camera Feed**: View security camera feeds with real-time detection overlays
- **Historical Data Analysis**: Review past incidents and demographic trends over time

## Project Structure

- `app.py`: FastAPI backend with computer vision integration
- `database.py`: Database models and operations
- `src/`: Frontend React application
  - `pages/`: React pages including SecurityDashboard and CustomerDemographics
  - `components/`: Reusable React components
  - `services/`: API service integrations

## Prerequisites

- Python 3.8+
- Node.js 14+
- PostgreSQL database
- CUDA-compatible GPU (recommended for real-time detection)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/secureview.git
cd secureview
```

2. Set up the Python environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Install YOLO models for detection:
```bash
mkdir -p models
pip install torch ultralytics
# Download pre-trained models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').save('models/yolov5s.pt')"
python -c "from ultralytics import YOLO; YOLO('yolov8n-face.pt').save('models/yolov8n-face.pt')"
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt').save('models/yolov8n-pose.pt')"
```

4. Set up the PostgreSQL database:
```bash
# Create a new database named 'security_dashboard'
# Update .env file with your database credentials
```

5. Install frontend dependencies:
```bash
npm install
```

6. Place a placeholder camera image:
```bash
# Place an image at public/images/placeholder-camera.jpg
```

## Running the Application

1. Start the backend server:
```bash
uvicorn app:app --reload
```

2. In a new terminal, start the frontend development server:
```bash
npm run dev
```

3. Open your browser and navigate to:
```
http://localhost:5173
```

## Security Dashboard

The Security Dashboard provides real-time monitoring of security incidents:

- View live camera feeds with detection overlays
- Track incident counts for loitering, theft, and product damage
- Review recent security alerts with confidence scores
- View demographics of customers in the store

## Customer Demographics

The Customer Demographics page offers detailed analysis of customer data:

- Filter by gender, age group, date, and time period
- View gender and age distribution charts
- Analyze hourly customer flow patterns
- Export demographic data for further analysis

## API Documentation

When the application is running, you can access the API documentation at:
```
http://localhost:8000/docs
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
