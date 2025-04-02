// src/App.jsx - Updated with API Provider
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { ApiProvider } from './contexts/ApiContext';
import Layout from './components/Layout';
import Home from './pages/Home';
import Cameras from './pages/Camera';
import SystemStatus from './pages/SystemStatus';
import Datacollection from './pages/Datacollection';
import DailyReport from './pages/DailyReport';
import AlertsPage from './pages/Alerts';
import Suspectlist from './pages/SuspectList';
import BillingActivityPage from './pages/BillingActivityPage';

// Global error boundary component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-100">
          <div className="bg-white p-8 rounded-lg shadow-lg max-w-md w-full">
            <h2 className="text-2xl font-bold text-red-600 mb-4">Something went wrong</h2>
            <p className="text-gray-700 mb-4">
              An error occurred in the application. Please try refreshing the page or contact support if the problem persists.
            </p>
            <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto max-h-40 mb-4">
              {this.state.error?.toString()}
            </pre>
            <button
              onClick={() => window.location.reload()}
              className="w-full bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
            >
              Refresh Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

function App() {
  return (
    <ErrorBoundary>
      <ApiProvider>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Home />} />
            <Route path="alerts" element={<AlertsPage />} />
            <Route path="suspectlist" element={<Suspectlist />} />
            {/* Uncomment these routes as needed */}
            {/* <Route path="Live-Feed" element={<Cameras />} />
            <Route path="dailyreport" element={<DailyReport />} />
            <Route path="System-Status" element={<SystemStatus />} />
            <Route path="datacollection" element={<Datacollection />} />
            <Route path="BillingActivityPage" element={<BillingActivityPage />} /> */}
          </Route>
        </Routes>
      </ApiProvider>
    </ErrorBoundary>
  );
}

export default App;