import React from 'react';
import { Routes, Route } from 'react-router-dom'; // Don't wrap with BrowserRouter here
import Layout from './components/Layout';
import Home from './pages/Home';
import Cameras from './pages/Camera';
import SystemStatus from './pages/SystemStatus';
import Datacollection from './pages/Datacollection';
import DailyReport from './pages/DailyReport';
import AlertsPage from './pages/Alerts';
import BillingActivityPage from './pages/BillingActivityPage';
import SecurityDashboard from './pages/SecurityDashboard';
import CustomerDemographics from './pages/CustomerDemographics';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="Live-Feed" element={<Cameras />} />
        <Route path="dailyreport" element={<DailyReport />} />
        <Route path="alerts" element={<AlertsPage />} />
        <Route path="System-Status" element={<SystemStatus />} />
        <Route path="datacollection" element={<Datacollection />} />
        <Route path="BillingActivityPage" element={<BillingActivityPage />} />
        <Route path="security-dashboard" element={<SecurityDashboard />} />
        <Route path="customer-demographics" element={<CustomerDemographics />} />
      </Route>
    </Routes>
  );
}

export default App;
