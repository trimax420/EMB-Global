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
      </Route>
    </Routes>
  );
}

export default App;
