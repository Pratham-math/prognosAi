// App.jsx
import React, { useState } from 'react';
import { ThemeProvider } from './context/Themecontext'; // Fixed import
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Forecast from './pages/Forecast';
import Playbook from './pages/Playbook';
import Settings from './pages/Settings';
import AlertModal from './components/AlertModal';

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [alertModal, setAlertModal] = useState({
    isOpen: true,
    title: 'High AQI Alert',
    message: 'Air Quality Index has reached unhealthy levels (156). Respiratory patient surge expected in 48 hours.',
    severity: 'high'
  });

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />;
      case 'forecast':
        return <Forecast />;
      case 'playbook':
        return <Playbook />;
      case 'settings':
        return <Settings />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <ThemeProvider>
      <div className="min-h-screen transition-colors duration-300 bg-gray-50 dark:bg-gray-900">
        <Navbar currentPage={currentPage} setCurrentPage={setCurrentPage} />
        <main className="container mx-auto px-4 py-6">
          {renderPage()}
        </main>
        <AlertModal
          isOpen={alertModal.isOpen}
          onClose={() => setAlertModal(prev => ({ ...prev, isOpen: false }))}
          title={alertModal.title}
          message={alertModal.message}
          severity={alertModal.severity}
        />
      </div>
    </ThemeProvider>
  );
}

export default App;