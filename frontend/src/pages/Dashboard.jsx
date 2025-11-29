// Updated Dashboard.jsx with dark theme
import React from 'react';
import StatsCard from '../components/StatsCard';
import ChartCard from '../components/ChartCard';
import { useTheme } from '../context/Themecontext';

const Dashboard = () => {
  const { isDark } = useTheme();

  const statsData = [
    {
      title: 'Current Occupancy',
      value: '78%',
      change: '+12%',
      trend: 'up',
      icon: '🛏️',
      color: 'blue'
    },
    {
      title: 'Available Doctors',
      value: '24',
      change: '-3',
      trend: 'down',
      icon: '👨‍⚕️',
      color: 'green'
    },
    {
      title: 'Available Nurses',
      value: '42',
      change: '+2',
      trend: 'up',
      icon: '👩‍⚕️',
      color: 'green'
    },
    {
      title: 'ICU Beds Available',
      value: '8',
      change: '-4',
      trend: 'down',
      icon: '💊',
      color: 'red'
    }
  ];

  const externalIndicators = [
    { name: 'Air Quality Index', value: '156', level: 'Unhealthy', color: 'red' },
    { name: 'Temperature', value: '28°C', level: 'Normal', color: 'green' },
    { name: 'Humidity', value: '65%', level: 'High', color: 'yellow' },
    { name: 'Viral Activity', value: 'Medium', level: 'Elevated', color: 'yellow' }
  ];

  const alerts = [
    {
      id: 1,
      type: 'warning',
      title: 'High AQI Detected',
      message: 'Respiratory patient surge expected in 48 hours',
      time: '2 hours ago'
    },
    {
      id: 2,
      type: 'info',
      title: 'Staffing Alert',
      message: 'Night shift doctors below optimal level',
      time: '1 hour ago'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className={`text-3xl font-bold transition-colors ${
            isDark ? 'text-white' : 'text-gray-800'
          }`}>
            Hospital Operations Dashboard
          </h1>
          <p className={`transition-colors ${
            isDark ? 'text-gray-400' : 'text-gray-600'
          }`}>
            Real-time monitoring and AI-powered predictions
          </p>
        </div>
        <div className={`px-4 py-2 rounded-lg border transition-colors ${
          isDark
            ? 'bg-orange-900 border-orange-700'
            : 'bg-orange-100 border-orange-300'
        }`}>
          <span className={`font-semibold transition-colors ${
            isDark ? 'text-orange-300' : 'text-orange-700'
          }`}>
            🔄 Live Updates
          </span>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statsData.map((stat, index) => (
          <StatsCard key={index} {...stat} />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Patient Inflow Chart */}
        <div className="lg:col-span-2">
          <ChartCard
            title="Predicted Patient Inflow (Next 72 hours)"
            chartType="line"
            data={[
              { hour: '24h', patients: 45 },
              { hour: '48h', patients: 78 },
              { hour: '72h', patients: 62 }
            ]}
          />
        </div>

        {/* External Indicators */}
        <div className={`rounded-xl shadow-sm border p-6 transition-colors ${
          isDark 
            ? 'bg-gray-800 border-gray-700' 
            : 'bg-white border-gray-200'
        }`}>
          <h3 className={`text-lg font-semibold mb-4 transition-colors ${
            isDark ? 'text-white' : 'text-gray-800'
          }`}>
            External Indicators
          </h3>
          <div className="space-y-3">
            {externalIndicators.map((indicator, index) => (
              <div key={index} className={`p-3 rounded-lg transition-colors ${
                isDark ? 'bg-gray-700' : 'bg-gray-50'
              }`}>
                <div className="flex justify-between items-center">
                  <span className={`transition-colors ${
                    isDark ? 'text-gray-300' : 'text-gray-700'
                  }`}>
                    {indicator.name}
                  </span>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      indicator.color === 'red' 
                        ? isDark ? 'bg-red-900 text-red-300' : 'bg-red-100 text-red-700' :
                      indicator.color === 'yellow' 
                        ? isDark ? 'bg-yellow-900 text-yellow-300' : 'bg-yellow-100 text-yellow-700' :
                        isDark ? 'bg-green-900 text-green-300' : 'bg-green-100 text-green-700'
                    }`}>
                      {indicator.value}
                    </span>
                    <span className={`text-sm transition-colors ${
                      isDark ? 'text-gray-500' : 'text-gray-500'
                    }`}>
                      {indicator.level}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Alerts Section */}
      <div className={`rounded-xl shadow-sm border p-6 transition-colors ${
        isDark 
          ? 'bg-gray-800 border-gray-700' 
          : 'bg-white border-gray-200'
      }`}>
        <h3 className={`text-lg font-semibold mb-4 transition-colors ${
          isDark ? 'text-white' : 'text-gray-800'
        }`}>
          Recent Alerts
        </h3>
        <div className="space-y-3">
          {alerts.map(alert => (
            <div key={alert.id} className={`p-4 rounded-lg border-l-4 transition-colors ${
              alert.type === 'warning' 
                ? isDark 
                  ? 'border-l-orange-600 bg-orange-900' 
                  : 'border-l-orange-500 bg-orange-50'
                : isDark
                ? 'border-l-blue-600 bg-blue-900'
                : 'border-l-blue-500 bg-blue-50'
            }`}>
              <div className="flex justify-between items-start">
                <div>
                  <h4 className={`font-semibold transition-colors ${
                    isDark ? 'text-white' : 'text-gray-800'
                  }`}>
                    {alert.title}
                  </h4>
                  <p className={`text-sm mt-1 transition-colors ${
                    isDark ? 'text-gray-300' : 'text-gray-600'
                  }`}>
                    {alert.message}
                  </p>
                </div>
                <span className={`text-xs transition-colors ${
                  isDark ? 'text-gray-500' : 'text-gray-500'
                }`}>
                  {alert.time}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;