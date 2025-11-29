// pages/Settings.jsx
import React, { useState } from 'react';
import { useTheme } from '../context/Themecontext';

const Settings = () => {
  const { isDark } = useTheme();
  const [thresholds, setThresholds] = useState({
    surgeAlert: 70,
    staffingCritical: 15,
    icuCritical: 5,
    aqiWarning: 100,
    aqiCritical: 150
  });

  const [integrations, setIntegrations] = useState({
    aqiSource: 'openweather',
    weatherAPI: 'enabled',
    hospitalDB: 'connected'
  });

  const handleThresholdChange = (key, value) => {
    setThresholds(prev => ({
      ...prev,
      [key]: parseInt(value)
    }));
  };

  const handleIntegrationToggle = (key, value) => {
    setIntegrations(prev => ({
      ...prev,
      [key]: value
    }));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className={`text-3xl font-bold transition-colors ${
          isDark ? 'text-white' : 'text-gray-800'
        }`}>
          System Settings
        </h1>
        <p className={`transition-colors ${
          isDark ? 'text-gray-400' : 'text-gray-600'
        }`}>
          Configure AI agent parameters and integrations
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Alert Thresholds */}
        <div className={`rounded-xl shadow-sm border p-6 transition-colors ${
          isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        }`}>
          <h2 className={`text-xl font-semibold mb-4 transition-colors ${
            isDark ? 'text-white' : 'text-gray-800'
          }`}>
            Alert Thresholds
          </h2>
          <div className="space-y-4">
            {[
              { key: 'surgeAlert', label: 'Surge Alert Threshold (%)', min: 50, max: 90 },
              { key: 'staffingCritical', label: 'Critical Staff Shortage', min: 5, max: 30 },
              { key: 'icuCritical', label: 'Critical ICU Beds', min: 2, max: 15 },
              { key: 'aqiWarning', label: 'AQI Warning Level', min: 50, max: 200 },
              { key: 'aqiCritical', label: 'AQI Critical Level', min: 100, max: 300 }
            ].map(setting => (
              <div key={setting.key} className="space-y-2">
                <label className={`block text-sm font-medium transition-colors ${
                  isDark ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  {setting.label}
                </label>
                <div className="flex items-center space-x-4">
                  <input
                    type="range"
                    min={setting.min}
                    max={setting.max}
                    value={thresholds[setting.key]}
                    onChange={(e) => handleThresholdChange(setting.key, e.target.value)}
                    className={`w-full h-2 rounded-lg appearance-none cursor-pointer transition-colors ${
                      isDark ? 'bg-gray-600' : 'bg-gray-200'
                    }`}
                  />
                  <span className={`text-sm font-medium w-12 transition-colors ${
                    isDark ? 'text-gray-300' : 'text-gray-600'
                  }`}>
                    {thresholds[setting.key]}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Integrations */}
        <div className={`rounded-xl shadow-sm border p-6 transition-colors ${
          isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        }`}>
          <h2 className={`text-xl font-semibold mb-4 transition-colors ${
            isDark ? 'text-white' : 'text-gray-800'
          }`}>
            Data Integrations
          </h2>
          <div className="space-y-4">
            {[
              { key: 'aqiSource', label: 'Air Quality Data Source', options: ['openweather', 'aqicn', 'government'] },
              { key: 'weatherAPI', label: 'Weather Data', options: ['enabled', 'disabled'] },
              { key: 'hospitalDB', label: 'Hospital Database', options: ['connected', 'disconnected'] }
            ].map(integration => (
              <div key={integration.key} className="space-y-2">
                <label className={`block text-sm font-medium transition-colors ${
                  isDark ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  {integration.label}
                </label>
                <div className="flex space-x-2">
                  {integration.options.map(option => (
                    <button
                      key={option}
                      onClick={() => handleIntegrationToggle(integration.key, option)}
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                        integrations[integration.key] === option
                          ? 'bg-blue-600 text-white shadow-md'
                          : isDark
                            ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                      }`}
                    >
                      {option.charAt(0).toUpperCase() + option.slice(1)}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Manual Controls */}
      <div className={`rounded-xl shadow-sm border p-6 transition-colors ${
        isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
      }`}>
        <h2 className={`text-xl font-semibold mb-4 transition-colors ${
          isDark ? 'text-white' : 'text-gray-800'
        }`}>
          Manual Controls
        </h2>
        <div className="space-y-4">
          <div className="flex space-x-4">
            <button className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors flex items-center space-x-2">
              <span>🔍</span>
              <span>Run Forecast Now</span>
            </button>
            <button className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors flex items-center space-x-2">
              <span>🔄</span>
              <span>Sync All Data</span>
            </button>
            <button className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors flex items-center space-x-2">
              <span>🤖</span>
              <span>Generate New Playbook</span>
            </button>
          </div>
          
          <div className={`rounded-lg p-4 border transition-colors ${
            isDark ? 'bg-yellow-900 border-yellow-700' : 'bg-yellow-50 border-yellow-200'
          }`}>
            <div className="flex items-center space-x-2">
              <span className={isDark ? 'text-yellow-300' : 'text-yellow-600'}>⚠️</span>
              <span className={`font-semibold transition-colors ${
                isDark ? 'text-yellow-300' : 'text-yellow-700'
              }`}>
                Manual Override Active
              </span>
            </div>
            <p className={`text-sm mt-1 transition-colors ${
              isDark ? 'text-yellow-200' : 'text-yellow-600'
            }`}>
              Manual controls are for testing and emergency use. AI automation is recommended for normal operations.
            </p>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className={`rounded-xl shadow-sm border p-6 transition-colors ${
        isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
      }`}>
        <h2 className={`text-xl font-semibold mb-4 transition-colors ${
          isDark ? 'text-white' : 'text-gray-800'
        }`}>
          System Status
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className={`p-4 rounded-lg border transition-colors ${
            isDark ? 'bg-green-900 border-green-700' : 'bg-green-50 border-green-200'
          }`}>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className={`font-semibold transition-colors ${
                isDark ? 'text-green-300' : 'text-green-700'
              }`}>
                AI Agent
              </span>
            </div>
            <p className={`text-sm mt-1 transition-colors ${
              isDark ? 'text-green-200' : 'text-green-600'
            }`}>
              Running optimally
            </p>
          </div>
          
          <div className={`p-4 rounded-lg border transition-colors ${
            isDark ? 'bg-green-900 border-green-700' : 'bg-green-50 border-green-200'
          }`}>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className={`font-semibold transition-colors ${
                isDark ? 'text-green-300' : 'text-green-700'
              }`}>
                Data Sources
              </span>
            </div>
            <p className={`text-sm mt-1 transition-colors ${
              isDark ? 'text-green-200' : 'text-green-600'
            }`}>
              All systems connected
            </p>
          </div>
          
          <div className={`p-4 rounded-lg border transition-colors ${
            isDark ? 'bg-blue-900 border-blue-700' : 'bg-blue-50 border-blue-200'
          }`}>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              <span className={`font-semibold transition-colors ${
                isDark ? 'text-blue-300' : 'text-blue-700'
              }`}>
                Last Update
              </span>
            </div>
            <p className={`text-sm mt-1 transition-colors ${
              isDark ? 'text-blue-200' : 'text-blue-600'
            }`}>
              {new Date().toLocaleString()}
            </p>
          </div>
        </div>
      </div>

      {/* Theme Settings */}
      <div className={`rounded-xl shadow-sm border p-6 transition-colors ${
        isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
      }`}>
        <h2 className={`text-xl font-semibold mb-4 transition-colors ${
          isDark ? 'text-white' : 'text-gray-800'
        }`}>
          Appearance
        </h2>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className={`font-medium transition-colors ${
                isDark ? 'text-gray-300' : 'text-gray-700'
              }`}>
                Dark Mode
              </h3>
              <p className={`text-sm transition-colors ${
                isDark ? 'text-gray-400' : 'text-gray-600'
              }`}>
                Toggle between light and dark themes
              </p>
            </div>
            <div className={`w-12 h-6 flex items-center rounded-full p-1 cursor-pointer transition-colors ${
              isDark ? 'bg-blue-600' : 'bg-gray-300'
            }`}>
              <div className={`bg-white w-4 h-4 rounded-full shadow-md transform transition-transform ${
                isDark ? 'translate-x-6' : 'translate-x-0'
              }`}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;