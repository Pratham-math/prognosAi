// Updated Navbar.jsx with theme toggle
import React from 'react';
import { useTheme } from '../context/Themecontext';

const Navbar = ({ currentPage, setCurrentPage }) => {
  const { isDark, toggleTheme } = useTheme();
  
  const navItems = [
    { id: 'forecast', label: 'Dashboard', icon: '🏠' },
    { id: 'playbook', label: 'AI Playbook', icon: '🤖' },
    { id: 'settings', label: 'Settings', icon: '⚙️' },
  ];

  return (
    <nav className={`shadow-lg border-b transition-colors duration-300 ${
      isDark 
        ? 'bg-gray-900 border-blue-800' 
        : 'bg-white border-blue-200'
    }`}>
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-lg transition-colors ${
              isDark ? 'bg-blue-700' : 'bg-blue-600'
            }`}>
              <span className="text-xl">🏥</span>
            </div>
            <div>
              <h1 className={`text-xl font-bold transition-colors ${
                isDark ? 'text-white' : 'text-gray-800'
              }`}>
                HealthSurge AI
              </h1>
              <p className={`text-xs transition-colors ${
                isDark ? 'text-gray-400' : 'text-gray-600'
              }`}>
                Autonomous Hospital Operations
              </p>
            </div>
          </div>

          {/* Navigation Items */}
          <div className="flex space-x-1">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setCurrentPage(item.id)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                  currentPage === item.id
                    ? isDark
                      ? 'bg-blue-700 text-white shadow-md'
                      : 'bg-blue-600 text-white shadow-md'
                    : isDark
                    ? 'text-gray-300 hover:bg-gray-800 hover:text-white'
                    : 'text-gray-600 hover:bg-blue-50 hover:text-blue-600'
                }`}
              >
                <span className="text-lg">{item.icon}</span>
                <span className="font-medium">{item.label}</span>
              </button>
            ))}
          </div>

          {/* Theme Toggle and Status */}
          <div className="flex items-center space-x-4">
            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className={`p-2 rounded-lg transition-colors ${
                isDark 
                  ? 'bg-gray-800 text-yellow-400 hover:bg-gray-700' 
                  : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
              }`}
            >
              {isDark ? '🌙' : '☀️'}
            </button>

            {/* Status Indicator */}
            <div className={`flex items-center space-x-2 px-3 py-2 rounded-full border transition-colors ${
              isDark
                ? 'bg-green-900 border-green-700'
                : 'bg-green-50 border-green-200'
            }`}>
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className={`text-sm font-medium transition-colors ${
                isDark ? 'text-green-300' : 'text-green-700'
              }`}>
                System Active
              </span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;