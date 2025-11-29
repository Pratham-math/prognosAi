// components/StatsCard.jsx
import React from 'react';
import { useTheme } from '../context/Themecontext'; // Fixed import path

const StatsCard = ({ title, value, change, trend, icon, color }) => {
  const { isDark } = useTheme(); // Now this will work
  
  const colorClasses = {
    blue: isDark ? 'from-blue-700 to-blue-800' : 'from-blue-500 to-blue-600',
    green: isDark ? 'from-green-700 to-green-800' : 'from-green-500 to-green-600',
    red: isDark ? 'from-red-700 to-red-800' : 'from-red-500 to-red-600',
    yellow: isDark ? 'from-yellow-600 to-yellow-700' : 'from-yellow-500 to-yellow-600'
  };

  return (
    <div className={`rounded-xl shadow-sm border p-6 transition-colors duration-300 ${
      isDark 
        ? 'bg-gray-800 border-gray-700' 
        : 'bg-white border-gray-200'
    }`}>
      <div className="flex items-center justify-between">
        <div>
          <p className={`text-sm font-medium transition-colors ${
            isDark ? 'text-gray-400' : 'text-gray-600'
          }`}>
            {title}
          </p>
          <p className={`text-2xl font-bold mt-1 transition-colors ${
            isDark ? 'text-white' : 'text-gray-800'
          }`}>
            {value}
          </p>
          <div className={`flex items-center space-x-1 mt-2 ${
            trend === 'up' 
              ? isDark ? 'text-green-400' : 'text-green-600'
              : isDark ? 'text-red-400' : 'text-red-600'
          }`}>
            <span>{trend === 'up' ? '↗' : '↘'}</span>
            <span className="text-sm font-medium">{change}</span>
            <span className={`text-sm transition-colors ${
              isDark ? 'text-gray-500' : 'text-gray-600'
            }`}>
              from yesterday
            </span>
          </div>
        </div>
        <div className={`p-3 rounded-lg bg-gradient-to-r ${colorClasses[color]}`}>
          <span className="text-2xl text-white">{icon}</span>
        </div>
      </div>
    </div>
  );
};

export default StatsCard;