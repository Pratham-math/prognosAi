// Updated ChartCard.jsx with dark theme
import React from 'react';
import { useTheme } from '../context/Themecontext';

const ChartCard = ({ title, chartType, data, dataKey, nameKey }) => {
  const { isDark } = useTheme();

  const renderChart = () => {
    switch (chartType) {
      case 'line':
        return (
          <div className="h-48 flex items-end space-x-2 justify-center">
            {data.map((item, index) => (
              <div key={index} className="flex flex-col items-center">
                <div 
                  className={`rounded-t w-8 transition-all duration-500 ${
                    isDark ? 'bg-blue-600' : 'bg-blue-500'
                  }`}
                  style={{ height: `${item[dataKey] * 2}px` }}
                ></div>
                <span className={`text-xs mt-2 transition-colors ${
                  isDark ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  {item[nameKey]}
                </span>
              </div>
            ))}
          </div>
        );
      case 'bar':
        return (
          <div className="h-48 flex items-end space-x-4 justify-center">
            {data.map((item, index) => (
              <div key={index} className="flex flex-col items-center">
                <div 
                  className={`rounded-t w-12 transition-all duration-500 ${
                    isDark 
                      ? 'bg-gradient-to-t from-blue-600 to-blue-500'
                      : 'bg-gradient-to-t from-blue-500 to-blue-400'
                  }`}
                  style={{ height: `${item[dataKey] * 2}px` }}
                ></div>
                <span className={`text-xs mt-2 transition-colors ${
                  isDark ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  {item[nameKey]}
                </span>
                <span className={`text-sm font-semibold mt-1 transition-colors ${
                  isDark ? 'text-white' : 'text-gray-800'
                }`}>
                  {item[dataKey]}
                </span>
              </div>
            ))}
          </div>
        );
      case 'scatter':
        return (
          <div className={`h-48 relative transition-colors ${
            isDark 
              ? 'border-l-2 border-b-2 border-gray-600' 
              : 'border-l-2 border-b-2 border-gray-300'
          }`}>
            {data.map((item, index) => (
              <div
                key={index}
                className={`absolute w-3 h-3 rounded-full transform -translate-x-1/2 -translate-y-1/2 ${
                  isDark ? 'bg-red-500' : 'bg-red-500'
                }`}
                style={{
                  left: `${(item[nameKey] / 300) * 100}%`,
                  bottom: `${(item[dataKey] / 100) * 100}%`
                }}
              ></div>
            ))}
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className={`rounded-xl shadow-sm border p-6 transition-colors duration-300 ${
      isDark 
        ? 'bg-gray-800 border-gray-700' 
        : 'bg-white border-gray-200'
    }`}>
      <h3 className={`text-lg font-semibold mb-4 transition-colors ${
        isDark ? 'text-white' : 'text-gray-800'
      }`}>
        {title}
      </h3>
      {renderChart()}
    </div>
  );
};

export default ChartCard;