
import React from 'react';

const AlertModal = ({ isOpen, onClose, title, message, severity }) => {
  if (!isOpen) return null;

  const severityStyles = {
    high: 'border-red-500 bg-red-50',
    medium: 'border-orange-500 bg-orange-50',
    low: 'border-yellow-500 bg-yellow-50'
  };

  const severityIcons = {
    high: '🚨',
    medium: '⚠️',
    low: 'ℹ️'
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className={`border-l-4 rounded-lg shadow-lg max-w-md w-full ${severityStyles[severity]}`}>
        <div className="bg-white p-6 rounded-r-lg">
          <div className="flex items-center space-x-3 mb-4">
            <span className="text-2xl">{severityIcons[severity]}</span>
            <h3 className="text-xl font-bold text-gray-800">{title}</h3>
          </div>
          
          <p className="text-gray-600 mb-6">{message}</p>
          
          <div className="flex space-x-3">
            <button
              onClick={onClose}
              className="flex-1 bg-gray-500 hover:bg-gray-600 text-white py-2 px-4 rounded-lg font-semibold transition-colors"
            >
              Dismiss
            </button>
            <button
              onClick={() => {
                onClose();
                // Navigate to playbook page
                window.location.hash = '#playbook';
              }}
              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg font-semibold transition-colors"
            >
              View Playbook
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AlertModal;