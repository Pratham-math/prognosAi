// components/PlaybookCard.jsx
import React from 'react';

const PlaybookCard = ({ title, description, priority, status, impact, effort }) => {
  const priorityStyles = {
    high: 'border-red-500 bg-red-50',
    medium: 'border-orange-500 bg-orange-50',
    low: 'border-yellow-500 bg-yellow-50'
  };

  const statusStyles = {
    pending: 'bg-gray-100 text-gray-700',
    'in-progress': 'bg-blue-100 text-blue-700',
    completed: 'bg-green-100 text-green-700'
  };

  return (
    <div className={`border-l-4 rounded-lg shadow-sm ${priorityStyles[priority]} p-4 h-full`}>
      <div className="flex justify-between items-start mb-2">
        <h3 className="font-semibold text-gray-800 text-lg">{title}</h3>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusStyles[status]}`}>
          {status.replace('-', ' ')}
        </span>
      </div>
      
      <p className="text-gray-600 text-sm mb-4">{description}</p>
      
      <div className="flex justify-between items-center text-xs">
        <div className="space-y-1">
          <div className="flex items-center space-x-1">
            <span className="text-gray-500">Impact:</span>
            <span className={`font-medium ${
              impact === 'Critical' ? 'text-red-600' : 
              impact === 'High' ? 'text-orange-600' : 
              'text-yellow-600'
            }`}>
              {impact}
            </span>
          </div>
          <div className="flex items-center space-x-1">
            <span className="text-gray-500">Effort:</span>
            <span className={`font-medium ${
              effort === 'High' ? 'text-red-600' : 
              effort === 'Medium' ? 'text-orange-600' : 
              'text-green-600'
            }`}>
              {effort}
            </span>
          </div>
        </div>
        
        <button className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-xs font-medium transition-colors">
          Execute
        </button>
      </div>
    </div>
  );
};

export default PlaybookCard;