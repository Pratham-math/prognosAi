// pages/Forecast.jsx (updated with API calls)
import React, { useState, useEffect } from 'react';
import ChartCard from '../components/ChartCard';
import { forecastAPI, getMockData } from '../services/Api';

const Forecast = () => {
  const [timeRange, setTimeRange] = useState('72h');
  const [forecastData, setForecastData] = useState({});
  const [correlationData, setCorrelationData] = useState([]);
  const [surgeJustification, setSurgeJustification] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadForecastData();
  }, [timeRange]);

  const loadForecastData = async () => {
    try {
      setLoading(true);
      
      // Use mock data for development
      const patientTypes = await getMockData('patientTypeForecast');
      const justification = "Rise in PM2.5 levels (AQI: 156) combined with sudden temperature drop and elevated humidity levels correlates with a 37% increase in respiratory cases. Historical data shows similar patterns during previous pollution spikes.";
      
      setForecastData(patientTypes);
      setSurgeJustification(justification);
      setCorrelationData([
        { aqi: 50, patients: 20 },
        { aqi: 100, patients: 35 },
        { aqi: 150, patients: 52 },
        { aqi: 200, patients: 68 },
        { aqi: 250, patients: 85 }
      ]);

    } catch (error) {
      console.error('Failed to load forecast data:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-800">Surge Forecast Analysis</h1>
          <p className="text-gray-600">AI-powered patient inflow predictions and correlations</p>
        </div>
        
        {/* Time Range Selector */}
        <div className="flex space-x-2 bg-white rounded-lg p-1 shadow-sm border">
          {['24h', '48h', '72h'].map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-md transition-all ${
                timeRange === range
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-4 text-gray-600">Loading forecast data...</p>
          </div>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Patient Type Forecast */}
            <ChartCard
              title={`Patient Type Forecast - ${timeRange}`}
              chartType="bar"
              data={forecastData[timeRange] || []}
              dataKey="patients"
              nameKey="type"
            />

            {/* AQI-Patient Correlation */}
            <ChartCard
              title="AQI - Patient Correlation"
              chartType="scatter"
              data={correlationData}
              dataKey="patients"
              nameKey="aqi"
            />
          </div>

          {/* Surge Justification Panel */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center space-x-3 mb-4">
              <div className="bg-purple-100 p-2 rounded-lg">
                <span className="text-2xl">🧠</span>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-800">AI Surge Justification</h3>
                <p className="text-sm text-gray-600">Explainable AI insights behind the forecast</p>
              </div>
            </div>
            
            <div className="bg-gray-50 rounded-lg p-4 border">
              <p className="text-gray-700 leading-relaxed">{surgeJustification}</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default Forecast;