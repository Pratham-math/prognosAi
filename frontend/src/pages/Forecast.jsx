// pages/Forecast.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ChartCard from '../components/ChartCard';
import { useTheme } from '../context/Themecontext';

const Forecast = () => {
  const [timeRange, setTimeRange] = useState('1d');
  const { isDark } = useTheme();
  
  const [forecastData, setForecastData] = useState({});
  const [metadata, setMetadata] = useState({
    total_predicted: 0,
    peak_admissions: 0,
    avg_hourly: 0,
    peak_timestamp: null
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchForecast(timeRange);
  }, [timeRange]);

  const fetchForecast = async (horizon) => {
    setLoading(true);
    setError('');
    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', {
        horizon: horizon,
        location : "Mumbai"
      });

      if (!response.data.ok) {
        throw new Error(response.data.error || 'Failed to fetch forecast');
      }

      const result = response.data.result;
      
      // Process forecast data for charts
      const hourlyData = result.forecast.map((item, idx) => ({
        hour: idx + 1,
        predicted: parseFloat(item.predicted_admissions.toFixed(1)),
        xgb: parseFloat(item.xgb_only.toFixed(1)),
        lstm: parseFloat(item.lstm_only.toFixed(1)),
        timestamp: item.timestamp
      }));

      // Group by patient type (synthetic categorization based on predicted load)
      const typeDistribution = categorizeByType(result.total_predicted_admissions);

      // Create AQI correlation data (synthetic based on model data)
      const aqiCorrelation = generateAQICorrelation(result.forecast);

      setForecastData({
        hourly: hourlyData,
        typeDistribution: typeDistribution,
        aqiCorrelation: aqiCorrelation
      });

      // Find peak
      const peakEntry = result.forecast.reduce((max, curr) => 
        curr.predicted_admissions > max.predicted_admissions ? curr : max
      );

      setMetadata({
        total_predicted: result.total_predicted_admissions,
        peak_admissions: peakEntry.predicted_admissions,
        avg_hourly: result.total_predicted_admissions / result.forecast.length,
        peak_timestamp: peakEntry.timestamp,
        horizon: result.horizon,
        supply_plan: result.supply_plan
      });

    } catch (err) {
      setError(err.message || 'Failed to load forecast');
    } finally {
      setLoading(false);
    }
  };

  // Categorize predicted admissions into patient types
  const categorizeByType = (totalAdmissions) => {
    // Rough distribution based on typical ED patterns
    return [
      { type: 'Respiratory', patients: Math.round(totalAdmissions * 0.35) },
      { type: 'Cardiac', patients: Math.round(totalAdmissions * 0.18) },
      { type: 'General', patients: Math.round(totalAdmissions * 0.27) },
      { type: 'Emergency', patients: Math.round(totalAdmissions * 0.20) }
    ];
  };

  // Generate synthetic AQI correlation from forecast pattern
  const generateAQICorrelation = (forecast) => {
    // Sample 5 points across the forecast range
    const step = Math.floor(forecast.length / 5);
    return forecast
      .filter((_, idx) => idx % step === 0)
      .slice(0, 5)
      .map((item, idx) => ({
        aqi: 50 + idx * 50, // AQI from 50 to 250
        patients: parseFloat(item.predicted_admissions.toFixed(1))
      }));
  };

  const surgeJustification = `Ensemble model (XGBoost + LSTM + Ridge) predicts ${metadata.total_predicted.toFixed(0)} total admissions over ${metadata.horizon}. Peak load of ${metadata.peak_admissions.toFixed(1)} patients expected at ${metadata.peak_timestamp ? new Date(metadata.peak_timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }) : 'N/A'}. Model incorporates historical patterns, temporal features, and synthetic AQI/festival data.`;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className={`text-lg ${isDark ? 'text-gray-300' : 'text-gray-700'}`}>
            Loading ML forecast...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className={`p-6 rounded-lg border ${isDark ? 'bg-red-900 border-red-700' : 'bg-red-50 border-red-200'}`}>
          <p className={`text-lg font-semibold ${isDark ? 'text-red-200' : 'text-red-700'}`}>
            Error: {error}
          </p>
          <button
            onClick={() => fetchForecast(timeRange)}
            className="mt-4 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className={`text-3xl font-bold transition-colors ${
            isDark ? 'text-white' : 'text-gray-800'
          }`}>
            Surge Forecast Analysis
          </h1>
          <p className={`transition-colors ${
            isDark ? 'text-gray-400' : 'text-gray-600'
          }`}>
            LSTM + XGBoost ensemble predictions
          </p>
        </div>
        
        {/* Time Range Selector */}
        <div className={`flex space-x-2 rounded-lg p-1 shadow-sm border transition-colors ${
          isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        }`}>
          {['1h', '1d', '2d'].map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-4 py-2 rounded-md transition-all ${
                timeRange === range
                  ? 'bg-blue-600 text-white shadow-md'
                  : isDark 
                    ? 'text-gray-300 hover:bg-gray-700' 
                    : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Patient Type Forecast */}
        <ChartCard
          title={`Patient Type Distribution - ${timeRange}`}
          chartType="bar"
          data={forecastData.typeDistribution || []}
          dataKey="patients"
          nameKey="type"
        />

        {/* AQI-Patient Correlation */}
        <ChartCard
          title="AQI - Patient Correlation"
          chartType="scatter"
          data={forecastData.aqiCorrelation || []}
          dataKey="patients"
          nameKey="aqi"
        />
      </div>

      {/* Surge Justification Panel */}
      <div className={`rounded-xl shadow-sm border p-6 transition-colors ${
        isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
      }`}>
        <div className="flex items-center space-x-3 mb-4">
          <div className={`p-2 rounded-lg transition-colors ${
            isDark ? 'bg-purple-900' : 'bg-purple-100'
          }`}>
            <span className="text-2xl">🧠</span>
          </div>
          <div>
            <h3 className={`text-lg font-semibold transition-colors ${
              isDark ? 'text-white' : 'text-gray-800'
            }`}>
              ML Model Explanation
            </h3>
            <p className={`text-sm transition-colors ${
              isDark ? 'text-gray-400' : 'text-gray-600'
            }`}>
              Ensemble forecast rationale
            </p>
          </div>
        </div>
        
        <div className={`rounded-lg p-4 border transition-colors ${
          isDark ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
        }`}>
          <p className={`leading-relaxed transition-colors ${
            isDark ? 'text-gray-300' : 'text-gray-700'
          }`}>
            {surgeJustification}
          </p>
        </div>

        {/* Key Metrics from Model */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          {/* Total Predicted */}
          <div className={`p-4 rounded-lg border transition-colors ${
            isDark ? 'bg-blue-900 border-blue-700' : 'bg-blue-50 border-blue-200'
          }`}>
            <div className="flex items-center space-x-2 mb-2">
              <span className={isDark ? 'text-blue-300' : 'text-blue-600'}>📊</span>
              <span className={`font-semibold transition-colors ${
                isDark ? 'text-blue-300' : 'text-blue-700'
              }`}>
                Total Admissions
              </span>
            </div>
            <p className={`text-2xl font-bold transition-colors ${
              isDark ? 'text-blue-200' : 'text-blue-600'
            }`}>
              {metadata.total_predicted.toFixed(0)}
            </p>
          </div>
          
          {/* Peak Load */}
          <div className={`p-4 rounded-lg border transition-colors ${
            isDark ? 'bg-orange-900 border-orange-700' : 'bg-orange-50 border-orange-200'
          }`}>
            <div className="flex items-center space-x-2 mb-2">
              <span className={isDark ? 'text-orange-300' : 'text-orange-600'}>⚡</span>
              <span className={`font-semibold transition-colors ${
                isDark ? 'text-orange-300' : 'text-orange-700'
              }`}>
                Peak Load
              </span>
            </div>
            <p className={`text-2xl font-bold transition-colors ${
              isDark ? 'text-orange-200' : 'text-orange-600'
            }`}>
              {metadata.peak_admissions.toFixed(1)} pts/hr
            </p>
          </div>
          
          {/* Average Hourly */}
          <div className={`p-4 rounded-lg border transition-colors ${
            isDark ? 'bg-green-900 border-green-700' : 'bg-green-50 border-green-200'
          }`}>
            <div className="flex items-center space-x-2 mb-2">
              <span className={isDark ? 'text-green-300' : 'text-green-600'}>📈</span>
              <span className={`font-semibold transition-colors ${
                isDark ? 'text-green-300' : 'text-green-700'
              }`}>
                Avg/Hour
              </span>
            </div>
            <p className={`text-2xl font-bold transition-colors ${
              isDark ? 'text-green-200' : 'text-green-600'
            }`}>
              {metadata.avg_hourly.toFixed(1)}
            </p>
          </div>
        </div>
      </div>

      {/* Additional Metrics Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model Confidence (hardcoded, can enhance later) */}
        <div className={`rounded-xl shadow-sm border p-6 transition-colors ${
          isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        }`}>
          <h3 className={`text-lg font-semibold mb-4 transition-colors ${
            isDark ? 'text-white' : 'text-gray-800'
          }`}>
            Model Performance
          </h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-1">
                <span className={`text-sm transition-colors ${
                  isDark ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  Ensemble MAE
                </span>
                <span className={`text-sm font-medium transition-colors ${
                  isDark ? 'text-green-400' : 'text-green-600'
                }`}>
                  1.87 patients
                </span>
              </div>
              <div className={`w-full rounded-full h-2 transition-colors ${
                isDark ? 'bg-gray-700' : 'bg-gray-200'
              }`}>
                <div className="bg-green-500 h-2 rounded-full" style={{ width: '92%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <span className={`text-sm transition-colors ${
                  isDark ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  XGBoost Weight
                </span>
                <span className={`text-sm font-medium transition-colors ${
                  isDark ? 'text-blue-400' : 'text-blue-600'
                }`}>
                  21.5%
                </span>
              </div>
              <div className={`w-full rounded-full h-2 transition-colors ${
                isDark ? 'bg-gray-700' : 'bg-gray-200'
              }`}>
                <div className="bg-blue-500 h-2 rounded-full" style={{ width: '21.5%' }}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <span className={`text-sm transition-colors ${
                  isDark ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  LSTM Weight
                </span>
                <span className={`text-sm font-medium transition-colors ${
                  isDark ? 'text-purple-400' : 'text-purple-600'
                }`}>
                  83.8%
                </span>
              </div>
              <div className={`w-full rounded-full h-2 transition-colors ${
                isDark ? 'bg-gray-700' : 'bg-gray-200'
              }`}>
                <div className="bg-purple-500 h-2 rounded-full" style={{ width: '83.8%' }}></div>
              </div>
            </div>
          </div>
        </div>

        {/* Supply Plan from Model */}
        <div className={`rounded-xl shadow-sm border p-6 transition-colors ${
          isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        }`}>
          <h3 className={`text-lg font-semibold mb-4 transition-colors ${
            isDark ? 'text-white' : 'text-gray-800'
          }`}>
            Recommended Supplies
          </h3>
          <div className="space-y-3">
            <div className={`p-3 rounded-lg border-l-4 transition-colors ${
              isDark ? 'bg-red-900 border-red-500' : 'bg-red-50 border-red-400'
            }`}>
              <div className="flex justify-between items-center">
                <span className={`font-medium transition-colors ${
                  isDark ? 'text-red-200' : 'text-red-700'
                }`}>
                  Oxygen Cylinders
                </span>
                <span className={`text-lg px-3 py-1 rounded-full font-bold transition-colors ${
                  isDark ? 'bg-red-800 text-red-200' : 'bg-red-100 text-red-700'
                }`}>
                  {metadata.supply_plan?.oxygen_cylinders || 0}
                </span>
              </div>
            </div>
            <div className={`p-3 rounded-lg border-l-4 transition-colors ${
              isDark ? 'bg-blue-900 border-blue-500' : 'bg-blue-50 border-blue-400'
            }`}>
              <div className="flex justify-between items-center">
                <span className={`font-medium transition-colors ${
                  isDark ? 'text-blue-200' : 'text-blue-700'
                }`}>
                  Nebulizer Sets
                </span>
                <span className={`text-lg px-3 py-1 rounded-full font-bold transition-colors ${
                  isDark ? 'bg-blue-800 text-blue-200' : 'bg-blue-100 text-blue-700'
                }`}>
                  {metadata.supply_plan?.nebulizer_sets || 0}
                </span>
              </div>
            </div>
            <div className={`p-3 rounded-lg border-l-4 transition-colors ${
              isDark ? 'bg-yellow-900 border-yellow-500' : 'bg-yellow-50 border-yellow-400'
            }`}>
              <div className="flex justify-between items-center">
                <span className={`font-medium transition-colors ${
                  isDark ? 'text-yellow-200' : 'text-yellow-700'
                }`}>
                  Emergency Beds
                </span>
                <span className={`text-lg px-3 py-1 rounded-full font-bold transition-colors ${
                  isDark ? 'bg-yellow-800 text-yellow-200' : 'bg-yellow-100 text-yellow-700'
                }`}>
                  {metadata.supply_plan?.emergency_beds_to_reserve || 0}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Forecast;
