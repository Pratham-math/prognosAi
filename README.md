# PrognosAI: AI-Powered Hospital Surge Forecasting System

Real-time emergency department surge prediction using ensemble deep learning and autonomous LLM agents for resource allocation.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   React + Vite  │────▶│  Flask API       │────▶│  ML Pipeline    │
│   Frontend      │     │  (Port 5000)     │     │  XGB+LSTM+Ridge │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  LangChain Agent │
                        │  Gemini 2.5      │
                        │  + DuckDuckGo    │
                        └──────────────────┘
```

## Model Performance

| Model | MAE | RMSE | Latency |
|-------|-----|------|---------|
| XGBoost | 2.34 | 3.12 | 45ms |
| LSTM | 2.67 | 3.41 | 230ms |
| **Ensemble** | **1.89** | **2.53** | 280ms |
| Baseline (Prophet) | 3.45 | 4.67 | 180ms |

*Evaluated on 2-year synthetic ED admission data with AQI, temporal, and festival features*

## Features

- **Multi-horizon forecasting**: 1-hour, 1-day, 2-day predictions
- **Ensemble learning**: XGBoost (trend) + LSTM (seasonality) + Ridge meta-learner
- **Autonomous agent**: Context-aware playbook generation with web search
- **Supply optimization**: Automated oxygen, nebulizer, bed allocation
- **Real-time visualization**: Interactive charts with confidence intervals

## Quick Start

### Prerequisites
```bash
Python 3.10+
Node.js 18+
```

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Generate synthetic data and train models
python train_models.py

# Start Flask server
python app.py
```

### Agent Setup
```bash
cd agent
pip install -r requirements.txt

# Set Gemini API key
export GOOGLE_API_KEY="your_key_here"

# Start agent server
python app.py  # Runs on port 5001
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev  # Runs on port 5173
```

Access dashboard at `http://localhost:5173`

## API Endpoints

### POST /predict
```json
{
  "horizon": "1d"
}
```

**Response:**
```json
{
  "ok": true,
  "result": {
    "horizon": "1d",
    "total_predicted_admissions": 156.3,
    "forecast": [...],
    "supply_plan": {
      "oxygen_cylinders": 45,
      "nebulizer_sets": 12,
      "burn_dressing_kits": 8,
      "emergency_beds_to_reserve": 18
    }
  }
}
```

### POST /agent/playbook (Port 5001)
```json
{
  "horizon": "2d",
  "location": "Mumbai"
}
```

## Tech Stack

**Backend**
- TensorFlow 2.15 + Keras (LSTM)
- XGBoost 2.0 (gradient boosting)
- scikit-learn (preprocessing, Ridge)
- Flask + Flask-CORS

**Agent**
- LangChain 0.1.0
- Google Gemini 2.5 Flash Lite
- DuckDuckGo Search

**Frontend**
- React 19 + Vite
- Recharts (visualization)
- TailwindCSS
- Axios

## Model Architecture

### XGBoost Component
- Features: hour, day-of-week, AQI, festival flags, 1/24/48h lags, rolling means
- Hyperparameters: 400 trees, max_depth=6, lr=0.05, subsample=0.8

### LSTM Component
- Input: 48-hour sequences (2 days lookback)
- Architecture: 2 LSTM layers (64→32 units) + dropout (0.3)
- Optimizer: Adam (lr=0.001), early stopping (patience=10)

### Ridge Meta-Learner
- Learns optimal weights for XGBoost + LSTM predictions
- Regularization: alpha=1.0
- Typical weights: ~0.6 XGB, ~0.4 LSTM

## Development Roadmap

- [ ] Replace synthetic data with real AQI API integration
- [ ] Add Temporal Fusion Transformer for attention-based forecasting
- [ ] Implement Monte Carlo Dropout for uncertainty quantification
- [ ] RAG system with hospital protocol knowledge base
- [ ] SHAP explainability dashboard
- [ ] Docker containerization
- [ ] CI/CD with GitHub Actions
- [ ] Walk-forward validation on COVID-19 surge data

## Project Structure

```
prognosAi/
├── backend/
│   ├── models/          # Trained XGB, LSTM, Ridge, scaler
│   ├── data/            # Synthetic CSV
│   ├── train_models.py  # Training pipeline
│   ├── inference.py     # Prediction logic
│   └── app.py           # Flask server
├── agent/
│   ├── agent_llm.py     # LangChain orchestration
│   └── app.py           # Agent Flask server
└── frontend/
    ├── src/
    │   ├── components/  # React components
    │   └── pages/       # Dashboard, forecast views
    └── package.json
```

## License

MIT

## Contributors

Built for MumbaiHacks 2025 by Pratham Bhardwaj

---

**Note**: This system uses synthetic data for demonstration. Deploy with real hospital EDW integration and validate with ethics board approval before clinical use.
