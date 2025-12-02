import { useState, useEffect } from 'react'; // Import useEffect
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

function App() {
  // --- FORECAST STATE ---
  const [weeks, setWeeks] = useState(12);
  const [forecast, setForecast] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // --- RETRAIN STATE ---
  const [nEstimators, setNEstimators] = useState(100);
  const [learningRate, setLearningRate] = useState(0.05);
  const [randomState, setRandomState] = useState(42);
  const [retrainLoading, setRetrainLoading] = useState(false);
  const [retrainMsg, setRetrainMsg] = useState(null);
  
  // --- NEW: CURRENT CONFIG STATE ---
  const [currentParams, setCurrentParams] = useState(null);

  // 1. FETCH CONFIGURATION (Runs on page load)
  const fetchConfig = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/config");
      const data = await response.json();
      setCurrentParams(data.Hyperparameters);
      
      // Optional: Pre-fill the inputs with the current values
      setNEstimators(data.Hyperparameters.n_estimators);
      setLearningRate(data.Hyperparameters.learning_rate);
      setRandomState(data.Hyperparameters.random_state);
    } catch (err) {
      console.error("Failed to fetch model config:", err);
    }
  };

  useEffect(() => {
    fetchConfig();
  }, []);

  // 2. FORECAST FUNCTION
  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://127.0.0.1:8000/predict?weeks=${weeks}`);
      if (!response.ok) throw new Error('Failed to connect to backend');
      
      const data = await response.json();
      setForecast(data.forecast);
    } catch (err) {
      setError("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // 3. RETRAIN FUNCTION
  const handleRetrain = async () => {
    setRetrainLoading(true);
    setRetrainMsg("⏳ Starting training job...");
    
    // Convert inputs to ensure types match Python
    const targetEstimators = parseInt(nEstimators);
    const targetLR = parseFloat(learningRate);
    const targetSeed = parseInt(randomState);

    const payload = {
      n_estimators: targetEstimators,
      learning_rate: targetLR,
      random_state: targetSeed
    };

    try {
      // 1. Trigger the job
      const response = await fetch("http://127.0.0.1:8000/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error("Retraining failed to start");
      const data = await response.json();
      setRetrainMsg(` ${data.message} Waiting for update...`);

      // 2. POLL FOR UPDATES
      let attempts = 0;
      const maxAttempts = 15; // Wait up to 30 seconds
      
      const pollInterval = setInterval(async () => {
        attempts++;
        try {
          const configResponse = await fetch("http://127.0.0.1:8000/config");
          const configData = await configResponse.json();
          const current = configData.Hyperparameters;

          console.log(`Polling ${attempts}:`, current); // Debug log

          // FLOAT SAFETY: Check if learning rate is "close enough" (within 0.0001)
          const isLRMatch = Math.abs(current.learning_rate - targetLR) < 0.0001;
          const isEstMatch = current.n_estimators === targetEstimators;
          const isSeedMatch = current.random_state === targetSeed;

          // CHECK: Did the model update?
          if (isEstMatch && isLRMatch && isSeedMatch) {
            
            // --- THE FIX IS HERE ---
            setRetrainMsg("✅ Model Successfully Updated!");
            
            // 1. Force React to update the "Active Model" UI immediately
            setCurrentParams(current); 
            
            // 2. (Optional) Re-fetch from server just to be 100% sure
            fetchConfig(); 

            setRetrainLoading(false);
            clearInterval(pollInterval); // Stop loop
          } 
          
          // Timeout
          if (attempts >= maxAttempts) {
             setRetrainMsg("⚠️ Update taking too long. Refresh page to check.");
             setRetrainLoading(false);
             clearInterval(pollInterval);
          }

        } catch (e) {
          console.error("Polling error", e);
        }
      }, 2000); 

    } catch (err) {
      setRetrainMsg(`❌ Error: ${err.message}`);
      setRetrainLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>🇧🇷 Olist Sales Predictor</h1>
      
      {/* --- RETRAIN PANEL --- */}
      <div className="retrain-panel">
        <h3 style={{ marginTop: 0 }}>⚙️ Model Configuration</h3>
        
        {/* CURRENT PARAMS DISPLAY */}
        {currentParams && (
          <div className="status-bar">
            <span><strong>Active Model:</strong> XGBoost Regressor</span>
            <div className="badges">
              <span className="badge">Est: {currentParams.n_estimators}</span>
              <span className="badge">LR: {currentParams.learning_rate}</span>
              <span className="badge">Seed: {currentParams.random_state}</span>
            </div>
          </div>
        )}

        <div className="retrain-grid">
          <div className="input-group">
            <label>N_Estimators</label>
            <input 
              type="number" 
              value={nEstimators} 
              onChange={(e) => setNEstimators(e.target.value)}
              className="number-input" style={{ width: '100px' }}
            />
          </div>

          <div className="input-group">
            <label>Learning Rate</label>
            <input 
              type="number" step="0.01" 
              value={learningRate} 
              onChange={(e) => setLearningRate(e.target.value)}
              className="number-input" style={{ width: '100px' }}
            />
          </div>

          <div className="input-group">
            <label>Random State</label>
            <input 
              type="number" 
              value={randomState} 
              onChange={(e) => setRandomState(e.target.value)}
              className="number-input" style={{ width: '100px' }}
            />
          </div>

          <button onClick={handleRetrain} disabled={retrainLoading} className="retrain-btn">
            {retrainLoading ? "Training..." : "Start Retraining"}
          </button>
        </div>

        {retrainMsg && <div className={retrainMsg.includes("Error") ? "error" : "success-msg"}>{retrainMsg}</div>}
      </div>

      {error && <p className="error">{error}</p>}

      {/* --- FORECAST PANEL --- */}
      <div className="controls-panel">
        <div className="input-group">
          <label>Forecast Horizon: <strong>{weeks} weeks</strong></label>
          <div className="slider-container">
            <input 
              type="range" min="1" max="52" value={weeks} 
              onChange={(e) => setWeeks(e.target.value)}
              className="slider"
            />
            <input 
              type="number" min="1" max="52" value={weeks} 
              onChange={(e) => setWeeks(e.target.value)}
              className="number-input"
            />
          </div>
        </div>
        <button onClick={handlePredict} disabled={loading} className="predict-btn">
          {loading ? "Calculating..." : "Generate Forecast"}
        </button>
      </div>

      {/* --- RESULTS GRID --- */}
      {forecast.length > 0 && (
        <div className="results-grid">
          <div className="chart-card">
            <h3>📈 Sales Trend</h3>
            <div style={{ width: '100%', height: 400 }}>
              <ResponsiveContainer>
                <AreaChart data={forecast} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorSales" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00a8ff" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#00a8ff" stopOpacity={0.05}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip formatter={(value) => `R$ ${Number(value).toFixed(2)}`} />
                  <Legend />
                  <Area type="monotone" dataKey="sales" stroke="#00a8ff" strokeWidth={3} fillOpacity={1} fill="url(#colorSales)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="table-card">
            <h3>📅 Tabular Data</h3>
            <div className="table-wrapper">
              <ul>
                {forecast.map((item, index) => (
                  <li key={index} className="data-item">
                    <span className="data-date">{item.date}</span>
                    <strong className="data-sales">R$ {Number(item.sales).toFixed(2)}</strong>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;