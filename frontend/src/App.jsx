import { useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const CLINICAL_FIELDS = [
  { name: "age",         label: "Age",                    unit: "years", min: 18,  max: 120, step: 1,   placeholder: "45" },
  { name: "cholesterol", label: "Total Cholesterol",      unit: "mg/dL", min: 100, max: 600, step: 1,   placeholder: "200" },
  { name: "systolic_bp", label: "Systolic Blood Pressure",unit: "mmHg",  min: 70,  max: 250, step: 1,   placeholder: "120" },
  { name: "max_heart_rate", label: "Max Heart Rate",      unit: "bpm",   min: 40,  max: 220, step: 1,   placeholder: "150" },
];

const LIFESTYLE_FIELDS = [
  { name: "smoking_packyears", label: "Smoking History",    unit: "pack-years",  min: 0,   max: 50,  step: 0.5, help: "Packs/day × years smoked. Enter 0 if never smoked." },
  { name: "sedentary_hours",   label: "Sedentary Hours",    unit: "hrs/day",     min: 2,   max: 16,  step: 0.5, help: "Hours per day spent sitting or physically inactive." },
  { name: "activity_mets",     label: "Physical Activity",  unit: "MET-hrs/wk",  min: 5,   max: 60,  step: 1,   help: "Weekly activity in MET-hours. Walking=3, Running=8 METs/hr." },
  { name: "sleep_regularity",  label: "Sleep Irregularity", unit: "hr variance", min: 0.2, max: 4.0, step: 0.1, help: "How much your sleep schedule varies night-to-night." },
  { name: "alcohol_units",     label: "Alcohol Consumption",unit: "units/wk",    min: 0,   max: 40,  step: 0.5, help: "Standard units per week (1 unit = 10ml pure alcohol)." },
];

const DEFAULT_VALUES = {
  age: 45, sex: 1, cholesterol: 200, systolic_bp: 120,
  fasting_blood_sugar: 0, max_heart_rate: 150,
  smoking_packyears: 0, sedentary_hours: 8,
  activity_mets: 25, sleep_regularity: 1.2, alcohol_units: 8,
};

const FEATURE_LABELS = {
  age: "Age", sex: "Sex", cholesterol: "Cholesterol",
  systolic_bp: "Systolic BP", fasting_blood_sugar: "Fasting Blood Sugar",
  max_heart_rate: "Max Heart Rate", LRS: "Lifestyle Risk Score",
};

function GaugeArc({ percent }) {
  const r = 80, cx = 110, cy = 100;
  const startAngle = Math.PI, endAngle = 2 * Math.PI;
  const fillAngle = startAngle + (endAngle - startAngle) * Math.min(percent / 100, 1);
  const toXY = (a) => ({ x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) });
  const start = toXY(startAngle), end = toXY(fillAngle), bgEnd = toXY(endAngle);
  const largeArc = fillAngle - startAngle > Math.PI ? 1 : 0;
  const bgPath   = `M ${start.x} ${start.y} A ${r} ${r} 0 1 1 ${bgEnd.x} ${bgEnd.y}`;
  const fillPath = fillAngle > startAngle + 0.01
    ? `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}` : "";
  const color = percent < 10 ? "#22c55e" : percent < 20 ? "#f59e0b" : percent < 30 ? "#f97316" : "#ef4444";
  return (
    <svg viewBox="0 0 220 120" className="gauge-svg">
      <path d={bgPath} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="14" strokeLinecap="round" />
      {fillPath && <path d={fillPath} fill="none" stroke={color} strokeWidth="14" strokeLinecap="round"
        style={{ filter: `drop-shadow(0 0 8px ${color}88)` }} />}
      <text x={cx} y={cy - 10} textAnchor="middle" className="gauge-value" fill={color}>{percent.toFixed(1)}%</text>
      <text x={cx} y={cy + 12} textAnchor="middle" className="gauge-label">10-year risk</text>
    </svg>
  );
}

function SliderField({ field, value, onChange }) {
  const pct = ((value - field.min) / (field.max - field.min)) * 100;
  return (
    <div className="slider-field">
      <div className="slider-header">
        <span className="slider-label">{field.label}</span>
        <span className="slider-value">{Number(value).toFixed(field.step < 1 ? 1 : 0)} <span className="slider-unit">{field.unit}</span></span>
      </div>
      <input type="range" min={field.min} max={field.max} step={field.step} value={value}
        onChange={(e) => onChange(field.name, parseFloat(e.target.value))}
        className="slider-input" style={{ "--pct": `${pct}%` }} />
      {field.help && <p className="slider-help">{field.help}</p>}
    </div>
  );
}

function SHAPBar({ item, maxAbs }) {
  const width   = Math.abs(item.shap_value) / maxAbs * 100;
  const isRisk  = item.direction === "increases_risk";
  const isNeutral = item.direction === "neutral";
  const color   = isNeutral ? "#6b7280" : isRisk ? "rgba(239,68,68,0.75)" : "rgba(34,197,94,0.75)";
  const label   = FEATURE_LABELS[item.feature] || item.feature;
  return (
    <div className="contrib-row">
      <div className="contrib-meta">
        <span className="contrib-label">{label}</span>
        <span className="contrib-raw">= {item.value}</span>
      </div>
      <div className="contrib-track">
        <div className="contrib-fill" style={{ width: `${width}%`, background: color }} />
      </div>
      <span className="contrib-val" style={{ color: isNeutral ? "#6b7280" : isRisk ? "#f87171" : "#4ade80" }}>
        {item.shap_value > 0 ? "+" : ""}{item.shap_value.toFixed(4)}
      </span>
    </div>
  );
}

export default function App() {
  const [values, setValues]   = useState(DEFAULT_VALUES);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [step, setStep]       = useState(0);

  const handleChange = (name, value) => setValues((p) => ({ ...p, [name]: value }));

  const handleSubmit = async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Prediction failed"); }
      setResult(await res.json());
      setStep(2);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const reset = () => { setResult(null); setError(null); setStep(0); setValues(DEFAULT_VALUES); };

  const maxAbs = result
    ? Math.max(...result.shap_contributions.map((c) => Math.abs(c.shap_value)), 0.0001)
    : 1;

  return (
    <div className="app">
      <div className="bg-grid" aria-hidden />
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <circle cx="14" cy="14" r="13" stroke="#ef4444" strokeWidth="1.5" />
              <path d="M6 14h3l2.5-6 3 12 2.5-8L19 14h3" stroke="#ef4444" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span>CVD Risk Predictor</span>
          </div>
          <div className="header-tag">ML-Powered · Research Tool</div>
        </div>
      </header>

      <main className="main">
        {step === 2 && result ? (
          <div className="results-wrap animate-in">
            <div className="results-grid">

              {/* Risk Gauge */}
              <div className="card card--gauge">
                <div className="card-label">10-Year CVD Risk</div>
                <GaugeArc percent={result.risk_probability * 100} />
                <div className="risk-band" style={{ color: result.risk_band.color, borderColor: result.risk_band.color + "44" }}>
                  {result.risk_band.label} Risk
                </div>
                <p className="risk-desc">{result.risk_band.description}</p>
              </div>

              {/* LRS Card */}
              <div className="card card--lrs">
                <div className="card-label">Lifestyle Risk Score</div>
                <div className="lrs-value">{result.lrs_percent}</div>
                <div className="lrs-bar-wrap">
                  <div className="lrs-bar-track">
                    <div className="lrs-bar-fill" style={{ width: result.lrs_percent }} />
                  </div>
                  <div className="lrs-ticks"><span>Low</span><span>Moderate</span><span>High</span></div>
                </div>
                <p className="lrs-desc">Composite of sleep, activity, smoking, sedentary time, and alcohol — normalised to [0,1].</p>
                <div className="input-summary">
                  <div className="summary-title">Clinical Inputs</div>
                  <div className="summary-grid">
                    <span>Age</span><span>{values.age} yrs</span>
                    <span>Sex</span><span>{values.sex === 1 ? "Male" : "Female"}</span>
                    <span>Cholesterol</span><span>{values.cholesterol} mg/dL</span>
                    <span>Systolic BP</span><span>{values.systolic_bp} mmHg</span>
                    <span>FBS &gt;120</span><span>{values.fasting_blood_sugar ? "Yes" : "No"}</span>
                    <span>Max HR</span><span>{values.max_heart_rate} bpm</span>
                  </div>
                </div>
              </div>

              {/* SHAP Contributions */}
              <div className="card card--contrib">
                <div className="card-label">SHAP Feature Contributions</div>
                <p className="contrib-intro">
                  Each bar shows how much that feature pushed the prediction above or below the model baseline
                  ({result.expected_value.toFixed(4)}). Red = increases risk, green = decreases risk.
                </p>
                <div className="contrib-list">
                  {result.shap_contributions.map((item) => (
                    <SHAPBar key={item.feature} item={item} maxAbs={maxAbs} />
                  ))}
                </div>
                <p className="contrib-note">
                  SHAP values computed via TreeExplainer on the base XGBoost model. Values represent
                  additive contributions in log-odds space.
                </p>
              </div>

            </div>

            <div className="disclaimer">
              ⚠ This tool is for research and educational purposes only. It does not constitute medical advice.
              Consult a qualified clinician for any health decisions.
            </div>
            <button className="btn btn--ghost" onClick={reset}>← Start Over</button>
          </div>

        ) : (
          <div className="form-wrap animate-in">
            <div className="form-header">
              <h1 className="form-title">Cardiovascular Risk Assessment</h1>
              <p className="form-subtitle">10-year CHD risk prediction using XGBoost with isotonic calibration and composite Lifestyle Risk Score</p>
            </div>

            <div className="steps">
              {["Clinical Data", "Lifestyle Data"].map((s, i) => (
                <button key={i} className={`step-btn ${step === i ? "step-btn--active" : ""}`} onClick={() => setStep(i)}>
                  <span className="step-num">{i + 1}</span>{s}
                </button>
              ))}
            </div>

            {step === 0 && (
              <div className="section animate-in">
                <div className="section-header">
                  <h2>Clinical Measurements</h2>
                  <p>Standard cardiovascular risk markers from clinical assessment</p>
                </div>
                <div className="field-grid">
                  {CLINICAL_FIELDS.map((f) => (
                    <div className="field-group" key={f.name}>
                      <label className="field-label">{f.label}</label>
                      <div className="input-wrap">
                        <input type="number" min={f.min} max={f.max} step={f.step}
                          value={values[f.name]}
                          onChange={(e) => handleChange(f.name, parseFloat(e.target.value))}
                          className="field-input" placeholder={f.placeholder} />
                        <span className="input-unit">{f.unit}</span>
                      </div>
                    </div>
                  ))}
                  <div className="field-group">
                    <label className="field-label">Biological Sex</label>
                    <div className="toggle-group">
                      {[{v:1,l:"Male"},{v:0,l:"Female"}].map(({v,l}) => (
                        <button key={v} className={`toggle-btn ${values.sex === v ? "toggle-btn--on" : ""}`}
                          onClick={() => handleChange("sex", v)}>{l}</button>
                      ))}
                    </div>
                  </div>
                  <div className="field-group">
                    <label className="field-label">Fasting Blood Sugar &gt; 120 mg/dL</label>
                    <div className="toggle-group">
                      {[{v:1,l:"Yes"},{v:0,l:"No"}].map(({v,l}) => (
                        <button key={v} className={`toggle-btn ${values.fasting_blood_sugar === v ? "toggle-btn--on" : ""}`}
                          onClick={() => handleChange("fasting_blood_sugar", v)}>{l}</button>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="form-actions">
                  <button className="btn btn--primary" onClick={() => setStep(1)}>Next: Lifestyle Data →</button>
                </div>
              </div>
            )}

            {step === 1 && (
              <div className="section animate-in">
                <div className="section-header">
                  <h2>Lifestyle Risk Factors</h2>
                  <p>Used to compute the composite Lifestyle Risk Score (LRS)</p>
                </div>
                <div className="slider-grid">
                  {LIFESTYLE_FIELDS.map((f) => (
                    <SliderField key={f.name} field={f} value={values[f.name]} onChange={handleChange} />
                  ))}
                </div>
                {error && <div className="error-banner"><strong>Error:</strong> {error}</div>}
                <div className="form-actions">
                  <button className="btn btn--ghost" onClick={() => setStep(0)}>← Back</button>
                  <button className="btn btn--primary" onClick={handleSubmit} disabled={loading}>
                    {loading ? <span className="loading-dots">Analysing<span>.</span><span>.</span><span>.</span></span>
                             : "Generate Risk Report →"}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}