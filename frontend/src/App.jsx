import React, { useEffect, useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const initial = {
  machine_type: "M",
  air_temperature_k: 300,
  process_temperature_k: 310,
  rotational_speed_rpm: 1500,
  torque_nm: 40,
  tool_wear_min: 120
};
const trend = Array.from({ length: 12 }, (_, i) => ({
  sample: `T-${11 - i}`,
  temperature: +(298 + i * 0.35 + Math.sin(i * 1.3) * 1.2).toFixed(1),
  vibration: +(2.1 + i * 0.08 + Math.cos(i * 0.8) * 0.3).toFixed(2)
}));

function Field({ label, value, min, max, step = 1, onChange, unit }) {
  return <label className="field">
    <span>{label} <small>{unit}</small></span>
    <div className="field-value">{value}</div>
    <input type="range" min={min} max={max} step={step} value={value}
      onChange={(e) => onChange(Number(e.target.value))} />
    <div className="range"><small>{min}</small><small>{max}</small></div>
  </label>;
}

export default function App() {
  const [reading, setReading] = useState(initial);
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState("Checking API");

  useEffect(() => {
    fetch(`${API}/health`).then(r => r.json())
      .then(d => setApiStatus(d.model_ready ? "API online · Model ready" : "API online · Train model first"))
      .catch(() => setApiStatus("API offline · Start FastAPI"));
    fetch(`${API}/metrics`).then(r => r.ok ? r.json() : null).then(setMetrics).catch(() => {});
  }, []);

  const update = (key, value) => setReading(prev => ({ ...prev, [key]: value }));
  const predict = async () => {
    setLoading(true); setError("");
    try {
      const response = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reading)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed");
      setResult(data);
    } catch (e) {
      setError(`${e.message}. Ensure the API is running and the model has been trained.`);
    } finally { setLoading(false); }
  };

  const probability = result ? Math.round(result.failure_probability * 100) : null;
  return <main className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">F</div><div><b>ForgeSight</b><small>INDUSTRIAL INTELLIGENCE</small></div></div>
      <div className="nav active">◫ <span>Overview</span></div>
      <div className="nav">⌁ <span>Machine health</span></div>
      <div className="nav">◷ <span>Failure predictions</span></div>
      <div className="nav">⚙ <span>Model evaluation</span></div>
      <div className="sidebar-bottom"><div className="pulse"></div><div><b>Demo environment</b><small>AI4I benchmark data</small></div></div>
    </aside>
    <section className="content">
      <header className="topbar">
        <div><div className="eyebrow">OPERATIONS / CONDITION MONITORING</div><h1>Predictive maintenance</h1><p>Machine health intelligence and failure-risk assessment</p></div>
        <div className="api-status"><span className={apiStatus.includes("online") ? "dot good" : "dot"}></span>{apiStatus}</div>
      </header>

      <div className="notice"><b>Benchmark demo</b><span>Predictions use the synthetic UCI AI4I dataset. Do not use for safety-critical maintenance decisions.</span></div>

      <div className="stats">
        <article className="stat"><span>MODEL</span><strong>Random Forest</strong><small>Binary failure classifier</small></article>
        <article className="stat"><span>INPUT SIGNALS</span><strong>6 features</strong><small>Operating conditions + wear</small></article>
        <article className="stat"><span>DATASET</span><strong>10,000 rows</strong><small>UCI AI4I 2020 benchmark</small></article>
        <article className="stat"><span>ANOMALY MODEL</span><strong>Isolation Forest</strong><small>Unusual operating patterns</small></article>
      </div>

      <div className="main-grid">
        <section className="panel">
          <div className="panel-head"><div><span className="eyebrow">INFERENCE WORKSPACE</span><h2>Machine sensor inputs</h2></div><span className="pill">MANUAL TEST</span></div>
          <div className="type-picker"><span>Machine quality type</span><div>{["L","M","H"].map(t => <button key={t} className={reading.machine_type === t ? "type selected" : "type"} onClick={() => update("machine_type", t)}>{t}<small>{t === "L" ? "Low" : t === "M" ? "Medium" : "High"}</small></button>)}</div></div>
          <div className="fields">
            <Field label="Air temperature" unit="K" value={reading.air_temperature_k} min={295} max={305} step={0.1} onChange={v => update("air_temperature_k", v)} />
            <Field label="Process temperature" unit="K" value={reading.process_temperature_k} min={305} max={315} step={0.1} onChange={v => update("process_temperature_k", v)} />
            <Field label="Rotational speed" unit="RPM" value={reading.rotational_speed_rpm} min={1000} max={3000} step={25} onChange={v => update("rotational_speed_rpm", v)} />
            <Field label="Torque" unit="Nm" value={reading.torque_nm} min={10} max={80} step={1} onChange={v => update("torque_nm", v)} />
            <Field label="Tool wear" unit="min" value={reading.tool_wear_min} min={0} max={250} step={5} onChange={v => update("tool_wear_min", v)} />
          </div>
          <button className="primary" onClick={predict} disabled={loading}>{loading ? "Analyzing sensor readings..." : "Run failure-risk assessment ↗"}</button>
          {error && <div className="error">{error}</div>}
        </section>

        <section className="panel result-panel">
          <div className="panel-head"><div><span className="eyebrow">MODEL OUTPUT</span><h2>Risk assessment</h2></div><span className="live-tag">● INFERENCE</span></div>
          {!result ? <div className="empty"><div className="empty-icon">⌁</div><h3>Awaiting sensor readings</h3><p>Adjust the machine inputs, then run an assessment to see its predicted failure risk and anomaly status.</p><button className="secondary" onClick={predict} disabled={loading}>Analyze default machine</button></div> :
          <>
            <div className="risk-circle" style={{"--progress": `${probability}%`}}><div><strong>{probability}%</strong><small>FAILURE RISK</small></div></div>
            <div className={`risk-label ${result.risk_level === "HIGH" ? "high" : "low"}`}>{result.risk_level} RISK</div>
            <p className="result-copy">{result.interpretation}</p>
            <div className="result-row"><span>Failure probability</span><b>{result.failure_probability}</b></div>
            <div className="result-row"><span>Configured threshold</span><b>{result.threshold}</b></div>
            <div className="result-row"><span>Anomaly detected</span><b className={result.anomaly_detected ? "warn" : ""}>{result.anomaly_detected ? "Yes" : "No"}</b></div>
            <div className="result-row"><span>Anomaly score</span><b>{result.anomaly_score}</b></div>
            <div className="result-foot">Anomaly score is a separate signal from failure probability.</div>
          </>}
        </section>
      </div>

      <div className="lower-grid">
        <section className="panel">
          <div className="panel-head"><div><span className="eyebrow">SAMPLE TELEMETRY</span><h2>Sensor trend preview</h2></div><span className="pill">ILLUSTRATIVE</span></div>
          <div className="chart-legend"><span><i className="temp"></i>Temperature (K)</span><span><i className="vib"></i>Vibration index (illustrative)</span></div>
          <div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={trend} margin={{top:10,right:8,left:-20,bottom:0}}><CartesianGrid strokeDasharray="3 3" stroke="#8ecced"/><XAxis dataKey="sample" tick={{fill:"#000000",fontSize:10}} axisLine={false} tickLine={false}/><YAxis tick={{fill:"#000000",fontSize:10}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{background:"#e3f5fd",border:"1px solid #72bedf",borderRadius:8,color:"#000000"}}/><Line type="monotone" dataKey="temperature" stroke="#0369a1" dot={false} strokeWidth={2}/><Line type="monotone" dataKey="vibration" stroke="#4f46e5" dot={false} strokeWidth={2}/></LineChart></ResponsiveContainer></div>
          <p className="muted small">Illustrative chart only. It is not historical telemetry returned by the trained dataset.</p>
        </section>
        <section className="panel">
          <div className="panel-head"><div><span className="eyebrow">VALIDATION</span><h2>Model evaluation</h2></div></div>
          {metrics ? <div className="metric-list">
            <div><span>Failure precision</span><b>{(metrics.precision_failure * 100).toFixed(1)}%</b></div>
            <div><span>Failure recall</span><b>{(metrics.recall_failure * 100).toFixed(1)}%</b></div>
            <div><span>Failure F1</span><b>{(metrics.f1_failure * 100).toFixed(1)}%</b></div>
            <div><span>PR-AUC / Avg. precision</span><b>{metrics.average_precision_pr_auc.toFixed(3)}</b></div>
            <div><span>ROC-AUC</span><b>{metrics.roc_auc.toFixed(3)}</b></div>
            <p className="muted small">Metrics are loaded from your local training run. Review the confusion matrix and false negatives before reporting performance.</p>
          </div> : <p className="muted">Train the model and restart the API to display evaluation metrics.</p>}
        </section>
      </div>
      <footer>ForgeSight · Portfolio prototype · Predictions are decision support, not a substitute for qualified maintenance review.</footer>
    </section>
  </main>;
}
