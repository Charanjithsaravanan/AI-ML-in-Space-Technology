"use client";

import { ChangeEvent, useState } from "react";
import GlassPanel from "../components/glass-panel";

type Analysis = {
  filename: string;
  device: string;
  prediction: { class_name: string; confidence: number; top_predictions: { class_name: string; probability: number }[] };
  cloud: { cloudy: boolean; cloud_score: number; source: string };
  sensor: { corrupted: boolean; likely_mode: string | null; sigma: number; dropped_columns: number; saturated_fraction: number };
  robustness: { mode: string; severity: number; prediction: string; confidence: number }[];
  heatmap: number[][];
};

const API_URL = process.env.NEXT_PUBLIC_GEOSAT_API_URL ?? "http://localhost:8000";
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

function Heatmap({ values }: { values: number[][] }) {
  return <div className="heatmap" aria-label="Occlusion saliency heatmap">{values.flatMap((row, rowIndex) => row.map((value, columnIndex) => <span key={`${rowIndex}-${columnIndex}`} style={{ opacity: 0.15 + value * 0.85 }} title={`importance ${value.toFixed(2)}`} />))}</div>;
}

export default function AnalysisWorkspace() {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setPreview(URL.createObjectURL(file));
    setBusy(true);
    setError(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch(`${API_URL}/analyze`, { method: "POST", body });
      if (!response.ok) throw new Error((await response.json()).detail ?? "Analysis failed");
      setAnalysis(await response.json());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not reach the analysis API");
    } finally {
      setBusy(false);
    }
  }

  return (
    <GlassPanel className="analysis panel">
      <div className="section-heading">
        <div><p className="eyebrow">Live science console</p><h2>Analyze an Earth-observation image</h2></div>
        <label className="upload-button">ATTACH IMAGE<input type="file" accept="image/*" onChange={submit} /></label>
      </div>
      <p className="muted">Upload one image for a real backend inference pass. The API resizes it to 64×64, processes it in memory, and does not retain the upload.</p>
      {busy && <p className="signal">Running classifier, cloud screen, sensor triage, robustness probes, and occlusion analysis…</p>}
      {error && <p className="error">{error}. Start the API with <code>uvicorn api.app:app --reload</code>.</p>}
      {analysis && (
        <div className="analysis-grid">
          {preview && <img className="preview" src={preview} alt="Uploaded satellite image" />}
          <div className="result-card"><span className="eyebrow">Primary classification</span><strong>{analysis.prediction.class_name}</strong><b>{percent(analysis.prediction.confidence)} confidence</b><small>executed on {analysis.device}</small></div>
          <div className="result-card"><span className="eyebrow">Cloud assessment</span><strong>{analysis.cloud.cloudy ? "CLOUDY" : "CLEAR"}</strong><b>{percent(analysis.cloud.cloud_score)} cloud score</b><small>{analysis.cloud.source}</small></div>
          <div className="result-card"><span className="eyebrow">Sensor triage</span><strong>{analysis.sensor.corrupted ? "DEGRADED" : "CLEAN"}</strong><b>{analysis.sensor.likely_mode ?? "no likely mode"}</b><small>σ {analysis.sensor.sigma.toFixed(4)} · dropout {percent(analysis.sensor.dropped_columns)}</small></div>
          <div className="result-card probability-card"><span className="eyebrow">Top predictions</span>{analysis.prediction.top_predictions.map((item) => <div className="probability" key={item.class_name}><span>{item.class_name}</span><div className="bar-track"><div className="bar selected" style={{ width: `${item.probability * 100}%` }} /></div><b>{percent(item.probability)}</b></div>)}</div>
          <div className="result-card heatmap-card"><span className="eyebrow">Occlusion importance</span><Heatmap values={analysis.heatmap} /><small>Brighter cells caused the largest target-confidence loss when masked.</small></div>
          <div className="result-card robustness-card"><span className="eyebrow">Synthetic sensor stress test</span>{analysis.robustness.map((item) => <div className="stress-row" key={`${item.mode}-${item.severity}`}><span>{item.mode} · {item.severity.toFixed(2)}</span><b>{item.prediction} · {percent(item.confidence)}</b></div>)}</div>
        </div>
      )}
    </GlassPanel>
  );
}
