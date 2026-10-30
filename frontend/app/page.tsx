import benchmark from "../public/data/benchmark.json";
import robustness from "../public/data/robustness.json";
import cloud from "../public/data/cloud_screener.json";
import InteractiveBackground from "./interactive-background";
import AnalysisWorkspace from "./analysis";
import GlassPanel from "../components/glass-panel";
import TelemetryBento from "../components/telemetry-bento";

type Condition = (typeof robustness.conditions)[number];

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const milliseconds = (value: number) => `${value.toFixed(3)} ms`;

function LatencyChart() {
  const entries = Object.entries(benchmark.latency_by_device);
  const fastest = benchmark.latency.device;
  return (
    <GlassPanel className="panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Execution profile</p>
          <h2>Device latency</h2>
        </div>
        <span className="badge">selected: {fastest}</span>
      </div>
      <div className="bars">
        {entries.map(([device, result]) => {
          const width = Math.max(12, (result.median_ms / Math.max(...entries.map(([, item]) => item.median_ms))) * 100);
          return (
            <div className="bar-row" key={device}>
              <span>{device.toUpperCase()}</span>
              <div className="bar-track"><div className={`bar ${device === fastest ? "selected" : ""}`} style={{ width: `${width}%` }} /></div>
              <b>{milliseconds(result.median_ms)}</b>
            </div>
          );
        })}
      </div>
      <p className="muted">Auto mode benchmarks complete CPU and CUDA paths, then keeps the faster one. The model is not split between devices for a single patch.</p>
    </GlassPanel>
  );
}

function RobustnessChart() {
  const conditions = robustness.conditions as Condition[];
  const clean = conditions[0];
  return (
    <GlassPanel className="panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Sensor resilience</p>
          <h2>Robustness conditions</h2>
        </div>
        <span className="badge">{conditions.length} conditions</span>
      </div>
      <div className="robustness-list">
        {conditions.map((condition) => (
          <div className="robustness-row" key={`${condition.mode}-${condition.severity}`}>
            <span>{condition.mode === "clean" ? "Clean" : `${condition.mode} · ${condition.severity.toFixed(2)}`}</span>
            <div className="bar-track"><div className="bar selected" style={{ width: `${condition.macro_f1 * 100}%` }} /></div>
            <b>{percent(condition.macro_f1)}</b>
          </div>
        ))}
      </div>
      <p className="muted">Clean baseline: {percent(clean.macro_f1)} macro F1 · {percent(clean.accuracy)} accuracy.</p>
    </GlassPanel>
  );
}

function ReliabilityPanel() {
  const conditions = robustness.conditions as Condition[];
  const clean = conditions.find((condition) => condition.mode === "clean") ?? conditions[0];
  const noisy = conditions.filter((condition) => condition.mode !== "clean");
  const worst = noisy.reduce((lowest, condition) => condition.macro_f1 < lowest.macro_f1 ? condition : lowest, noisy[0]);
  const averageNoisy = noisy.reduce((total, condition) => total + condition.macro_f1, 0) / noisy.length;
  const degradation = clean.macro_f1 - worst.macro_f1;
  const recommendation = worst.macro_f1 >= 0.70 ? "USABLE WITH GUARDRAILS" : "RESEARCH / PILOT ONLY";

  return (
    <GlassPanel className="panel reliability-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Reliability & usability</p>
          <h2>Evidence before deployment</h2>
        </div>
        <span className="badge">{recommendation}</span>
      </div>
      <div className="reliability-grid">
        <div><span>Clean macro F1</span><strong>{percent(clean.macro_f1)}</strong><small>{percent(clean.accuracy)} accuracy · {robustness.classes.length} classes</small></div>
        <div><span>Average noisy F1</span><strong>{percent(averageNoisy)}</strong><small>all {noisy.length} synthetic stress views</small></div>
        <div><span>Worst stress case</span><strong>{percent(worst.macro_f1)}</strong><small>{worst.mode} at severity {worst.severity.toFixed(2)}</small></div>
        <div><span>Worst-case drop</span><strong>{percent(degradation)}</strong><small>relative to the clean validation baseline</small></div>
      </div>
      <p className="reliability-callout">
        <b>Recommended use:</b> suitable for analyst assistance, triage, and onboard
        screening when a human or downstream rule can review uncertain predictions.
        Do not use this evidence alone for safety-critical or fully autonomous
        decisions: the stress tests are synthetic and the evaluation set is a
        fixed 400-patch holdout.
      </p>
    </GlassPanel>
  );
}

export default function Dashboard() {
  return (
    <main>
      <InteractiveBackground />
      <div className="shell">
        <section className="landing" aria-labelledby="site-title">
          <div className="landing-copy">
            <p className="eyebrow">EARTH OBSERVATION / MODEL TELEMETRY</p>
            <h1 id="site-title">GEOSAT<span>.</span></h1>
            <p className="lede">Read-only mission intelligence for a compact satellite imaging model.</p>
            <a className="scroll-cue" href="#dashboard">
              <span className="scroll-line" />
              SCROLL TO EXPLORE
            </a>
          </div>
          <div className="status"><i /> DATA SNAPSHOTS CONNECTED</div>
        </section>

        <div id="dashboard" className="dashboard-content">
          <TelemetryBento metrics={[
            { id: "f1", label: "MODEL QUALITY", title: percent(robustness.conditions[0].macro_f1), description: "Clean macro F1 · 400 held-out patches" },
            { id: "footprint", label: "MODEL FOOTPRINT", title: `${benchmark.torchscript_mb.toFixed(3)} MiB`, description: `${benchmark.parameters.toLocaleString()} trainable parameters` },
            { id: "latency", label: "EXECUTION", title: milliseconds(benchmark.latency.median_ms), description: `${benchmark.latency.device.toUpperCase()} · batch 1 · selected path` },
            { id: "cloud", label: "CLOUD SCREENING", title: percent(cloud.macro_f1), description: cloud.source === "eurosat-synthetic" ? "EuroSAT-derived fallback" : "CloudSEN12 signal" },
          ]} />

          <div className="dashboard-grid">
            <LatencyChart />
            <GlassPanel className="panel cloud-panel">
              <p className="eyebrow">Cloud screening</p>
              <h2>Signal provenance</h2>
              <div className="signal"><span className="signal-dot" />{cloud.source === "eurosat-synthetic" ? "EuroSAT-derived signal" : "CloudSEN12 masks"}</div>
              <dl>
                <div><dt>Accuracy</dt><dd>{percent(cloud.accuracy)}</dd></div>
                <div><dt>Macro F1</dt><dd>{percent(cloud.macro_f1)}</dd></div>
                <div><dt>Training patches</dt><dd>{cloud.n_train.toLocaleString()}</dd></div>
                <div><dt>Test patches</dt><dd>{cloud.n_test.toLocaleString()}</dd></div>
              </dl>
              <p className="muted">This dashboard only retrieves the saved screener metadata. It does not download CloudSEN12 or retrain the model.</p>
            </GlassPanel>
          </div>

          <RobustnessChart />
          <ReliabilityPanel />
          <AnalysisWorkspace />

          <footer>Source: <code>results/*.json</code> and <code>runs/cloud_screener.json</code> · Run <code>npm run sync-data</code> after generating new Python results.</footer>
        </div>
      </div>
    </main>
  );
}
