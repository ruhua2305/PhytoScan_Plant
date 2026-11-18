import { useState } from "react";
import { getDiseaseInfo, SEVERITY_CONFIG } from "../utils/diseaseInfo";

export default function DiagnosisCard({ result, previewUrl, onReset }) {
  const [showTreatment, setShowTreatment] = useState(false);
  const [showAlternatives, setShowAlternatives] = useState(false);

  const { top_prediction, top_5, inference_ms, plant, disease, is_healthy } = result;
  const info = getDiseaseInfo(top_prediction.label);
  const severity = SEVERITY_CONFIG[info.severity] || SEVERITY_CONFIG.unknown;
  const confidence = Math.round(top_prediction.confidence * 100);

  return (
    <div className="diagnosis-card" role="region" aria-label="Diagnosis results">
      {/* ── Image preview ──────────────────────────────────────────────── */}
      <div className="diagnosis-image-wrap">
        <div className="diagnosis-image-frame">
          <img src={previewUrl} alt={`Uploaded ${plant} leaf`} className="diagnosis-image" />
          {/* Scan ring overlay */}
          <div className={`scan-ring ${is_healthy ? "ring-healthy" : "ring-disease"}`} aria-hidden="true" />
        </div>
        <div className="inference-badge" title="Model inference time">
          ⚡ {inference_ms} ms
        </div>
      </div>

      {/* ── Main result ────────────────────────────────────────────────── */}
      <div className="diagnosis-body">
        {/* Status badge */}
        <div
          className="severity-badge"
          style={{ color: severity.color, background: severity.bg, borderColor: severity.ring }}
        >
          {info.emoji} {severity.label}
        </div>

        {/* Plant + disease name */}
        <div className="diagnosis-heading">
          <span className="plant-name">{plant}</span>
          <span className="disease-name">{disease}</span>
        </div>

        {/* Confidence meter */}
        <div className="confidence-section" aria-label={`Confidence: ${confidence}%`}>
          <div className="confidence-header">
            <span className="confidence-label">Confidence</span>
            <span className="confidence-value">{confidence}%</span>
          </div>
          <div className="confidence-track" role="progressbar" aria-valuenow={confidence} aria-valuemin={0} aria-valuemax={100}>
            <div
              className="confidence-fill"
              style={{
                width: `${confidence}%`,
                background: confidence > 85
                  ? severity.color
                  : confidence > 60
                  ? "#f59e0b"
                  : "#9ca3af",
              }}
            />
          </div>
          {confidence < 70 && (
            <p className="confidence-warning">
              ⚠ Low confidence — try a clearer, better-lit photo of the leaf.
            </p>
          )}
        </div>

        {/* Description */}
        <p className="disease-description">{info.description}</p>

        {/* Treatment advice (expandable) */}
        <button
          className={`expand-btn ${showTreatment ? "expanded" : ""}`}
          onClick={() => setShowTreatment((v) => !v)}
          aria-expanded={showTreatment}
        >
          {is_healthy ? "Care Tips" : "Treatment Steps"}
          <ChevronIcon />
        </button>
        {showTreatment && (
          <ol className="treatment-list" role="list">
            {info.treatment.map((step, i) => (
              <li key={i} className="treatment-step">
                <span className="step-num" aria-hidden="true">{i + 1}</span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
        )}

        {/* Divider */}
        <hr className="diagnosis-divider" />

        {/* Alternative predictions */}
        <button
          className={`expand-btn ${showAlternatives ? "expanded" : ""}`}
          onClick={() => setShowAlternatives((v) => !v)}
          aria-expanded={showAlternatives}
        >
          Other possibilities
          <ChevronIcon />
        </button>
        {showAlternatives && (
          <div className="alternatives-list">
            {top_5.slice(1).map((item, i) => {
              const altInfo = getDiseaseInfo(item.label);
              const altPct = Math.round(item.confidence * 100);
              return (
                <div key={i} className="alt-item">
                  <span className="alt-emoji">{altInfo.emoji}</span>
                  <div className="alt-info">
                    <span className="alt-label">{altInfo.plant} — {altInfo.displayName}</span>
                    <div className="alt-track">
                      <div className="alt-fill" style={{ width: `${altPct}%` }} />
                    </div>
                  </div>
                  <span className="alt-pct">{altPct}%</span>
                </div>
              );
            })}
          </div>
        )}

        {/* Reset button */}
        <button className="reset-btn" onClick={onReset}>
          Examine another leaf
        </button>
      </div>
    </div>
  );
}

function ChevronIcon() {
  return (
    <svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true" className="chevron-icon">
      <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clipRule="evenodd" />
    </svg>
  );
}
