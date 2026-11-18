import React, { useState, useCallback } from "react";
import Header from "./components/Header";
import UploadZone from "./components/UploadZone";
import LoadingSpinner from "./components/LoadingSpinner";
import DiagnosisCard from "./components/DiagnosisCard";
import HowItWorks from "./components/HowItWorks";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// App states
const STATE = {
  IDLE: "idle",
  PREVIEWING: "previewing",
  LOADING: "loading",
  RESULT: "result",
  ERROR: "error",
};

export default function App() {
  const [appState, setAppState] = useState(STATE.IDLE);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");

  // Called by UploadZone when user selects/drops an image
  const handleImageSelected = useCallback((file, url) => {
    setSelectedFile(file);
    setPreviewUrl(url);
    setAppState(STATE.PREVIEWING);
  }, []);

  // Sends the image to the FastAPI backend
  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;

    setAppState(STATE.LOADING);
    setErrorMsg("");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Unknown server error." }));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      setAppState(STATE.RESULT);
    } catch (err) {
      setErrorMsg(err.message || "Could not reach the server. Is the backend running?");
      setAppState(STATE.ERROR);
    }
  }, [selectedFile]);

  // Reset to initial state
  const handleReset = useCallback(() => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setErrorMsg("");
    setAppState(STATE.IDLE);
  }, [previewUrl]);

  return (
    <div className="app">
      <Header />

      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <main className="hero">
        <div className="hero-bg" aria-hidden="true">
          {/* Decorative botanical SVG background */}
          <BotanicalBg />
        </div>

        <div className="hero-content">
          <div className="hero-eyebrow">AI-powered plant diagnostics</div>
          <h1 className="hero-title">
            Diagnose your plant<br />
            <span className="hero-title-accent">in seconds</span>
          </h1>
          <p className="hero-subtitle">
            Upload a photo of a leaf and let our EfficientNet-B2 model identify diseases
            across 14 crops — from apple to tomato.
          </p>

          {/* ── Scanner panel ──────────────────────────────────────────── */}
          <div className="scanner-panel">

            {/* IDLE: show upload zone */}
            {appState === STATE.IDLE && (
              <UploadZone onImageSelected={handleImageSelected} disabled={false} />
            )}

            {/* PREVIEWING: image selected, ready to analyze */}
            {appState === STATE.PREVIEWING && (
              <div className="preview-stage">
                <div className="preview-frame">
                  <img
                    src={previewUrl}
                    alt="Selected leaf"
                    className="preview-img"
                  />
                  <button
                    className="preview-change"
                    onClick={handleReset}
                    aria-label="Choose a different image"
                  >
                    ✕
                  </button>
                </div>
                <div className="preview-actions">
                  <p className="preview-filename">{selectedFile?.name}</p>
                  <button className="analyze-btn" onClick={handleAnalyze}>
                    <span>Analyze leaf</span>
                    <MicroscopeIcon />
                  </button>
                  <button className="change-btn" onClick={handleReset}>
                    Choose different photo
                  </button>
                </div>
              </div>
            )}

            {/* LOADING: running inference */}
            {appState === STATE.LOADING && (
              <LoadingSpinner previewUrl={previewUrl} />
            )}

            {/* RESULT: show diagnosis card */}
            {appState === STATE.RESULT && result && (
              <DiagnosisCard
                result={result}
                previewUrl={previewUrl}
                onReset={handleReset}
              />
            )}

            {/* ERROR */}
            {appState === STATE.ERROR && (
              <div className="error-panel" role="alert">
                <div className="error-icon" aria-hidden="true">⚠</div>
                <h2 className="error-title">Something went wrong</h2>
                <p className="error-body">{errorMsg}</p>
                <button className="analyze-btn" onClick={handleReset}>
                  Try again
                </button>
              </div>
            )}
          </div>

          {/* Supported plants */}
          {(appState === STATE.IDLE) && (
            <div className="plant-chips" aria-label="Supported plant species">
              {["🍎 Apple", "🌽 Corn", "🍇 Grape", "🍅 Tomato", "🥔 Potato",
                "🍑 Peach", "🫑 Pepper", "🍓 Strawberry", "🍊 Orange", "🌱 More"].map((p) => (
                <span className="plant-chip" key={p}>{p}</span>
              ))}
            </div>
          )}
        </div>
      </main>

      {/* ── How it works ──────────────────────────────────────────────────── */}
      <HowItWorks />

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="site-footer">
        <p>
          Built with EfficientNet-B2 · PlantVillage dataset ·{" "}
          <a href="https://fastapi.tiangolo.com" target="_blank" rel="noopener noreferrer">
            FastAPI
          </a>{" "}
          +{" "}
          <a href="https://vitejs.dev" target="_blank" rel="noopener noreferrer">
            Vite
          </a>
        </p>
        <p className="footer-disclaimer">
          For educational and research purposes only. Always consult a certified agronomist for
          critical crop decisions.
        </p>
      </footer>
    </div>
  );
}

function MicroscopeIcon() {
  return (
    <svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true" className="btn-icon">
      <path fillRule="evenodd" d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z" clipRule="evenodd" />
    </svg>
  );
}

function BotanicalBg() {
  return (
    <svg
      className="botanical-bg"
      viewBox="0 0 1200 800"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
    >
      {/* Large background leaves - top left */}
      <ellipse cx="100" cy="150" rx="180" ry="90" transform="rotate(-30 100 150)" fill="currentColor" opacity="0.04" />
      <path d="M100 60 Q200 120 100 240 Q0 120 100 60Z" fill="currentColor" opacity="0.06" />
      <line x1="100" y1="60" x2="100" y2="240" stroke="currentColor" strokeWidth="1" opacity="0.1" />

      {/* Top right cluster */}
      <ellipse cx="1100" cy="100" rx="160" ry="80" transform="rotate(20 1100 100)" fill="currentColor" opacity="0.04" />
      <path d="M1100 30 Q1180 100 1100 180 Q1020 100 1100 30Z" fill="currentColor" opacity="0.06" />
      <line x1="1100" y1="30" x2="1100" y2="180" stroke="currentColor" strokeWidth="1" opacity="0.1" />

      {/* Bottom left */}
      <ellipse cx="150" cy="700" rx="200" ry="100" transform="rotate(15 150 700)" fill="currentColor" opacity="0.04" />

      {/* Bottom right */}
      <ellipse cx="1050" cy="720" rx="180" ry="90" transform="rotate(-20 1050 720)" fill="currentColor" opacity="0.04" />

      {/* Mid scattered dots — microscope slide grid feel */}
      {[...Array(12)].map((_, i) => (
        <circle
          key={i}
          cx={200 + i * 80}
          cy={400 + (i % 3) * 30}
          r="2"
          fill="currentColor"
          opacity="0.08"
        />
      ))}
    </svg>
  );
}
