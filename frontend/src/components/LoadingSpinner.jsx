export default function LoadingSpinner({ previewUrl }) {
  return (
    <div className="loading-wrap" role="status" aria-live="polite">
      <div className="loading-image-frame">
        {previewUrl && (
          <img src={previewUrl} alt="Leaf being analyzed" className="loading-preview" />
        )}
        {/* Animated scan line */}
        <div className="scan-line" aria-hidden="true" />
        {/* Pulsing ring */}
        <div className="scan-ring-pulse ring-1" aria-hidden="true" />
        <div className="scan-ring-pulse ring-2" aria-hidden="true" />
      </div>

      <div className="loading-text-wrap">
        <p className="loading-title">Analyzing leaf…</p>
        <div className="loading-dots" aria-hidden="true">
          <span /><span /><span />
        </div>
        <p className="loading-sub">
          Running EfficientNet-B2 · Checking 38 disease patterns
        </p>
      </div>
    </div>
  );
}
