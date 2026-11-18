const STEPS = [
  {
    icon: "📸",
    title: "Upload a leaf photo",
    body: "Take a clear, well-lit close-up of the affected leaf and drag it into the scanner — or tap to browse.",
  },
  {
    icon: "🔬",
    title: "AI scans for disease",
    body: "EfficientNet-B2 runs 38 disease pattern checks against your image in under a second.",
  },
  {
    icon: "💊",
    title: "Get your diagnosis",
    body: "Receive the disease name, confidence score, and actionable treatment steps to protect your crop.",
  },
];

export default function HowItWorks() {
  return (
    <section className="how-it-works" id="how-it-works" aria-labelledby="hiw-heading">
      <h2 className="hiw-heading" id="hiw-heading">How it works</h2>
      <p className="hiw-sub">Three steps from photo to treatment plan.</p>

      <div className="hiw-steps">
        {STEPS.map((step, i) => (
          <div className="hiw-step" key={i}>
            <div className="hiw-step-icon" aria-hidden="true">
              {step.icon}
            </div>
            <div className="hiw-step-content">
              <h3 className="hiw-step-title">{step.title}</h3>
              <p className="hiw-step-body">{step.body}</p>
            </div>
            {i < STEPS.length - 1 && (
              <div className="hiw-connector" aria-hidden="true">→</div>
            )}
          </div>
        ))}
      </div>

      {/* Model info strip */}
      <div className="model-strip" id="about">
        <div className="model-stat">
          <span className="model-num">38</span>
          <span className="model-label">disease classes</span>
        </div>
        <div className="model-divider" aria-hidden="true" />
        <div className="model-stat">
          <span className="model-num">14</span>
          <span className="model-label">plant species</span>
        </div>
        <div className="model-divider" aria-hidden="true" />
        <div className="model-stat">
          <span className="model-num">B2</span>
          <span className="model-label">EfficientNet arch</span>
        </div>
        <div className="model-divider" aria-hidden="true" />
        <div className="model-stat">
          <span className="model-num">&lt;1s</span>
          <span className="model-label">inference time</span>
        </div>
      </div>
    </section>
  );
}
