import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import DiagnosisCard from "./DiagnosisCard";

function result(overrides = {}) {
  return {
    top_prediction: { label: "Tomato___Early_blight", confidence: 0.91 },
    top_5: [
      { label: "Tomato___Early_blight", confidence: 0.91 },
      { label: "Tomato___Late_blight", confidence: 0.06 },
      { label: "Tomato___healthy", confidence: 0.03 },
    ],
    inference_ms: 42,
    plant: "Tomato",
    disease: "Early Blight",
    is_healthy: false,
    ...overrides,
  };
}

describe("DiagnosisCard", () => {
  it("renders plant, disease, and confidence", () => {
    render(
      <DiagnosisCard result={result()} previewUrl="blob:leaf" onReset={vi.fn()} />
    );

    expect(screen.getByText("Tomato")).toBeInTheDocument();
    expect(screen.getByText("Early Blight")).toBeInTheDocument();
    expect(screen.getByText("91%")).toBeInTheDocument();
    expect(screen.getByText(/42 ms/)).toBeInTheDocument();
    expect(screen.queryByText(/low confidence/i)).not.toBeInTheDocument();
  });

  it("warns when confidence is below 70%", () => {
    render(
      <DiagnosisCard
        result={result({
          top_prediction: { label: "Tomato___Early_blight", confidence: 0.62 },
        })}
        previewUrl="blob:leaf"
        onReset={vi.fn()}
      />
    );

    expect(screen.getByText("62%")).toBeInTheDocument();
    expect(screen.getByText(/low confidence/i)).toBeInTheDocument();
  });

  it("expands treatment steps and alternative predictions", async () => {
    const user = userEvent.setup();
    render(
      <DiagnosisCard result={result()} previewUrl="blob:leaf" onReset={vi.fn()} />
    );

    await user.click(screen.getByRole("button", { name: /treatment steps/i }));
    expect(screen.getByRole("list")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /other possibilities/i }));
    expect(screen.getByText(/Late Blight/)).toBeInTheDocument();
    expect(screen.getByText("6%")).toBeInTheDocument();
  });
});
