import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const prediction = {
  top_prediction: { label: "Tomato___Early_blight", confidence: 0.88 },
  top_5: [
    { label: "Tomato___Early_blight", confidence: 0.88 },
    { label: "Tomato___healthy", confidence: 0.12 },
  ],
  inference_ms: 18,
  plant: "Tomato",
  disease: "Early Blight",
  is_healthy: false,
};

function uploadInput() {
  return document.querySelector('input[type="file"]');
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("goes from idle to preview to a diagnosis", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => prediction,
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);

    expect(screen.getByText(/drop a leaf photo/i)).toBeInTheDocument();

    const file = new File(["leaf"], "tomato.jpg", { type: "image/jpeg" });
    await user.upload(uploadInput(), file);

    expect(screen.getByText("tomato.jpg")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /analyze leaf/i }));

    expect(await screen.findByText("Early Blight")).toBeInTheDocument();
    expect(screen.getByText("88%")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0][0])).toMatch(/\/predict$/);
  });

  it("shows the server error and can reset", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        json: async () => ({ detail: "Model is not loaded." }),
      })
    );
    const user = userEvent.setup();
    render(<App />);

    await user.upload(
      uploadInput(),
      new File(["leaf"], "leaf.png", { type: "image/png" })
    );
    await user.click(screen.getByRole("button", { name: /analyze leaf/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/model is not loaded/i);
    await user.click(screen.getByRole("button", { name: /try again/i }));
    expect(screen.getByText(/drop a leaf photo/i)).toBeInTheDocument();
  });
});
