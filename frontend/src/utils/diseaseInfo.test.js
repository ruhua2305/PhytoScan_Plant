import { describe, expect, it } from "vitest";
import { getDiseaseInfo, SEVERITY_CONFIG } from "./diseaseInfo";

describe("getDiseaseInfo", () => {
  it("returns the catalog entry for a known class", () => {
    const info = getDiseaseInfo("Tomato___Early_blight");
    expect(info.plant).toBe("Tomato");
    expect(info.displayName).toBe("Early Blight");
    expect(info.severity).toBe("medium");
    expect(info.treatment.length).toBeGreaterThan(0);
  });

  it("marks healthy classes as severity none", () => {
    const info = getDiseaseInfo("Apple___healthy");
    expect(info.displayName).toBe("Healthy");
    expect(info.severity).toBe("none");
    expect(SEVERITY_CONFIG[info.severity].label).toBe("Healthy");
  });

  it("falls back for an unknown label", () => {
    const info = getDiseaseInfo("Mango___Leaf_spot");
    expect(info.plant).toBe("Mango");
    expect(info.displayName).toBe("Leaf spot");
    expect(info.severity).toBe("unknown");
    expect(info.emoji).toBe("🌿");
  });
});
