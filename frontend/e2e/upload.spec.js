import { test, expect } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = path.join(__dirname, "fixtures", "leaf.jpg");

test.describe("Upload flow", () => {
  test("shows a validation error for non-image files", async ({ page }) => {
    await page.goto("/");

    const invalid = path.join(__dirname, "fixtures", "not-an-image.txt");
    const fs = await import("node:fs/promises");
    await fs.writeFile(invalid, "hello, not an image");

    await page.setInputFiles('input[type="file"]', invalid);

    const alert = page.getByRole("alert");
    await expect(alert).toBeVisible();
    await expect(alert).toContainText(/JPEG, PNG, or WebP/i);
  });

  test("uploads a leaf and renders diagnosis (mocked backend)", async ({ page }) => {
    await page.route("**/predict", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          top_prediction: { label: "Tomato___Early_blight", confidence: 0.91 },
          top_5: [
            { label: "Tomato___Early_blight", confidence: 0.91 },
            { label: "Tomato___Late_blight", confidence: 0.05 },
          ],
          inference_ms: 42.3,
          plant: "Tomato",
          disease: "Early Blight",
          is_healthy: false,
        }),
      });
    });

    await page.goto("/");
    await page.setInputFiles('input[type="file"]', FIXTURE);

    const analyze = page.getByRole("button", { name: /analyze leaf/i });
    await expect(analyze).toBeVisible();
    await analyze.click();

    await expect(page.getByText(/Tomato/i).first()).toBeVisible();
    await expect(page.getByText(/Early Blight/i).first()).toBeVisible();
    await expect(page.getByText(/91%/)).toBeVisible();
  });
});
