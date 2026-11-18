import { test, expect } from "@playwright/test";

test("backend /health responds through the frontend container", async ({ request }) => {
  const res = await request.get("/health");
  expect(res.status()).toBe(200);
  const body = await res.json();
  expect(body).toMatchObject({ status: "ok" });
  expect(typeof body.model_loaded).toBe("boolean");
});
