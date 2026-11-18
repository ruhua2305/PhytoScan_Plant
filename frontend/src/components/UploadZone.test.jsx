import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import UploadZone from "./UploadZone";

function fileInput() {
  return document.querySelector('input[type="file"]');
}

function dropFile(input, file) {
  Object.defineProperty(input, "files", { value: [file], configurable: true });
  fireEvent.change(input);
}

describe("UploadZone", () => {
  it("accepts a JPEG and reports the file with a preview URL", async () => {
    const onImageSelected = vi.fn();
    const user = userEvent.setup();
    render(<UploadZone onImageSelected={onImageSelected} />);

    const file = new File(["leaf"], "leaf.jpg", { type: "image/jpeg" });
    await user.upload(fileInput(), file);

    expect(onImageSelected).toHaveBeenCalledTimes(1);
    expect(onImageSelected.mock.calls[0][0]).toBe(file);
    expect(String(onImageSelected.mock.calls[0][1])).toMatch(/^blob:/);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("rejects a non-image file before calling the parent", () => {
    const onImageSelected = vi.fn();
    render(<UploadZone onImageSelected={onImageSelected} />);

    const file = new File(["notes"], "notes.txt", { type: "text/plain" });
    dropFile(fileInput(), file);

    expect(onImageSelected).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(/jpeg, png, or webp/i);
  });

  it("rejects a file larger than 10 MB", () => {
    const onImageSelected = vi.fn();
    render(<UploadZone onImageSelected={onImageSelected} />);

    const file = new File(["x"], "huge.jpg", { type: "image/jpeg" });
    Object.defineProperty(file, "size", { value: 11 * 1024 * 1024 });
    dropFile(fileInput(), file);

    expect(onImageSelected).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(/maximum size is 10 mb/i);
  });
});
