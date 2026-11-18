import React, { useState, useRef, useCallback } from "react";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_MB = 10;

export default function UploadZone({ onImageSelected, disabled }) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  const validate = (file) => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      return "Please upload a JPEG, PNG, or WebP image.";
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      return `File too large. Maximum size is ${MAX_MB} MB.`;
    }
    return null;
  };

  const handleFile = useCallback(
    (file) => {
      if (!file) return;
      const err = validate(file);
      if (err) {
        setError(err);
        return;
      }
      setError("");
      const previewUrl = URL.createObjectURL(file);
      onImageSelected(file, previewUrl);
    },
    [onImageSelected]
  );

  // Drag handlers
  const onDragOver = (e) => { e.preventDefault(); if (!disabled) setIsDragging(true); };
  const onDragLeave = () => setIsDragging(false);
  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const onInputChange = (e) => handleFile(e.target.files[0]);

  return (
    <div className="upload-zone-wrapper">
      <div
        className={`upload-zone ${isDragging ? "dragging" : ""} ${disabled ? "disabled" : ""}`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={disabled ? -1 : 0}
        onKeyDown={(e) => e.key === "Enter" && !disabled && inputRef.current?.click()}
        aria-label="Upload leaf image for disease detection"
      >
        {/* Corner crosshair markers — specimen-slide aesthetic */}
        <span className="corner tl" aria-hidden="true" />
        <span className="corner tr" aria-hidden="true" />
        <span className="corner bl" aria-hidden="true" />
        <span className="corner br" aria-hidden="true" />

        <div className="upload-inner">
          <LeafUploadIcon isDragging={isDragging} />
          <p className="upload-title">
            {isDragging ? "Release to examine" : "Drop a leaf photo here"}
          </p>
          <p className="upload-sub">or click to browse — JPEG, PNG, WebP up to 10 MB</p>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.webp"
          onChange={onInputChange}
          style={{ display: "none" }}
          tabIndex={-1}
        />
      </div>

      {error && (
        <p className="upload-error" role="alert">
          ⚠ {error}
        </p>
      )}
    </div>
  );
}

function LeafUploadIcon({ isDragging }) {
  return (
    <div className={`upload-icon ${isDragging ? "icon-bounce" : ""}`}>
      <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        {/* Leaf shape */}
        <path
          d="M32 8C20 8 10 20 10 34C10 46 20 56 32 56C44 56 54 46 54 34C54 20 44 8 32 8Z"
          fill="currentColor"
          opacity="0.15"
        />
        <path
          d="M32 10C21 10 12 21 12 34C12 45.5 21 54 32 54"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          fill="none"
        />
        <path
          d="M32 10C43 10 52 21 52 34C52 45.5 43 54 32 54"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          fill="none"
        />
        {/* Center vein */}
        <path d="M32 54V14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        {/* Secondary veins */}
        <path d="M32 28L24 22M32 36L22 32M32 44L24 40" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.7" />
        <path d="M32 28L40 22M32 36L42 32M32 44L40 40" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.7" />
        {/* Upload arrow */}
        <path d="M32 4L28 9H31V16H33V9H36L32 4Z" fill="currentColor" />
      </svg>
    </div>
  );
}
