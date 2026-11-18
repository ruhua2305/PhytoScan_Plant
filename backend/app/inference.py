"""
inference.py — ONNX model loader and image preprocessor for EfficientNet-B2
"""

import numpy as np
import onnxruntime as ort
from PIL import Image

# ── PlantVillage 38-class labels ───────────────────────────────────────────────
CLASS_NAMES: list[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ImageNet normalisation (standard for EfficientNet fine-tuned from ImageNet)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PlantDiseasePredictor:
    """
    Wraps an ONNX EfficientNet-B2 model for plant disease classification.

    The model's expected input size is read directly from its metadata,
    so this class works even if you retrain on a different resolution.
    """

    def __init__(self, model_path: str) -> None:
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Use all available CPU threads
        sess_opts.intra_op_num_threads = 0

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        # Read expected H×W from model input shape (NCHW)
        inp = self.session.get_inputs()[0]
        _, _, h, w = inp.shape          # e.g. [1, 3, 260, 260]
        self.input_size = (int(h), int(w))
        self.input_name = inp.name
        self.output_name = self.session.get_outputs()[0].name

    # ── Public API ─────────────────────────────────────────────────────────────
    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        """
        Run inference on a PIL image.

        Returns a list of dicts sorted by confidence (descending):
            [{"label": str, "confidence": float}, ...]
        """
        tensor = self._preprocess(image)
        logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
        probabilities = self._softmax(logits[0])

        # Pair labels with probabilities and sort
        ranked = sorted(
            enumerate(probabilities),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [
            {"label": CLASS_NAMES[idx], "confidence": round(float(prob), 6)}
            for idx, prob in ranked
        ]

    # ── Private helpers ────────────────────────────────────────────────────────
    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Resize → float32 → normalise → CHW → add batch dim
        """
        h, w = self.input_size

        # Resize with high-quality downsampling
        image = image.resize((w, h), Image.BICUBIC)

        # HWC → float32 in [0, 1]
        arr = np.array(image, dtype=np.float32) / 255.0

        # Normalise per-channel
        arr = (arr - _MEAN) / _STD          # HWC

        # HWC → CHW
        arr = arr.transpose(2, 0, 1)        # CHW

        # Add batch dimension → NCHW
        return arr[np.newaxis, ...]         # [1, 3, H, W]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max()     # numerical stability
        exp = np.exp(shifted)
        return exp / exp.sum()
