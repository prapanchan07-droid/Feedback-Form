import os
import time
import numpy as np
from typing import Dict, Any, List, Optional
from backend.app.core.config import settings
from backend.app.ml.models.resnet_digit import ResNet18Recognizer

class CNNInferenceService:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.CNN_MODEL_PATH
        self.recognizer = ResNet18Recognizer(weights_path=self.model_path)

    def is_ready(self) -> bool:
        return self.recognizer.weights_loaded

    def get_status(self) -> Dict[str, Any]:
        weights_exist = os.path.exists(self.model_path)
        if self.is_ready():
            return {
                "status": "READY",
                "weights_exist": True,
                "architecture": "ResNet-18",
                "notice": "CNN verification engine active."
            }
        else:
            return {
                "status": "NOT_TRAINED",
                "weights_exist": weights_exist,
                "architecture": "ResNet-18",
                "notice": "CNN verification unavailable until a trained model is provided."
            }

    def predict_phone_number(self, box_crops: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs CNN inference box-by-box for student mobile number (10 boxes).
        """
        start_time = time.time()
        if not self.is_ready():
            return {
                "value": None,
                "confidence": 0.0,
                "per_character_results": [],
                "status": "CNN_UNAVAILABLE",
                "notice": "CNN verification unavailable until a trained model is provided."
            }

        char_results = []
        digits = []
        confidences = []

        for b in box_crops:
            idx = b.get("index", len(digits))
            crop = b["crop"]
            pred = self.recognizer.predict_digit_crop(crop)
            digit_val = pred["digit"] or "0"
            conf = pred["confidence"]
            
            digits.append(digit_val)
            confidences.append(conf)
            char_results.append({
                "box_index": idx,
                "digit": digit_val,
                "confidence": conf,
                "probabilities": pred.get("probabilities", {})
            })

        reconstructed_phone = "".join(digits)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        latency = (time.time() - start_time) * 1000.0

        return {
            "value": reconstructed_phone,
            "confidence": round(avg_confidence, 4),
            "per_character_results": char_results,
            "status": "COMPLETED",
            "latency_ms": round(latency, 2)
        }

    def predict_cutoff_marks(self, cutoff_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs CNN inference for cutoff marks (digit boxes + decimal point detection).
        """
        start_time = time.time()
        if not self.is_ready():
            return {
                "value": None,
                "confidence": 0.0,
                "per_character_results": [],
                "status": "CNN_UNAVAILABLE",
                "notice": "CNN verification unavailable until a trained model is provided."
            }

        boxes = cutoff_data.get("boxes", [])
        char_results = []
        chars = []
        confidences = []

        for b in boxes:
            idx = b.get("index", len(chars))
            b_type = b.get("type", "digit")
            if b_type == "decimal":
                chars.append(".")
                confidences.append(1.0)
                char_results.append({
                    "box_index": idx,
                    "type": "decimal",
                    "digit": ".",
                    "confidence": 1.0,
                    "probabilities": {"10": 1.0}
                })
            else:
                crop = b["crop"]
                pred = self.recognizer.predict_digit_crop(crop)
                digit_val = pred["digit"] or "0"
                conf = pred["confidence"]
                
                chars.append(digit_val)
                confidences.append(conf)
                char_results.append({
                    "box_index": idx,
                    "type": "digit",
                    "digit": digit_val,
                    "confidence": conf,
                    "probabilities": pred.get("probabilities", {})
                })

        reconstructed_cutoff = "".join(chars)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        latency = (time.time() - start_time) * 1000.0

        return {
            "value": reconstructed_cutoff,
            "confidence": round(avg_confidence, 4),
            "per_character_results": char_results,
            "status": "COMPLETED",
            "latency_ms": round(latency, 2)
        }
