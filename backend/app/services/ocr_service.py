import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from backend.app.core.config import get_form_template_config
from backend.app.ml.inference.cnn_service import CNNInferenceService
from backend.app.services.field_detector import FieldDetector

class OCRService:
    """
    Numeric Recognition Subsystem using Pure Computer Vision + CNN.
    STRICTLY NO OCR (No Tesseract, EasyOCR, PaddleOCR, or external OCR APIs).
    """
    def __init__(self, template_config: Optional[Dict[str, Any]] = None):
        self.config = template_config or get_form_template_config()
        self.cnn_service = CNNInferenceService()
        self.field_detector = FieldDetector()

    def process_numeric_field_cv_cnn(
        self,
        image: np.ndarray,
        field_name: str
    ) -> Dict[str, Any]:
        """
        Processes a numeric field using FieldDetector -> DigitSegmenter -> CNNInferenceService.
        """
        crop, meta = self.field_detector.detect_and_crop_field(image, field_name)
        if not meta.get("is_valid") or crop is None:
            return {
                "field_name": field_name,
                "consensus_text": None,
                "confidence": 0.0,
                "status": "FAILED",
                "reason": meta.get("reason", "Field crop invalid"),
                "ocr_engine": "PURE_CV_CNN"
            }

        cnn_res = self.cnn_service.process_numeric_field(crop)
        val = cnn_res.get("value")
        conf = cnn_res.get("confidence", 0.0)

        return {
            "field_name": field_name,
            "consensus_text": val,
            "confidence": conf,
            "variant_agreement": conf,  # Map confidence to agreement metric
            "bounding_box": meta.get("bbox", [0, 0, 0, 0]),
            "detector_status": meta.get("detector_status"),
            "per_digit_results": cnn_res.get("per_digit_results", []),
            "ocr_engine": "PURE_CV_CNN",
            "status": "VALID" if val else "Digit segmentation failed"
        }

    def run_multi_variant_ocr(
        self,
        variants: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Runs pure CV + CNN recognition across defined numeric fields.
        Strictly NO OCR engines used.
        """
        orig = variants.get("original")
        if orig is None and len(variants) > 0:
            orig = list(variants.values())[0]

        results = {}
        numeric_fields = ["student_mobile", "parent_mobile", "cutoff_marks", "date"]

        for field_name in numeric_fields:
            res = self.process_numeric_field_cv_cnn(orig, field_name)
            results[field_name] = res

        return results
