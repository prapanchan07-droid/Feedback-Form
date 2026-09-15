import os
import shutil
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import pytesseract
from backend.app.core.config import get_form_template_config
from backend.app.services.preprocessing import ImagePreprocessor

# Configure Tesseract binary path on Windows if not already in PATH
if not shutil.which("tesseract"):
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break

class OCRService:
    def __init__(self, template_config: Optional[Dict[str, Any]] = None):
        self.config = template_config or get_form_template_config()

    @staticmethod
    def _run_tesseract(crop: np.ndarray, psm: int = 6, digits_only: bool = False) -> Tuple[str, float]:
        """
        Runs Tesseract OCR on a single image crop.
        Returns (extracted_text, average_confidence).
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return "", 0.0

        # Build custom config
        config = f"--psm {psm}"
        if digits_only:
            config += " -c tessedit_char_whitelist=0123456789."

        try:
            # Use image_to_data for confidence scores
            data = pytesseract.image_to_data(crop, config=config, output_type=pytesseract.Output.DICT)
            texts = []
            confs = []
            for text, conf in zip(data["text"], data["conf"]):
                clean_t = text.strip()
                if clean_t:
                    texts.append(clean_t)
                    try:
                        c_val = float(conf)
                        if c_val >= 0:
                            confs.append(c_val)
                    except (ValueError, TypeError):
                        pass

            full_text = " ".join(texts).strip()
            avg_conf = (sum(confs) / len(confs) / 100.0) if confs else (0.75 if full_text else 0.0)
            return full_text, min(1.0, max(0.0, avg_conf))
        except Exception as e:
            # Fallback to image_to_string
            try:
                raw_text = pytesseract.image_to_string(crop, config=config).strip()
                conf = 0.70 if raw_text else 0.0
                return raw_text, conf
            except Exception:
                return "", 0.0

    def extract_field_consensus(
        self,
        field_name: str,
        variants: Dict[str, np.ndarray],
        img_w: int,
        img_h: int
    ) -> Dict[str, Any]:
        """
        Extracts OCR text for a specific field across image variants (original, grayscale, enhanced, sharpened).
        Calculates multi-variant consensus and variant agreement.
        """
        fields_cfg = self.config.get("fields", {})
        cfg = fields_cfg.get(field_name, {})
        field_type = cfg.get("type")

        # Determine PSM and digits_only flags
        digits_only = field_name in ("student_mobile", "parent_mobile", "cutoff_marks", "date")
        psm = 7 if field_type == "character_boxes" or digits_only else 6

        # Determine bounding box
        if "normalized_bbox" in cfg:
            x1, y1, x2, y2 = ImagePreprocessor.map_bbox_norm(cfg["normalized_bbox"], img_w, img_h)
        else:
            x1, y1, x2, y2 = 0, 0, img_w, img_h

        variant_results = {}
        variant_texts = []
        variant_confs = []

        # Evaluate across relevant variants
        eval_variants = ["original", "grayscale", "enhanced", "sharpened"]
        for v_name in eval_variants:
            v_img = variants.get(v_name)
            if v_img is None:
                continue

            crop = v_img[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
            txt, conf = self._run_tesseract(crop, psm=psm, digits_only=digits_only)

            # Normalize digits for mobile or cutoff
            if digits_only and txt:
                txt = "".join(c for c in txt if c.isdigit() or c == ".")

            variant_results[v_name] = {"text": txt, "confidence": conf}
            if txt:
                variant_texts.append(txt)
                variant_confs.append(conf)

        if not variant_texts:
            return {
                "field_name": field_name,
                "consensus_text": None,
                "ocr_confidence": 0.0,
                "variant_agreement": 0.0,
                "variant_extractions": {k: v["text"] for k, v in variant_results.items()},
                "status": "EMPTY"
            }

        # Calculate consensus text (mode text)
        counts = Counter(variant_texts)
        consensus_text, count = counts.most_common(1)[0]
        total_eval = len(eval_variants)
        variant_agreement = count / float(total_eval)
        avg_confidence = sum(variant_confs) / float(len(variant_confs)) if variant_confs else 0.0

        return {
            "field_name": field_name,
            "consensus_text": consensus_text,
            "ocr_confidence": round(avg_confidence, 2),
            "variant_agreement": round(variant_agreement, 2),
            "variant_extractions": {k: v["text"] for k, v in variant_results.items()},
            "status": "VALID" if consensus_text else "EMPTY"
        }

    def run_multi_variant_ocr(
        self,
        variants: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Runs multi-variant OCR across all defined fields in the form template.
        Returns dictionary of OCR consensus results per field.
        """
        orig = variants.get("original")
        if orig is None and len(variants) > 0:
            orig = list(variants.values())[0]
        h, w = orig.shape[:2]

        results = {}
        fields_cfg = self.config.get("fields", {})

        for field_name in fields_cfg.keys():
            if field_name == "questionnaire":
                # OMR handles questionnaire separately
                continue
            consensus = self.extract_field_consensus(field_name, variants, w, h)
            results[field_name] = consensus

        return results
