import re
from typing import Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
from backend.app.validation.validator import FieldValidator

class FieldReconciler:
    """
    Field-level evidence reconciliation engine.
    Compares independent outputs from Qwen VLM and Pure CV + CNN Digit Recognizer.
    Strictly NO OCR engines used.
    
    Default priority: Qwen VLM > Digit Recognizer.
    Exception: High-confidence digit recognition overrides Qwen VLM
    when Qwen VLM is uncertain, invalid, or hallucinated.
    """

    @staticmethod
    def calculate_string_similarity(str1: str, str2: str) -> float:
        if not str1 or not str2:
            return 0.0
        s1 = re.sub(r'[^a-zA-Z0-9]', '', str1.lower())
        s2 = re.sub(r'[^a-zA-Z0-9]', '', str2.lower())
        if not s1 or not s2:
            return 0.0
        return SequenceMatcher(None, s1, s2).ratio()

    @classmethod
    def reconcile_field(
        cls,
        field_name: str,
        qwen_val: Optional[str],
        cnn_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Reconciles Qwen VLM output with Pure CV + CNN Recognition output for a specific field.
        Returns a rich provenance dictionary per field.
        """
        cnn_text = cnn_result.get("consensus_text") if cnn_result else None
        cnn_conf = cnn_result.get("ocr_confidence", cnn_result.get("confidence", 0.0)) if cnn_result else 0.0
        cnn_agreement = cnn_result.get("variant_agreement", 1.0) if cnn_result else 0.0

        qwen_clean = str(qwen_val).strip() if qwen_val is not None and str(qwen_val).strip() not in ("null", "None", "") else None
        cnn_clean = str(cnn_text).strip() if cnn_text is not None and str(cnn_text).strip() not in ("null", "None", "") else None

        # 1. Non-Hallucination check: If both are absent, output blank/null
        if not qwen_clean and not cnn_clean:
            return {
                "field_name": field_name,
                "final_value": None,
                "source": "none",
                "qwen_value": None,
                "ocr_value": None,
                "vlm_confidence": 0.0,
                "ocr_confidence": 0.0,
                "ocr_variant_agreement": 0.0,
                "final_confidence": 0.0,
                "status": "BLANK",
                "reasoning": "Field left blank by student (both Qwen VLM and CV/CNN detected no content)."
            }

        # 2. Specialized handling by field category

        # A. Mobile Numbers (student_mobile_number, parent_mobile_number)
        if "mobile" in field_name:
            g_valid, g_norm, g_err = FieldValidator.validate_phone_number(qwen_clean)
            c_valid, c_norm, c_err = FieldValidator.validate_phone_number(cnn_clean)

            if g_valid and c_valid:
                if g_norm == c_norm:
                    return {
                        "field_name": field_name,
                        "final_value": g_norm,
                        "source": "ocr+vlm",
                        "qwen_value": g_norm,
                        "ocr_value": c_norm,
                        "vlm_confidence": 0.90,
                        "ocr_confidence": cnn_conf,
                        "ocr_variant_agreement": cnn_agreement,
                        "final_confidence": 0.98,
                        "status": "CORROBORATED",
                        "reasoning": "Qwen VLM and CV/CNN Digit Recognizer match perfectly on valid 10-digit mobile number."
                    }
                else:
                    if cnn_conf >= 0.85:
                        return {
                            "field_name": field_name,
                            "final_value": c_norm,
                            "source": "ocr_override",
                            "qwen_value": g_norm,
                            "ocr_value": c_norm,
                            "vlm_confidence": 0.85,
                            "ocr_confidence": cnn_conf,
                            "ocr_variant_agreement": cnn_agreement,
                            "final_confidence": 0.92,
                            "status": "OCR_OVERRIDE",
                            "reasoning": f"CV/CNN Digit Recognizer override selected due to high confidence (conf={cnn_conf:.2f}). Qwen was {g_norm}."
                        }
                    else:
                        return {
                            "field_name": field_name,
                            "final_value": g_norm,
                            "source": "qwen_vlm",
                            "qwen_value": g_norm,
                            "ocr_value": c_norm,
                            "vlm_confidence": 0.85,
                            "ocr_confidence": cnn_conf,
                            "ocr_variant_agreement": cnn_agreement,
                            "final_confidence": 0.80,
                            "status": "VLM_SELECTED",
                            "reasoning": f"Qwen VLM selected for mobile number. Disagreement with CV/CNN ({c_norm})."
                        }

            if g_valid and not c_valid:
                return {
                    "field_name": field_name,
                    "final_value": g_norm,
                    "source": "qwen_vlm",
                    "qwen_value": g_norm,
                    "ocr_value": cnn_clean,
                    "vlm_confidence": 0.88,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.88,
                    "status": "VLM_SELECTED",
                    "reasoning": "Qwen VLM passed 10-digit validation while CV/CNN digit recognition was invalid/unconfident."
                }

            if c_valid and not g_valid:
                return {
                    "field_name": field_name,
                    "final_value": c_norm,
                    "source": "ocr_override",
                    "qwen_value": qwen_clean,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.40,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.90,
                    "status": "OCR_OVERRIDE",
                    "reasoning": f"CV/CNN Digit Recognizer passed 10-digit validation ({c_norm}) while Qwen VLM failed."
                }

        # B. Cutoff Marks (cut_off_marks)
        if "cutoff" in field_name:
            g_valid, g_norm, g_err = FieldValidator.validate_cutoff_marks(qwen_clean)
            c_valid, c_norm, c_err = FieldValidator.validate_cutoff_marks(cnn_clean)

            if g_valid and c_valid and g_norm == c_norm:
                return {
                    "field_name": field_name,
                    "final_value": g_norm,
                    "source": "ocr+vlm",
                    "qwen_value": g_norm,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.90,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.98,
                    "status": "CORROBORATED",
                    "reasoning": "Qwen VLM and CV/CNN Digit Recognizer match on valid cut-off marks."
                }

            if g_valid and not c_valid:
                return {
                    "field_name": field_name,
                    "final_value": g_norm,
                    "source": "qwen_vlm",
                    "qwen_value": g_norm,
                    "ocr_value": cnn_clean,
                    "vlm_confidence": 0.88,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.88,
                    "status": "VLM_SELECTED",
                    "reasoning": f"Qwen VLM cut-off marks valid ({g_norm})."
                }

            if c_valid and not g_valid:
                return {
                    "field_name": field_name,
                    "final_value": c_norm,
                    "source": "ocr_override",
                    "qwen_value": qwen_clean,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.40,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.90,
                    "status": "OCR_OVERRIDE",
                    "reasoning": f"CV/CNN Digit Recognizer passed cut-off validation ({c_norm}) while Qwen VLM failed."
                }

        # C. Date Validation
        if field_name == "date":
            g_valid, g_norm, _ = FieldValidator.validate_date(qwen_clean)
            c_valid, c_norm, _ = FieldValidator.validate_date(cnn_clean)

            if g_valid and c_valid and g_norm == c_norm:
                return {
                    "field_name": field_name,
                    "final_value": g_norm,
                    "source": "ocr+vlm",
                    "qwen_value": g_norm,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.90,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.98,
                    "status": "CORROBORATED",
                    "reasoning": "Qwen VLM and CV/CNN Digit Recognizer match on valid date."
                }

            if c_valid and (not g_valid or cnn_conf >= 0.85):
                return {
                    "field_name": field_name,
                    "final_value": c_norm,
                    "source": "ocr_override",
                    "qwen_value": qwen_clean,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.50 if qwen_clean else 0.0,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.92,
                    "status": "OCR_OVERRIDE",
                    "reasoning": f"CV/CNN Digit Recognizer agreed on date ({c_norm})."
                }

        # D. General Text / Name / School / Email
        if qwen_clean and cnn_clean:
            sim = cls.calculate_string_similarity(qwen_clean, cnn_clean)
            if sim >= 0.80 or qwen_clean.lower() == cnn_clean.lower():
                return {
                    "field_name": field_name,
                    "final_value": qwen_clean,
                    "source": "ocr+vlm",
                    "qwen_value": qwen_clean,
                    "ocr_value": cnn_clean,
                    "vlm_confidence": 0.90,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": 0.98,
                    "status": "CORROBORATED",
                    "reasoning": "Qwen VLM and CV/CNN corroborated."
                }

        # Default Qwen VLM priority for text & OMR fields
        if qwen_clean:
            return {
                "field_name": field_name,
                "final_value": qwen_clean,
                "source": "qwen_vlm",
                "qwen_value": qwen_clean,
                "ocr_value": cnn_clean,
                "vlm_confidence": 0.88,
                "ocr_confidence": cnn_conf,
                "ocr_variant_agreement": cnn_agreement,
                "final_confidence": 0.88,
                "status": "VLM_SELECTED",
                "reasoning": "Qwen VLM selected as primary visual extraction source."
            }

        # Fallback to CNN if Qwen is missing
        if cnn_clean:
            return {
                "field_name": field_name,
                "final_value": cnn_clean,
                "source": "ocr_override",
                "qwen_value": None,
                "ocr_value": cnn_clean,
                "vlm_confidence": 0.0,
                "ocr_confidence": cnn_conf,
                "ocr_variant_agreement": cnn_agreement,
                "final_confidence": 0.80,
                "status": "OCR_OVERRIDE",
                "reasoning": f"CV/CNN Digit Recognizer extracted digits (conf={cnn_conf:.2f}) while Qwen VLM returned null."
            }

        return {
            "field_name": field_name,
            "final_value": None,
            "source": "none",
            "qwen_value": qwen_clean,
            "ocr_value": cnn_clean,
            "vlm_confidence": 0.0,
            "ocr_confidence": cnn_conf,
            "ocr_variant_agreement": cnn_agreement,
            "final_confidence": 0.0,
            "status": "BLANK",
            "reasoning": "Unable to confidently recognize digits or text across vision models."
        }
