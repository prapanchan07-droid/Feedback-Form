import re
from typing import Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
from backend.app.validation.validator import FieldValidator

class FieldReconciler:
    """
    Field-level evidence reconciliation engine.
    Compares independent outputs from Qwen VLM and Multi-Variant OCR.
    
    Default priority: Qwen VLM > OCR.
    Exception: High-confidence, multi-variant-consistent OCR overrides Qwen VLM
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
        ocr_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Reconciles Qwen VLM output with Multi-Variant OCR consensus for a specific field.
        Returns a rich provenance dictionary per field.
        """
        ocr_text = ocr_result.get("consensus_text") if ocr_result else None
        ocr_conf = ocr_result.get("ocr_confidence", 0.0) if ocr_result else 0.0
        ocr_agreement = ocr_result.get("variant_agreement", 0.0) if ocr_result else 0.0

        # Clean raw inputs
        qwen_clean = str(qwen_val).strip() if qwen_val is not None and str(qwen_val).strip() not in ("null", "None", "") else None
        ocr_clean = str(ocr_text).strip() if ocr_text is not None and str(ocr_text).strip() not in ("null", "None", "") else None

        # 1. Non-Hallucination check: If both are absent, output blank/null
        if not qwen_clean and not ocr_clean:
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
                "reasoning": "Field left blank by student (both Qwen VLM and OCR detected no content)."
            }

        # 2. Specialized handling by field category
        
        # A. Mobile Numbers (student_mobile_number, parent_mobile_number)
        if "mobile" in field_name:
            g_valid, g_norm, g_err = FieldValidator.validate_phone_number(qwen_clean)
            c_valid, c_norm, c_err = FieldValidator.validate_phone_number(ocr_clean)

            if g_valid and c_valid:
                if g_norm == c_norm:
                    return {
                        "field_name": field_name,
                        "final_value": g_norm,
                        "source": "ocr+vlm",
                        "qwen_value": g_norm,
                        "ocr_value": c_norm,
                        "vlm_confidence": 0.90,
                        "ocr_confidence": ocr_conf,
                        "ocr_variant_agreement": ocr_agreement,
                        "final_confidence": 0.98,
                        "status": "CORROBORATED",
                        "reasoning": "Qwen VLM and Multi-Variant OCR match perfectly on valid 10-digit mobile number."
                    }
                else:
                    # Both valid but differ on specific digits
                    if ocr_conf >= 0.85 and ocr_agreement >= 0.75:
                        return {
                            "field_name": field_name,
                            "final_value": c_norm,
                            "source": "ocr_override",
                            "qwen_value": g_norm,
                            "ocr_value": c_norm,
                            "vlm_confidence": 0.85,
                            "ocr_confidence": ocr_conf,
                            "ocr_variant_agreement": ocr_agreement,
                            "final_confidence": 0.91,
                            "status": "OCR_OVERRIDE",
                            "reasoning": f"Multi-variant OCR override selected due to high consensus ({ocr_agreement*100:.0f}%, conf={ocr_conf:.2f}). Qwen was {g_norm}."
                        }
                    else:
                        return {
                            "field_name": field_name,
                            "final_value": g_norm,
                            "source": "qwen_vlm",
                            "qwen_value": g_norm,
                            "ocr_value": c_norm,
                            "vlm_confidence": 0.85,
                            "ocr_confidence": ocr_conf,
                            "ocr_variant_agreement": ocr_agreement,
                            "final_confidence": 0.80,
                            "status": "VLM_SELECTED",
                            "reasoning": f"Qwen VLM selected for mobile number. Disagreement with OCR ({c_norm})."
                        }
            
            if g_valid and not c_valid:
                return {
                    "field_name": field_name,
                    "final_value": g_norm,
                    "source": "qwen_vlm",
                    "qwen_value": g_norm,
                    "ocr_value": ocr_clean,
                    "vlm_confidence": 0.88,
                    "ocr_confidence": ocr_conf,
                    "ocr_variant_agreement": ocr_agreement,
                    "final_confidence": 0.88,
                    "status": "VLM_SELECTED",
                    "reasoning": f"Qwen VLM passed 10-digit validation while OCR failed ({c_err})."
                }

            if c_valid and not g_valid:
                if ocr_agreement >= 0.50:
                    return {
                        "field_name": field_name,
                        "final_value": c_norm,
                        "source": "ocr_override",
                        "qwen_value": qwen_clean,
                        "ocr_value": c_norm,
                        "vlm_confidence": 0.40,
                        "ocr_confidence": ocr_conf,
                        "ocr_variant_agreement": ocr_agreement,
                        "final_confidence": 0.90,
                        "status": "OCR_OVERRIDE",
                        "reasoning": f"Multi-variant OCR passed 10-digit validation ({c_norm}) while Qwen VLM failed ({g_err})."
                    }

        # B. Cutoff Marks (cut_off_marks)
        if "cutoff" in field_name:
            g_valid, g_norm, g_err = FieldValidator.validate_cutoff_marks(qwen_clean)
            c_valid, c_norm, c_err = FieldValidator.validate_cutoff_marks(ocr_clean)

            if g_valid and c_valid and g_norm == c_norm:
                return {
                    "field_name": field_name,
                    "final_value": g_norm,
                    "source": "ocr+vlm",
                    "qwen_value": g_norm,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.90,
                    "ocr_confidence": ocr_conf,
                    "ocr_variant_agreement": ocr_agreement,
                    "final_confidence": 0.98,
                    "status": "CORROBORATED",
                    "reasoning": "Qwen VLM and Multi-Variant OCR match perfectly on valid cut-off marks."
                }

            if g_valid:
                return {
                    "field_name": field_name,
                    "final_value": g_norm,
                    "source": "qwen_vlm",
                    "qwen_value": g_norm,
                    "ocr_value": ocr_clean,
                    "vlm_confidence": 0.88,
                    "ocr_confidence": ocr_conf,
                    "ocr_variant_agreement": ocr_agreement,
                    "final_confidence": 0.88,
                    "status": "VLM_SELECTED",
                    "reasoning": f"Qwen VLM cut-off marks valid ({g_norm})."
                }

            if c_valid and ocr_agreement >= 0.50:
                return {
                    "field_name": field_name,
                    "final_value": c_norm,
                    "source": "ocr_override",
                    "qwen_value": qwen_clean,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.40,
                    "ocr_confidence": ocr_conf,
                    "ocr_variant_agreement": ocr_agreement,
                    "final_confidence": 0.90,
                    "status": "OCR_OVERRIDE",
                    "reasoning": f"Multi-variant OCR passed cut-off validation ({c_norm}) while Qwen VLM failed."
                }

        # C. General Text / Name / School / Email / Date
        if qwen_clean and ocr_clean:
            sim = cls.calculate_string_similarity(qwen_clean, ocr_clean)
            if sim >= 0.80 or qwen_clean.lower() == ocr_clean.lower():
                return {
                    "field_name": field_name,
                    "final_value": qwen_clean,
                    "source": "ocr+vlm",
                    "qwen_value": qwen_clean,
                    "ocr_value": ocr_clean,
                    "vlm_confidence": 0.90,
                    "ocr_confidence": ocr_conf,
                    "ocr_variant_agreement": ocr_agreement,
                    "final_confidence": 0.98,
                    "status": "CORROBORATED",
                    "reasoning": f"Qwen VLM and Multi-Variant OCR corroborated (similarity {sim*100:.0f}%)."
                }

        # Default Qwen VLM priority for text & OMR fields
        if qwen_clean:
            return {
                "field_name": field_name,
                "final_value": qwen_clean,
                "source": "qwen_vlm",
                "qwen_value": qwen_clean,
                "ocr_value": ocr_clean,
                "vlm_confidence": 0.88,
                "ocr_confidence": ocr_conf,
                "ocr_variant_agreement": ocr_agreement,
                "final_confidence": 0.88,
                "status": "VLM_SELECTED",
                "reasoning": "Qwen VLM selected as primary extraction source."
            }

        # Fallback to OCR if Qwen is missing but OCR has consensus
        if ocr_clean and ocr_agreement >= 0.50:
            return {
                "field_name": field_name,
                "final_value": ocr_clean,
                "source": "ocr_override",
                "qwen_value": None,
                "ocr_value": ocr_clean,
                "vlm_confidence": 0.0,
                "ocr_confidence": ocr_conf,
                "ocr_variant_agreement": ocr_agreement,
                "final_confidence": 0.80,
                "status": "OCR_OVERRIDE",
                "reasoning": f"Multi-variant OCR extracted text with {ocr_agreement*100:.0f}% variant consensus while Qwen VLM returned null."
            }

        # Fallback if both unreadable
        return {
            "field_name": field_name,
            "final_value": None,
            "source": "none",
            "qwen_value": qwen_clean,
            "ocr_value": ocr_clean,
            "vlm_confidence": 0.0,
            "ocr_confidence": ocr_conf,
            "ocr_variant_agreement": ocr_agreement,
            "final_confidence": 0.0,
            "status": "BLANK",
            "reasoning": "Field unreadable across vision and OCR models."
        }
