import re
from typing import Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
from backend.app.validation.validator import FieldValidator
from backend.app.validation.contact_validator import ContactValidator

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

    EVIDENCE_SCORES = {
        "HIGH": 0.9,
        "MEDIUM": 0.6,
        "LOW": 0.3,
        "NONE": 0.0,
    }

    @staticmethod
    def _passes_agree(p1: Any, p2: Any, fname: str) -> bool:
        if p1 is None or p2 is None:
            return False
        s1 = str(p1).strip()
        s2 = str(p2).strip()
        if not s1 or not s2:
            return False
        if "mobile" in fname:
            d1 = re.sub(r'\D', '', s1)
            d2 = re.sub(r'\D', '', s2)
            return len(d1) == 10 and len(d2) == 10 and d1 == d2
        elif "cutoff" in fname or "cut_off" in fname:
            try:
                return abs(float(s1) - float(s2)) < 0.01
            except (ValueError, TypeError):
                return s1 == s2
        elif fname == "date":
            return s1 == s2
        return s1.lower() == s2.lower()

    @staticmethod
    def _passes_differ(p1: Any, p2: Any, fname: str) -> bool:
        if p1 is None or p2 is None:
            return False
        s1 = str(p1).strip()
        s2 = str(p2).strip()
        if not s1 or not s2:
            return False
        if "mobile" in fname:
            d1 = re.sub(r'\D', '', s1)
            d2 = re.sub(r'\D', '', s2)
            if len(d1) == 10 and len(d2) == 10:
                return d1 != d2
            return s1 != s2
        elif "cutoff" in fname or "cut_off" in fname:
            try:
                return abs(float(s1) - float(s2)) >= 0.01
            except (ValueError, TypeError):
                return s1 != s2
        elif fname == "date":
            return s1 != s2
        return s1.lower() != s2.lower()

    @classmethod
    def reconcile_field(
        cls,
        field_name: str,
        qwen_val: Optional[str],
        cnn_result: Optional[Dict[str, Any]] = None,
        field_meta: Optional[Dict[str, Any]] = None,
        validation_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Reconciles field extraction evidence using honest, transparent evidence levels.
        Supports both Qwen-only path (cnn_result is None) and hybrid CNN path (backward compatibility).

        Returns rich provenance dictionary per field with ordinal confidence scores:
          HIGH: 0.9, MEDIUM: 0.6, LOW: 0.3, NONE: 0.0
        """
        meta = field_meta or {}
        val_info = validation_info or {}
        pass1 = meta.get("pass1")
        pass2 = meta.get("pass2")
        pass1_str = str(pass1) if pass1 is not None else None
        pass2_str = str(pass2) if pass2 is not None else None

        qwen_clean = str(qwen_val).strip() if qwen_val is not None and str(qwen_val).strip() not in ("null", "None", "") else None

        # ---------------------------------------------------------
        # Path 1: Qwen-only path (cnn_result is None)
        # ---------------------------------------------------------
        if cnn_result is None:
            # 1. Non-Hallucination check: If no value, return BLANK with evidence_level NONE
            if not qwen_clean:
                return {
                    "field_name": field_name,
                    "final_value": None,
                    "raw_value": None,
                    "source": "none",
                    "evidence_level": "NONE",
                    "final_confidence": 0.0,
                    "status": "BLANK",
                    "reasoning": "Field left blank by student or unreadable (no value extracted).",
                    "first_pass_value": pass1_str,
                    "verification_value": pass2_str,
                }

            # 2. Validate using FieldValidator based on field type
            final_value = qwen_clean
            val_passed: Optional[bool] = None  # None indicates no specific validator applied
            val_warning = False
            val_reason = ""

            # Check external validation_info if provided
            if val_info:
                ext_stat = str(val_info.get("status", "")).lower()
                if ext_stat == "invalid":
                    val_passed = False
                    val_reason = val_info.get("reason", "Validation failed")
                elif ext_stat == "needs_review":
                    val_warning = True
                    val_reason = val_info.get("reason", "Validation warning")
                elif ext_stat == "valid":
                    val_passed = True

            # Validate based on field type
            if "mobile" in field_name:
                v_ok, v_norm, v_warn, v_err = ContactValidator.validate_mobile(qwen_clean)
                if v_ok:
                    final_value = v_norm
                    if val_passed is not False:
                        val_passed = True
                    if v_warn:
                        val_warning = True
                        val_reason = v_err or "Mobile number warning"
                else:
                    val_passed = False
                    val_reason = v_err or "Invalid 10-digit mobile number"
            elif "cutoff" in field_name or "cut_off" in field_name:
                v_ok, v_norm, v_err = FieldValidator.validate_cutoff_marks(qwen_clean)
                if v_ok:
                    final_value = v_norm
                    if val_passed is not False:
                        val_passed = True
                else:
                    val_passed = False
                    val_reason = v_err or "Invalid cutoff marks"
            elif field_name == "date":
                v_ok, v_norm, v_err = FieldValidator.validate_date(qwen_clean)
                if v_ok:
                    final_value = v_norm
                    if val_passed is not False:
                        val_passed = True
                else:
                    val_passed = False
                    val_reason = v_err or "Invalid date format"
            elif field_name == "email":
                v_ok, v_norm, v_warn, v_err = ContactValidator.validate_email(qwen_clean)
                if v_ok:
                    final_value = v_norm
                    if val_passed is not False:
                        val_passed = True
                    if v_warn:
                        val_warning = True
                        val_reason = v_err or "Email warning"
                else:
                    val_passed = False
                    val_reason = v_err or "Invalid email format"
            elif field_name in ("interested_in_bit", "interested_in_choice_filling_assistance",
                                "career_guidance_usefulness", "speaker_effectiveness") or \
                 "interested" in field_name or "guidance" in field_name or "speaker" in field_name:
                if qwen_clean.lower() == "unclear":
                    val_warning = True
                    val_reason = f"Selection is unclear for {field_name}"
                else:
                    v_ok, v_norm, v_err = FieldValidator.validate_questionnaire_option(field_name, qwen_clean)
                    if v_ok:
                        final_value = v_norm
                        if val_passed is not False:
                            val_passed = True
                    else:
                        val_passed = False
                        val_reason = v_err or f"Invalid option for {field_name}"

            # 3. Determine evidence_level from field_meta
            meta_evidence = meta.get("evidence_level")
            if meta_evidence:
                evidence_level = str(meta_evidence).upper()
            elif cls._passes_agree(pass1, pass2, field_name):
                evidence_level = "HIGH"
            elif pass1 is not None or pass2 is not None:
                evidence_level = "MEDIUM"
            else:
                evidence_level = "MEDIUM"

            # 4. Determine status:
            #   - BLANK: no value
            #   - VERIFIED: evidence=HIGH AND validation passes
            #   - EXTRACTED: evidence=MEDIUM AND validation passes (or no validation needed)
            #   - NEEDS_REVIEW: evidence=LOW OR validation has warnings
            #   - CONFLICT: pass1 != pass2 (from field_meta)
            #   - INVALID: hard validation failure
            is_conflict = cls._passes_differ(pass1, pass2, field_name) or meta.get("status") == "CONFLICT"

            if val_passed is False:
                status = "INVALID"
                evidence_level = "LOW"
                reasoning = f"Validation FAILED: {val_reason}"
            elif is_conflict:
                status = "CONFLICT"
                evidence_level = "LOW"
                reasoning = f"Extraction passes conflict: pass1='{pass1_str}' vs pass2='{pass2_str}'."
            elif evidence_level == "LOW" or val_warning or meta.get("status") == "NEEDS_REVIEW":
                status = "NEEDS_REVIEW"
                evidence_level = "LOW"
                warning_text = f" Validation warning: {val_reason}" if val_warning and val_reason else ""
                reasoning = f"Flagged for review (evidence={evidence_level}).{warning_text}"
            elif evidence_level == "HIGH" and (val_passed is True or val_passed is None):
                status = "VERIFIED"
                reasoning = f"Both passes agree and validated: {final_value}."
            elif evidence_level == "MEDIUM" and (val_passed is True or val_passed is None):
                status = "EXTRACTED"
                reasoning = f"Single-pass extraction validated: {final_value}."
            else:
                status = "EXTRACTED"
                reasoning = f"Extracted value: {final_value}."

            final_confidence = cls.EVIDENCE_SCORES.get(evidence_level, 0.6)

            return {
                "field_name": field_name,
                "final_value": final_value,
                "raw_value": qwen_clean,
                "source": "qwen_vlm",
                "evidence_level": evidence_level,
                "final_confidence": final_confidence,
                "status": status,
                "reasoning": reasoning,
                "first_pass_value": pass1_str,
                "verification_value": pass2_str,
            }

        # ---------------------------------------------------------
        # Path 2: Hybrid CNN path (cnn_result is NOT None)
        # ---------------------------------------------------------
        cnn_text = cnn_result.get("consensus_text") if cnn_result else None
        cnn_conf = cnn_result.get("ocr_confidence", cnn_result.get("confidence", 0.0)) if cnn_result else 0.0
        cnn_agreement = cnn_result.get("variant_agreement", 1.0) if cnn_result else 0.0
        cnn_clean = str(cnn_text).strip() if cnn_text is not None and str(cnn_text).strip() not in ("null", "None", "") else None

        # 1. Non-Hallucination check: If both are absent, output blank/null
        if not qwen_clean and not cnn_clean:
            return {
                "field_name": field_name,
                "final_value": None,
                "raw_value": None,
                "source": "none",
                "evidence_level": "NONE",
                "first_pass_value": pass1_str,
                "verification_value": pass2_str,
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
                    evidence_level = "HIGH"
                    return {
                        "field_name": field_name,
                        "final_value": g_norm,
                        "raw_value": qwen_clean,
                        "source": "ocr+vlm",
                        "evidence_level": evidence_level,
                        "first_pass_value": pass1_str,
                        "verification_value": pass2_str,
                        "qwen_value": g_norm,
                        "ocr_value": c_norm,
                        "vlm_confidence": 0.9,
                        "ocr_confidence": cnn_conf,
                        "ocr_variant_agreement": cnn_agreement,
                        "final_confidence": cls.EVIDENCE_SCORES[evidence_level],
                        "status": "CORROBORATED",
                        "reasoning": "Qwen VLM and CV/CNN Digit Recognizer match perfectly on valid 10-digit mobile number."
                    }
                else:
                    if cnn_conf >= 0.85:
                        evidence_level = "HIGH"
                        return {
                            "field_name": field_name,
                            "final_value": c_norm,
                            "raw_value": qwen_clean,
                            "source": "ocr_override",
                            "evidence_level": evidence_level,
                            "first_pass_value": pass1_str,
                            "verification_value": pass2_str,
                            "qwen_value": g_norm,
                            "ocr_value": c_norm,
                            "vlm_confidence": 0.6,
                            "ocr_confidence": cnn_conf,
                            "ocr_variant_agreement": cnn_agreement,
                            "final_confidence": cls.EVIDENCE_SCORES[evidence_level],
                            "status": "OCR_OVERRIDE",
                            "reasoning": f"CV/CNN Digit Recognizer override selected due to high confidence (conf={cnn_conf:.2f}). Qwen was {g_norm}."
                        }
                    else:
                        evidence_level = "MEDIUM"
                        return {
                            "field_name": field_name,
                            "final_value": g_norm,
                            "raw_value": qwen_clean,
                            "source": "qwen_vlm",
                            "evidence_level": evidence_level,
                            "first_pass_value": pass1_str,
                            "verification_value": pass2_str,
                            "qwen_value": g_norm,
                            "ocr_value": c_norm,
                            "vlm_confidence": 0.6,
                            "ocr_confidence": cnn_conf,
                            "ocr_variant_agreement": cnn_agreement,
                            "final_confidence": cls.EVIDENCE_SCORES[evidence_level],
                            "status": "VLM_SELECTED",
                            "reasoning": f"Qwen VLM selected for mobile number. Disagreement with CV/CNN ({c_norm})."
                        }

            if g_valid and not c_valid:
                evidence_level = "MEDIUM"
                return {
                    "field_name": field_name,
                    "final_value": g_norm,
                    "raw_value": qwen_clean,
                    "source": "qwen_vlm",
                    "evidence_level": evidence_level,
                    "first_pass_value": pass1_str,
                    "verification_value": pass2_str,
                    "qwen_value": g_norm,
                    "ocr_value": cnn_clean,
                    "vlm_confidence": 0.6,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": cls.EVIDENCE_SCORES[evidence_level],
                    "status": "VLM_SELECTED",
                    "reasoning": "Qwen VLM passed 10-digit validation while CV/CNN digit recognition was invalid/unconfident."
                }

            if c_valid and not g_valid:
                evidence_level = "HIGH" if cnn_conf >= 0.85 else "MEDIUM"
                return {
                    "field_name": field_name,
                    "final_value": c_norm,
                    "raw_value": qwen_clean,
                    "source": "ocr_override",
                    "evidence_level": evidence_level,
                    "first_pass_value": pass1_str,
                    "verification_value": pass2_str,
                    "qwen_value": qwen_clean,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.3,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": cls.EVIDENCE_SCORES[evidence_level],
                    "status": "OCR_OVERRIDE",
                    "reasoning": f"CV/CNN Digit Recognizer passed 10-digit validation ({c_norm}) while Qwen VLM failed."
                }

        # B. Cutoff Marks (cut_off_marks)
        if "cutoff" in field_name or "cut_off" in field_name:
            g_valid, g_norm, g_err = FieldValidator.validate_cutoff_marks(qwen_clean)
            c_valid, c_norm, c_err = FieldValidator.validate_cutoff_marks(cnn_clean)

            if g_valid and c_valid and g_norm == c_norm:
                evidence_level = "HIGH"
                return {
                    "field_name": field_name,
                    "final_value": g_norm,
                    "raw_value": qwen_clean,
                    "source": "ocr+vlm",
                    "evidence_level": evidence_level,
                    "first_pass_value": pass1_str,
                    "verification_value": pass2_str,
                    "qwen_value": g_norm,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.9,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": cls.EVIDENCE_SCORES[evidence_level],
                    "status": "CORROBORATED",
                    "reasoning": "Qwen VLM and CV/CNN Digit Recognizer match on valid cut-off marks."
                }

            if g_valid and not c_valid:
                evidence_level = "MEDIUM"
                return {
                    "field_name": field_name,
                    "final_value": g_norm,
                    "raw_value": qwen_clean,
                    "source": "qwen_vlm",
                    "evidence_level": evidence_level,
                    "first_pass_value": pass1_str,
                    "verification_value": pass2_str,
                    "qwen_value": g_norm,
                    "ocr_value": cnn_clean,
                    "vlm_confidence": 0.6,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": cls.EVIDENCE_SCORES[evidence_level],
                    "status": "VLM_SELECTED",
                    "reasoning": f"Qwen VLM cut-off marks valid ({g_norm})."
                }

            if c_valid and not g_valid:
                evidence_level = "HIGH" if cnn_conf >= 0.85 else "MEDIUM"
                return {
                    "field_name": field_name,
                    "final_value": c_norm,
                    "raw_value": qwen_clean,
                    "source": "ocr_override",
                    "evidence_level": evidence_level,
                    "first_pass_value": pass1_str,
                    "verification_value": pass2_str,
                    "qwen_value": qwen_clean,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.3,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": cls.EVIDENCE_SCORES[evidence_level],
                    "status": "OCR_OVERRIDE",
                    "reasoning": f"CV/CNN Digit Recognizer passed cut-off validation ({c_norm}) while Qwen VLM failed."
                }

        # C. Date Validation
        if field_name == "date":
            g_valid, g_norm, _ = FieldValidator.validate_date(qwen_clean)
            c_valid, c_norm, _ = FieldValidator.validate_date(cnn_clean)

            if g_valid and c_valid and g_norm == c_norm:
                evidence_level = "HIGH"
                return {
                    "field_name": field_name,
                    "final_value": g_norm,
                    "raw_value": qwen_clean,
                    "source": "ocr+vlm",
                    "evidence_level": evidence_level,
                    "first_pass_value": pass1_str,
                    "verification_value": pass2_str,
                    "qwen_value": g_norm,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.9,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": cls.EVIDENCE_SCORES[evidence_level],
                    "status": "CORROBORATED",
                    "reasoning": "Qwen VLM and CV/CNN Digit Recognizer match on valid date."
                }

            if c_valid and (not g_valid or cnn_conf >= 0.85):
                evidence_level = "HIGH" if cnn_conf >= 0.85 else "MEDIUM"
                return {
                    "field_name": field_name,
                    "final_value": c_norm,
                    "raw_value": qwen_clean,
                    "source": "ocr_override",
                    "evidence_level": evidence_level,
                    "first_pass_value": pass1_str,
                    "verification_value": pass2_str,
                    "qwen_value": qwen_clean,
                    "ocr_value": c_norm,
                    "vlm_confidence": 0.3 if qwen_clean else 0.0,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": cls.EVIDENCE_SCORES[evidence_level],
                    "status": "OCR_OVERRIDE",
                    "reasoning": f"CV/CNN Digit Recognizer agreed on date ({c_norm})."
                }

        # D. General Text / Name / School / Email
        if qwen_clean and cnn_clean:
            sim = cls.calculate_string_similarity(qwen_clean, cnn_clean)
            if sim >= 0.80 or qwen_clean.lower() == cnn_clean.lower():
                evidence_level = "HIGH"
                return {
                    "field_name": field_name,
                    "final_value": qwen_clean,
                    "raw_value": qwen_clean,
                    "source": "ocr+vlm",
                    "evidence_level": evidence_level,
                    "first_pass_value": pass1_str,
                    "verification_value": pass2_str,
                    "qwen_value": qwen_clean,
                    "ocr_value": cnn_clean,
                    "vlm_confidence": 0.9,
                    "ocr_confidence": cnn_conf,
                    "ocr_variant_agreement": cnn_agreement,
                    "final_confidence": cls.EVIDENCE_SCORES[evidence_level],
                    "status": "CORROBORATED",
                    "reasoning": "Qwen VLM and CV/CNN corroborated."
                }

        # Default Qwen VLM priority for text & OMR fields
        if qwen_clean:
            evidence_level = "MEDIUM"
            return {
                "field_name": field_name,
                "final_value": qwen_clean,
                "raw_value": qwen_clean,
                "source": "qwen_vlm",
                "evidence_level": evidence_level,
                "first_pass_value": pass1_str,
                "verification_value": pass2_str,
                "qwen_value": qwen_clean,
                "ocr_value": cnn_clean,
                "vlm_confidence": 0.6,
                "ocr_confidence": cnn_conf,
                "ocr_variant_agreement": cnn_agreement,
                "final_confidence": cls.EVIDENCE_SCORES[evidence_level],
                "status": "VLM_SELECTED",
                "reasoning": "Qwen VLM selected as primary visual extraction source."
            }

        # Fallback to CNN if Qwen is missing
        if cnn_clean:
            evidence_level = "MEDIUM"
            return {
                "field_name": field_name,
                "final_value": cnn_clean,
                "raw_value": None,
                "source": "ocr_override",
                "evidence_level": evidence_level,
                "first_pass_value": pass1_str,
                "verification_value": pass2_str,
                "qwen_value": None,
                "ocr_value": cnn_clean,
                "vlm_confidence": 0.0,
                "ocr_confidence": cnn_conf,
                "ocr_variant_agreement": cnn_agreement,
                "final_confidence": cls.EVIDENCE_SCORES[evidence_level],
                "status": "OCR_OVERRIDE",
                "reasoning": f"CV/CNN Digit Recognizer extracted digits (conf={cnn_conf:.2f}) while Qwen VLM returned null."
            }

        return {
            "field_name": field_name,
            "final_value": None,
            "raw_value": None,
            "source": "none",
            "evidence_level": "NONE",
            "first_pass_value": pass1_str,
            "verification_value": pass2_str,
            "qwen_value": qwen_clean,
            "ocr_value": cnn_clean,
            "vlm_confidence": 0.0,
            "ocr_confidence": cnn_conf,
            "ocr_variant_agreement": cnn_agreement,
            "final_confidence": 0.0,
            "status": "BLANK",
            "reasoning": "Unable to confidently recognize digits or text across vision models."
        }

    @classmethod
    def reconcile_qwen_only(
        cls,
        field_name: str,
        qwen_val: Optional[str],
        field_meta: Optional[Dict[str, Any]] = None,
        validation_info: Optional[Dict[str, Any]] = None,
        image_quality: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evidence-based reconciliation for Qwen-only extraction path.
        
        Uses real signals from 2-pass consensus and validation to compute
        honest ordinal evidence scores. No fabricated scores.
        
        Args:
            field_name: Schema field name
            qwen_val: Extracted value from Qwen VLM
            field_meta: Per-field metadata from 2-pass consensus
                       {pass1, pass2, status, evidence_level}
            validation_info: Per-field validation result
                            {status: valid/invalid/needs_review, reason}
            image_quality: Image quality assessment
                          {blur_score, contrast, resolution, is_adequate}
        """
        rec = cls.reconcile_field(
            field_name=field_name,
            qwen_val=qwen_val,
            cnn_result=None,
            field_meta=field_meta,
            validation_info=validation_info
        )
        if image_quality and image_quality.get("is_adequate") is False:
            if rec["evidence_level"] == "HIGH":
                rec["evidence_level"] = "MEDIUM"
                rec["final_confidence"] = cls.EVIDENCE_SCORES["MEDIUM"]
            elif rec["evidence_level"] == "MEDIUM":
                rec["evidence_level"] = "LOW"
                rec["final_confidence"] = cls.EVIDENCE_SCORES["LOW"]
                if rec["status"] == "EXTRACTED":
                    rec["status"] = "NEEDS_REVIEW"
            rec["reasoning"] += " Image quality is poor."
        return rec

