import os
import cv2
import json
import datetime
import numpy as np
from sqlalchemy.orm import Session

from backend.app.workers.celery_app import celery_app
from backend.app.db.session import SessionLocal
from backend.app.models.domain import (
    Document, Page, ExtractionJob, FieldExtraction, GeminiResult, CNNResult, VerificationDecision
)
from backend.app.services.scratch_manager import ScratchManager
from backend.app.services.preprocessing import ImagePreprocessor
from backend.app.services.alignment import FormAligner
from backend.app.services.roi_extractor import ROIExtractor
from backend.app.services.gemini_service import GeminiVisionService
from backend.app.decision_engine.field_reconciler import FieldReconciler
from backend.app.validation.validator import FieldValidator
# CNN/OCR imports kept for import safety but inference is bypassed in
# this path.  Do not remove; other modules (model-status endpoint) still
# reference CNNInferenceService indirectly.
from backend.app.ml.inference.cnn_service import CNNInferenceService as _CNNInferenceService  # noqa: F401

TEMPLATE_TO_SCHEMA_MAP = {
    "student_name": "student_name",
    "school_name": "school_name",
    "date": "date",
    "student_mobile": "student_mobile_number",
    "parent_mobile": "parent_mobile_number",
    "email": "email",
    "cutoff_marks": "cut_off_marks"
}

ALL_SCHEMA_FIELDS = [
    "student_name",
    "school_name",
    "date",
    "student_mobile_number",
    "parent_mobile_number",
    "email",
    "cut_off_marks",
    "interested_in_bit",
    "career_guidance_usefulness",
    "speaker_effectiveness",
    "interested_in_choice_filling_assistance"
]


def _auto_crop_form(image: np.ndarray) -> np.ndarray:
    """Detect and crop to the white paper region of the form photo.
    
    Many phone photos include desk/background around the form. Cropping to
    just the paper gives digit boxes 2-3× more pixels when encoded at the
    same max_dim, dramatically improving digit recognition accuracy.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # Threshold to find white paper region (paper is light, desk/background is darker)
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    # Morphological close to fill gaps in text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find the largest contour (should be the paper)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"[AUTO-CROP] No contours found — using original image")
        return image

    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (h * w)

    # Only crop if the paper takes up less than 90% of the image
    # (otherwise the photo is already tightly framed)
    if area_ratio > 0.90:
        print(f"[AUTO-CROP] Paper fills {area_ratio:.0%} of image — no cropping needed")
        return image

    # SAFETY CHECK: if the detected region is too small, it's NOT the paper
    # (it might be a label, box, or other small white region)
    if area_ratio < 0.30:
        print(f"[AUTO-CROP] Detected region too small ({area_ratio:.0%}) — not paper, using original image")
        return image

    x, y, cw, ch = cv2.boundingRect(largest)

    # Additional safety: bounding rect must be at least 50% of image dimensions
    if cw < w * 0.5 or ch < h * 0.5:
        print(f"[AUTO-CROP] Bounding rect too small ({cw}x{ch} vs {w}x{h}) — using original image")
        return image

    # Add small padding (2% of dimensions)
    pad_x = int(w * 0.02)
    pad_y = int(h * 0.02)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + cw + pad_x)
    y2 = min(h, y + ch + pad_y)

    cropped = image[y1:y2, x1:x2]
    print(f"[AUTO-CROP] Cropped form area: {w}x{h} → {x2-x1}x{y2-y1} "
          f"(paper={area_ratio:.0%} of original)")
    return cropped


def process_document_pipeline_sync(job_id: str, document_id: int):
    """
    Qwen-only extraction pipeline:
    1. Prepares job-isolated scratch paths under scratch/<job_id>/
    2. EXIF orientation fix + deskew
    3. Form alignment (marker homography → ORB feature → letterbox fallback)
    4. ROI debug overlay saved to scratch/<job_id>/debug/
    5. CNN inference BYPASSED — cnn_results is always {}
    6. Qwen VLM full-form extraction (qwen/qwen-2.5-vl-72b-instruct via OpenRouter)
    7. Optional second-pass Qwen crop for low-confidence fields
    8. Field reconciliation with cnn_result=None (Qwen value or BLANK)
    9. Database persistence of extraction results and reasoning
    """
    db: Session = SessionLocal()
    try:
        job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not job or not doc:
            return

        print(f"[QWEN-ONLY PIPELINE STARTED] job_id={job_id} | document_id={doc.id} | file={doc.filename}")

        # Reset any top-level stale scratch
        ScratchManager.reset_stale_scratch()

        job.status = "RUNNING"
        job.stage = "PREPROCESSING"
        job.progress = 10.0
        doc.status = "PROCESSING"   # distinguish queued vs actively running
        db.commit()


        aligner = FormAligner()
        roi_extractor = ROIExtractor()
        vision_service = GeminiVisionService()
        # CNN inference is intentionally not instantiated in this code path.
        # To re-enable: restore OCRService / CNNInferenceService calls below.

        pages = db.query(Page).filter(Page.document_id == doc.id).order_by(Page.page_number).all()
        success_count = 0
        failed_count = 0

        for idx, page in enumerate(pages):
            try:
                # Clear stale records for page before reprocessing
                db.query(GeminiResult).filter(GeminiResult.page_id == page.id).delete(synchronize_session='fetch')
                db.query(CNNResult).filter(CNNResult.page_id == page.id).delete(synchronize_session='fetch')
                db.query(FieldExtraction).filter(FieldExtraction.page_id == page.id).delete(synchronize_session='fetch')
                db.query(VerificationDecision).filter(VerificationDecision.page_id == page.id).delete(synchronize_session='fetch')
                db.commit()

                job.stage = f"PROCESSING_PAGE_{page.page_number}"
                job.progress = 10.0 + (80.0 * (idx / max(1, len(pages))))
                db.commit()

                print(f"[PAGE PROCESSING] page_id={page.id} | page_number={page.page_number} | original_file={page.original_image_path}")

                # 1. Load image & apply EXIF orientation fix
                with open(page.original_image_path, "rb") as f:
                    file_bytes = f.read()
                img_exif_fixed = ImagePreprocessor.load_and_fix_exif(file_bytes)
                ScratchManager.save_image(job_id, "original", f"page_{page.id}_original.png", img_exif_fixed)

                # 2. Deskew
                gray_exif = cv2.cvtColor(img_exif_fixed, cv2.COLOR_BGR2GRAY)
                deskewed_gray, skew_angle = ImagePreprocessor.deskew_image(gray_exif)
                if abs(skew_angle) > 0.5:
                    (h_e, w_e) = img_exif_fixed.shape[:2]
                    M = cv2.getRotationMatrix2D((w_e // 2, h_e // 2), skew_angle, 1.0)
                    img_deskewed = cv2.warpAffine(img_exif_fixed, M, (w_e, h_e), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                else:
                    img_deskewed = img_exif_fixed.copy()

                # 3. Alignment & Homography
                aligned, H, is_warped = aligner.align(img_deskewed)

                # Save aligned image
                aligned_path = page.original_image_path.replace(".png", "_aligned.png").replace(".jpg", "_aligned.png")
                cv2.imwrite(aligned_path, aligned)
                page.aligned_image_path = aligned_path

                # 4. Multi-Variant Preprocessing
                # Variants are generated only for Qwen (colour image).
                # CNN variant generation is bypassed.

                # Save ROI overlay debug stage to debug/
                roi_overlay = ImagePreprocessor.draw_roi_overlay(aligned)
                ScratchManager.save_image(job_id, "debug", f"page_{page.id}_roi_overlay.png", roi_overlay)

                # 5. CNN Digit Recognition — BYPASSED (Qwen-only path)
                # CNN model is not loaded or invoked in this extraction path.
                # All cnn_results remain empty; reconciler will use Qwen values only.
                cnn_results: dict = {}
                print(f"[CNN BYPASS] CNN inference skipped for page_id={page.id} — Qwen-only extraction path active.")

                # 6. Independent Qwen VLM Vision Extraction
                job.stage = "VISION_EXTRACTION"
                db.commit()
                qwen_data = {}
                vision_latency = 0.0
                try:
                    # Auto-crop to form paper area for maximum digit resolution
                    qwen_input = _auto_crop_form(img_deskewed)
                    h_qwen, w_qwen = qwen_input.shape[:2]
                    print(f"[QWEN VLM START] Submitting page_id={page.id} to Qwen Vision API | image_size=({w_qwen}x{h_qwen})")
                    
                    # 2-pass extraction:
                    # Pass 1: auto-cropped image → all 11 fields
                    # Pass 2: reference-strip composite from ORIGINAL image → mobile numbers
                    # We pass the original deskewed image so the reference strip at
                    # top-right is preserved for digit verification.
                    qwen_data, vision_latency = vision_service.extract_with_consensus(
                        qwen_input, original_image=img_deskewed
                    )
                    print(f"[QWEN VLM SUCCESS] 2-pass extraction completed | "
                          f"Latency={vision_latency:.2f} ms | fields_extracted={len(qwen_data)}")

                    db.add(GeminiResult(
                        page_id=page.id,
                        raw_json=json.dumps(qwen_data),
                        model_name=getattr(vision_service.qwen_service, 'model_name', getattr(vision_service, 'model_name', 'qwen-vl')),
                        latency_ms=vision_latency
                    ))
                except Exception as e_qwen:
                    print(f"[QWEN VLM WARNING] Qwen Vision extraction failed/unconfigured: {e_qwen}")

                # Log extraction status
                has_qwen_valid = bool(qwen_data)
                if not has_qwen_valid:
                    print(f"[PAGE WARNING] Qwen VLM returned no data for page_id={page.id}. All fields will be BLANK.")

                # 7. Field-Level Comparison & Intelligent Evidence Reconciliation
                job.stage = "FIELD_RECONCILIATION"
                db.commit()

                # Extract per-field metadata from 2-pass consensus (if available)
                field_meta = qwen_data.pop("_field_meta", {}) if isinstance(qwen_data, dict) else {}
                validation_info = qwen_data.pop("_validation", {}) if isinstance(qwen_data, dict) else {}

                reconciled_results = {}
                for schema_field in ALL_SCHEMA_FIELDS:
                    # Find corresponding template field name
                    tmpl_field = None
                    for t_name, s_name in TEMPLATE_TO_SCHEMA_MAP.items():
                        if s_name == schema_field:
                            tmpl_field = t_name
                            break

                    cnn_res = cnn_results.get(tmpl_field) if tmpl_field else None
                    qwen_val = qwen_data.get(schema_field)

                    # Normalize Qwen response values to match schema expectations
                    if qwen_val is not None and isinstance(qwen_val, str):
                        val_lower = qwen_val.strip().lower()
                        # yes/no fields
                        if schema_field in ("interested_in_bit", "interested_in_choice_filling_assistance"):
                            if val_lower in ("yes", "no", "unclear"):
                                qwen_val = val_lower
                        # Rating fields: "Good (4)" → "good", "Excellent (5)" → "excellent"
                        elif schema_field in ("career_guidance_usefulness", "speaker_effectiveness"):
                            for label in ("excellent", "good", "satisfactory", "fair", "poor", "unclear"):
                                if label in val_lower:
                                    qwen_val = label
                                    break

                    meta = field_meta.get(schema_field, {})
                    val_info = validation_info.get(schema_field, {})

                    # Reconcile field evidence (Qwen VLM vs Pure CV/CNN)
                    rec = FieldReconciler.reconcile_field(
                        schema_field, qwen_val, cnn_res,
                        field_meta=meta,
                        validation_info=val_info
                    )

                    reconciled_results[schema_field] = rec

                    # Determine the raw and normalized values
                    raw = str(qwen_val) if qwen_val is not None else (cnn_res.get("consensus_text") if cnn_res else "")
                    normalized = rec["final_value"]

                    # Store in FieldExtraction DB table
                    db.add(FieldExtraction(
                        page_id=page.id,
                        field_name=schema_field,
                        raw_value=raw,
                        normalized_value=normalized,
                        source=rec["source"],
                        confidence=rec["final_confidence"],
                        status=rec["status"]
                    ))

                    # Store in VerificationDecision DB table
                    pass2_val_str = str(meta.get("pass2")) if meta.get("pass2") is not None else None
                    db.add(VerificationDecision(
                        page_id=page.id,
                        field_name=schema_field,
                        gemini_value=str(qwen_val) if qwen_val is not None else None,
                        cnn_value=pass2_val_str,  # Store Pass 2 value in cnn_value column for reference
                        final_value=normalized,
                        decision=rec["status"],
                        confidence=rec["final_confidence"],
                        reason=f"[{rec.get('evidence_level', 'MEDIUM')}] {rec['reasoning']}"
                    ))

                page.status = "COMPLETED"
                success_count += 1
                print(f"[PAGE COMPLETED] page_id={page.id} | Reconciled fields: {len(reconciled_results)}")

            except Exception as e:
                db.rollback()
                page.status = "FAILED"
                page.error_message = str(e)
                failed_count += 1
                print(f"[PAGE EXTRACTION FAILED] page_id={page.id} | error={e}")

            db.commit()

        job.progress = 100.0
        job.stage = "COMPLETED"
        if failed_count == 0:
            job.status = "COMPLETED"
            doc.status = "COMPLETED"
        elif success_count > 0:
            job.status = "PARTIAL_FAILURE"
            doc.status = "PARTIAL_FAILURE"
        else:
            job.status = "FAILED"
            doc.status = "FAILED"

        doc.completed_at = datetime.datetime.utcnow()
        db.commit()
        
        # Clean temporary heavy variants while keeping debug logs
        ScratchManager.clean_job_scratch(job_id, keep_debug=True)
        print(f"[JOB COMPLETED] job_id={job_id} | status={job.status}")

    finally:
        db.close()

@celery_app.task(name="tasks.process_document")
def process_document_task(job_id: str, document_id: int):
    process_document_pipeline_sync(job_id, document_id)
