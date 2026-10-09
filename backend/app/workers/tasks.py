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
from backend.app.services.ocr_service import OCRService
from backend.app.services.gemini_service import GeminiVisionService
from backend.app.services.field_detector import FieldDetector
from backend.app.ml.digit_segmenter import DigitSegmenter
from backend.app.ml.inference.cnn_service import CNNInferenceService
from backend.app.ml.debug_visualizer import DebugVisualizer
from backend.app.decision_engine.field_reconciler import FieldReconciler
from backend.app.validation.validator import FieldValidator

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

def process_document_pipeline_sync(job_id: str, document_id: int):
    """
    Synchronous Pure CV + CNN + Qwen VLM Pipeline Runner:
    1. Prepares job-isolated scratch paths under scratch/<job_id>/
    2. Orientation & form alignment
    3. Multi-variant preprocessing
    4. Pure CV + CNN Digit Recognition Subsystem (15-step 28x28 preprocessing, batch CNN inference). STRICTLY NO OCR.
    5. Independent Qwen VLM Vision Subsystem
    6. Visual Debug Artifact generation under scratch/<job_id>/debug/
    7. Field-Level Evidence Reconciliation & Scoring Engine (Qwen VLM + CV/CNN Digit Recognizer)
    8. Database persistence of provenances & detailed decision rationale
    """
    db: Session = SessionLocal()
    try:
        job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not job or not doc:
            return

        print(f"[CV-CNN-VLM PIPELINE STARTED] job_id={job_id} | document_id={doc.id} | file={doc.filename}")

        # Reset any top-level stale scratch
        ScratchManager.reset_stale_scratch()

        job.status = "RUNNING"
        job.stage = "PREPROCESSING"
        job.progress = 10.0
        db.commit()

        aligner = FormAligner()
        roi_extractor = ROIExtractor()
        ocr_service = OCRService()
        vision_service = GeminiVisionService()
        field_detector = FieldDetector()
        digit_segmenter = DigitSegmenter()
        cnn_service = CNNInferenceService()
        debug_visualizer = DebugVisualizer()

        pages = db.query(Page).filter(Page.document_id == doc.id).order_by(Page.page_number).all()
        success_count = 0
        failed_count = 0

        for idx, page in enumerate(pages):
            try:
                # Clear stale records for page before reprocessing
                db.query(GeminiResult).filter(GeminiResult.page_id == page.id).delete()
                db.query(FieldExtraction).filter(FieldExtraction.page_id == page.id).delete()
                db.query(VerificationDecision).filter(VerificationDecision.page_id == page.id).delete()
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
                variants = ImagePreprocessor.generate_variants(aligned)
                for v_name, v_img in variants.items():
                    if v_img is not None:
                        ScratchManager.save_image(job_id, "variants", f"page_{page.id}_variant_{v_name}.png", v_img)

                # Save ROI overlay debug stage to debug/
                roi_overlay = ImagePreprocessor.draw_roi_overlay(aligned)
                ScratchManager.save_image(job_id, "debug", f"page_{page.id}_roi_overlay.png", roi_overlay)

                # 5. Pure Computer Vision + CNN Digit Recognition Subsystem (STRICTLY NO OCR)
                job.stage = "PURE_CV_CNN_DIGIT_RECOGNITION"
                db.commit()
                cnn_results = {}
                try:
                    print(f"[PURE CV + CNN START] Running digit segmentation & batch CNN recognition for page_id={page.id}...")
                    cnn_results = ocr_service.run_multi_variant_ocr(variants)

                    # Generate Debug Visualizations for every numeric field
                    numeric_fields_tmpl = ["student_mobile", "parent_mobile", "cutoff_marks", "date"]
                    for f_tmpl in numeric_fields_tmpl:
                        f_crop, f_meta = field_detector.detect_and_crop_field(aligned, f_tmpl)
                        if f_crop is not None:
                            d_crops, d_bboxes, canvases, b_tensor = digit_segmenter.segment_digits(f_crop)
                            c_res = cnn_service.process_numeric_field(f_crop)
                            debug_visualizer.visualize_numeric_extraction(
                                job_id, f_tmpl, aligned, f_meta.get("bbox"), f_crop, d_bboxes, canvases, c_res
                            )

                    # Save CNN results log to scratch
                    with open(os.path.join(ScratchManager.get_job_dir(job_id, "ocr"), f"page_{page.id}_cnn_results.json"), "w") as f:
                        json.dump(cnn_results, f, indent=2)
                except Exception as e_cnn:
                    print(f"[PURE CV + CNN WARNING] Digit recognition encountered an error: {e_cnn}")

                # 6. Independent Qwen VLM Vision Extraction
                job.stage = "VISION_EXTRACTION"
                db.commit()
                qwen_data = {}
                vision_latency = 0.0
                try:
                    print(f"[QWEN VLM START] Submitting page_id={page.id} to Qwen Vision API...")
                    qwen_data, vision_latency = vision_service.extract_form_fields(aligned)
                    print(f"[QWEN VLM SUCCESS] Latency={vision_latency:.2f} ms")
                    db.add(GeminiResult(
                        page_id=page.id,
                        raw_json=json.dumps(qwen_data),
                        model_name=getattr(vision_service.qwen_service, 'model_name', getattr(vision_service, 'model_name', 'qwen-vl')),
                        latency_ms=vision_latency
                    ))
                except Exception as e_qwen:
                    print(f"[QWEN VLM WARNING] Qwen Vision extraction failed/unconfigured: {e_qwen}")

                # Log status of subsystems
                has_cnn_valid = any(v.get("status") in ("VALID", "COMPLETED") or v.get("consensus_text") for v in cnn_results.values()) if cnn_results else False
                has_qwen_valid = bool(qwen_data)
                if not has_cnn_valid and not has_qwen_valid:
                    print(f"[PAGE WARNING] Neither Qwen VLM nor CV/CNN produced values for page_id={page.id}. Proceeding to reconcile fields as BLANK.")

                # 7. Field-Level Comparison & Intelligent Evidence Reconciliation
                job.stage = "FIELD_RECONCILIATION"
                db.commit()

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

                    # Reconcile field evidence (Qwen VLM vs Pure CV/CNN)
                    rec = FieldReconciler.reconcile_field(schema_field, qwen_val, cnn_res)

                    # 8. Targeted Second-Pass Crop Verification (ONLY generated when required for unreadable/conflicted fields)
                    needs_targeted_verification = (
                        rec["final_confidence"] < 0.90 or
                        rec["status"] in ("CNN_OVERRIDE", "VLM_SELECTED") and schema_field in ("student_mobile_number", "cut_off_marks")
                    )

                    if needs_targeted_verification:
                        crop_img, crop_meta = roi_extractor.crop_single_field(aligned, tmpl_field or schema_field, page.page_number)
                        if crop_meta["is_valid"] and crop_img is not None:
                            crop_filename = f"crop_{schema_field}_p{page.id}.png"
                            crop_path = ScratchManager.save_image(job_id, "crops", crop_filename, crop_img)

                            try:
                                pass2_val, _ = vision_service.extract_field_crop(crop_img, schema_field)
                                if pass2_val:
                                    pass2_rec = FieldReconciler.reconcile_field(schema_field, pass2_val, cnn_res)
                                    if pass2_rec["final_confidence"] > rec["final_confidence"]:
                                        rec = pass2_rec
                                        rec["reasoning"] += " (Verified via targeted second-pass field crop inspection)."
                            except Exception as e_crop:
                                print(f"[CROP VERIFICATION WARNING] Field crop vision verification skipped: {e_crop}")

                    reconciled_results[schema_field] = rec

                    # Store in FieldExtraction DB table
                    db.add(FieldExtraction(
                        page_id=page.id,
                        field_name=schema_field,
                        raw_value=str(qwen_val) if qwen_val is not None else (cnn_res.get("consensus_text") if cnn_res else ""),
                        normalized_value=rec["final_value"],
                        source=rec["source"],
                        confidence=rec["final_confidence"],
                        status=rec["status"]
                    ))

                    # Store in VerificationDecision DB table
                    db.add(VerificationDecision(
                        page_id=page.id,
                        field_name=schema_field,
                        gemini_value=str(qwen_val) if qwen_val is not None else None,
                        cnn_value=cnn_res.get("consensus_text") if cnn_res else None,
                        final_value=rec["final_value"],
                        decision=rec["status"],
                        confidence=rec["final_confidence"],
                        reason=rec["reasoning"]
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
