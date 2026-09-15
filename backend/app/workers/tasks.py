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
    Synchronous hybrid extraction pipeline runner:
    1. Prepares job-isolated scratch paths under scratch/<job_id>/
    2. Orientation & form alignment
    3. Multi-variant preprocessing (safeguarding thin handwriting strokes)
    4. Independent Multi-Variant OCR Extraction & Multi-Variant Consensus Aggregation
    5. Independent Qwen VLM Vision Extraction
    6. Field-Level Evidence Reconciliation & Scoring Engine (Qwen VLM + Multi-Variant OCR)
    7. Targeted Second-Pass Crop Verification (generated ONLY when required for unreadable/conflicted fields)
    8. Database persistence of rich provenances & detailed decision rationale
    """
    db: Session = SessionLocal()
    try:
        job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not job or not doc:
            return

        print(f"[HYBRID PIPELINE STARTED] job_id={job_id} | document_id={doc.id} | file={doc.filename}")

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

        pages = db.query(Page).filter(Page.document_id == doc.id).order_by(Page.page_number).all()
        success_count = 0
        failed_count = 0

        for idx, page in enumerate(pages):
            try:
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

                # 4. Multi-Variant Preprocessing (Safeguards thin handwriting strokes)
                variants = ImagePreprocessor.generate_variants(aligned)
                for v_name, v_img in variants.items():
                    if v_img is not None:
                        ScratchManager.save_image(job_id, "variants", f"page_{page.id}_variant_{v_name}.png", v_img)

                # Save ROI overlay debug stage to debug/
                roi_overlay = ImagePreprocessor.draw_roi_overlay(aligned)
                ScratchManager.save_image(job_id, "debug", f"page_{page.id}_roi_overlay.png", roi_overlay)

                # 5. Independent Multi-Variant OCR Subsystem
                job.stage = "MULTI_VARIANT_OCR"
                db.commit()
                print(f"[OCR SERVICE START] Running multi-variant OCR across fields for page_id={page.id}...")
                ocr_results = ocr_service.run_multi_variant_ocr(variants)
                
                # Save OCR raw consensus log to scratch
                with open(os.path.join(ScratchManager.get_job_dir(job_id, "ocr"), f"page_{page.id}_ocr_results.json"), "w") as f:
                    json.dump(ocr_results, f, indent=2)

                # 6. Independent Qwen VLM Vision Extraction
                job.stage = "VISION_EXTRACTION"
                db.commit()
                print(f"[QWEN VLM START] Submitting page_id={page.id} to Qwen Vision API...")
                qwen_data, vision_latency = vision_service.extract_form_fields(aligned)
                print(f"[QWEN VLM SUCCESS] Latency={vision_latency:.2f} ms")

                db.add(GeminiResult(
                    page_id=page.id,
                    raw_json=json.dumps(qwen_data),
                    model_name=getattr(vision_service.qwen_service, 'model_name', vision_service.model_name),
                    latency_ms=vision_latency
                ))

                # 7. Field-Level Comparison & Intelligent Evidence Reconciliation
                job.stage = "FIELD_RECONCILIATION"
                db.commit()

                # Pre-crop ROIs dictionary in case targeted crop is required
                rois = roi_extractor.crop_rois(aligned)

                reconciled_results = {}
                for schema_field in ALL_SCHEMA_FIELDS:
                    # Find corresponding template field name
                    tmpl_field = None
                    for t_name, s_name in TEMPLATE_TO_SCHEMA_MAP.items():
                        if s_name == schema_field:
                            tmpl_field = t_name
                            break

                    ocr_res = ocr_results.get(tmpl_field) if tmpl_field else None
                    qwen_val = qwen_data.get(schema_field)

                    # Reconcile field evidence
                    rec = FieldReconciler.reconcile_field(schema_field, qwen_val, ocr_res)

                    # 8. Targeted Second-Pass Crop Verification (ONLY when required for low confidence/conflicts)
                    if schema_field in ("student_mobile_number", "cut_off_marks") and rec["status"] in ("CONFLICT", "VLM_SELECTED", "OCR_OVERRIDE") and rec["final_confidence"] < 0.95:
                        crop_dict = rois.get(tmpl_field, {})
                        crop_img = crop_dict.get("crop") if "crop" in crop_dict else None
                        
                        if crop_img is None and "boxes" in crop_dict and crop_dict["boxes"]:
                            # Stitch box crops if line crop not directly available
                            crops = [b["crop"] for b in crop_dict["boxes"] if b.get("crop") is not None]
                            if crops:
                                crop_img = np.hstack(crops)

                        if crop_img is not None and crop_img.size > 0:
                            # Save targeted field crop ONLY when required
                            crop_filename = f"crop_{schema_field}_p{page.id}.png"
                            crop_path = ScratchManager.save_image(job_id, "crops", crop_filename, crop_img)
                            print(f"[TARGETED CROP GENERATED] Saved required crop: {crop_path}")

                            pass2_val, _ = vision_service.extract_field_crop(crop_img, schema_field)
                            if pass2_val:
                                pass2_rec = FieldReconciler.reconcile_field(schema_field, pass2_val, ocr_res)
                                if pass2_rec["final_confidence"] > rec["final_confidence"]:
                                    rec = pass2_rec
                                    rec["reasoning"] += " (Verified via targeted second-pass field crop inspection)."

                    reconciled_results[schema_field] = rec

                    # Store in FieldExtraction DB table
                    db.add(FieldExtraction(
                        page_id=page.id,
                        field_name=schema_field,
                        raw_value=str(qwen_val) if qwen_val is not None else (ocr_res.get("consensus_text") if ocr_res else ""),
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
                        cnn_value=ocr_res.get("consensus_text") if ocr_res else None,
                        final_value=rec["final_value"],
                        decision=rec["status"],
                        confidence=rec["final_confidence"],
                        reason=rec["reasoning"]
                    ))

                page.status = "COMPLETED"
                success_count += 1
                print(f"[PAGE COMPLETED] page_id={page.id} | Reconciled fields: {len(reconciled_results)}")

            except Exception as e:
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
        print(f"[JOB COMPLETED] job_id={job_id} | status={job.status}")

    finally:
        db.close()

@celery_app.task(name="tasks.process_document")
def process_document_task(job_id: str, document_id: int):
    process_document_pipeline_sync(job_id, document_id)
