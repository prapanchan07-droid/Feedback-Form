import os
import cv2
import json
import datetime
from sqlalchemy.orm import Session

from backend.app.workers.celery_app import celery_app
from backend.app.db.session import SessionLocal
from backend.app.models.domain import (
    Document, Page, ExtractionJob, FieldExtraction, GeminiResult, CNNResult, VerificationDecision
)
from backend.app.services.preprocessing import ImagePreprocessor
from backend.app.services.alignment import FormAligner
from backend.app.services.roi_extractor import ROIExtractor
from backend.app.services.gemini_service import GeminiVisionService
from backend.app.ml.inference.cnn_service import CNNInferenceService
from backend.app.decision_engine.comparator import DecisionEngine
from backend.app.validation.validator import FieldValidator

def process_document_pipeline_sync(job_id: str, document_id: int):
    """
    Synchronous pipeline runner.
    Processes pages independently with validated homography, safe letterboxing fallback,
    high-res RGB vision extraction, diagnostic logging, and 6 debug stage image outputs.
    """
    db: Session = SessionLocal()
    try:
        job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not job or not doc:
            return

        print(f"[EXTRACTION JOB STARTED] job_id={job_id} | document_id={doc.id} | file={doc.filename}")

        job.status = "RUNNING"
        job.stage = "PREPROCESSING"
        job.progress = 10.0
        db.commit()

        aligner = FormAligner()
        roi_extractor = ROIExtractor()
        vision_service = GeminiVisionService()
        cnn_service = CNNInferenceService()

        pages = db.query(Page).filter(Page.document_id == doc.id).order_by(Page.page_number).all()
        success_count = 0
        failed_count = 0

        for idx, page in enumerate(pages):
            try:
                job.stage = f"PROCESSING_PAGE_{page.page_number}"
                job.progress = 10.0 + (80.0 * (idx / max(1, len(pages))))
                db.commit()

                print(f"[PAGE PROCESSING] page_id={page.id} | page_number={page.page_number} | original_file={page.original_image_path}")

                # 1. Load image & apply EXIF orientation fix exactly once
                with open(page.original_image_path, "rb") as f:
                    file_bytes = f.read()
                img_exif_fixed = ImagePreprocessor.load_and_fix_exif(file_bytes)

                # 2. Optional deskew
                gray_exif = cv2.cvtColor(img_exif_fixed, cv2.COLOR_BGR2GRAY)
                deskewed_gray, skew_angle = ImagePreprocessor.deskew_image(gray_exif)
                if abs(skew_angle) > 0.5:
                    (h_e, w_e) = img_exif_fixed.shape[:2]
                    M = cv2.getRotationMatrix2D((w_e // 2, h_e // 2), skew_angle, 1.0)
                    img_deskewed = cv2.warpAffine(img_exif_fixed, M, (w_e, h_e), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                else:
                    img_deskewed = img_exif_fixed.copy()

                # 3. Alignment & Preprocessing with Validation
                aligned, H, is_warped = aligner.align(img_deskewed)

                # Save aligned/final preprocessed image
                aligned_path = page.original_image_path.replace(".png", "_aligned.png").replace(".jpg", "_aligned.png")
                cv2.imwrite(aligned_path, aligned)
                page.aligned_image_path = aligned_path

                # 4. Save 6 Preprocessing Debug Stages
                debug_dir = os.path.join(os.path.dirname(page.original_image_path), f"page_{page.id}_debug")
                roi_overlay = ImagePreprocessor.draw_roi_overlay(aligned)

                stages_dict = {
                    "01_original": cv2.imread(page.original_image_path),
                    "02_orientation_corrected": img_exif_fixed,
                    "03_deskewed": img_deskewed,
                    "04_perspective_corrected": aligned if is_warped else None,
                    "05_final_preprocessed": aligned,
                    "06_ROI_overlay": roi_overlay
                }
                ImagePreprocessor.save_debug_stages(debug_dir, stages_dict)

                # 5. Crop ROIs
                rois = roi_extractor.crop_rois(aligned)

                # 6. Vision Extraction (Qwen / Gemini on high-quality RGB preprocessed image)
                job.stage = "VISION_EXTRACTION"
                db.commit()
                
                print(f"[VISION EXTRACTION START] job_id={job_id} | page_id={page.id} | provider={vision_service.qwen_service.__class__.__name__ if vision_service.qwen_service else 'GeminiVisionService'}")
                
                vision_data, vision_latency = vision_service.extract_form_fields(aligned)

                print(f"[VISION EXTRACTION SUCCESS] job_id={job_id} | page_id={page.id} | latency={vision_latency:.2f} ms")

                g_res = GeminiResult(
                    page_id=page.id,
                    raw_json=json.dumps(vision_data),
                    model_name=getattr(vision_service.qwen_service, 'model_name', vision_service.model_name),
                    latency_ms=vision_latency
                )
                db.add(g_res)

                # 7. CNN Inference (Optional non-blocking verification for phone + cutoff)
                job.stage = "CNN_EXTRACTION"
                db.commit()
                mob_crops = rois.get("student_mobile", {}).get("boxes", [])
                cnn_mob_res = cnn_service.predict_phone_number(mob_crops)

                db.add(CNNResult(
                    page_id=page.id,
                    field_name="student_mobile",
                    value=cnn_mob_res.get("value"),
                    per_character_results=cnn_mob_res.get("per_character_results"),
                    confidence=cnn_mob_res.get("confidence"),
                    latency_ms=cnn_mob_res.get("latency_ms")
                ))

                cutoff_crops = rois.get("cutoff_marks", {})
                cnn_cutoff_res = cnn_service.predict_cutoff_marks(cutoff_crops)

                db.add(CNNResult(
                    page_id=page.id,
                    field_name="cutoff_marks",
                    value=cnn_cutoff_res.get("value"),
                    per_character_results=cnn_cutoff_res.get("per_character_results"),
                    confidence=cnn_cutoff_res.get("confidence"),
                    latency_ms=cnn_cutoff_res.get("latency_ms")
                ))

                # 8. Decision Engine & Validation
                job.stage = "COMPARING_AND_VALIDATING"
                db.commit()

                # Phone Decision
                mob_dec = DecisionEngine.evaluate_phone_number(
                    vision_data.get("student_mobile_number"), cnn_mob_res
                )
                db.add(VerificationDecision(
                    page_id=page.id,
                    field_name="student_mobile_number",
                    gemini_value=mob_dec["gemini_value"],
                    cnn_value=mob_dec["cnn_value"],
                    final_value=mob_dec["final_value"],
                    decision=mob_dec["decision"],
                    confidence=mob_dec["confidence"],
                    reason=mob_dec["reason"]
                ))
                db.add(FieldExtraction(
                    page_id=page.id,
                    field_name="student_mobile_number",
                    raw_value=vision_data.get("student_mobile_number"),
                    normalized_value=mob_dec["final_value"],
                    source="decision_engine",
                    confidence=mob_dec["confidence"],
                    status="VERIFIED" if mob_dec["decision"] == "AGREEMENT" else "CONFLICT"
                ))

                # Cutoff Decision
                cutoff_dec = DecisionEngine.evaluate_cutoff_marks(
                    vision_data.get("cut_off_marks"), cnn_cutoff_res
                )
                db.add(VerificationDecision(
                    page_id=page.id,
                    field_name="cut_off_marks",
                    gemini_value=cutoff_dec["gemini_value"],
                    cnn_value=cutoff_dec["cnn_value"],
                    final_value=cutoff_dec["final_value"],
                    decision=cutoff_dec["decision"],
                    confidence=cutoff_dec["confidence"],
                    reason=cutoff_dec["reason"]
                ))
                db.add(FieldExtraction(
                    page_id=page.id,
                    field_name="cut_off_marks",
                    raw_value=vision_data.get("cut_off_marks"),
                    normalized_value=cutoff_dec["final_value"],
                    source="decision_engine",
                    confidence=cutoff_dec["confidence"],
                    status="VERIFIED" if cutoff_dec["decision"] == "AGREEMENT" else "CONFLICT"
                ))

                # Store remaining Vision fields for CURRENT page
                for fname, val in vision_data.items():
                    if fname not in ("student_mobile_number", "cut_off_marks"):
                        db.add(FieldExtraction(
                            page_id=page.id,
                            field_name=fname,
                            raw_value=str(val) if val is not None else "",
                            normalized_value=str(val) if val is not None else "",
                            source="vision_model",
                            confidence=0.85,
                            status="EXTRACTED"
                        ))

                page.status = "COMPLETED"
                success_count += 1
                print(f"[EXTRACTION JOB COMPLETED] page_id={page.id} | status=COMPLETED")
            except Exception as e:
                page.status = "FAILED"
                page.error_message = str(e)
                failed_count += 1
                print(f"[EXTRACTION JOB FAILED] page_id={page.id} | error={e}")

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
    finally:
        db.close()

@celery_app.task(name="tasks.process_document")
def process_document_task(job_id: str, document_id: int):
    process_document_pipeline_sync(job_id, document_id)
