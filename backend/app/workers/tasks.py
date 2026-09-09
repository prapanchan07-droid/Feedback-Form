import os
import cv2
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
    Synchronous pipeline runner usable directly or inside Celery worker task.
    Processes pages independently and records all model outputs and decisions in DB.
    """
    db: Session = SessionLocal()
    try:
        job = db.query(ExtractionJob).filter(ExtractionJob.id == job_id).first()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not job or not doc:
            return

        job.status = "RUNNING"
        job.stage = "PREPROCESSING"
        job.progress = 10.0
        db.commit()

        aligner = FormAligner()
        roi_extractor = ROIExtractor()
        gemini_service = GeminiVisionService()
        cnn_service = CNNInferenceService()

        pages = db.query(Page).filter(Page.document_id == doc.id).order_by(Page.page_number).all()
        success_count = 0
        failed_count = 0

        for idx, page in enumerate(pages):
            try:
                # Update progress
                job.stage = f"PROCESSING_PAGE_{page.page_number}"
                job.progress = 10.0 + (80.0 * (idx / max(1, len(pages))))
                db.commit()

                # 1. Load image
                img_bgr = cv2.imread(page.original_image_path)
                if img_bgr is None:
                    raise ValueError(f"Could not load image at {page.original_image_path}")

                # 2. Alignment & Preprocessing
                aligned, H, align_success = aligner.align(img_bgr)
                aligned_path = page.original_image_path.replace(".png", "_aligned.png").replace(".jpg", "_aligned.png")
                cv2.imwrite(aligned_path, aligned)
                page.aligned_image_path = aligned_path

                # 3. Crop ROIs
                rois = roi_extractor.crop_rois(aligned)

                # 4. Gemini Extraction
                job.stage = "GEMINI_EXTRACTION"
                db.commit()
                gemini_data, gemini_latency = gemini_service.extract_form_fields(aligned)

                g_res = GeminiResult(
                    page_id=page.id,
                    raw_json=str(gemini_data),
                    model_name=gemini_service.model_name,
                    latency_ms=gemini_latency
                )
                db.add(g_res)

                # 5. CNN Inference
                job.stage = "CNN_EXTRACTION"
                db.commit()
                mob_crops = rois.get("student_mobile", {}).get("boxes", [])
                cnn_mob_res = cnn_service.predict_phone_number(mob_crops)

                cnn_res_db = CNNResult(
                    page_id=page.id,
                    field_name="student_mobile",
                    value=cnn_mob_res.get("value"),
                    per_character_results=cnn_mob_res.get("per_character_results"),
                    confidence=cnn_mob_res.get("confidence"),
                    latency_ms=cnn_mob_res.get("latency_ms")
                )
                db.add(cnn_res_db)

                cutoff_crops = rois.get("cutoff_marks", {})
                cnn_cutoff_res = cnn_service.predict_cutoff_marks(cutoff_crops)

                cnn_cutoff_db = CNNResult(
                    page_id=page.id,
                    field_name="cutoff_marks",
                    value=cnn_cutoff_res.get("value"),
                    per_character_results=cnn_cutoff_res.get("per_character_results"),
                    confidence=cnn_cutoff_res.get("confidence"),
                    latency_ms=cnn_cutoff_res.get("latency_ms")
                )
                db.add(cnn_cutoff_db)

                # 6. Decision Engine & Validation
                job.stage = "COMPARING_AND_VALIDATING"
                db.commit()

                # Phone Decision
                mob_dec = DecisionEngine.evaluate_phone_number(
                    gemini_data.get("student_mobile_number"), cnn_mob_res
                )
                v_dec_mob = VerificationDecision(
                    page_id=page.id,
                    field_name="student_mobile_number",
                    gemini_value=mob_dec["gemini_value"],
                    cnn_value=mob_dec["cnn_value"],
                    final_value=mob_dec["final_value"],
                    decision=mob_dec["decision"],
                    confidence=mob_dec["confidence"],
                    reason=mob_dec["reason"]
                )
                db.add(v_dec_mob)

                fe_mob = FieldExtraction(
                    page_id=page.id,
                    field_name="student_mobile_number",
                    raw_value=gemini_data.get("student_mobile_number"),
                    normalized_value=mob_dec["final_value"],
                    source="decision_engine",
                    confidence=mob_dec["confidence"],
                    status="VERIFIED" if mob_dec["decision"] == "AGREEMENT" else "CONFLICT"
                )
                db.add(fe_mob)

                # Cutoff Decision
                cutoff_dec = DecisionEngine.evaluate_cutoff_marks(
                    gemini_data.get("cut_off_marks"), cnn_cutoff_res
                )
                v_dec_cutoff = VerificationDecision(
                    page_id=page.id,
                    field_name="cut_off_marks",
                    gemini_value=cutoff_dec["gemini_value"],
                    cnn_value=cutoff_dec["cnn_value"],
                    final_value=cutoff_dec["final_value"],
                    decision=cutoff_dec["decision"],
                    confidence=cutoff_dec["confidence"],
                    reason=cutoff_dec["reason"]
                )
                db.add(v_dec_cutoff)

                fe_cutoff = FieldExtraction(
                    page_id=page.id,
                    field_name="cut_off_marks",
                    raw_value=gemini_data.get("cut_off_marks"),
                    normalized_value=cutoff_dec["final_value"],
                    source="decision_engine",
                    confidence=cutoff_dec["confidence"],
                    status="VERIFIED" if cutoff_dec["decision"] == "AGREEMENT" else "CONFLICT"
                )
                db.add(fe_cutoff)

                # Store remaining Gemini fields
                for fname, val in gemini_data.items():
                    if fname not in ("student_mobile_number", "cut_off_marks"):
                        db.add(FieldExtraction(
                            page_id=page.id,
                            field_name=fname,
                            raw_value=str(val) if val is not None else "",
                            normalized_value=str(val) if val is not None else "",
                            source="gemini",
                            confidence=0.85,
                            status="EXTRACTED"
                        ))

                page.status = "COMPLETED"
                success_count += 1
            except Exception as e:
                page.status = "FAILED"
                page.error_message = str(e)
                failed_count += 1

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
