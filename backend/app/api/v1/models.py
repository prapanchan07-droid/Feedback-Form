from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.ml.inference.cnn_service import CNNInferenceService
from backend.app.models.domain import ModelVersion
from backend.app.schemas.pydantic_models import APIResponse

router = APIRouter(prefix="/models", tags=["models"])

@router.get("/cnn/status", response_model=APIResponse)
def get_cnn_status():
    service = CNNInferenceService()
    status_info = service.get_status()
    return APIResponse(data=status_info)

@router.get("", response_model=APIResponse)
def list_model_versions(db: Session = Depends(get_db)):
    versions = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    out = []
    for v in versions:
        out.append({
            "id": v.id,
            "version_name": v.version_name,
            "architecture": v.architecture,
            "is_active": v.is_active,
            "metrics": v.metrics_json,
            "created_at": v.created_at.isoformat()
        })
    return APIResponse(data=out)
