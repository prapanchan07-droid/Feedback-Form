import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from backend.app.core.config import get_form_template_config

class ROIExtractor:
    def __init__(self, template_config: Optional[Dict[str, Any]] = None):
        self.config = template_config or get_form_template_config()

    def crop_rois(self, aligned_image: np.ndarray) -> Dict[str, Any]:
        """
        Crops all form fields from aligned canvas image using template coordinates.
        Supports character boxes, text regions, and questionnaire checkboxes/bubbles.
        """
        h, w = aligned_image.shape[:2]
        fields_cfg = self.config.get("fields", {})
        cropped = {}

        for field_name, cfg in fields_cfg.items():
            fieldType = cfg.get("type")
            
            if fieldType == "character_boxes":
                boxes_crops = []
                for box_item in cfg.get("boxes", []):
                    idx = box_item.get("index", 0)
                    box_type = box_item.get("type", "digit")
                    
                    if "normalized_bbox" in box_item:
                        x1_n, y1_n, x2_n, y2_n = box_item["normalized_bbox"]
                        x1, y1 = int(x1_n * w), int(y1_n * h)
                        x2, y2 = int(x2_n * w), int(y2_n * h)
                    elif "box" in box_item:
                        bx, by, bw, bh = box_item["box"]
                        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
                    else:
                        continue

                    crop = aligned_image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    boxes_crops.append({
                        "index": idx,
                        "type": box_type,
                        "crop": crop,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "normalized_bbox": [x1 / w, y1 / h, x2 / w, y2 / h]
                    })
                cropped[field_name] = {"boxes": boxes_crops}

            elif fieldType == "text_region":
                if "normalized_bbox" in cfg:
                    x1_n, y1_n, x2_n, y2_n = cfg["normalized_bbox"]
                    x1, y1 = int(x1_n * w), int(y1_n * h)
                    x2, y2 = int(x2_n * w), int(y2_n * h)
                elif "bounding_box" in cfg:
                    bx, by, bw, bh = cfg["bounding_box"]
                    x1, y1, x2, y2 = bx, by, bx + bw, by + bh
                else:
                    continue

                crop = aligned_image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                cropped[field_name] = {
                    "crop": crop,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "normalized_bbox": [x1 / w, y1 / h, x2 / w, y2 / h]
                }

            elif fieldType in ("checkbox_group", "omr_group"):
                questions_out = []
                for q in cfg.get("questions", []):
                    q_id = q.get("question_id")
                    q_label = q.get("label")
                    opts_out = []
                    for opt in q.get("options", []):
                        opt_label = opt.get("label")
                        if "center_normalized" in opt:
                            cx_n, cy_n = opt["center_normalized"]
                            r_n = opt.get("radius_normalized", 0.01)
                            cx, cy = int(cx_n * w), int(cy_n * h)
                            r = int(r_n * w)
                        elif "center" in opt:
                            cx, cy = opt["center"]
                            r = opt.get("radius", 25)
                        else:
                            continue

                        x1, y1 = max(0, cx - r - 5), max(0, cy - r - 5)
                        x2, y2 = min(w, cx + r + 5), min(h, cy + r + 5)
                        bubble_crop = aligned_image[y1:y2, x1:x2]
                        opts_out.append({
                            "label": opt_label,
                            "crop": bubble_crop,
                            "center": [cx, cy],
                            "radius": r,
                            "bbox": [x1, y1, x2 - x1, y2 - y1]
                        })
                    questions_out.append({
                        "question_id": q_id,
                        "label": q_label,
                        "options": opts_out
                    })
                cropped[field_name] = {"questions": questions_out}

        return cropped
