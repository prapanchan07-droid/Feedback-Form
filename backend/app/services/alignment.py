import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from backend.app.core.config import get_form_template_config

class FormAligner:
    def __init__(self, template_config: Optional[Dict[str, Any]] = None):
        self.config = template_config or get_form_template_config()
        canvas_cfg = self.config.get("canvas", {"width": 2479, "height": 3508})
        self.canvas_width = canvas_cfg.get("width", 2479)
        self.canvas_height = canvas_cfg.get("height", 3508)

        # Target registration markers in template space
        reg = self.config.get("registration_markers", {})
        if reg:
            tl = reg["top_left"]
            tr = reg["top_right"]
            br = reg["bottom_right"]
            bl = reg["bottom_left"]
            
            # If coordinates are normalized [0, 1]
            if isinstance(tl, list) and len(tl) == 4 and tl[0] <= 1.0:
                tl_x = (tl[0] + tl[2]) / 2.0 * self.canvas_width
                tl_y = (tl[1] + tl[3]) / 2.0 * self.canvas_height
                tr_x = (tr[0] + tr[2]) / 2.0 * self.canvas_width
                tr_y = (tr[1] + tr[3]) / 2.0 * self.canvas_height
                br_x = (br[0] + br[2]) / 2.0 * self.canvas_width
                br_y = (br[1] + br[3]) / 2.0 * self.canvas_height
                bl_x = (bl[0] + bl[2]) / 2.0 * self.canvas_width
                bl_y = (bl[1] + bl[3]) / 2.0 * self.canvas_height
            else:
                tl_x, tl_y = reg["top_left"]["x"], reg["top_left"]["y"]
                tr_x, tr_y = reg["top_right"]["x"], reg["top_right"]["y"]
                br_x, br_y = reg["bottom_right"]["x"], reg["bottom_right"]["y"]
                bl_x, bl_y = reg["bottom_left"]["x"], reg["bottom_left"]["y"]

            self.dst_markers = np.array([
                [tl_x, tl_y],
                [tr_x, tr_y],
                [br_x, br_y],
                [bl_x, bl_y]
            ], dtype=np.float32)
        else:
            self.dst_markers = np.array([
                [140, 140],
                [2339, 140],
                [2339, 3368],
                [140, 3368]
            ], dtype=np.float32)

    def detect_markers(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        img_h, img_w = gray.shape[:2]
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        min_area = (img_h * img_w) * 0.00005
        max_area = (img_h * img_w) * 0.02

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = float(M["m10"] / M["m00"])
                        cy = float(M["m01"] / M["m00"])
                        candidates.append((cx, cy, area))

        if len(candidates) < 4:
            # Fallback document boundary detection
            return self._detect_document_corners(gray)

        # Classify candidate centers into 4 quadrants
        mid_x, mid_y = img_w / 2.0, img_h / 2.0
        quad_tl, quad_tr, quad_br, quad_bl = [], [], [], []

        for cx, cy, area in candidates:
            if cx < mid_x and cy < mid_y:
                quad_tl.append((cx, cy))
            elif cx >= mid_x and cy < mid_y:
                quad_tr.append((cx, cy))
            elif cx >= mid_x and cy >= mid_y:
                quad_br.append((cx, cy))
            elif cx < mid_x and cy >= mid_y:
                quad_bl.append((cx, cy))

        def best_corner(pts, tx, ty):
            if not pts:
                return None
            return min(pts, key=lambda p: (p[0] - tx)**2 + (p[1] - ty)**2)

        tl = best_corner(quad_tl, 0, 0)
        tr = best_corner(quad_tr, img_w, 0)
        br = best_corner(quad_br, img_w, img_h)
        bl = best_corner(quad_bl, 0, img_h)

        if None in (tl, tr, br, bl):
            return self._detect_document_corners(gray)

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _detect_document_corners(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        # Return standard canvas corners as fallback
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    def align(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Warps input image to canvas template dimensions (2479x3508).
        Returns (warped_image, homography_matrix, success_flag).
        """
        try:
            src_pts = self.detect_markers(image)
            H, _ = cv2.findHomography(src_pts, self.dst_markers, cv2.RANSAC, 5.0)
            if H is None:
                warped = cv2.resize(image, (self.canvas_width, self.canvas_height))
                return warped, np.eye(3), False
            warped = cv2.warpPerspective(image, H, (self.canvas_width, self.canvas_height))
            return warped, H, True
        except Exception:
            warped = cv2.resize(image, (self.canvas_width, self.canvas_height))
            return warped, np.eye(3), False
