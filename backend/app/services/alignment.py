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

    def is_valid_homography(self, H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> bool:
        """
        Validates homography matrix H to prevent destructive warping.
        Checks determinant, scale change ratio, and corner ordering.
        """
        if H is None or H.shape != (3, 3):
            return False

        det = np.linalg.det(H)
        if det <= 0.01 or det > 100.0:
            return False  # Degenerate or negative determinant

        # Check transformed corners scale ratio
        try:
            pts_homo = np.hstack([src_pts, np.ones((4, 1))])
            transformed = (H @ pts_homo.T).T
            transformed = transformed[:, :2] / transformed[:, 2:]

            # Compare widths and heights of src vs dst
            src_w = np.linalg.norm(src_pts[1] - src_pts[0])
            dst_w = np.linalg.norm(transformed[1] - transformed[0])
            if src_w == 0 or not (0.4 <= (dst_w / src_w) <= 2.5):
                return False

            src_h = np.linalg.norm(src_pts[3] - src_pts[0])
            dst_h = np.linalg.norm(transformed[3] - transformed[0])
            if src_h == 0 or not (0.4 <= (dst_h / src_h) <= 2.5):
                return False

            return True
        except Exception:
            return False

    def detect_markers(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects registration corner markers. Returns None if markers cannot be located with high confidence.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        img_h, img_w = gray.shape[:2]
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        min_area = (img_h * img_w) * 0.0001
        max_area = (img_h * img_w) * 0.015

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
            return None  # Unreliable candidate count, reject marker warping

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
            return None

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def letterbox_to_canvas(self, image: np.ndarray) -> np.ndarray:
        """
        Scales image to fit target template canvas (2479x3508) while strictly preserving aspect ratio.
        Adds equal padding (letterboxing/pillarboxing).
        """
        img_h, img_w = image.shape[:2]
        target_w, target_h = self.canvas_width, self.canvas_height

        scale = min(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create padded white canvas
        if len(image.shape) == 3:
            canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        else:
            canvas = np.full((target_h, target_w), 255, dtype=np.uint8)

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas

    def align(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """
        Aligns form image to template canvas (2479x3508).
        Validates homography matrix. If invalid or unconfident, falls back to letterboxing to preserve geometry.
        Returns (aligned_image, homography_matrix, is_warped_flag).
        """
        try:
            src_pts = self.detect_markers(image)
            if src_pts is not None:
                H, _ = cv2.findHomography(src_pts, self.dst_markers, cv2.RANSAC, 5.0)
                if self.is_valid_homography(H, src_pts, self.dst_markers):
                    warped = cv2.warpPerspective(image, H, (self.canvas_width, self.canvas_height))
                    return warped, H, True

            # Safe Fallback: Letterbox to canvas without aspect ratio distortion
            letterboxed = self.letterbox_to_canvas(image)
            return letterboxed, None, False
        except Exception:
            letterboxed = self.letterbox_to_canvas(image)
            return letterboxed, None, False
