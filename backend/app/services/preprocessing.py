import cv2
import numpy as np
from PIL import Image, ImageOps
import io
from typing import Dict, Tuple, Any

class ImagePreprocessor:
    @staticmethod
    def load_and_fix_exif(file_bytes: bytes) -> np.ndarray:
        """
        Loads image bytes, applies EXIF rotation fix via PIL, and converts to OpenCV BGR format.
        """
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return bgr

    @staticmethod
    def generate_variants(image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generates original, grayscale, contrast-enhanced, and adaptive thresholded variants.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Denoising
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        # Contrast Normalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Adaptive Thresholding
        thresholded = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2
        )

        return {
            "original": image,
            "grayscale": gray,
            "enhanced": enhanced,
            "thresholded": thresholded
        }

    @staticmethod
    def deskew_image(gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculates skew angle via Minimum Area Bounding Box on thresholded contours and rotates gray image.
        """
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return gray, 0.0

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.2 or abs(angle) > 20.0:
            return gray, 0.0  # Ignore negligible or extreme angles

        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, angle
