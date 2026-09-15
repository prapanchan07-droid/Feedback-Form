import os
import json
import time
import base64
import cv2
import numpy as np
import httpx
from typing import Dict, Tuple, Any, Optional
from backend.app.core.config import settings

class QwenVisionService:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.QWEN_API_KEY or settings.GEMINI_API_KEY or os.getenv("QWEN_API_KEY", "")
        
        configured_model = model_name or settings.QWEN_MODEL or settings.GEMINI_MODEL or os.getenv("QWEN_MODEL", "")
        
        # Sanitize model slug for OpenRouter / Qwen API compatibility
        if not configured_model or "gemini" in configured_model.lower() or "instruct" in configured_model.lower():
            if "free" in configured_model.lower():
                self.model_name = "qwen/qwen-2.5-vl-72b-instruct:free"
            elif "/" in configured_model:
                self.model_name = configured_model
            else:
                self.model_name = "qwen/qwen-2.5-vl-72b-instruct"
        else:
            self.model_name = configured_model

        # Determine endpoint URL based on key structure
        if self.api_key.startswith("sk-or-v1-"):
            self.endpoint_url = "https://openrouter.ai/api/v1/chat/completions"
        elif "dashscope" in self.api_key.lower():
            self.endpoint_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        else:
            self.endpoint_url = "https://openrouter.ai/api/v1/chat/completions"

        print(f"[QWEN SERVICE] Initialized | QWEN_API_KEY configured: {bool(self.api_key)} | model={self.model_name} | endpoint={self.endpoint_url}")

    def encode_image_to_base64(self, image: np.ndarray, max_dim: int = 1280) -> str:
        """
        Encodes OpenCV BGR image to base64 JPEG data string, downscaling if max_dim exceeded
        to keep vision patch tokens within budget while preserving document legibility.
        """
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = image

        success, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            raise ValueError("Failed to encode image to JPEG format")
        return base64.b64encode(buffer).decode("utf-8")

    def extract_form_fields(self, image: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """
        Submits high-quality preprocessed RGB form image to Qwen Vision API for structured JSON field extraction.
        Returns (extracted_dict, latency_ms).
        Raises RuntimeError on API failure without returning silent mock fallbacks.
        """
        if not self.api_key:
            raise RuntimeError("Qwen Vision API key is not configured in environment (GEMINI_API_KEY / QWEN_API_KEY)")

        start_time = time.time()
        base64_img = self.encode_image_to_base64(image, max_dim=1280)

        prompt = """You are an expert OCR and vision document analysis AI specializing in Tamil Nadu school career feedback forms for Bannari Amman Institute of Technology (BIT).

Extract ALL handwritten fields from the provided feedback form into a JSON object.

Form Layout & Specialized Field Instructions:
1. student_name: Handwritten student full name.
2. school_name: Handwritten school name.
3. date: Date written on the form.
4. student_mobile_number: Read ONLY the Student Mobile Number. Read the individual digit boxes from left to right. Each box corresponds to one digit. Do not read the Parent Mobile Number. Do not read unrelated numbers. Do not infer missing digits. Do not guess visually ambiguous digits. Preserve each visible digit exactly.
5. parent_mobile_number: 10-digit handwritten parent mobile number. Distinguish clearly from Student Mobile Number.
6. email: Student email ID. Do not merge with nearby text.
7. cut_off_marks: Read ONLY the Cut Off Marks field. Preserve every visible digit. Preserve decimal points. Do not round. Do not truncate. Do not add missing digits. Examples of valid formats include: 195, 184.56, 167.5.
8. interested_in_bit: "yes", "no", or "unclear" for Question 1 (Are you interested in joining BIT for B.E/B.Tech courses?). Interpret OMR/checkbox ticks accurately.
9. career_guidance_usefulness: "excellent", "good", "satisfactory", "fair", "poor", or "unclear" for Question 2.
10. speaker_effectiveness: "excellent", "good", "satisfactory", "fair", "poor", or "unclear" for Question 3.
11. interested_in_choice_filling_assistance: "yes", "no", or "unclear" for Question 4.

Strict Accuracy Rules:
- Extract ONLY information visibly present on THIS image.
- NEVER use information from previous images or sample/default values.
- NEVER hallucinate or guess unreadable handwriting. Return null when information is unreadable.
- Preserve handwritten digits and decimal points exactly. Do not merge nearby fields.
- Respond ONLY with valid JSON matching the specified keys without prose or markdown wrappers.
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if "openrouter" in self.endpoint_url:
            headers["HTTP-Referer"] = "https://bannari-amman.ac.in"
            headers["X-Title"] = "BIT Form Digitization"

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        }
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 1024
        }

        print(f"[QWEN EXTRACTION REQUEST] Sending fresh image ({image.shape[1]}x{image.shape[0]}) to model={self.model_name}")

        max_retries = 3
        backoff = 1.0
        last_exception = None

        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(self.endpoint_url, headers=headers, json=payload)
                    
                    if resp.status_code != 200:
                        print(f"[QWEN EXTRACTION API ERROR] Status {resp.status_code}: {resp.text}")
                        # Fallback try free model slug if paid slug returned 404 or 402 payment required
                        if resp.status_code in (402, 404) and not self.model_name.endswith(":free"):
                            self.model_name = "qwen/qwen-2.5-vl-72b-instruct:free"
                            payload["model"] = self.model_name
                            continue

                    resp.raise_for_status()
                    res_json = resp.json()
                    content_str = res_json["choices"][0]["message"]["content"]

                    # Clean markdown codeblocks
                    if "```json" in content_str:
                        content_str = content_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in content_str:
                        content_str = content_str.split("```")[1].split("```")[0].strip()

                    parsed_dict = json.loads(content_str)
                    latency = (time.time() - start_time) * 1000.0
                    print(f"[QWEN EXTRACTION RESPONSE RECEIVED] Successfully parsed structured JSON in {latency:.2f} ms")
                    return parsed_dict, latency

            except Exception as e:
                last_exception = e
                print(f"[QWEN EXTRACTION ATTEMPT FAILED] Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2.0

        raise RuntimeError(f"Qwen Vision API extraction failed after {max_retries} attempts: {last_exception}")

    def extract_field_crop(self, crop_image: np.ndarray, field_type: str) -> Tuple[Optional[str], float]:
        """
        Sends high-quality field ROI crop image to Qwen for independent second-pass verification of a specific field.
        Does NOT show previous predictions. Instructs model to independently inspect visible characters.
        """
        if not self.api_key:
            raise RuntimeError("Qwen Vision API key is not configured")

        start_time = time.time()
        base64_crop = self.encode_image_to_base64(crop_image, max_dim=1280)

        if field_type == "student_mobile_number":
            prompt = """Independently inspect the ten handwritten digit boxes in this image.
Determine the digit visible in each box from left to right.
Do not use or infer a previous prediction.
Return ONLY a JSON object: {"value": "10-digit-string"}
Example: {"value": "9876543210"}
"""
        elif field_type == "cut_off_marks":
            prompt = """Independently inspect the handwritten cut off marks field in this image.
Preserve every visible digit and decimal point exactly. Do not round or truncate.
Return ONLY a JSON object: {"value": "cutoff-string"}
Example: {"value": "184.56"}
"""
        else:
            prompt = f"Independently inspect the handwritten {field_type} field in this image and return JSON: {{\"value\": \"extracted-value\"}}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_crop}"
                            }
                        }
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 512
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(self.endpoint_url, headers=headers, json=payload)
                resp.raise_for_status()
                content_str = resp.json()["choices"][0]["message"]["content"]
                if "```json" in content_str:
                    content_str = content_str.split("```json")[1].split("```")[0].strip()
                elif "```" in content_str:
                    content_str = content_str.split("```")[1].split("```")[0].strip()
                parsed = json.loads(content_str)
                latency = (time.time() - start_time) * 1000.0
                return parsed.get("value"), latency
        except Exception as e:
            print(f"[QWEN CROP EXTRACTION FAILED] {field_type}: {e}")
            return None, (time.time() - start_time) * 1000.0
