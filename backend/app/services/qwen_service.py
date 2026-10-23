import os
import json
import time
import base64
import cv2
import numpy as np
import httpx
from typing import Dict, Tuple, Any, Optional
from backend.app.core.config import settings, get_form_template_config

class QwenVisionService:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = (
            api_key
            or settings.QWEN_API_KEY
            or settings.GEMINI_API_KEY
            or os.getenv("QWEN_API_KEY", "")
        )
        # Model selection: try .env first, fallback to default
        _DEFAULT_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
        _env_model = None
        try:
            import pathlib
            env_path = pathlib.Path(__file__).resolve().parents[3] / ".env"
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("QWEN_MODEL="):
                        _env_model = line.split("=", 1)[1].strip()
                        break
        except Exception:
            pass
        
        self.model_name = model_name or _env_model or _DEFAULT_MODEL
        
        # Fallback models if primary fails (404 or unavailable)
        self.fallback_models = [
            "qwen/qwen3-vl-32b-instruct",
        ]

        # Determine endpoint URL based on key structure
        if self.api_key.startswith("sk-or-v1-"):
            self.endpoint_url = "https://openrouter.ai/api/v1/chat/completions"
        elif "dashscope" in self.api_key.lower():
            self.endpoint_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        else:
            self.endpoint_url = "https://openrouter.ai/api/v1/chat/completions"

        print(f"[QWEN SERVICE] Initialized | QWEN_API_KEY configured: {bool(self.api_key)} | model={self.model_name} | endpoint={self.endpoint_url}")

    @staticmethod
    def to_grayscale_enhanced(image: np.ndarray) -> np.ndarray:
        """Convert image to high-contrast grayscale for cleaner VLM reading.
        
        Grayscale removes color noise and makes handwritten ink stand out
        as clear dark strokes against a uniform white background.
        Returns 3-channel grayscale (VLM APIs expect RGB).
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize background: morphological closing estimates the paper background
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Divide by background → removes shadows, uneven lighting
        # Result: uniform white paper, dark ink
        normalized = cv2.divide(gray, bg, scale=255)
        
        # Gentle contrast stretch
        p2, p98 = np.percentile(normalized, (2, 98))
        if p98 - p2 > 20:
            stretched = np.clip((normalized.astype(np.float32) - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        else:
            stretched = normalized
        
        # Convert back to 3-channel (API expects RGB)
        return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)

    def extract_with_consensus(self, image: np.ndarray, original_image: np.ndarray = None) -> Tuple[Dict[str, Any], float]:
        """Smart 2-pass extraction with per-field confidence tracking.
        
        Pass 1: Full form image → all 11 fields (original color)
        Pass 2: Precise bbox-cropped digit fields + reference strip
        
        Returns (merged_data, total_latency_ms).
        merged_data includes '_field_meta' key with per-field evidence info.
        """
        import re as _re
        
        start_time = time.time()
        
        # Pass 1: Full form extraction — all fields (original color, no grayscale)
        data1, lat1 = self.extract_form_fields(image)
        
        if not data1:
            data1 = {}
        
        # Track per-field metadata: {field_name: {"pass1": v, "pass2": v, "status": ..., "evidence_level": ...}}
        field_meta = {}
        for k, v in data1.items():
            field_meta[k] = {"pass1": v, "pass2": None, "status": "EXTRACTED", "evidence_level": "MEDIUM"}
        
        # Pass 2: Precise bbox-cropped digit verification
        DIGIT_FIELDS = {"student_mobile_number", "parent_mobile_number", "date", "cut_off_marks"}
        try:
            verify_img = original_image if original_image is not None else image
            digit_verified = self._verify_digits_focused(verify_img)
            
            if digit_verified:
                for field in DIGIT_FIELDS:
                    v1_raw = data1.get(field)
                    v2_raw = digit_verified.get(field)
                    
                    if field not in field_meta:
                        field_meta[field] = {"pass1": v1_raw, "pass2": v2_raw, "status": "EXTRACTED", "evidence_level": "MEDIUM"}
                    else:
                        field_meta[field]["pass2"] = v2_raw
                    
                    # ---------- Ambiguity detection ----------
                    needs_refine = False
                    suspect_reason = ""
                    # Mobile numbers: format validation & ambiguous characters
                    if field in ("student_mobile_number", "parent_mobile_number"):
                        raw_str = str(v1_raw or "")
                        digits = _re.sub(r'\\D', '', raw_str)
                        if not (len(digits) == 10 and digits[0] in '6789'):
                            needs_refine = True
                            suspect_reason = "invalid format"
                        elif any(ch in "OlI1" for ch in raw_str):
                            needs_refine = True
                            suspect_reason = "ambiguous character"
                    # Date fields: basic format validation & ambiguous characters
                    elif field == "date":
                        raw_str = str(v1_raw or "")
                        if not _re.match(r"\d{1,2}[/.\\-]\d{1,2}[/.\\-]\d{4}", raw_str):
                            needs_refine = True
                            suspect_reason = "invalid date format"
                        elif any(ch in "OlI1" for ch in raw_str):
                            needs_refine = True
                            suspect_reason = "ambiguous character"
                    # Cut‑off marks: numeric range & parseability
                    elif field == "cut_off_marks":
                        try:
                            cf_val = float(str(v1_raw or ""))
                            if not (0 <= cf_val <= 200):
                                needs_refine = True
                                suspect_reason = "out of range"
                        except Exception:
                            needs_refine = True
                            suspect_reason = "non‑numeric"
                    if needs_refine:
                        field_meta[field]["status"] = "NEEDS_REVIEW"
                        field_meta[field]["evidence_level"] = "LOW"
                        field_meta[field]["reason"] = f"[AMBIGUITY] {suspect_reason}"
                        print(f"[AMBIGUITY] {field}: {suspect_reason} – flagged for review")
                        continue
                    # Normalize for comparison
                    if field in ("student_mobile_number", "parent_mobile_number"):
                        v1 = _re.sub(r'\D', '', str(v1_raw or ''))
                        v2 = _re.sub(r'\D', '', str(v2_raw or ''))
                        
                        v2_valid = len(v2) == 10 and v2[0] in '6789'
                        v1_valid = len(v1) == 10 and v1[0] in '6789'
                        
                        if v2_valid and v1_valid and v1 == v2:
                            # Both passes agree → high confidence
                            field_meta[field]["status"] = "VERIFIED"
                            field_meta[field]["evidence_level"] = "HIGH"
                            print(f"[REF-STRIP VERIFY] {field}: '{v1}' ✓ confirmed by both passes")
                        elif v2_valid and v1_valid and v1 != v2:
                            # Disagree — flag for review, keep pass1
                            field_meta[field]["status"] = "NEEDS_REVIEW"
                            field_meta[field]["evidence_level"] = "LOW"
                            print(f"[REF-STRIP VERIFY] {field}: CONFLICT pass1='{v1}' vs pass2='{v2}' — flagging for review")
                            # Don't auto-override, let the user decide
                        elif v2_valid and not v1_valid:
                            data1[field] = v2
                            field_meta[field]["status"] = "EXTRACTED"
                            field_meta[field]["evidence_level"] = "MEDIUM"
                            print(f"[REF-STRIP VERIFY] {field}: pass1 invalid, using pass2='{v2}'")
                        elif not v2_valid and v1_valid:
                            field_meta[field]["status"] = "EXTRACTED"
                            field_meta[field]["evidence_level"] = "MEDIUM"
                            print(f"[REF-STRIP VERIFY] {field}: pass2 invalid '{v2}', keeping pass1='{v1}'")
                    
                    elif field == "date":
                        v1_str = str(v1_raw or '')
                        v2_str = str(v2_raw or '')
                        
                        def _valid_date(d):
                            parts = _re.split(r'[/.\-]', d)
                            if len(parts) != 3 or not all(p.isdigit() for p in parts):
                                return False
                            dd, mm = int(parts[0]), int(parts[1])
                            return 1 <= dd <= 31 and 1 <= mm <= 12
                        
                        v1_ok = _valid_date(v1_str)
                        v2_ok = _valid_date(v2_str)
                        
                        if v2_ok and v1_ok and v1_str == v2_str:
                            field_meta[field]["status"] = "VERIFIED"
                            field_meta[field]["evidence_level"] = "HIGH"
                            print(f"[REF-STRIP VERIFY] date: '{v1_str}' ✓ confirmed")
                        elif v2_ok and v1_ok and v1_str != v2_str:
                            field_meta[field]["status"] = "NEEDS_REVIEW"
                            field_meta[field]["evidence_level"] = "LOW"
                            print(f"[REF-STRIP VERIFY] date: CONFLICT pass1='{v1_str}' vs pass2='{v2_str}'")
                        elif v2_ok and not v1_ok:
                            data1["date"] = v2_str
                            field_meta[field]["status"] = "EXTRACTED"
                            field_meta[field]["evidence_level"] = "MEDIUM"
                            print(f"[REF-STRIP VERIFY] date: pass1 invalid, using pass2='{v2_str}'")
                    
                    elif field == "cut_off_marks":
                        try:
                            cf1 = float(str(v1_raw or 0))
                            cf2 = float(str(v2_raw or 0))
                            cf1_ok = 0 <= cf1 <= 200
                            cf2_ok = 0 <= cf2 <= 200
                            
                            if cf2_ok and cf1_ok and abs(cf1 - cf2) < 0.01:
                                field_meta[field]["status"] = "VERIFIED"
                                field_meta[field]["evidence_level"] = "HIGH"
                                # Normalize to 2 decimal places
                                data1["cut_off_marks"] = f"{cf1:.2f}"
                                print(f"[REF-STRIP VERIFY] cut_off_marks: '{cf1}' ✓ confirmed")
                            elif cf2_ok and cf1_ok and abs(cf1 - cf2) >= 0.01:
                                field_meta[field]["status"] = "NEEDS_REVIEW"
                                field_meta[field]["evidence_level"] = "LOW"
                                data1["cut_off_marks"] = f"{cf1:.2f}"
                                print(f"[REF-STRIP VERIFY] cut_off_marks: CONFLICT pass1='{cf1}' vs pass2='{cf2}'")
                            elif cf2_ok and not cf1_ok:
                                data1["cut_off_marks"] = f"{cf2:.2f}"
                                field_meta[field]["status"] = "EXTRACTED"
                                field_meta[field]["evidence_level"] = "MEDIUM"
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            print(f"[REF-STRIP VERIFY WARNING] Skipped: {e}")
        
        # Attach metadata for downstream consumption
        data1["_field_meta"] = field_meta
        
        total_latency = (time.time() - start_time) * 1000.0
        return data1, total_latency

    def _verify_digits_focused(self, image: np.ndarray) -> dict:
        """Pass 2: Use form template bounding boxes for precise digit field crops.
        
        Loads normalized bboxes from form_template.yaml and crops each digit field
        individually at full resolution. Sends reference strip + individual field
        crops as separate images for maximum accuracy.
        """
        from backend.app.core.config import get_form_template_config
        
        h, w = image.shape[:2]
        template = get_form_template_config()
        fields_cfg = template.get("fields", {})
        
        # 1. Crop reference strip from known location (top-right corner)
        ref_y_end = int(h * 0.10)
        ref_x_start = int(w * 0.55)
        ref_strip = image[0:ref_y_end, ref_x_start:]
        ref_b64 = self.encode_image_to_base64(ref_strip, max_dim=1024)
        
        # 2. Crop each digit field using template bboxes (precise coordinates)
        # Use a deterministic order to ensure the model receives images in the same
        # sequence as the field names listed in the prompt. This prevents accidental
        # reuse of a previous crop when the template contains duplicate or missing
        # entries.
        from collections import OrderedDict
        field_crops = OrderedDict()
        # Mapping from template field keys to schema field names
        bbox_fields = {
            "date": "date",
            "student_mobile": "student_mobile_number",
            "parent_mobile": "parent_mobile_number",
            "cutoff_marks": "cut_off_marks",
        }
        # Desired order of processing (consistent with prompt generation)
        ordered_template_keys = ["date", "student_mobile", "parent_mobile", "cutoff_marks"]
        for tmpl_name in ordered_template_keys:
            schema_name = bbox_fields.get(tmpl_name)
            cfg = fields_cfg.get(tmpl_name)
            if not cfg:
                continue
            bbox = cfg.get("normalized_bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1 = max(0, int(bbox[0] * w) - 10)
            y1 = max(0, int(bbox[1] * h) - 10)
            x2 = min(w, int(bbox[2] * w) + 10)
            y2 = min(h, int(bbox[3] * h) + 10)
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                field_crops[schema_name] = self.encode_image_to_base64(crop, max_dim=2048)
                print(f"[DIGIT VERIFY] Cropped {schema_name}: {x2-x1}x{y2-y1}px")
        
        if not field_crops:
            print("[DIGIT VERIFY] No field crops generated")
            # Return dict with all expected fields set to None to indicate N/A
            return {"date": None, "student_mobile_number": None, "parent_mobile_number": None, "cut_off_marks": None}
        
        # 3. Build multi-image request: ref strip + each field crop
        content_parts = []
        content_parts.append({"type": "text", "text": self._build_digit_verify_prompt(list(field_crops.keys()))})
        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}})
        
        for i, (field_name, b64) in enumerate(field_crops.items()):
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if "openrouter" in self.endpoint_url:
            headers["HTTP-Referer"] = "https://bannari-amman.ac.in"
            headers["X-Title"] = "BIT Form Digitization"
        
        payload = {
            "model": self.model_name,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content_parts}],
            "temperature": 0.0
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(self.endpoint_url, headers=headers, json=payload)
            
            if resp.status_code != 200:
                print(f"[DIGIT VERIFY] API returned {resp.status_code}")
                return {}
            
            parsed, ok = self._parse_json_from_response(resp.json())
            return parsed if ok else {}
        except Exception as e:
            print(f"[DIGIT VERIFY] Request failed: {e}")
            return {}
    
    @staticmethod
    def _build_digit_verify_prompt(field_names: list) -> str:
        """Build the prompt for digit verification based on which fields are being verified."""
        field_descriptions = {
            "date": '"Date:" — DD/MM/YYYY (8 digit boxes)',
            "student_mobile_number": '"Student Mobile Number:" — 10 digit boxes (starts with 6/7/8/9)',
            "parent_mobile_number": '"Parent Mobile Number:" — 10 digit boxes (starts with 6/7/8/9)',
            "cut_off_marks": '"Cut Off Marks (out of 200):" — format NNN.NN (0-200)'
        }
        
        field_list = "\n".join(f"- Image {i+2}: {field_descriptions.get(f, f)}" for i, f in enumerate(field_names))
        
        json_template = {f: None for f in field_names}
        
        return f"""Output ONLY a raw JSON object. No markdown, no explanation.

Image 1: Handwritten digit REFERENCE STRIP showing "0 1 2 3 4 5 6 7 8 9". This shows how THIS student writes each digit. Use it to identify digits in the following images.

{field_list}

For EACH digit box in each image, compare the handwritten digit to the reference strip in Image 1.

KEY DIGIT SHAPES:
- "1" = SINGLE THIN VERTICAL STROKE. No loops, no curves. Just a vertical line.
- "0" = CLOSED OVAL (fully enclosed loop)
- "8" = TWO closed loops stacked vertically
- "6" = OPEN top, closed bottom loop  
- "9" = CLOSED top loop, open bottom
- "4" = OPEN top with angled lines

{json.dumps(json_template)}
"""

    def encode_image_to_base64(self, image: np.ndarray, max_dim: int = 3072) -> str:
        """Encode image to base64 JPEG, capping the longest edge at max_dim.
        
        Default raised to 3072 to preserve fine handwriting detail.
        JPEG quality at 93 for sharper digit/text edges.
        """
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = image

        success, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 93])
        if not success:
            raise ValueError("Failed to encode image to JPEG format")
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def _parse_json_from_response(res_json: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        choices = res_json.get("choices", [])
        if not choices:
            return {}, False
        msg = choices[0].get("message", {})
        content_str = msg.get("content") or msg.get("reasoning") or ""
        if not content_str and "text" in choices[0]:
            content_str = choices[0]["text"] or ""

        # Handle content that is a list (some models return array of content parts)
        if isinstance(content_str, list):
            text_parts = [p.get("text", "") for p in content_str if isinstance(p, dict) and p.get("type") == "text"]
            content_str = " ".join(text_parts)

        if not isinstance(content_str, str) or not content_str.strip():
            return {}, False

        import re
        cleaned = content_str.strip()

        # 1. Strip markdown code fences: ```json ... ``` or ``` ... ```
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', cleaned, re.DOTALL)
        if fence_match:
            inner = fence_match.group(1).strip()
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict):
                    return parsed, True
            except Exception:
                pass

        # 2. Try parsing the entire content directly
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed, True
        except Exception:
            pass

        # 3. Smart extraction: find ALL { positions and try parsing from each one
        #    (searching from the END, since reasoning models put JSON last)
        #    Look for JSON objects that contain our expected field keys
        EXPECTED_KEYS = {"student_name", "school_name", "student_mobile_number", "cut_off_marks"}
        brace_positions = [m.start() for m in re.finditer(r'\{', cleaned)]

        # Try from last { to first (JSON is usually at the end of reasoning text)
        for start_pos in reversed(brace_positions):
            # Find the matching closing brace by counting depth
            depth = 0
            end_pos = None
            for i in range(start_pos, len(cleaned)):
                if cleaned[i] == '{':
                    depth += 1
                elif cleaned[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end_pos = i
                        break
            if end_pos is None:
                continue

            candidate = cleaned[start_pos:end_pos + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    # Check if this looks like our form data (has at least 2 expected keys)
                    matching_keys = EXPECTED_KEYS.intersection(parsed.keys())
                    if len(matching_keys) >= 2:
                        return parsed, True
                    # Accept any dict with 5+ keys as plausible form data
                    if len(parsed) >= 5:
                        return parsed, True
            except Exception:
                continue

        # Log failure for debugging
        print(f"[QWEN PARSE DEBUG] Failed to parse JSON from response (len={len(content_str)}): {content_str[:300]}...{content_str[-200:]}")
        return {}, False

    def extract_form_fields(self, image: np.ndarray) -> Tuple[Dict[str, Any], float]:
        if not self.api_key:
            print("[QWEN VISION WARNING] No Qwen/Gemini API key configured in environment.")
            return {}, 0.0

        start_time = time.time()

        # Send ORIGINAL color image — grayscale destroys thin digit strokes like '1'
        base64_img = self.encode_image_to_base64(image, max_dim=3072)
        print(f"[QWEN IMAGE] Sending original color | "
              f"size={image.shape[1]}x{image.shape[0]} → encoded max_dim=3072")

        prompt = """\
CRITICAL: Output ONLY a single raw JSON object. No analysis, no explanation, no markdown. Start with { and end with }.

Extract fields from this BANNARI AMMAN INSTITUTE OF TECHNOLOGY Career Guidance Feedback form:

{"student_name":null,"school_name":null,"date":null,"student_mobile_number":null,"parent_mobile_number":null,"email":null,"cut_off_marks":null,"interested_in_bit":null,"career_guidance_usefulness":null,"speaker_effectiveness":null,"interested_in_choice_filling_assistance":null}

DIGIT CALIBRATION:
The top-right corner has a handwritten reference strip "0 1 2 3 4 5 6 7 8 9" showing EXACTLY how this person writes each digit. Compare every digit you read against this strip.

DIGIT BOX READING (one box = one digit, read left to right):
- Date: DD/MM/YYYY format. Read the actual digits written — do NOT assume or constrain the year.
- Student Mobile Number: 10 boxes. Indian mobiles start with 6, 7, 8, or 9.
- Parent Mobile Number: 10 boxes. Indian mobiles start with 6, 7, 8, or 9.
- Cut-off marks: maximum is 200.00. Value MUST be 0.00-200.00.

CRITICAL DIGIT RULES — read VERY carefully:
- The digit "1" is JUST A SINGLE THIN VERTICAL LINE. If a box contains only a thin vertical stroke with NO loops, NO curves, NO crossbar, it is "1". Do NOT misread "1" as 0, 6, 7, or 8.
- "0" is a FULL CLOSED OVAL/CIRCLE shape. It is NOT a thin line.
- "8" has TWO FULLY CLOSED LOOPS stacked vertically. If you don't see two loops, it is NOT 8.
- "4" has an OPEN TOP with angled lines meeting. It is NOT 8 or 9.
- "6" has an OPEN TOP with a closed bottom loop. "9" has a closed top loop with an open bottom.
- "5" vs "6": 5 has a straight horizontal top + curve at bottom. 6 has a curved stroke from top.
- "7" has a horizontal top bar + diagonal stroke. "1" does NOT have a horizontal bar.

RADIO BUTTONS / CHECKBOXES (look carefully for filled circles ●, ticks ✓, or crosses ✗):
- "interested in joining BIT": look for ● next to "Yes" or "No"
- "usefulness of Career Guidance": look for ● next to one of: Excellent (5), Good (4), Satisfactory (3), Fair (2), Poor (1)
- "effectiveness of the speakers": same scale as above
- "choice filling assistance": look for ● next to "Yes" or "No"
- An UNFILLED circle ○ means NOT selected. Only return the option with a FILLED ● circle.

OTHER RULES:
- Extract ONLY what is physically written. Do NOT invent or correct values.
- If blank or unreadable, use null.
- Email: read exactly as written including dots, @, domain.

Output ONLY the JSON object.
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if "openrouter" in self.endpoint_url:
            headers["HTTP-Referer"] = "https://bannari-amman.ac.in"
            headers["X-Title"] = "BIT Form Digitization"

        # Use the configured Qwen model + fallbacks
        candidate_models = [self.model_name] + getattr(self, 'fallback_models', [])

        # Retry parameters for rate-limiting (429) and transient errors
        MAX_RETRIES = 3
        BASE_BACKOFF_SECONDS = 5  # 5s → 10s → 20s

        for current_model in candidate_models:
            payload = {
                "model": current_model,
                "max_tokens": 4000,  # allow space for reasoning
                "response_format": {"type": "json_object"},
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
                "temperature": 0.0
            }

            print(f"[QWEN EXTRACTION REQUEST] Trying model={current_model}")

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    with httpx.Client(timeout=90.0) as client:
                        resp = client.post(self.endpoint_url, headers=headers, json=payload)

                        if resp.status_code == 429:
                            backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                            print(f"[QWEN RATE LIMITED] 429 on attempt {attempt}/{MAX_RETRIES}. "
                                  f"Backing off {backoff}s before retry...")
                            time.sleep(backoff)
                            continue  # retry

                        if resp.status_code == 402:
                            print(f"[QWEN CREDITS EXHAUSTED] OpenRouter credits depleted. "
                                  f"Add credits at https://openrouter.ai/credits")
                            break

                        # Retry on 5xx server errors (504 timeout, 502 bad gateway, etc.)
                        if resp.status_code >= 500:
                            backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                            print(f"[QWEN SERVER ERROR] {resp.status_code} on attempt {attempt}/{MAX_RETRIES}. "
                                  f"Backing off {backoff}s before retry...")
                            time.sleep(backoff)
                            continue  # retry

                        if resp.status_code == 404:
                            print(f"[QWEN MODEL UNAVAILABLE] {current_model} not found. Trying next model...")
                            break  # skip to next model

                        if resp.status_code != 200:
                            print(f"[QWEN EXTRACTION API ERROR] Model {current_model} returned "
                                  f"Status {resp.status_code}: {resp.text[:200]}")
                            break  # non-retryable error → skip to next model

                        res_json = resp.json()
                        # Log which model the auto-router actually used
                        actual_model = res_json.get("model", current_model)
                        if actual_model != current_model:
                            print(f"[QWEN AUTO-ROUTED] {current_model} → {actual_model}")

                        parsed_dict, ok = self._parse_json_from_response(res_json)
                        if ok:
                            # Post-extraction validation
                            parsed_dict = self._validate_extracted_fields(parsed_dict)
                            latency = (time.time() - start_time) * 1000.0
                            print(f"[QWEN EXTRACTION RESPONSE RECEIVED] Successfully parsed structured JSON "
                                  f"with model={actual_model} in {latency:.2f} ms (attempt {attempt})")
                            return parsed_dict, latency
                        else:
                            # Parse failure — skip to next candidate model
                            print(f"[QWEN PARSE FAIL] Model {actual_model} returned 200 but "
                                  f"no valid JSON found. Trying next model...")
                            break  # try next candidate model

                except httpx.TimeoutException:
                    print(f"[QWEN TIMEOUT] Attempt {attempt}/{MAX_RETRIES} timed out for {current_model}")
                    if attempt < MAX_RETRIES:
                        time.sleep(BASE_BACKOFF_SECONDS)
                    continue
                except Exception as e:
                    print(f"[QWEN EXTRACTION ATTEMPT FAILED] Model {current_model} failed: {e}")
                    break  # unexpected error → skip to next model

        print("[QWEN SERVICE WARNING] All Qwen/Gemini vision model endpoints failed. Pipeline will rely on Pure CV + CNN digit recognizer.")
        return {}, (time.time() - start_time) * 1000.0

    @staticmethod
    def _validate_extracted_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-extraction validation. Non-destructive: flags invalid values but preserves raw data.
        
        Adds '_validation' key to data with per-field validation status.
        Only nullifies values that are clearly impossible (e.g., mobile with wrong digit count).
        """
        import re as _re
        
        validation = {}

        # Cut-off marks: must be 0-200
        cutoff = data.get("cut_off_marks")
        if cutoff is not None:
            try:
                val = float(str(cutoff))
                if val > 200.0:
                    validation["cut_off_marks"] = {"status": "invalid", "reason": f"exceeds max 200 (got {val})"}
                    print(f"[VALIDATION] cut_off_marks={cutoff} exceeds max 200. Flagged.")
                elif val < 0:
                    validation["cut_off_marks"] = {"status": "invalid", "reason": f"negative value {val}"}
                    print(f"[VALIDATION] cut_off_marks={cutoff} is negative. Flagged.")
                else:
                    # Normalize to 2 decimal places
                    data["cut_off_marks"] = f"{val:.2f}"
                    validation["cut_off_marks"] = {"status": "valid"}
            except (ValueError, TypeError):
                validation["cut_off_marks"] = {"status": "invalid", "reason": "not a number"}

        # Date validation
        date_val = data.get("date")
        if date_val and isinstance(date_val, str):
            parts = _re.split(r'[/.\-]', date_val)
            if len(parts) == 3 and all(p.isdigit() for p in parts):
                dd, mm = int(parts[0]), int(parts[1])
                if dd > 31 or dd < 1:
                    validation["date"] = {"status": "needs_review", "reason": f"day={dd} invalid"}
                    print(f"[VALIDATION] date day={dd} suspicious. Flagged for review.")
                elif mm > 12 or mm < 1:
                    validation["date"] = {"status": "needs_review", "reason": f"month={mm} invalid"}
                    print(f"[VALIDATION] date month={mm} suspicious. Flagged for review.")
                else:
                    validation["date"] = {"status": "valid"}
            else:
                validation["date"] = {"status": "needs_review", "reason": "unparseable format"}

        # Phone number validation
        for phone_field in ("student_mobile_number", "parent_mobile_number"):
            phone = data.get(phone_field)
            if phone is not None:
                digits = _re.sub(r'\D', '', str(phone))
                if len(digits) == 10:
                    data[phone_field] = digits  # normalize to digits only
                    if digits[0] in '6789':
                        validation[phone_field] = {"status": "valid"}
                    else:
                        validation[phone_field] = {"status": "needs_review", "reason": f"starts with {digits[0]}, expected 6-9"}
                else:
                    validation[phone_field] = {"status": "invalid", "reason": f"got {len(digits)} digits, expected 10"}
                    print(f"[VALIDATION] {phone_field}='{phone}' is not 10 digits. Flagged.")

        data["_validation"] = validation
        return data

    def extract_field_crop(self, crop_image: np.ndarray, field_type: str) -> Tuple[Optional[str], float]:
        """
        Second-pass crop verification.
        DISABLED in Qwen-only path to prevent latency from unauthorized free-tier model.
        This method is kept for interface compatibility but returns immediately.
        """
        # STUB: Not yet implemented — returns None
        # Second-pass crop calls are disabled. The primary extract_form_fields() response
        # covers all fields. Re-enable here if a dedicated crop-verification model is
        # configured and tested separately.
        return None, 0.0
