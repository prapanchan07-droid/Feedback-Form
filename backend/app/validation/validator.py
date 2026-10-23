import re
import datetime
import logging
from typing import Dict, Any, Tuple, Optional
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class FieldValidator:
    @staticmethod
    def validate_phone_number(phone_str: Optional[str]) -> Tuple[bool, str, Optional[str]]:
        """
        Validates 10-digit Indian mobile number format (starts with 6-9).
        Returns (is_valid, normalized_phone, error_message).
        """
        if not phone_str:
            return False, "", "Phone number is empty or missing"

        # Remove spaces, hyphens, and optional country code prefix
        cleaned = re.sub(r"[^\d]", "", phone_str)
        if len(cleaned) == 12 and cleaned.startswith("91"):
            cleaned = cleaned[2:]

        if cleaned and cleaned[0] in "012345":
            logger.warning(f"Phone number '{cleaned}' starts with invalid prefix {cleaned[0]} (0-5)")
            print(f"[VALIDATION] Warning: Phone number '{cleaned}' starts with invalid prefix {cleaned[0]} (0-5)")

        if len(cleaned) != 10:
            return False, cleaned, f"Phone number must be exactly 10 digits (got {len(cleaned)})"

        if not re.match(r"^[6-9]\d{9}$", cleaned):
            return False, cleaned, "Invalid Indian mobile number prefix (must start with 6, 7, 8, or 9)"

        return True, cleaned, None

    @staticmethod
    def validate_cutoff_marks(cutoff_str: Optional[str], min_val: float = 0.0, max_val: float = 200.0) -> Tuple[bool, str, Optional[str]]:
        """
        Validates engineering cut off mark (0.0 <= cutoff <= 200.0). Normalizes to 2 decimal places.
        """
        if not cutoff_str:
            return False, "", "Cutoff mark is empty"

        cleaned = cutoff_str.strip()
        try:
            val = float(cleaned)
            if val < min_val or val > max_val:
                return False, cleaned, f"Cutoff mark {val} out of valid range [{min_val}, {max_val}]"
            normalized = f"{val:.2f}"
            return True, normalized, None
        except ValueError:
            return False, cleaned, f"Cutoff mark '{cleaned}' is not a valid numeric decimal"

    @staticmethod
    def validate_email(email_str: Optional[str]) -> Tuple[bool, str, Optional[str]]:
        if not email_str:
            return False, "", "Email is empty"
        cleaned = email_str.strip()
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if re.match(pattern, cleaned):
            return True, cleaned, None
        return False, cleaned, "Invalid email address format"

    @staticmethod
    def validate_date(date_str: Optional[str]) -> Tuple[bool, str, Optional[str]]:
        if not date_str:
            return False, "", "Date is empty"
        cleaned = date_str.strip()

        parsed_dt = None
        # Check patterns with day/month/year components to distinguish calendar errors
        m_dmy = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$", cleaned)
        m_ymd = re.match(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$", cleaned)

        if m_dmy:
            day, month, year = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
            try:
                parsed_dt = datetime.datetime(year, month, day)
            except ValueError as e:
                return False, cleaned, f"Impossible calendar date: {e}"
        elif m_ymd:
            year, month, day = int(m_ymd.group(1)), int(m_ymd.group(2)), int(m_ymd.group(3))
            try:
                parsed_dt = datetime.datetime(year, month, day)
            except ValueError as e:
                return False, cleaned, f"Impossible calendar date: {e}"
        else:
            for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y"):
                try:
                    parsed_dt = datetime.datetime.strptime(cleaned, fmt)
                    break
                except ValueError:
                    continue

        if parsed_dt is None:
            return False, cleaned, "Invalid date format"

        if parsed_dt.year < 1900:
            return False, cleaned, f"Impossible date: year {parsed_dt.year} is out of realistic range"

        now = datetime.datetime.now()
        two_years_later = now + datetime.timedelta(days=730)
        one_year_later = now + datetime.timedelta(days=365)

        if parsed_dt > two_years_later:
            return False, cleaned, f"Date '{cleaned}' is in the far future (>2 years)"

        if parsed_dt > one_year_later:
            return True, cleaned, f"Warning: Date '{cleaned}' is more than 1 year in the future"

        return True, cleaned, None

    @staticmethod
    def validate_questionnaire_option(field_name: str, val: Optional[str]) -> Tuple[bool, str, Optional[str]]:
        if val is None:
            return False, "", "Questionnaire option is empty"
        cleaned = str(val).strip().lower()
        if not cleaned:
            return False, "", "Questionnaire option is empty"

        boolean_map = {
            "yes": "Yes",
            "y": "Yes",
            "true": "Yes",
            "1": "Yes",
            "no": "No",
            "n": "No",
            "false": "No",
            "0": "No",
        }

        rating_map = {
            "excellent": "Excellent",
            "good": "Good",
            "satisfactory": "Satisfactory",
            "fair": "Fair",
            "poor": "Poor",
            "5": "Excellent",
            "4": "Good",
            "3": "Satisfactory",
            "2": "Fair",
            "1": "Poor",
        }

        if field_name in ("interested_in_bit", "interested_in_choice_filling_assistance", "q1", "q4"):
            if cleaned in boolean_map:
                return True, boolean_map[cleaned], None
        elif field_name in ("career_guidance_usefulness", "speaker_effectiveness", "q2", "q3"):
            if cleaned in rating_map:
                return True, rating_map[cleaned], None
            if cleaned.endswith(".0") and cleaned[:-2] in rating_map:
                return True, rating_map[cleaned[:-2]], None

        return False, cleaned, f"Invalid selection '{val}' for field '{field_name}'"
