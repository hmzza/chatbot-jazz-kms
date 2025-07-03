"""
services/analyzer.py - Query analysis and category detection for JazzBot
"""

import re
from typing import List, Optional
from utils.logger import logger


class CategoryDetector:
    """Category detection"""

    CATEGORY_KEYWORDS = {
        "B2C": ["b2c", "consumer", "individual", "personal"],
        "B2B": ["b2b", "business", "corporate", "enterprise"],
        "prepaid": ["prepaid", "pre-paid", "recharge", "top-up"],
        "postpaid": ["postpaid", "post-paid", "monthly", "contract"],
        "data": ["data", "internet", "mb", "gb", "4g", "5g"],
        "voice": ["call", "calls", "minutes", "voice", "talk"],
        "sms": ["sms", "text", "message", "messaging"],
        "facebook": ["facebook", "fb"],
        "youtube": ["youtube", "yt"],
        "whatsapp": ["whatsapp", "wa"],
    }

    @classmethod
    def detect_categories(cls, query: str) -> List[str]:
        """Detect multiple categories from the query"""
        query_lower = query.lower()
        detected_categories = []

        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_categories.append(category)

        return detected_categories

    @classmethod
    def detect_primary_category(cls, query: str) -> Optional[str]:
        """Detect primary category"""
        categories = cls.detect_categories(query)
        return categories[0] if categories else None


class QueryAnalyzer:
    """Query analysis"""

    GREETING_PATTERNS = [
        r"^(hi|hello|hey|aoa|salam|assalam|good\s+(morning|afternoon|evening))$",
        r"^(how\s+are\s+you|what\'s\s+up)$",
    ]

    FOLLOWUP_PATTERNS = [
        r"^(yes|yeah|yep|sure|ok|okay)$",
        r"^(show\s+me\s+)?(more\s+)?(details?|info)",
        r"^(tell\s+me\s+more|give\s+me\s+more)$",
        r"^(what\s+about|how\s+about)$",
        r"^(can\s+you\s+explain|explain\s+more)$",
    ]

    PACKAGE_EXTRACTION_PATTERNS = [
        r"(?:we have|offers?|try)\s+(?:a\s+|the\s+)?([^.]+?)(?:\s+(?:offer|bundle|package|plan))",
        r"(facebook|youtube|whatsapp|instagram)\s+(?:premium\s+|daily\s+|weekly\s+|monthly\s+)?(?:offer|bundle|package)",
        r"(?:jazz\s+)?(super\s+card|smart\s+bundle)",
    ]

    @classmethod
    def is_greeting(cls, query: str) -> bool:
        """Check if query is a greeting"""
        query_clean = query.strip().lower()
        return any(
            re.match(pattern, query_clean, re.IGNORECASE)
            for pattern in cls.GREETING_PATTERNS
        )

    @classmethod
    def is_followup(cls, query: str, last_response: str = "") -> bool:
        """Check if query is a follow-up question"""
        query_clean = query.strip().lower()
        is_followup_pattern = any(
            re.match(pattern, query_clean, re.IGNORECASE)
            for pattern in cls.FOLLOWUP_PATTERNS
        )
        return is_followup_pattern

    @classmethod
    def extract_package_name(cls, text: str) -> Optional[str]:
        """Extract package name from text"""
        for pattern in cls.PACKAGE_EXTRACTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None


# Global instances
category_detector = CategoryDetector()
query_analyzer = QueryAnalyzer()
