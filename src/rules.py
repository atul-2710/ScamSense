# src/rules.py
"""
ScamSense - Rule-Based Warning Indicators.

These rule checks complement the ML model by detecting common high-risk patterns.
They do not constitute a complete scam or phishing detection system.
"""

import re
from typing import List

WARNING_PATTERNS = {
    "Suspicious link": (
        r"https?://|www\.|bit\.ly|tinyurl\.com|"
        r"\b[\w-]+\.(?:com|net|org|info|xyz)\b"
    ),
    "Urgency or pressure": (
        r"\b(?:urgent|immediately|act now|"
        r"expires today|limited time|final warning)\b"
    ),
    "Prize or reward language": (
        r"\b(?:you(?:'| a)?ve won|winner|"
        r"claim your prize|free entry|lottery|"
        r"won a prize)\b"
    ),
    "Sensitive information request": (
        r"\b(?:password|otp|one[\s\.-]?time password|"
        r"pin number|cvv|bank details)\b"
    ),
    "Payment-related language": (
        r"\b(?:pay now|send money|processing fee|"
        r"payment required|account suspended)\b"
    ),
}


def find_warning_signals(message: str) -> List[str]:
    """
    Scans an input string against predefined regular expressions.
    Returns a list of matched warning category names.
    """
    if not message or not isinstance(message, str):
        return []

    signals = []
    for name, pattern in WARNING_PATTERNS.items():
        if re.search(pattern, message, flags=re.IGNORECASE):
            signals.append(name)

    return signals