# tests/test_rules.py
import pytest
from src.rules import find_warning_signals


def test_normal_personal_message_no_signals():
    msg = "Hey, are we still meeting at the library at 5 PM to review lab notes?"
    assert find_warning_signals(msg) == []


def test_empty_or_none_input():
    assert find_warning_signals("") == []
    assert find_warning_signals(None) == []


def test_prize_wording_variations():
    msg1 = "Congratulations! You have won a prize of $1000."
    msg2 = "You've won a brand new car! Claim your prize now."
    msg3 = "You are the lucky winner of our lottery!"

    assert "Prize or reward language" in find_warning_signals(msg1)
    assert "Prize or reward language" in find_warning_signals(msg2)
    assert "Prize or reward language" in find_warning_signals(msg3)


def test_otp_and_sensitive_info_requests():
    msg1 = "Your bank OTP is 482910. Do not share your password with anyone."
    msg2 = "Please send your CVV and PIN number to confirm your identity."
    msg3 = "Please submit your one-time password to verify."

    assert "Sensitive information request" in find_warning_signals(msg1)
    assert "Sensitive information request" in find_warning_signals(msg2)
    assert "Sensitive information request" in find_warning_signals(msg3)


def test_suspicious_urls():
    msg1 = "Track your parcel at http://delivery-tracking.info/pkg"
    msg2 = "Claim benefits here: www.claim-portal.xyz"
    msg3 = "Verify account via bit.ly/secure-login"

    assert "Suspicious link" in find_warning_signals(msg1)
    assert "Suspicious link" in find_warning_signals(msg2)
    assert "Suspicious link" in find_warning_signals(msg3)


def test_urgency_language():
    msg = "Act now! Your subscription expires today. Final warning."
    signals = find_warning_signals(msg)
    assert "Urgency or pressure" in signals


def test_payment_related_language():
    msg = "Payment required: your account suspended unless you pay now."
    signals = find_warning_signals(msg)
    assert "Payment-related language" in signals


def test_case_insensitive_matching():
    msg = "URGENT: YOU HAVE WON A PRIZE. SEND MONEY NOW."
    signals = find_warning_signals(msg)
    assert "Urgency or pressure" in signals
    assert "Prize or reward language" in signals
    assert "Payment-related language" in signals


def test_multi_indicator_trigger():
    msg = (
        "URGENT: Your account suspended. Pay now via bit.ly/auth or "
        "enter your OTP to confirm."
    )
    signals = find_warning_signals(msg)
    expected = {
        "Urgency or pressure",
        "Payment-related language",
        "Suspicious link",
        "Sensitive information request",
    }
    assert expected.issubset(set(signals))