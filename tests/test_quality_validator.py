"""Unit tests for quality_validator (no API required)."""

import pytest

from src.evaluation.quality_validator import (
    extract_summary_payload,
    topic_coverage,
    check_cross_domain,
    detect_summary_flags,
    heuristic_score,
    validate_task,
    ValidationResult,
)


def test_extract_summary_payload():
    step = {
        "summary": {
            "key_findings": ["Solar efficiency 20%"],
            "conclusion": "Solar is viable.",
        }
    }
    obj, text = extract_summary_payload(step)
    assert "Solar efficiency" in text
    assert "viable" in text


def test_topic_coverage():
    cov, hit, miss = topic_coverage(
        "SQL injection and index optimization for PostgreSQL",
        ["sql", "injection", "index", "missing_topic"],
    )
    assert cov == 0.75
    assert "sql" in hit
    assert "missing_topic" in miss


def test_topic_synonym_solar_photovoltaic():
    """'solar' topic matches answer that only says photovoltaic (v0.6)."""
    cov, hit, miss = topic_coverage(
        "Monocrystalline photovoltaic panels achieve 20% efficiency.",
        ["solar", "photovoltaic", "efficiency"],
    )
    assert "solar" in hit
    assert "photovoltaic" in hit
    assert cov >= 0.66


def test_cross_domain_security():
    ok, hits = check_cross_domain(
        "Perovskite solar panels achieve high photovoltaic efficiency",
        "security",
    )
    assert not ok
    assert hits


def test_error_flag_fails_heuristic():
    task = {
        "task_id": "t1",
        "description": "Analyze SQL injection patterns in Python web apps",
        "tags": ["security"],
        "expected_topics": ["sql", "injection"],
        "domain": "security",
    }
    step = {
        "summary": {
            "key_findings": [],
            "conclusion": "error",
            "_error": True,
        }
    }

    def fake_encode(_):
        return [1.0] + [0.0] * 383

    v, _, _ = validate_task(
        task, step, fake_encode, use_llm_judge=False, pass_threshold=0.65
    )
    assert v.has_error_flag
    assert not v.composite_pass


def test_good_security_answer_passes_heuristic():
    task = {
        "task_id": "t2",
        "description": "Analyze SQL injection and command injection in Python",
        "tags": ["security"],
        "expected_topics": ["sql", "injection", "security", "python"],
        "domain": "security",
    }
    answer = (
        "SQL injection via string formatting is detectable with static analysis. "
        "Command injection using os.system requires code review. "
        "Python security best practices include parameterized queries."
    )
    step = {"summary": {"key_findings": [answer], "conclusion": answer}}

    def encode(text):
        # Simple deterministic: same text → high self-similarity
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        return [float(h[i % 16]) / 255.0 for i in range(384)]

    v, _, text = validate_task(
        task, step, encode, use_llm_judge=False, pass_threshold=0.5
    )
    assert v.answer_nonempty
    assert v.topic_coverage >= 0.5
    assert not v.has_error_flag
