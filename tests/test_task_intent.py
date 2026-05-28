"""Tests for task intent gating."""

from src.task_intent import (
    TaskIntent,
    blocks_executor_dropout,
    cache_intents_compatible,
    detect_intent,
    requires_fresh_synthesis,
)


def test_detect_compare_from_tags():
    intent = detect_intent(
        "Solar and wind LCOE",
        tags=["solar", "wind", "comparison"],
    )
    assert intent == TaskIntent.COMPARE


def test_detect_compare_from_text():
    intent = detect_intent(
        "Compare solar and wind energy across multiple dimensions",
        tags=["energy"],
    )
    assert intent == TaskIntent.COMPARE


def test_wind_analyze_not_compare():
    intent = detect_intent(
        "Research wind energy technologies including onshore and offshore turbines",
        tags=["wind", "energy"],
    )
    assert intent == TaskIntent.ANALYZE


def test_cache_incompatible_compare_vs_analyze():
    wind = "Research wind energy technologies for onshore and offshore deployment"
    compare = "Compare solar and wind energy across LCOE and land use"
    assert not cache_intents_compatible(wind, compare)
    assert not cache_intents_compatible(compare, wind)


def test_cache_compatible_same_compare():
    a = "Compare solar and wind across LCOE"
    b = "Compare solar and wind across capacity factors"
    assert cache_intents_compatible(a, b, ["comparison"], ["comparison"])


def test_requires_fresh_synthesis():
    assert requires_fresh_synthesis(
        "Compare solar and wind", tags=["comparison"]
    )
    assert not requires_fresh_synthesis(
        "Research wind turbines", tags=["wind"]
    )


def test_blocks_executor_dropout():
    assert not blocks_executor_dropout("Compare A and B", ["comparison"])
    assert blocks_executor_dropout("Code review the following snippet", ["review"])
    assert not blocks_executor_dropout("Research wind energy", ["wind"])
    assert blocks_executor_dropout(
        "Outline SOC2 and ISO27001 control mapping",
        ["security", "compliance"],
    )


def test_detect_compliance_from_tags_and_text():
    x3 = (
        "Outline SOC2 and ISO27001 control mapping for a Python SaaS backend: "
        "access control, logging, change management."
    )
    assert detect_intent(x3, ["security", "compliance"]) == TaskIntent.COMPLIANCE


def test_detect_vulnerability_from_text():
    s1 = (
        "Analyze common Python security vulnerabilities: SQL injection via string "
        "formatting, command injection via os.system()."
    )
    assert detect_intent(s1, ["security", "python"]) == TaskIntent.VULNERABILITY


def test_cache_incompatible_compliance_vs_vulnerability():
    compliance = (
        "Outline SOC2 and ISO27001 control mapping for a Python SaaS backend"
    )
    vuln = "Analyze Python security vulnerabilities including SQL injection"
    assert not cache_intents_compatible(
        compliance, vuln, ["security", "compliance"], ["security", "python"]
    )
    assert not cache_intents_compatible(
        vuln, compliance, ["security", "python"], ["security", "compliance"]
    )


def test_compliance_requires_fresh_synthesis():
    desc = "Outline SOC2 control mapping for SaaS"
    assert requires_fresh_synthesis(desc, ["security", "compliance"])


def test_review_requires_fresh_synthesis():
    desc = "Perform a comprehensive web application security audit"
    assert requires_fresh_synthesis(desc, ["security", "web"])
