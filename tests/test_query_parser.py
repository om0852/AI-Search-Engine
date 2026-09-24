import pytest
from datetime import datetime, timezone
from ai_search import NaturalQueryParser

def test_typo_correction_and_time_parser():
    parser = NaturalQueryParser()
    ref_time = datetime(2026, 9, 24, 23, 0, 0, tzinfo=timezone.utc)
    
    res = parser.parse_query("oayement issue occur in last 2 days", reference_time=ref_time)
    
    assert res["corrected_query"] == "payment issue"
    assert res["detected_category"] == "Finance & Payment Issue"
    assert res["time_filter"]["expression"] == "last 2 days"
    assert "upi" in res["expanded_keywords"]
    assert "#payment_issue" in res["suggested_tags"]

def test_relative_time_expressions():
    parser = NaturalQueryParser()
    ref_time = datetime(2026, 9, 24, 23, 0, 0, tzinfo=timezone.utc)
    
    # 1. Past 24 hours
    res1 = parser.parse_query("critical bug reported in past 24 hours", reference_time=ref_time)
    assert res1["corrected_query"] == "critical bug reported"
    assert res1["detected_category"] == "Tech & Software Bugs"
    assert res1["time_filter"]["expression"] == "past 24 hours"

    # 2. Yesterday
    res2 = parser.parse_query("customer care complaint yesterday", reference_time=ref_time)
    assert res2["detected_category"] == "Customer Support & Service"
    assert res2["time_filter"]["expression"] == "yesterday"
