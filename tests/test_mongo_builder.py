import pytest
from datetime import datetime, timezone
from ai_search import MongoQueryBuilder

def test_mongo_query_builder():
    builder = MongoQueryBuilder()
    ref_time = datetime(2026, 9, 24, 23, 0, 0, tzinfo=timezone.utc)
    
    ui_filters = {"platform": "twitter", "sentiment": "negative"}
    res = builder.build_query("oayement issue occur in last 2 days", ui_filters=ui_filters, reference_time=ref_time)
    
    mongo_q = res["mongo_query"]
    assert mongo_q["platform"] == "twitter"
    assert mongo_q["sentiment.label"] == "negative"
    assert "$gte" in mongo_q["created_at"]
    assert "$or" in mongo_q
