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

def test_numeric_price_query_builder():
    builder = MongoQueryBuilder()
    res = builder.build_query("Payment issue having price more than 2 thousand")
    
    mongo_q = res["mongo_query"]
    parsed = res["parsed_intent"]
    
    assert "price" in mongo_q
    assert mongo_q["price"] == {"$gt": 2000}
    assert parsed["numeric_filter"]["operator"] == "$gt"
    assert parsed["numeric_filter"]["value"] == 2000
    assert "$or" in mongo_q
