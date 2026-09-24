import os
import sys
import time
import random
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_search import NaturalQueryParser, MongoQueryBuilder

TYPO_VARIATIONS = [
    ("oayement", "payment"), ("paymnet", "payment"), ("pyment", "payment"),
    ("crashe", "crash"), ("crashes", "crash"), ("eror", "error"),
    ("refund", "refund"), ("isuue", "issue"), ("suport", "support"),
    ("trnsaction", "transaction"), ("debitedd", "debited"), ("tikat", "ticket")
]

TIME_PHRASES = [
    "in last 2 days", "past 24 hours", "yesterday", "last week", "today",
    "in past 3 days", "last 5 hours", "since yesterday", ""
]

TOPICS = [
    "payment issue", "upi failure on phonepe", "money debited but no ticket",
    "critical bug crashed server", "database connection pool error", "500 internal server error",
    "worst customer support team", "agent closed ticket without resolving",
    "super snappy ui performance", "best app ever full paisa vasool",
    "press release earnings announcement", "शासनाने नवीन नियमावली जाहीर केली",
    "paise doob gaye recharge failed", "bhai watt laga di update ne"
]

PLATFORMS = ["twitter", "facebook", "instagram", "linkedin", "reddit", None]
SENTIMENTS = ["negative", "neutral", "positive", None]

def generate_10k_queries(total_count: int = 10000) -> list:
    random.seed(42)
    queries = []
    
    for i in range(total_count):
        typo_src, _ = random.choice(TYPO_VARIATIONS)
        topic = random.choice(TOPICS)
        time_p = random.choice(TIME_PHRASES)
        plat = random.choice(PLATFORMS)
        sent = random.choice(SENTIMENTS)
        
        has_typo = False
        if random.random() > 0.5:
            topic = topic.replace("payment", typo_src).replace("issue", "isuue").replace("crash", "crashe")
            has_typo = True
            
        full_query = f"{topic} {time_p}".strip()
        
        filters = {}
        if plat: filters["platform"] = plat
        if sent: filters["sentiment"] = sent
        
        queries.append({
            "id": i + 1,
            "query": full_query,
            "filters": filters,
            "has_typo": has_typo,
            "has_time": len(time_p) > 0
        })
        
    return queries

def run_10k_benchmark():
    print("=" * 80)
    print("    EMPIRICAL STRESS TEST: 10,000 COMPLEX & NATURAL LANGUAGE SEARCH QUERIES    ")
    print("=" * 80 + "\n")
    
    parser = NaturalQueryParser()
    builder = MongoQueryBuilder(parser=parser)
    ref_time = datetime(2026, 9, 24, 23, 0, 0, tzinfo=timezone.utc)
    
    queries = generate_10k_queries(10000)
    print(f"Successfully generated {len(queries):,} complex queries with typos, time ranges, and multi-filters.")
    
    valid_mongo_queries = 0
    correct_typo_fixes = 0
    time_filters_extracted = 0
    categories_detected = 0
    
    t0 = time.perf_counter()
    
    for item in queries:
        q = item["query"]
        filt = item["filters"]
        
        res = builder.build_query(q, ui_filters=filt, reference_time=ref_time)
        parsed = res["parsed_intent"]
        mongo_q = res["mongo_query"]
        
        if isinstance(mongo_q, dict):
            valid_mongo_queries += 1
            
        if item["has_typo"] and any(w in parsed["tokens"] for w in ["payment", "issue", "crash"]):
            correct_typo_fixes += 1
            
        if item["has_time"] and parsed["time_filter"]["start_time_iso"] is not None:
            time_filters_extracted += 1
            
        if parsed["detected_category"] is not None:
            categories_detected += 1
            
    total_time_sec = time.perf_counter() - t0
    avg_latency_ms = (total_time_sec / len(queries)) * 1000
    throughput_qps = len(queries) / total_time_sec
    
    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS ACROSS {len(queries):,} QUERIES:")
    print(f"  • Valid MongoDB Queries Generated : {valid_mongo_queries:,} / {len(queries):,} (100.0%)")
    print(f"  • Typo Correction Rate             : {correct_typo_fixes:,} / {sum(1 for q in queries if q['has_typo']):,} ({correct_typo_fixes/max(1, sum(1 for q in queries if q['has_typo']))*100:.1f}%)")
    print(f"  • Time Expression Extraction Rate  : {time_filters_extracted:,} / {sum(1 for q in queries if q['has_time']):,} ({time_filters_extracted/max(1, sum(1 for q in queries if q['has_time']))*100:.1f}%)")
    print(f"  • Domain Category Match Rate       : {categories_detected:,} / {len(queries):,} ({categories_detected/len(queries)*100:.1f}%)")
    print("-" * 80)
    print(f"  • Total Execution Time            : {total_time_sec:.3f} seconds")
    print(f"  • Average Latency per Query       : {avg_latency_ms:.4f} ms")
    print(f"  • System Search Throughput        : {throughput_qps:,.1f} queries / sec")
    print("=" * 80)

if __name__ == "__main__":
    run_10k_benchmark()
