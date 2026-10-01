import os
import sys
import time
import random
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_search import NaturalQueryParser, MongoQueryBuilder, VectorRanker, HybridVectorEngine

DOMAINS_AND_QUERIES = [
    {
        "domain": "Finance & UPI",
        "queries": ["paise cut gaya payment failed", "upi debited no recharge", "refund not received in bank", "gpay transaction error"],
        "corpus": [
            {"id": 101, "title": "UPI Transaction Failure", "content": "Money debited from bank account but payment failed on PhonePe app"},
            {"id": 102, "title": "Bank Refund Update", "content": "Refund of Rs 500 processed successfully to customer account"},
            {"id": 103, "title": "Weather Notice", "content": "Heavy rain forecast in Pune city today"}
        ],
        "target_id": 101
    },
    {
        "domain": "Software & Server Bugs",
        "queries": ["500 internal server error", "database pool exhausted crash", "backend API latency lag", "memory leak freeze"],
        "corpus": [
            {"id": 201, "title": "Production Server Crash", "content": "Critical 500 internal error due to database connection pool exhaustion"},
            {"id": 202, "title": "New UI Release", "content": "Frontend theme updated with dark mode"},
            {"id": 203, "title": "Hiring Announcement", "content": "We are hiring senior software engineers"}
        ],
        "target_id": 201
    },
    {
        "domain": "Logistics & Delivery",
        "queries": ["stolen package delivery delayed", "courier tracking parcel not arrived", "damaged box shipping issue"],
        "corpus": [
            {"id": 301, "title": "Delivery Delay Complaint", "content": "Order tracking shows package delayed by courier partner in transit"},
            {"id": 302, "title": "Restaurant Review", "content": "Delicious food and fast service at local dining"},
            {"id": 303, "title": "Flight Schedule", "content": "Flight timings updated for morning departure"}
        ],
        "target_id": 301
    },
    {
        "domain": "Telecom & 5G",
        "queries": ["wifi signal call drop 5g", "broadband net down disconnected", "sim card data pack issue"],
        "corpus": [
            {"id": 401, "title": "Network Outage Warning", "content": "Wifi broadband signal constantly dropping causing 5G call disconnects"},
            {"id": 402, "title": "Movie Release", "content": "New blockbuster film releasing this Friday in theaters"},
            {"id": 403, "title": "Stock Market Report", "content": "Sensex gains 300 points in morning trading session"}
        ],
        "target_id": 401
    },
    {
        "domain": "Travel & Aviation",
        "queries": ["flight cancelled pnr refund", "airline booking cancellation error", "hotel reservation reschedule"],
        "corpus": [
            {"id": 501, "title": "Flight Cancellation Error", "content": "Flight cancelled by airline without PNR refund or reschedule option"},
            {"id": 502, "title": "Laptop Repair", "content": "Screen replacement service available at repair center"},
            {"id": 503, "title": "Fitness Tips", "content": "Daily workout routine for health and stamina"}
        ],
        "target_id": 501
    },
    {
        "domain": "Crypto & Web3",
        "queries": ["bitcoin wallet gas fee high", "ethereum transaction pending", "crypto exchange withdrawal stuck"],
        "corpus": [
            {"id": 601, "title": "Crypto Gas Fee Spike", "content": "Bitcoin and Ethereum wallet gas fees reached record high during network congestion"},
            {"id": 602, "title": "Book Review", "content": "Fascinating novel on mystery and adventure"},
            {"id": 603, "title": "School Admission", "content": "Admissions open for primary and secondary classes"}
        ],
        "target_id": 601
    }
]

def run_1000_hybrid_benchmark(total_count: int = 1000):
    print("=" * 80)
    print("    HYBRID VECTOR SEARCH BENCHMARK: 1,000 MULTI-DOMAIN STRESS QUERIES    ")
    print("=" * 80 + "\n")

    ranker = VectorRanker(vector_dim=128)
    random.seed(42)

    queries = []
    for i in range(total_count):
        domain_data = random.choice(DOMAINS_AND_QUERIES)
        q_text = random.choice(domain_data["queries"])
        
        # Add random typo variations occasionally
        if random.random() > 0.6:
            q_text = q_text.replace("payment", "paymnet").replace("server", "servr").replace("delay", "delaye")

        queries.append({
            "id": i + 1,
            "query": q_text,
            "domain": domain_data["domain"],
            "corpus": domain_data["corpus"],
            "target_id": domain_data["target_id"]
        })

    print(f"Generated {len(queries):,} test queries across {len(DOMAINS_AND_QUERIES)} distinct industry domains.\n")

    top1_correct = 0
    dense_hits = 0
    sparse_hits = 0
    total_latency_ms = 0.0

    t0 = time.perf_counter()

    for item in queries:
        q = item["query"]
        corpus = item["corpus"]
        target = item["target_id"]

        q_t0 = time.perf_counter()
        results = ranker.rank_candidates(q, corpus, top_k=3)
        q_elapsed = (time.perf_counter() - q_t0) * 1000
        total_latency_ms += q_elapsed

        if results and results[0]["id"] == target:
            top1_correct += 1

        if results and results[0]["dense_score"] > 0.4:
            dense_hits += 1

        if results and results[0]["sparse_score"] > 0.0:
            sparse_hits += 1

    total_time_sec = time.perf_counter() - t0
    avg_latency = total_latency_ms / len(queries)
    throughput = len(queries) / total_time_sec

    top1_acc = (top1_correct / len(queries)) * 100.0
    dense_recall = (dense_hits / len(queries)) * 100.0
    sparse_precision = (sparse_hits / len(queries)) * 100.0

    print("=" * 80)
    print("EMPIRICAL BENCHMARK RESULTS (1,000 HYBRID SEARCH QUERIES):")
    print(f"  • Hybrid RRF Top-1 Precision (Accurate Item) : {top1_correct:,} / {len(queries):,} ({top1_acc:.2f}%)")
    print(f"  • Dense Vector Semantic Recall Rate          : {dense_hits:,} / {len(queries):,} ({dense_recall:.2f}%)")
    print(f"  • Sparse Lexical BM25 Precision Rate          : {sparse_hits:,} / {len(queries):,} ({sparse_precision:.2f}%)")
    print("-" * 80)
    print(f"  • Total Execution Time                        : {total_time_sec:.3f} seconds")
    print(f"  • Average Hybrid Search Latency               : {avg_latency:.4f} ms")
    print(f"  • Search Throughput                           : {throughput:,.1f} queries / sec")
    print("=" * 80)

if __name__ == '__main__':
    run_1000_hybrid_benchmark(1000)
