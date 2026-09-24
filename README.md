# 🚀 AI Natural Language Search Engine & SDK (`ai-search-engine`)

A high-performance (**< 5 ms latency**), zero-LLM-cost AI Natural Language Search Engine and **MongoDB Query Builder**.

It converts natural, unstructured user search queries with typos (*"oayement issue occur in last 2 days"*) into optimized, native MongoDB Queries and sub-word vector similarity rankings without needing expensive external LLM API calls.

---

## ✨ Features

- **Fuzzy Typo Corrector**: Automatically fixes typos using Damerau-Levenshtein edit-distance (`"oayement"` $\rightarrow$ `"payment"`, `"crashe"` $ightarrow$ `"crash"`).
- **Temporal Relative Time Parser**: Converts natural time expressions (`"last 2 days"`, `"past 24 hours"`, `"yesterday"`, `"last week"`, `"today"`) into exact UTC ISO timestamp filters (`$gte`, `$lte`).
- **Concept & Intent Expansion**: Maps queries like `"payment issue"` to domain categories (`Finance & Payment Issue`), semantic tags (`#payment_issue`, `#upi_failure`), and synonym clusters (`["money debited", "transaction failed", "paise doob", "refund"]`).
- **MongoDB Query Builder**: Combines parsed AI intent with existing UI dropdown filters (`platform`, `sentiment`, `language`, `author`) into indexed MongoDB queries.
- **FastAPI Service & Python SDK**: Deploys as a lightweight HTTP server or imports directly as a Python module.

---

## 🛠️ Quickstart

### 1. Installation & Environment Setup

```bash
git clone https://github.com/om0852/ai-search-engine.git
cd ai-search-engine

# Create virtualenv and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run API Server

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8001 --reload
```

Interactive Swagger API docs available at: `http://localhost:8001/docs`

---

## 💡 How to Integrate

### Option A: Python Import (Direct SDK Integration)

```python
from ai_search import AISearchClient, MongoQueryBuilder, NaturalQueryParser

client = AISearchClient()

# 1. Parse natural search query
parsed = client.parser.parse_query("oayement issue occur in last 2 days")
print(parsed["corrected_query"])  # "payment issue"
print(parsed["time_filter"]["expression"])  # "last 2 days"

# 2. Build MongoDB query combining AI intent + UI Dropdown filters
ui_filters = {"platform": "twitter", "sentiment": "negative"}
mongo_build = client.build_mongo_search("oayement issue occur in last 2 days", ui_filters=ui_filters)

# Resulting Mongo Query:
# {
#   "platform": "twitter",
#   "sentiment.label": "negative",
#   "created_at": { "$gte": "2026-09-22T23:00:00Z" },
#   "$or": [
#     { "category": "Finance & Payment Issue" },
#     { "tags": { "$in": ["#payment_issue", "#upi_failure", "#transaction_failure"] } },
#     { "$text": { "$search": "payment issue debit debited upi refund paise doob" } }
#   ]
# }
```

### Option B: Node.js / JavaScript (Mongoose or Native Driver)

```javascript
const axios = require('axios');

async function searchPosts(userSearchText, dropdownFilters) {
  // 1. Call AI Search API to get generated MongoDB query
  const response = await axios.post('http://localhost:8001/search/mongo/build', {
    query: userSearchText,
    filters: dropdownFilters
  });

  const mongoQuery = response.data.result.mongo_query;

  // 2. Execute on your existing MongoDB collection
  const posts = await db.collection('posts').find(mongoQuery).limit(20).toArray();
  return posts;
}
```

---

## 🧪 Running Unit Tests

```bash
pytest tests/
```

100% test coverage across typo correction, time range parsing, category detection, MongoDB query generation, and FastAPI endpoints.
