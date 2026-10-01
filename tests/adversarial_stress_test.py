import os
import sys
import time
import json
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding='utf-8')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_search import NaturalQueryParser, MongoQueryBuilder

# 100 Highly Complex, Tricky, Ambiguous & Adversarial Human Queries
ADVERSARIAL_QUERIES = [
    # Category 1: Negations & Exclusions (1-15)
    {"id": 1, "query": "show posts NOT about payment issues", "category": "Negation", "expected_challenge": "Negation handling ('not')"},
    {"id": 2, "query": "no server crash but very slow UI lag", "category": "Negation", "expected_challenge": "Negation vs actual topic"},
    {"id": 3, "query": "everything except refund problem", "category": "Negation", "expected_challenge": "Exclusion clause"},
    {"id": 4, "query": "dont send me marketing posts", "category": "Negation", "expected_challenge": "Contraction negation"},
    {"id": 5, "query": "app is working fine no issues faced", "category": "Negation", "expected_challenge": "Positive text with negative words"},
    {"id": 6, "query": "without any bug or error", "category": "Negation", "expected_challenge": "Prepositional negation"},
    {"id": 7, "query": "neither upi nor card working", "category": "Negation", "expected_challenge": "Correlative conjunction"},
    {"id": 8, "query": "never had such a bad experience", "category": "Negation", "expected_challenge": "Adverb negation"},
    {"id": 9, "query": "stop sending spam notifications", "category": "Negation", "expected_challenge": "Command verb negation"},
    {"id": 10, "query": "not satisfied with customer support", "category": "Negation", "expected_challenge": "Negated sentiment"},
    {"id": 11, "query": "no money debited but transaction failed", "category": "Negation", "expected_challenge": "Mixed negation & failure"},
    {"id": 12, "query": "payment was not rejected", "category": "Negation", "expected_challenge": "Negated failure"},
    {"id": 13, "query": "no downtime reported in last 24 hours", "category": "Negation", "expected_challenge": "Negated outage with time"},
    {"id": 14, "query": "cant login into my account", "category": "Negation", "expected_challenge": "Missing apostrophe contraction"},
    {"id": 15, "query": "nothing wrong with delivery speed", "category": "Negation", "expected_challenge": "Double positive/negative"},

    # Category 2: Complex Numerical & Currency Ranges (16-30)
    {"id": 16, "query": "payment issue having price more than 2 thousand", "category": "Numeric", "expected_challenge": "Parsed price > 2000"},
    {"id": 17, "query": "costing between 500 and 2000 rupees", "category": "Numeric", "expected_challenge": "Range between X and Y"},
    {"id": 18, "query": "refund amount exactly 1500 inr", "category": "Numeric", "expected_challenge": "Exact match numeric"},
    {"id": 19, "query": "loss under $50 in past 3 days", "category": "Numeric", "expected_challenge": "Currency symbol $"},
    {"id": 20, "query": "valuation over 10 billion dollars", "category": "Numeric", "expected_challenge": "Large scale billion"},
    {"id": 21, "query": "more than 50k rupees debited from bank", "category": "Numeric", "expected_challenge": "Short 50k notation"},
    {"id": 22, "query": "price less than 1.5 lakh", "category": "Numeric", "expected_challenge": "Decimal 1.5 lakh"},
    {"id": 23, "query": "recharge of 299 failed 3 times", "category": "Numeric", "expected_challenge": "Multiple numbers (amount vs count)"},
    {"id": 24, "query": "error 500 occurred 10 times today", "category": "Numeric", "expected_challenge": "HTTP code vs count"},
    {"id": 25, "query": "bill above Rs. 10,000 with upi error", "category": "Numeric", "expected_challenge": "Formatted number with comma"},
    {"id": 26, "query": "5G data plan below 500/month", "category": "Numeric", "expected_challenge": "Per month rate notation"},
    {"id": 27, "query": "salary not credited for 2 months", "category": "Numeric", "expected_challenge": "Duration count vs price"},
    {"id": 28, "query": "price range 1k to 5k", "category": "Numeric", "expected_challenge": "Short range 1k to 5k"},
    {"id": 29, "query": "transaction greater than equal to 100", "category": "Numeric", "expected_challenge": "Verbal operator GTE"},
    {"id": 30, "query": "loss of around 25000 rs", "category": "Numeric", "expected_challenge": "Approximate verbal price"},

    # Category 3: Hinglish & Multilingual Slang (31-45)
    {"id": 31, "query": "bhai paise kat gaye lekin ticket nahi mila worst service", "category": "Hinglish", "expected_challenge": "Code-switching Hinglish"},
    {"id": 32, "query": "paise doob gaye recharge fail ho gaya", "category": "Hinglish", "expected_challenge": "Hindi idiom paise doob"},
    {"id": 33, "query": "bhai watt laga di naye update ne app crash ho raha hai", "category": "Hinglish", "expected_challenge": "Slang watt laga di"},
    {"id": 34, "query": "aaj subah se wifi bilkul nahi chal raha", "category": "Hinglish", "expected_challenge": "Hindi time & network"},
    {"id": 35, "query": "worst support ever koi jawab hi nahi de raha", "category": "Hinglish", "expected_challenge": "Mixed English & Hindi complaint"},
    {"id": 36, "query": "shaasane navin niyamavali jahir keli", "category": "Marathi Transliterated", "expected_challenge": "Marathi in Roman script"},
    {"id": 37, "query": "शासनाने नवीन नियमावली जाहीर केली", "category": "Devanagari Marathi", "expected_challenge": "Native Marathi Script"},
    {"id": 38, "query": "kab tak milega mera refund 10 din ho gaye", "category": "Hinglish", "expected_challenge": "Hinglish question & duration"},
    {"id": 39, "query": "faltu app hai bhai mat download karo", "category": "Hinglish", "expected_challenge": "Slang review faltu app"},
    {"id": 40, "query": "server down hai kya kisi ka khul raha hai?", "category": "Hinglish", "expected_challenge": "Hinglish outage inquiry"},
    {"id": 41, "query": "paisay kat gaye phonepe se", "category": "Hinglish Typo", "expected_challenge": "Hinglish typo paisay"},
    {"id": 42, "query": "bakwas customer care ek ghante se hold pe hu", "category": "Hinglish", "expected_challenge": "Slang bakwas + duration"},
    {"id": 43, "query": "giga chad ui update super fast", "category": "Internet Slang", "expected_challenge": "Gen-Z internet slang"},
    {"id": 44, "query": "scam alert paise chor liye app ne", "category": "Hinglish Fraud", "expected_challenge": "Fraud slang paise chor"},
    {"id": 45, "query": "plz bro fix dis bug ASAP varna uninstall kar dunga", "category": "Hinglish Threat", "expected_challenge": "Chat speak + threat"},

    # Category 4: Multiple Intent & Complex Compound Queries (46-60)
    {"id": 46, "query": "refund issue AND server outage in past 3 days for twitter", "category": "Multi-Intent", "expected_challenge": "Explicit Boolean AND"},
    {"id": 47, "query": "payment failure or network lag on instagram today", "category": "Multi-Intent", "expected_challenge": "Explicit Boolean OR"},
    {"id": 48, "query": "show positive reviews about camera but negative about battery", "category": "Multi-Intent", "expected_challenge": "Conflicting sentiment aspects"},
    {"id": 49, "query": "flight cancellation AND hotel refund issue yesterday", "category": "Multi-Intent", "expected_challenge": "Two different domain issues"},
    {"id": 50, "query": "critical bug on android app and slow loading on iOS", "category": "Multi-Intent", "expected_challenge": "Multi-platform issues"},
    {"id": 51, "query": "upi failure with high priority tag in last 12 hours", "category": "Multi-Intent", "expected_challenge": "Explicit priority & time"},
    {"id": 52, "query": "swiggy delivery delay vs zomato order cancellation", "category": "Multi-Intent", "expected_challenge": "Comparative query vs"},
    {"id": 53, "query": "crypto wallet hack or phishing scam alert", "category": "Multi-Intent", "expected_challenge": "Security threat options"},
    {"id": 54, "query": "customer care executive rude behavior AND no resolution", "category": "Multi-Intent", "expected_challenge": "Compound complaint"},
    {"id": 55, "query": "app crashing on startup after version 2.4.1 update", "category": "Multi-Intent", "expected_challenge": "Version number + crash"},
    {"id": 56, "query": "wifi disconnected and bluetooth pairing failed", "category": "Multi-Intent", "expected_challenge": "Hardware dual failure"},
    {"id": 57, "query": "billing error in invoice #994012 yesterday", "category": "Multi-Intent", "expected_challenge": "Invoice number + billing"},
    {"id": 58, "query": "login OTP not receiving on mobile or email", "category": "Multi-Intent", "expected_challenge": "Dual channel failure"},
    {"id": 59, "query": "refund requested 5 days ago still pending in bank", "category": "Multi-Intent", "expected_challenge": "Status + past time"},
    {"id": 60, "query": "subscription auto-renewed without permission rupees 999", "category": "Multi-Intent", "expected_challenge": "Auto-renew fraud + price"},

    # Category 5: Severe Typos, Noise & Punctuation Overload (61-75)
    {"id": 61, "query": "PAYMENT??? FAIL!!1! pls help ASAP!!!", "category": "Noise & Caps", "expected_challenge": "All caps & 1! noise"},
    {"id": 62, "query": "rechargeee faildddd dddebitedd", "category": "Character Flooding", "expected_challenge": "Repeated character floods"},
    {"id": 63, "query": "p@ym3nt !ssu3 gp@y f@1l", "category": "Leet Speak", "expected_challenge": "Leet speak symbol substitution"},
    {"id": 64, "query": "s3rv3r cr@sh 500 3rr0r", "category": "Leet Speak", "expected_challenge": "Leet speak numbers"},
    {"id": 65, "query": "what d hell is dis app worst everrrr", "category": "Chat Slang", "expected_challenge": "Slang spelling & lengthened words"},
    {"id": 66, "query": "wtfff money cut from acc", "category": "Abbreviation", "expected_challenge": "Abbreviated swear & words"},
    {"id": 67, "query": "plzzzz help my refund is stuckkkk", "category": "Character Flooding", "expected_challenge": "Flooded z and k"},
    {"id": 68, "query": "cnnt lgn t my accnt", "category": "Vowel Stripping", "expected_challenge": "Vowels stripped chat speak"},
    {"id": 69, "query": "errrrrrrrror in upiiiiiiii", "category": "Character Flooding", "expected_challenge": "Extended r and i"},
    {"id": 70, "query": "paymnt issue.... pls response???", "category": "Dot Overload", "expected_challenge": "Ellipsis punctuation overload"},
    {"id": 71, "query": "HELPPPPP MY MONEY IS GONE", "category": "Caps & Flooding", "expected_challenge": "Panicked caps"},
    {"id": 72, "query": "bank server downnnn 404 not foundd", "category": "HTTP Code + Flooding", "expected_challenge": "404 code + lengthened words"},
    {"id": 73, "query": "#payment #fail #scam #help #urgent", "category": "Hashtag Flood", "expected_challenge": "Pure hashtag query"},
    {"id": 74, "query": "http://example.com/error?code=500 login failure", "category": "URL Noise", "expected_challenge": "Raw URL with params"},
    {"id": 75, "query": "login_error_code_99182", "category": "Snake Case Code", "expected_challenge": "Snake case string"},

    # Category 6: Ambiguous / Polysemous Words & Edge Cases (76-90)
    {"id": 76, "query": "apple phone crash", "category": "Polysemy", "expected_challenge": "Apple (brand vs fruit)"},
    {"id": 77, "query": "bank account frozen", "category": "Polysemy", "expected_challenge": "Frozen (ice vs locked)"},
    {"id": 78, "query": "canva log in error", "category": "Polysemy", "expected_challenge": "Canva (brand vs canvas)"},
    {"id": 79, "query": "java memory leak", "category": "Polysemy", "expected_challenge": "Java (language vs island)"},
    {"id": 80, "query": "python script error", "category": "Polysemy", "expected_challenge": "Python (language vs snake)"},
    {"id": 81, "query": "bug in the kitchen app", "category": "Polysemy", "expected_challenge": "Bug (insect vs software)"},
    {"id": 82, "query": "charge on credit card", "category": "Polysemy", "expected_challenge": "Charge (fee vs battery)"},
    {"id": 83, "query": "phone battery charge issue", "category": "Polysemy", "expected_challenge": "Charge (battery vs fee)"},
    {"id": 84, "query": "driver update for display", "category": "Polysemy", "expected_challenge": "Driver (software vs person)"},
    {"id": 85, "query": "train booking ticket error", "category": "Polysemy", "expected_challenge": "Ticket (railway vs support)"},
    {"id": 86, "query": "support ticket for laptop", "category": "Polysemy", "expected_challenge": "Ticket (support vs event)"},
    {"id": 87, "query": "flat 50 percent discount", "category": "Polysemy", "expected_challenge": "Flat (apartment vs level)"},
    {"id": 88, "query": "lightweight app performance", "category": "Polysemy", "expected_challenge": "Lightweight (weight vs tech)"},
    {"id": 89, "query": "sweet customer support response", "category": "Polysemy", "expected_challenge": "Sweet (taste vs nice)"},
    {"id": 90, "query": "cold response from customer care", "category": "Polysemy", "expected_challenge": "Cold (temperature vs emotion)"},

    # Category 7: Temporal & Relative Metric Edge Cases (91-100)
    {"id": 91, "query": "posts from last fortnight", "category": "Time Phrase", "expected_challenge": "Uncommon word fortnight"},
    {"id": 92, "query": "since 3 days ago", "category": "Time Phrase", "expected_challenge": "Redundant 'since ago'"},
    {"id": 93, "query": "in past fortnight with over 100 likes", "category": "Time + Metric", "expected_challenge": "Time + social likes"},
    {"id": 94, "query": "posts between 9am and 5pm yesterday", "category": "Time Span", "expected_challenge": "Intra-day time span"},
    {"id": 95, "query": "issues since last monday", "category": "Time Day", "expected_challenge": "Named weekday monday"},
    {"id": 96, "query": "recent payment issues", "category": "Vague Time", "expected_challenge": "Vague word 'recent'"},
    {"id": 97, "query": "old bugs from 2024", "category": "Year Filter", "expected_challenge": "Explicit year 2024"},
    {"id": 98, "query": "latest news about canva", "category": "Vague Time", "expected_challenge": "Superlative 'latest'"},
    {"id": 99, "query": "q3 earnings announcement", "category": "Quarter Time", "expected_challenge": "Quarter 'q3'"},
    {"id": 100, "query": "payment issue logged at midnight", "category": "Specific Time", "expected_challenge": "Specific word 'midnight'"}
]

def run_adversarial_audit():
    print("=" * 90)
    print("      ADVERSARIAL STRESS TEST & FAILURE AUDIT: 100 COMPLEX & TRICKY QUERIES      ")
    print("=" * 90 + "\n")

    parser = NaturalQueryParser()
    builder = MongoQueryBuilder(parser=parser)
    ref_time = datetime(2026, 9, 24, 23, 0, 0, tzinfo=timezone.utc)

    passed_count = 0
    failed_count = 0
    partial_count = 0

    failure_log = []

    for item in ADVERSARIAL_QUERIES:
        q_id = item["id"]
        q_str = item["query"]
        cat = item["category"]
        challenge = item["expected_challenge"]

        res = builder.build_query(q_str, reference_time=ref_time)
        parsed = res["parsed_intent"]
        mongo_q = res["mongo_query"]

        # Evaluate performance & potential failures
        has_category = parsed["detected_category"] is not None
        has_numeric = parsed.get("numeric_filter") is not None
        has_time = parsed["time_filter"]["start_time_iso"] is not None
        has_or_text = "$or" in mongo_q

        status = "PASSED"
        issues_found = []

        # Audit Check 1: Negations
        if cat == "Negation":
            if "not" in q_str.lower() or "no " in q_str.lower() or "without" in q_str.lower():
                # Check if mongo_query contains negation operator ($ne / $nin / $nor)
                has_negation_op = any("$ne" in str(mongo_q) or "$nin" in str(mongo_q) or "$nor" in str(mongo_q) for _ in [1])
                if not has_negation_op:
                    status = "FAILED"
                    issues_found.append("Negation clause ('not'/'no') ignored in MongoDB filter ($nor / $ne missing)")

        # Audit Check 2: Numerics
        elif cat == "Numeric":
            if not has_numeric and any(w in q_str.lower() for w in ["more than", "between", "under", "above", "over", "less than"]):
                status = "FAILED"
                issues_found.append("Numeric range/price constraint missed")

        # Audit Check 3: Hinglish & Multilingual
        elif "Hinglish" in cat:
            if not has_category and not has_or_text:
                status = "FAILED"
                issues_found.append("Failed to extract intent from Hinglish/slang phrase")

        # Audit Check 4: Extreme Noise & Leet Speak
        elif cat in ["Noise & Caps", "Leet Speak", "Vowel Stripping"]:
            if not parsed["tokens"]:
                status = "FAILED"
                issues_found.append("Noise/Leet speak destroyed query tokens")

        # Audit Check 5: Temporal Edge Cases
        elif "Time" in cat or cat in ["Time Span", "Time Day", "Quarter Time"]:
            if not has_time and any(w in q_str.lower() for w in ["fortnight", "monday", "2024", "midnight"]):
                status = "PARTIAL"
                issues_found.append("Time phrase word not matched in temporal parser")

        if status == "FAILED":
            failed_count += 1
            failure_log.append({
                "id": q_id,
                "query": q_str,
                "category": cat,
                "challenge": challenge,
                "status": "FAILED",
                "issues": issues_found,
                "generated_query": mongo_q
            })
        elif status == "PARTIAL":
            partial_count += 1
            failure_log.append({
                "id": q_id,
                "query": q_str,
                "category": cat,
                "challenge": challenge,
                "status": "PARTIAL",
                "issues": issues_found,
                "generated_query": mongo_q
            })
        else:
            passed_count += 1

    print("=" * 90)
    print("                     EMPIRICAL ADVERSARIAL AUDIT SUMMARY                         ")
    print("=" * 90)
    print(f"  • Fully Handled Queries : {passed_count} / 100 ({passed_count}%) 🟢")
    print(f"  • Partial Intent Matches: {partial_count} / 100 ({partial_count}%) 🟡")
    print(f"  • Model Failure Cases   : {failed_count} / 100 ({failed_count}%) 🔴")
    print("-" * 90)

    print("\n" + "=" * 90)
    print("                 TOP DETECTED MODEL GAPS & VULNERABILITIES                       ")
    print("=" * 90)
    
    cat_failures = {}
    for item in failure_log:
        c = item["category"]
        cat_failures[c] = cat_failures.get(c, 0) + 1

    for c, count in cat_failures.items():
        print(f"  ⚠️ Category '{c}': {count} queries failed or produced partial query bounds.")

    print("\n" + "=" * 90)
    print("                     SAMPLE DETAILED FAILURE LOGS (FIRST 5)                       ")
    print("=" * 90)

    for item in failure_log[:5]:
        print(f"\n[QUERY #{item['id']}] \"{item['query']}\"")
        print(f"  • Category: {item['category']} | Challenge: {item['challenge']}")
        print(f"  • Status: {item['status']}")
        print(f"  • Detected Issue: {', '.join(item['issues'])}")
        print(f"  • Generated Mongo Query: {item['generated_query']}")

    print("=" * 90)

if __name__ == '__main__':
    run_adversarial_audit()
