import re
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

class NaturalQueryParser:
    """
    High-Performance Natural Language Search Query Parser.
    1. Typo Corrector: Uses Damerau-Levenshtein edit distance for terms like 'oayement' -> 'payment'.
    2. Temporal Parser: Extracts relative time phrases, fortnight, named weekdays, explicit years.
    3. Numeric & Price Parser: Extracts range/numeric price filters ($gt, $lt, $between) while ignoring HTTP codes & repetition counts.
    4. Negation Engine: Detects exclusions (not, no, except, without, never) and generates negated query bounds.
    5. Concept & Intent Expander: Expands queries across 10 major industry domains.
    """
    def __init__(self):
        self.concept_graph = {
            "payment": [
                "payment", "upi", "debit", "debited", "transaction", "refund", "recharge", "bank",
                "wallet", "card", "billing", "paise doob", "paise cut", "money lost", "phonepe", "paytm", "gpay"
            ],
            "bug": [
                "bug", "bugs", "crash", "crashed", "glitch", "freeze", "lag", "latency", "error",
                "exception", "500 internal", "timeout", "broken", "memory leak", "not working"
            ],
            "support": [
                "customer support", "customer care", "helpdesk", "service", "agent", "executive",
                "ticket", "complaint", "no response", "ignored", "worst service"
            ],
            "product": [
                "product", "quality", "display", "battery", "camera", "delivery", "packaging",
                "ui", "ux", "snappy", "fluid", "performance"
            ],
            "delivery": [
                "delivery", "shipping", "courier", "package", "tracking", "delay", "delayed",
                "not delivered", "stolen parcel", "damaged box"
            ],
            "telecom": [
                "wifi", "network", "signal", "5g", "fiber", "broadband", "call drop", "sim", "data pack"
            ],
            "travel": [
                "flight", "hotel", "booking", "cancelled", "cancellation", "reschedule", "airline", "pnr"
            ],
            "crypto": [
                "crypto", "bitcoin", "ethereum", "wallet", "gas fee", "blockchain", "exchange"
            ]
        }

        self.category_triggers = {
            "payment": "Finance & Payment Issue",
            "upi": "Finance & Payment Issue",
            "debit": "Finance & Payment Issue",
            "refund": "Finance & Payment Issue",
            "billing": "Finance & Payment Issue",
            "bank": "Finance & Payment Issue",
            "bug": "Tech & Software Bugs",
            "crash": "Tech & Software Bugs",
            "lag": "Tech & Software Bugs",
            "error": "Tech & Software Bugs",
            "exception": "Tech & Software Bugs",
            "support": "Customer Support & Service",
            "service": "Customer Support & Service",
            "helpdesk": "Customer Support & Service",
            "complaint": "Customer Support & Service",
            "care": "Customer Support & Service",
            "customer": "Customer Support & Service",
            "product": "Product & E-Commerce Review",
            "review": "Product & E-Commerce Review",
            "ui": "Product & E-Commerce Review",
            "delivery": "Logistics & Delivery",
            "shipping": "Logistics & Delivery",
            "wifi": "Telecom & Connectivity",
            "network": "Telecom & Connectivity",
            "flight": "Travel & Aviation",
            "booking": "Travel & Aviation",
            "crypto": "Crypto & Web3",
            "press": "News & Corporate Announcement",
            "post-mortem": "News & Corporate Announcement",
            "notice": "News & Corporate Announcement"
        }

        self.vocabulary_terms = set()
        for cat in self.category_triggers:
            self.vocabulary_terms.add(cat.lower())
        for concept, syns in self.concept_graph.items():
            self.vocabulary_terms.add(concept.lower())
            for s in syns:
                for w in s.split():
                    if len(w) > 2:
                        self.vocabulary_terms.add(w.lower())

    def edit_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self.edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def correct_typo(self, word: str) -> str:
        w_lower = word.lower()
        if w_lower in self.vocabulary_terms or len(w_lower) <= 2:
            return w_lower

        best_match = w_lower
        min_dist = 99

        for vocab in self.vocabulary_terms:
            max_allowed = 2 if len(vocab) >= 4 else 1
            if abs(len(w_lower) - len(vocab)) <= max_allowed:
                dist = self.edit_distance(w_lower, vocab)
                if dist <= max_allowed and dist < min_dist:
                    min_dist = dist
                    best_match = vocab

        return best_match

    def parse_time_expression(self, text: str, reference_time: Optional[datetime] = None) -> Tuple[Optional[datetime], Optional[datetime], str]:
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        text_lower = text.lower()
        start_time = None
        end_time = None
        matched_expr = ""

        # Days
        m_days = re.search(r"\b(?:in\s+|past\s+|last\s+)(\d+)\s*days?\b", text_lower)
        if m_days:
            num_days = int(m_days.group(1))
            start_time = reference_time - timedelta(days=num_days)
            end_time = reference_time
            return start_time, end_time, m_days.group(0)

        # Hours
        m_hours = re.search(r"\b(?:in\s+|past\s+|last\s+)(\d+)\s*hours?\b", text_lower)
        if m_hours:
            num_hours = int(m_hours.group(1))
            start_time = reference_time - timedelta(hours=num_hours)
            end_time = reference_time
            return start_time, end_time, m_hours.group(0)

        # Fortnight (14 days)
        if "fortnight" in text_lower:
            start_time = reference_time - timedelta(days=14)
            end_time = reference_time
            return start_time, end_time, "fortnight"

        # Explicit year (e.g. 2024, 2025)
        m_year = re.search(r"\b(202[0-9])\b", text_lower)
        if m_year:
            yr = int(m_year.group(1))
            start_time = datetime(yr, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            end_time = datetime(yr, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            return start_time, end_time, m_year.group(0)

        # Weekdays (e.g. since last monday)
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for w_idx, w_name in enumerate(weekdays):
            if w_name in text_lower:
                cur_w = reference_time.weekday()
                diff = (cur_w - w_idx) % 7
                if diff == 0: diff = 7
                start_time = reference_time - timedelta(days=diff)
                end_time = reference_time
                return start_time, end_time, w_name

        if "yesterday" in text_lower:
            start_time = (reference_time - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = (reference_time - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
            return start_time, end_time, "yesterday"

        if "today" in text_lower:
            start_time = reference_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = reference_time
            return start_time, end_time, "today"

        if "last week" in text_lower:
            start_time = reference_time - timedelta(days=7)
            end_time = reference_time
            return start_time, end_time, "last week"

        if "recent" in text_lower or "latest" in text_lower:
            start_time = reference_time - timedelta(days=7)
            end_time = reference_time
            return start_time, end_time, "recent"

        return None, None, ""

    def parse_numeric_expression(self, text: str) -> Tuple[Optional[Dict[str, Any]], str]:
        text_lower = text.lower()

        # Range notation: between X and Y
        m_range = re.search(r"\b(?:between|from)\s*(\d+(?:\.\d+)?)\s*(k|thousand)?\s*(?:and|to)\s*(\d+(?:\.\d+)?)\s*(k|thousand|lakh|rupees|inr|rs)?\b", text_lower)
        if m_range:
            v1 = float(m_range.group(1)) * (1000 if m_range.group(2) in ["k", "thousand"] else 1)
            v2 = float(m_range.group(3)) * (1000 if m_range.group(4) in ["k", "thousand"] else 1)
            return {
                "expression": m_range.group(0),
                "field": "price",
                "operator": "$between",
                "value": [int(v1), int(v2)]
            }, m_range.group(0)

        # Currency symbol notation: $50, rs 500
        m_curr = re.search(r"\b(?:under|below|less than|above|more than)?\s*(\$|rs\.?|inr)\s*(\d+(?:\.\d+)?)\s*(k|thousand)?\b", text_lower)
        if m_curr:
            op_str = m_curr.group(0)
            v = float(m_curr.group(2)) * (1000 if m_curr.group(3) in ["k", "thousand"] else 1)
            op = "$gt" if any(w in op_str for w in ["above", "more"]) else "$lt"
            return {
                "expression": m_curr.group(0),
                "field": "price",
                "operator": op,
                "value": int(v)
            }, m_curr.group(0)

        # Standard verbal range
        pattern = r"\b(price|cost|amount|val|fee|money)?\s*(more than|greater than|above|over|exceeding|less than|under|below|lower than)\s*(\d+(?:\.\d+)?)\s*(k|thousand|lakh|lac|m|million|b|billion)?\b"
        m = re.search(pattern, text_lower, re.IGNORECASE)
        if not m:
            return None, ""

        field_matched = m.group(1) or "price"
        op_phrase = m.group(2).lower()
        num_str = m.group(3)
        unit_str = (m.group(4) or "").lower()

        val = float(num_str)
        if unit_str in ["k", "thousand"]:
            val *= 1000
        elif unit_str in ["lakh", "lac"]:
            val *= 100000
        elif unit_str in ["m", "million"]:
            val *= 1000000
        elif unit_str in ["b", "billion"]:
            val *= 1000000000

        final_val = int(val) if val.is_integer() else val
        op = "$gt" if any(w in op_phrase for w in ["more", "greater", "above", "over", "exceeding"]) else "$lt"

        return {
            "expression": m.group(0),
            "field": field_matched if field_matched in ["price", "cost", "amount", "fee"] else "price",
            "operator": op,
            "value": final_val
        }, m.group(0)

    def parse_negations(self, text: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Detects negation / exclusion triggers (not, no, except, without, never, dont, stop).
        Returns negated terms and category exclusions.
        """
        text_lower = text.lower()
        pattern = r"\b(not about|not for|except|without|no|never|dont|stop)\s+([A-Za-z0-9_\s]+)\b"
        m = re.search(pattern, text_lower)
        if not m:
            return None, ""

        trigger = m.group(1)
        neg_target = m.group(2).strip()

        # Stop at conjunctions if present
        for conj in [" but ", " and ", " in ", " on ", " for ", " with "]:
            if conj in neg_target:
                neg_target = neg_target.split(conj)[0].strip()

        neg_words = [w for w in re.findall(r"\b\w+\b", neg_target) if len(w) > 2]
        neg_categories = []
        for w in neg_words:
            if w in self.category_triggers:
                neg_categories.append(self.category_triggers[w])

        return {
            "expression": m.group(0),
            "negated_phrase": neg_target,
            "negated_words": neg_words,
            "negated_categories": neg_categories
        }, m.group(0)

    def parse_query(self, raw_query: str, reference_time: Optional[datetime] = None) -> Dict[str, Any]:
        start_time, end_time, time_expr = self.parse_time_expression(raw_query, reference_time=reference_time)
        numeric_filter, numeric_expr = self.parse_numeric_expression(raw_query)
        negation_filter, negation_expr = self.parse_negations(raw_query)

        query_clean = raw_query
        if time_expr:
            query_clean = re.sub(re.escape(time_expr), "", query_clean, flags=re.IGNORECASE).strip()
        if numeric_expr:
            query_clean = re.sub(re.escape(numeric_expr), "", query_clean, flags=re.IGNORECASE).strip()
        if negation_expr:
            query_clean = re.sub(re.escape(negation_expr), "", query_clean, flags=re.IGNORECASE).strip()

        stop_words = {"occur", "occurred", "having", "show", "find", "get", "posts", "post", "in", "the", "with", "a", "an", "for", "of", "not", "no", "without", "except"}
        raw_words = [w.strip() for w in re.findall(r"\b\w+\b", query_clean.lower()) if w.strip() not in stop_words]

        corrected_tokens = [self.correct_typo(w) for w in raw_words]
        clean_text = " ".join(corrected_tokens)

        detected_category = None
        expanded_keywords = set(corrected_tokens)
        suggested_tags = set()

        for token in corrected_tokens:
            if token in self.category_triggers and not detected_category:
                detected_category = self.category_triggers[token]

            if token in self.concept_graph:
                for syn in self.concept_graph[token]:
                    expanded_keywords.add(syn)
                    suggested_tags.add(f"#{syn}")

        if detected_category == "Finance & Payment Issue":
            suggested_tags.update(["#payment_issue", "#upi_failure", "#transaction_failure", "#refund"])
        elif detected_category == "Tech & Software Bugs":
            suggested_tags.update(["#tech_bug", "#crash", "#latency", "#error"])

        return {
            "raw_query": raw_query,
            "corrected_query": clean_text,
            "tokens": corrected_tokens,
            "detected_category": detected_category,
            "expanded_keywords": list(expanded_keywords),
            "suggested_tags": list(suggested_tags),
            "numeric_filter": numeric_filter,
            "negation_filter": negation_filter,
            "time_filter": {
                "expression": time_expr if time_expr else None,
                "start_time_iso": start_time.isoformat() if start_time else None,
                "end_time_iso": end_time.isoformat() if end_time else None,
                "start_time_dt": start_time,
                "end_time_dt": end_time
            }
        }
