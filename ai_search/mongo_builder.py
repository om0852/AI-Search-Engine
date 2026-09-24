from typing import Dict, Any, List, Optional
from datetime import datetime
from .query_parser import NaturalQueryParser

class MongoQueryBuilder:
    """
    Translates Natural Language queries and UI dropdown filters into native,
    highly indexed MongoDB query JSON objects for PyMongo or Mongoose.
    """
    def __init__(self, parser: Optional[NaturalQueryParser] = None):
        self.parser = parser or NaturalQueryParser()

    def build_query(self, natural_query: str, ui_filters: Optional[Dict[str, Any]] = None, reference_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Combines parsed AI natural query parameters with user's UI dropdown filters.
        Example UI filters: {'platform': 'twitter', 'sentiment': 'negative', 'language': 'marathi'}
        """
        parsed = self.parser.parse_query(natural_query, reference_time=reference_time)
        ui_filters = ui_filters or {}

        mongo_filter: Dict[str, Any] = {}

        # A. Apply UI Dropdown Filters (e.g. platform, sentiment, author)
        for key, val in ui_filters.items():
            if val is not None and val != "":
                if key == "sentiment":
                    mongo_filter["sentiment.label"] = val.lower()
                else:
                    mongo_filter[key] = val

        # B. Apply Time Range Filter from Natural Query
        tf = parsed["time_filter"]
        if tf["start_time_iso"] or tf["end_time_iso"]:
            time_cond = {}
            if tf["start_time_iso"]:
                time_cond["$gte"] = tf["start_time_iso"]
            if tf["end_time_iso"]:
                time_cond["$lte"] = tf["end_time_iso"]
            mongo_filter["created_at"] = time_cond

        # C. Apply Concept & Intent Expansion Filter ($or conditions)
        or_conditions = []

        if parsed["detected_category"]:
            or_conditions.append({"category": parsed["detected_category"]})

        if parsed["suggested_tags"]:
            or_conditions.append({"tags": {"$in": parsed["suggested_tags"]}})

        if parsed["expanded_keywords"]:
            search_str = " ".join(parsed["expanded_keywords"])
            or_conditions.append({"$text": {"$search": search_str}})

        if or_conditions:
            mongo_filter["$or"] = or_conditions

        return {
            "parsed_intent": parsed,
            "mongo_query": mongo_filter
        }
