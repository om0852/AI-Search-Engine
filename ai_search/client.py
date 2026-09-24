from typing import Dict, Any, List, Optional
from .query_parser import NaturalQueryParser
from .mongo_builder import MongoQueryBuilder
from .vector_ranker import VectorRanker

class AISearchClient:
    """
    Python SDK Client for AI Natural Language Search Engine.
    Usable directly inside any Python social listening or analytics service.
    """
    def __init__(self):
        self.parser = NaturalQueryParser()
        self.builder = MongoQueryBuilder(parser=self.parser)
        self.ranker = VectorRanker()

    def build_mongo_search(self, natural_query: str, ui_filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.builder.build_query(natural_query, ui_filters=ui_filters)

    def search_and_rank(self, natural_query: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        parsed = self.parser.parse_query(natural_query)
        ranked = self.ranker.rank_candidates(parsed["tokens"], candidates)
        return {
            "parsed_intent": parsed,
            "total_candidates": len(candidates),
            "results": ranked
        }
