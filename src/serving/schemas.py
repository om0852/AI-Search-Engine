from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class SearchParseRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "oayement issue occur in last 2 days"})

class MongoBuildRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "oayement issue occur in last 2 days"})
    filters: Optional[Dict[str, Any]] = Field(default=None, json_schema_extra={"example": {"platform": "twitter", "sentiment": "negative"}})

class MongoExecuteRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "oayement issue occur in last 2 days"})
    filters: Optional[Dict[str, Any]] = Field(default=None, json_schema_extra={"example": {"platform": "twitter", "sentiment": "negative"}})
    mongo_uri: Optional[str] = Field(default=None, json_schema_extra={"example": "mongodb://localhost:27017"})
    database: Optional[str] = Field(default="social_listening", json_schema_extra={"example": "social_listening"})
    collection: Optional[str] = Field(default="posts", json_schema_extra={"example": "posts"})
    limit: Optional[int] = Field(default=20, json_schema_extra={"example": 20})
