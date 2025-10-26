from pydantic import BaseModel
from typing import List, Tuple


class UserQuery(BaseModel):
    user_query: str


class QueryVariant(BaseModel):
    text: str
    score: float


class OptimizedQueries(BaseModel):
    result: List[QueryVariant]