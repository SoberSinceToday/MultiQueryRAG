import json
import asyncio
from fastapi import FastAPI
from models import *
from utils import *
from sentence_transformers import SentenceTransformer

app = FastAPI(title="QueryOptimizer")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


@app.post("/optimize_query", response_model=OptimizedQueries)
async def optimize_query(user_query: UserQuery) -> OptimizedQueries:
    resp_json = await generate(user_query.user_query)
    content = resp_json["choices"][0]["message"]["content"]
    queries_json = json.loads(content)
    response = queries_json.values()
    scored_queries = [
        QueryVariant(text=q, score=await get_similarity_score(user_query.user_query, q, embedding_model))
        for q in response
    ]
    return OptimizedQueries(result=scored_queries)
