import json

import docs
from docs import index
from utils import *
from models import *
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


async def main():
    query = input("User query: ")
    resp_json = await generate(query)
    content = resp_json["choices"][0]["message"]["content"]
    queries_json = json.loads(content)
    response = queries_json.values()
    scored_queries = [
        QueryVariant(text=q, score=await get_similarity_score(query, q, embedding_model))
        for q in response
    ]
    distances, indices = index.search(embedding_model.encode([query]), k=10)
    print(f"Default query results ({len(indices[0])}): {[docs.texts[x] for x in indices[0]]}")
    optimized_results = set()
    for q in scored_queries:
        distances, indices = index.search(embedding_model.encode([q.text]), k=10)
        for x in indices[0]:
            optimized_results.add(docs.texts[x])
    print(f"Optimized results({len(optimized_results)}): {list(optimized_results)}")


if __name__ == "__main__":
    asyncio.run(main())
