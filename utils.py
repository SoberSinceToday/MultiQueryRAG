import asyncio
import os
import httpx
from typing import Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


async def generate(user_query: str) -> Any:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mistralai/mistral-7b-instruct",
                "extra_body": {
                    "models": ["x-ai/grok-4-fast"]
                },
                "messages": [
                    {
                        "role": "developer",
                        "content": "You are an assistant whose task is, based on the user's query, to generate 3 alternative formulations of the query (multi-query) to improve recall when searching over a vector index."
                    },
                    {
                        "role": "system",
                        "content": "Requirements:\n1. Generate exactly 3 query variants.\n2. Preserve the meaning of the original query, using synonyms and expanded formulations.\n3. Response format — JSON"
                    },
                    {
                        "role": "user",
                        "content": user_query,
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "rephrased_queries",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "query1": {
                                    "type": "string",
                                    "description": "First alternative formulation of the user's original query"
                                },
                                "query2": {
                                    "type": "string",
                                    "description": "Second alternative formulation of the user's original query"
                                },
                                "query3": {
                                    "type": "string",
                                    "description": "Third alternative formulation of the user's original query"
                                }
                            },
                            "required": ["query1", "query2", "query3"],
                            "additionalProperties": False
                        }
                    }
                },
            }
        )
        return response.json()


def _get_similarity_score(user_query: str, generated_query: str, embedding_model: SentenceTransformer) -> float:
    embeds = embedding_model.encode([user_query, generated_query])
    return round(float(embedding_model.similarity(embeds[0], embeds[1])[0][0]), 2)


async def get_similarity_score(user_query: str, generated_query: str, embedding_model) -> float:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        _get_similarity_score,
        user_query,
        generated_query,
        embedding_model,
    )
