# extractor/ogx_rag.py
import os
from openai import OpenAI

def get_ogx():
    return OpenAI(
        base_url=os.environ.get("OGX_SERVER", "http://localhost:8321/v1"),
        api_key="fake"  # OGX self-hosted doesn't require a real key
    )

def ingest_to_ogx(file_path: str) -> str:
    """Upload extracted text to OGX vector store. Returns vector_store_id."""
    ogx = get_ogx()
    with open(file_path, "rb") as f:
        uploaded = ogx.files.create(file=f, purpose="assistants")
    vs = ogx.vector_stores.create(
        name=f"automotive_{os.path.basename(file_path)}",
        file_ids=[uploaded.id]
    )
    return vs.id

def query_documents(vector_store_ids: list, question: str, model: str = "ollama/llama3.2:3b") -> str:
    """Cross-document RAG query via OGX Responses API."""
    ogx = get_ogx()
    response = ogx.responses.create(
        model=model,
        input=question,
        tools=[{
            "type": "file_search",
            "vector_store_ids": vector_store_ids
        }]
    )
    return response.output_text