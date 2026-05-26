# extractor/ogx_rag.py
import os
import uuid
from pathlib import Path

from ogx_client import OgxClient


def get_client() -> OgxClient:
    return OgxClient(
        base_url=os.environ.get("OGX_CLIENT_BASE_URL", "http://localhost:8321"),
        api_key=os.environ.get("OGX_CLIENT_API_KEY", "fake"),
    )


def ingest_file(file_path: str) -> str:
    """
    Upload a file to OGX and attach it to a vector store.
    Returns the vector_store_id.
    """
    client = get_client()

    # 1. Upload the file via client.files.create()
    uploaded = client.files.create(
        file=Path(file_path),
        purpose="assistants",
    )

    # 2. Create a vector store via client.vector_stores.create()
    vs = client.vector_stores.create(
        name=f"automotive_{uuid.uuid4().hex[:6]}",
    )

    # 3. Attach the uploaded file to the vector store
    #    via client.vector_stores.files.create()
    client.vector_stores.files.create(
        vector_store_id=vs.id,
        file_id=uploaded.id,
    )

    return vs.id


def query_with_agent(vector_store_ids: list[str], question: str) -> str:
    """
    Use the OGX Responses API with built-in file_search RAG to answer
    a question across all ingested vector stores.
    Returns the agent's answer as a string.
    """
    client = get_client()
    model = os.environ.get("OGX_MODEL", "llama-3.3-70b")

    response = client.responses.create(
        model=model,
        instructions=(
            "You are an automotive document expert. "
            "Use the retrieved document context to answer accurately. "
            "If the answer is not found in the documents, say so clearly."
        ),
        input=question,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": vector_store_ids,
            }
        ],
    )

    # Extract the text content from the response output items
    text_parts: list[str] = []
    for item in response.output:
        # Each output item may be a message with content blocks
        if hasattr(item, "content"):
            for block in item.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)

    return "\n".join(text_parts) if text_parts else ""