import asyncio
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from common.llm.base_bedrock import GeminiClient, GeminiClientError
from common.llm.business_analyst import (
    build_business_analyst_graph,
    BusinessAnalysisState,
)
from common.shields.agent_shields import BUSINESS_ANALYSIS_SHIELD

load_dotenv()
logger = logging.getLogger("server")


client: Optional[GeminiClient] = None
graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise Bedrock client, register shields, compile the agent graph."""
    global client, graph

    provider_data = {}
    # if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
    #     provider_data["aws_bearer_token_bedrock"] = os.environ["AWS_BEARER_TOKEN_BEDROCK"]

    try:
        client = GeminiClient(
            base_url=os.getenv("LLAMA_STACK_URL", "http://localhost:8321"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
        )
        # Register shields
        await client.register_shields([BUSINESS_ANALYSIS_SHIELD])
        # Compile the business analyst graph
        graph = build_business_analyst_graph(client)
        logger.info("Application started successfully.")
    except Exception as e:
        logger.error(f"Failed to initialise Bedrock client: {e}")
        client = None
        graph = None

    yield  # server runs here

    client = None
    graph = None


app = FastAPI(title="Automotive Business Analyst", lifespan=lifespan)


class AnalysisRequest(BaseModel):
    user_input: str
    github_token: Optional[str] = None   # optional, used when MCP tools are needed


class GitHubConnectRequest(BaseModel):
    pat: str


@app.post("/analyze")
async def analyze_endpoint(request: AnalysisRequest):
    """
    Run the business analyst agent on the user’s input.
    (Optionally receives a GitHub token for future MCP integration.)
    """
    if graph is None:
        raise HTTPException(status_code=503, detail="Service not ready – Bedrock client unavailable")

    initial_state: BusinessAnalysisState = {
        "messages": [],
        "user_input": request.user_input,
        "analysis_result": "",
        "final_answer": "",
        "shield_violation": None,
        "error": None,
        # The token can be stored in the state and later forwarded to MCP tools
        # when we extend the agent to use GitHub/Jira.
        "github_token": request.github_token,
    }

    try:
        final_state = await graph.ainvoke(initial_state)
    except GeminiClientError:
        raise HTTPException(status_code=401, detail="AWS Bedrock token expired. Please refresh.")
    except Exception as e:
        logger.exception("Unexpected agent error")
        raise HTTPException(status_code=500, detail=f"Internal agent error: {e}")

    return {
        "answer": final_state.get("final_answer", "No answer generated."),
        "shield_alert": final_state.get("shield_violation"),
    }


@app.get("/mcp/status")
async def mcp_status(pat: Optional[str] = Query(None)):
    """
    Return Jira / GitHub connection status.
    - Jira: checks environment variables.
    - GitHub: if `pat` is provided, validates that token; otherwise uses the
      default GITHUB_TOKEN from the environment.
    """
    # Jira status
    jira_connected = bool(
        os.getenv("JIRA_URL") and
        os.getenv("JIRA_USERNAME") and
        os.getenv("JIRA_API_TOKEN")
    )

    # GitHub status
    token_to_check = pat or os.getenv("GITHUB_TOKEN")
    github_status = "disconnected"

    if token_to_check:
        try:
            resp = requests.get(
                "https://api.github.com/user",
                headers={"Authorization": f"token {token_to_check}"},
                timeout=5,
            )
            if resp.status_code == 200:
                user_data = resp.json()
                username = user_data.get("login")
                if username:
                    github_status = f"connected ({username})"
        except Exception:
            logger.warning("GitHub status check failed")

    return {
        "jira": "connected" if jira_connected else "disconnected",
        "github": github_status,
    }


@app.post("/github/connect")
async def github_connect(request: GitHubConnectRequest):
    """
    Validate a GitHub Personal Access Token and return the associated username.
    (No server‑side session – the caller is responsible for storing the token.)
    """
    try:
        resp = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"token {request.pat}"},
            timeout=10,
        )
        if resp.status_code == 200:
            user_data = resp.json()
            username = user_data.get("login")
            if username:
                return {"username": username, "message": "Token is valid"}
        raise HTTPException(status_code=401, detail="Invalid GitHub token")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"GitHub API unreachable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    if client is None:
        raise HTTPException(status_code=503, detail="Llama Stack client not connected")
    return {"status": "ok"}