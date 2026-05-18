import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from common.llm.business_analyst import BusinessAnalystAgent
from common.llm.ollama_client import OllamaClient
from common.mcp.mcp_engine import MCPEngine

load_dotenv()
logger = logging.getLogger("auto_business_analyst")

agent: Optional[BusinessAnalystAgent] = None
mcp_client_wrapper: Optional[object] = None


# ... (imports remain the same)

class MCPToolClient:
    """Wrapper around MultiServerMCPClient to provide a call_tool method."""
    def __init__(self, client):
        self._client = client

    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool via the underlying MultiServerMCPClient."""
        # The client's call_tool method expects the server name prefix?
        # Assuming tool_name includes server prefix (e.g., "github_...").
        # We'll just call directly.
        try:
            return await self._client.call_tool(tool_name, arguments)
        except Exception as e:
            logger.error(f"Tool call failed for {tool_name}: {e}")
            return {"status": "error", "message": str(e)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the Business Analyst Agent with MCP tools (if available)."""
    global agent, mcp_client_wrapper

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("OLLAMA_MODEL", "phi3.5:latest")

    agent = BusinessAnalystAgent(base_url=ollama_base_url, model=model)

    # MCP tool loading (only if ENABLE_MCP is true)
    tool_definitions = None
    mcp_client_wrapper = None
    if os.getenv("ENABLE_MCP", "false").lower() == "true":
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from common.mcp.config import MCPServerConfig

            server_configs = MCPServerConfig.get_server_configs()
            # Create client without context manager (new API)
            client = MultiServerMCPClient(server_configs, tool_name_prefix=True)
            # Get tools – this connects automatically
            discovered_tools = await asyncio.wait_for(client.get_tools(), timeout=30)
            if discovered_tools:
                tool_definitions = OllamaClient.convert_langchain_tools_to_openai(discovered_tools)
                mcp_client_wrapper = MCPToolClient(client)
                logger.info(f"Loaded {len(discovered_tools)} MCP tools")
            else:
                logger.warning("No MCP tools discovered")
        except Exception as e:
            logger.error(f"MCP tool loading failed: {e}")
            # Continue without tools

    await agent.initialize(mcp_client=mcp_client_wrapper, tool_definitions=tool_definitions)

    if not await agent.health_check():
        logger.error(f"Agent health check failed – model {model} not available")
        agent = None
    else:
        logger.info("BusinessAnalystAgent ready")

    yield  # server runs

    # Cleanup: MCP client has no explicit close; it's fine to let it be GC'd
    agent = None

app = FastAPI(
    title="Automotive Business Analyst",
    description="Automotive business analysis with GitHub/Jira MCP tools",
    lifespan=lifespan,
)


# ---------- Request/Response Models ----------
class AnalysisRequest(BaseModel):
    user_input: str


class AnalysisResponse(BaseModel):
    answer: str
    shield_alert: Optional[str] = None


class GitHubConnectRequest(BaseModel):
    pat: str


class GitHubConnectResponse(BaseModel):
    username: Optional[str] = None
    message: Optional[str] = None


class JiraConnectRequest(BaseModel):
    url: str
    username: str
    api_token: str


class JiraConnectResponse(BaseModel):
    account_id: Optional[str] = None
    display_name: Optional[str] = None
    message: Optional[str] = None


# ---------- Endpoints ----------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """Run the automotive business analyst on the user's input."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised – check Ollama connection")

    try:
        final_answer = None
        shield_violation = None
        async for event in agent.generate(request.user_input):
            if event.get("type") == "final":
                final_answer = event.get("content")
            elif event.get("type") == "error":
                raise HTTPException(status_code=500, detail=event.get("error"))
            elif event.get("type") == "progress":
                # Can be logged or forwarded to a WebSocket if needed
                logger.debug(f"Progress: {event.get('node')} -> {event.get('state')}")
        if final_answer is None:
            final_answer = "No answer generated."
        return AnalysisResponse(answer=final_answer, shield_alert=shield_violation)
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/status")
async def mcp_status():
    """Return connection status of Jira and GitHub (based on environment)."""
    jira_connected = bool(
        os.getenv("JIRA_URL") and
        os.getenv("JIRA_USERNAME") and
        os.getenv("JIRA_API_TOKEN")
    )
    jira_status = "connected" if jira_connected else "disconnected"

    github_token = os.getenv("GITHUB_TOKEN")
    github_status = "disconnected"
    if github_token:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"token {github_token}"}
                async with session.get("https://api.github.com/user", headers=headers, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        username = data.get("login")
                        github_status = f"connected ({username})"
                    else:
                        github_status = "disconnected"
        except Exception as e:
            logger.warning(f"GitHub status check failed: {e}")
    return {"jira": jira_status, "github": github_status}


@app.post("/github/connect")
async def github_connect(request: GitHubConnectRequest):
    """Validate a GitHub personal access token."""
    if not request.pat:
        raise HTTPException(status_code=400, detail="PAT required")
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"token {request.pat}"}
            async with session.get("https://api.github.com/user", headers=headers, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    username = data.get("login")
                    if username:
                        return GitHubConnectResponse(username=username)
                    else:
                        raise HTTPException(status_code=401, detail="Invalid token")
                else:
                    error_text = await resp.text()
                    raise HTTPException(status_code=resp.status, detail=f"GitHub API error: {error_text}")
    except aiohttp.ClientError as e:
        logger.error(f"GitHub connection error: {e}")
        raise HTTPException(status_code=503, detail=f"GitHub unreachable: {e}")
    except Exception as e:
        logger.exception("Unexpected error in /github/connect")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jira/connect")
async def jira_connect(request: JiraConnectRequest):
    """Validate Jira credentials by calling the Jira API."""
    if not request.url or not request.username or not request.api_token:
        raise HTTPException(status_code=400, detail="URL, username, and API token are required")

    base_url = request.url.rstrip('/')
    myself_url = f"{base_url}/rest/api/2/myself"

    try:
        async with aiohttp.ClientSession() as session:
            auth = aiohttp.BasicAuth(request.username, request.api_token)
            async with session.get(myself_url, auth=auth, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    account_id = data.get("accountId")
                    display_name = data.get("displayName")
                    return JiraConnectResponse(
                        account_id=account_id,
                        display_name=display_name,
                        message="Connected successfully"
                    )
                else:
                    error_text = await resp.text()
                    raise HTTPException(status_code=resp.status, detail=f"Jira API error: {error_text}")
    except aiohttp.ClientError as e:
        logger.error(f"Jira connection error: {e}")
        raise HTTPException(status_code=503, detail=f"Jira unreachable: {e}")
    except Exception as e:
        logger.exception("Unexpected error in /jira/connect")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")
    try:
        healthy = await agent.health_check()
        if not healthy:
            raise HTTPException(status_code=503, detail="Model not available")
        return {"status": "ok", "model": agent.model}
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=503, detail=str(e))