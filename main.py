# main.py
import os
import threading
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# Import our custom modules
from common.mcp.config import MCPConfig
from agents.business_analyst import create_ba_agent
from workflow.workflow_graph import create_workflow


# Pydantic models for request/response
class AnalyzeRequest(BaseModel):
    user_input: str

class AnalyzeResponse(BaseModel):
    answer: str
    shield_alert: str | None = None   # Kept for compatibility

class GitHubConnectRequest(BaseModel):
    pat: str

class StatusResponse(BaseModel):
    jira: str
    github: str


# Global agent and workflow state
_current_agent = None
_current_workflow = None
_agent_lock = threading.Lock()

def rebuild_agent_and_workflow(github_token: str | None = None):
    """Rebuild the ADK agent and LangGraph workflow using the given GitHub token."""
    global _current_agent, _current_workflow
    
    # Temporarily set/override GITHUB_TOKEN in environment
    if github_token is not None:
        os.environ["GITHUB_TOKEN"] = github_token
    elif "GITHUB_TOKEN" not in os.environ:
        os.environ["GITHUB_TOKEN"] = ""  # ensure it exists
    
    # Create the ADK agent (this uses MCPConfig, which reads GITHUB_TOKEN from env)
    agent = create_ba_agent()
    
    # Create the LangGraph workflow that uses this agent
    workflow = create_workflow(agent)
    
    with _agent_lock:
        _current_agent = agent
        _current_workflow = workflow

def get_current_workflow():
    """Thread-safe getter for the current workflow."""
    with _agent_lock:
        if _current_workflow is None:
            # Initial build using environment variable from .env or default
            rebuild_agent_and_workflow()
        return _current_workflow


# FastAPI app
app = FastAPI(title="Automotive Business Analyst API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: runs before the server starts
    get_current_workflow()
    yield

# Endpoints
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Process user input through the LangGraph agentic workflow."""
    workflow = get_current_workflow()
    
    # Prepare initial state for the graph
    initial_state = {
        "messages": [],
        "input": request.user_input,
        "output": ""
    }
    
    try:
        result = workflow.invoke(initial_state)
        answer = result.get("output", "No response generated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
    
    # shield_alert is omitted – always None
    return AnalyzeResponse(answer=answer, shield_alert=None)

@app.get("/mcp/status", response_model=StatusResponse)
async def mcp_status():
    """Return connection status of Jira and GitHub MCP servers."""
    # Jira: static configuration – assume it's always connectable
    jira_status = "connected"
    
    # GitHub: check if a token is present and if the MCP toolset can be initialised
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if github_token:
        # Optional: perform a lightweight check (e.g., list tools)
        # For now, assume token presence means connected
        # You could fetch the username from GitHub API to show it
        try:
            # Simulate getting username (you would call GitHub MCP or API)
            username = "user"  # placeholder
            github_status = f"connected ({username})"
        except:
            github_status = "disconnected"
    else:
        github_status = "disconnected"
    
    return StatusResponse(jira=jira_status, github=github_status)

@app.post("/github/connect")
async def connect_github(request: GitHubConnectRequest):
    """Store the provided GitHub PAT and rebuild the agent with it."""
    pat = request.pat
    if not pat:
        raise HTTPException(status_code=400, detail="PAT cannot be empty")
    
    # Optional: verify the token works by calling GitHub API
    # (You can add a quick validation here)
    
    # Rebuild the agent with the new token
    try:
        rebuild_agent_and_workflow(github_token=pat)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild agent: {str(e)}")
    
    # Return a simple success message with a placeholder username
    # In a real scenario, you would fetch the actual GitHub username
    return {"username": "authenticated_user"}


# Run the server (if executed directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)