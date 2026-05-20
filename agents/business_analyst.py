from google.adk.agents import LlmAgent
from common.llm.ollama_client import get_ollama_model
from common.mcp.config import MCPConfig
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset

def create_ba_agent():
    # Map the config methods to their corresponding MCPToolset
    mcp_toolsets = [
        MCPToolset(connection_params=MCPConfig.atlassian_params()),
        MCPToolset(connection_params=MCPConfig.github_params()),
    ]

    business_analyst_agent = LlmAgent(
        name="business_analyst_agent",
        model=get_ollama_model("gemma3:12b"),
        instruction="""
        You are a helpful, AI-powered business analyst assistant.

        Your purpose is to assist users with their project management and development tasks only in Automotive domain.

        You have access to the following tools:
        - Jira tools to query, create, and update tickets.
        - GitHub tools to search code, check out pull requests, and manage repositories.

        When a user asks a question, think step-by-step about which tool(s) you need to use.
        Explain your reasoning clearly, present the information you find in a structured way,
        and offer to help with the next steps.
        """,
        tools=mcp_toolsets, # Connecting to mcp toolsets for Jira and GitHub
    )