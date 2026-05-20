import os
from mcp import StdioServerParameters
from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
#from dotenv import load_dotenv
#load_dotenv()  # Load environment variables from .env file, if present

class MCPConfig:
    """
    Provides unmodifiable structural params for our underlying MCP integrations. This includes things like server command, args, and env vars. The idea is to have a single source of truth for these params, and to avoid hardcoding them in multiple places throughout the codebase.
    """

    @staticmethod
    def atlassian_params() -> StdioConnectionParams:

        jira_params = StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=["-y", "mcp-remote", "https://mcp.atlassian.com/v1/mcp"]
            ),
            timeout=30
        )

        # return {
        #     "jira": jira_params
        # }
        return jira_params

    @staticmethod
    def github_params() -> StreamableHTTPConnectionParams:
        GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
        github_params = StreamableHTTPConnectionParams(
                url="https://api.githubcopilot.com/mcp/",
                headers={
                    "Authorization": f"Bearer {GITHUB_TOKEN}",
                    "X-MCP-Toolsets": "all",
                    "X-MCP-Readonly": "true"
                },
        )
        return github_params