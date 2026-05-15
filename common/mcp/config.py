import os
from typing import Mapping
from dotenv import load_dotenv
from mcp import StdioServerParameters

load_dotenv()


class MCPServerConfig:
    """
    Centralized MCP server configuration registry.
    """

    @staticmethod
    def get_server_configs() -> Mapping[str, StdioServerParameters]:

        jira_params = StdioServerParameters(
            command="npx",
            args=["-y", "mcp-atlassian"],
            env={
                **os.environ,
                "JIRA_URL": os.getenv("JIRA_URL"),
                "JIRA_USERNAME": os.getenv("JIRA_USERNAME"),
                "JIRA_API_TOKEN": os.getenv("JIRA_API_TOKEN"),
            }
        )

        github_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={
                **os.environ,
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")
            }
        )

        return {
            "jira": jira_params,
            "github": github_params
        }