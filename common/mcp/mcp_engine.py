import asyncio
import logging

from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from common.mcp.config import MCPServerConfig

logger = logging.getLogger("mcp_engine")


class MCPEngine:
    """
    Handles MCP server lifecycle management,
    tool discovery, and cleanup.
    """

    def __init__(self):
        self.server_configs = MCPServerConfig.get_server_configs()

    @asynccontextmanager
    async def connect_servers(self) -> AsyncGenerator[List[BaseTool], None]:
        logger.info("Initializing MultiServerMCPClient...")
        try:
            async with MultiServerMCPClient(
                self.server_configs,
                tool_name_prefix=True
            ) as client:
                logger.info("Connecting to MCP servers...")
                discovered_tools = await asyncio.wait_for(
                    client.get_tools(),
                    timeout=30
                )
                if not discovered_tools:
                    raise RuntimeError("No tools discovered from MCP servers.")
                logger.info(f"Successfully loaded {len(discovered_tools)} tools.")
                yield discovered_tools
        except asyncio.TimeoutError:
            logger.exception("Timeout while connecting to MCP servers.")
            raise
        except Exception as error:
            logger.exception(f"MCP Engine initialization failed: {error}")
            raise
        finally:
            logger.info("MCP Engine cleanup complete.")
