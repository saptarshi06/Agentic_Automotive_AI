import asyncio
import logging
from typing import Optional, List

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)

from mcp_engine import MCPEngine
from common.llm.ollama_client import OllamaClient as BedrockClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_agent")


def _convert_messages_to_bedrock(messages: List[BaseMessage]) -> List[dict]:
    """Convert LangChain messages to the dict list expected by BedrockClient."""
    bedrock_msgs = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            bedrock_msgs.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            bedrock_msgs.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            msg_dict = {"role": "assistant", "content": msg.content or ""}
            if msg.additional_kwargs.get("tool_calls"):
                msg_dict["tool_calls"] = msg.additional_kwargs["tool_calls"]
            bedrock_msgs.append(msg_dict)
        elif isinstance(msg, ToolMessage):
            # Tool response: Bedrock expects a "tool" role with tool_call_id
            bedrock_msgs.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            })
    return bedrock_msgs


class LangGraphAgent:
    """LangGraph agent using BedrockClient directly (no extra wrapper)."""

    def __init__(self, bedrock_client: BedrockClient):
        self.client = bedrock_client
        self.mcp_engine = MCPEngine()
        self.graph = None
        self._tools_list: Optional[List] = None  # will store LangChain tools

    async def initialize_graph(self):
        """Connect MCP servers, compile the graph (tools will be passed at runtime)."""
        async with self.mcp_engine.connect_servers() as tools:
            self._tools_list = tools
            # Convert tools once to Bedrock format
            bedrock_tools = BedrockClient.convert_langchain_tools_to_bedrock(tools)

            workflow = StateGraph(MessagesState)

            # The agent node will use the client directly
            workflow.add_node("agent", self._create_agent_node(bedrock_tools))
            workflow.add_node("tools", ToolNode(tools))

            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges(
                "agent",
                self._router,
                {"tools": "tools", END: END},
            )
            workflow.add_edge("tools", "agent")

            self.graph = workflow.compile()
            logger.info("LangGraph compilation successful.")

    def _create_agent_node(self, bedrock_tools: List[dict]):
        """Return an async node function that calls BedrockClient directly."""
        async def agent_node(state: MessagesState):
            messages_dict = _convert_messages_to_bedrock(state["messages"])
            try:
                response = await self.client.chat_completion(
                    messages=messages_dict,
                    tools=bedrock_tools if bedrock_tools else None,
                    tool_choice="auto",
                    temperature=0.7,
                )
                ai_msg = self.client.parse_chat_completion_to_aimessage(response)
                return {"messages": [ai_msg]}
            except Exception as error:
                logger.exception(f"Agent node execution failed: {error}")
                raise
        return agent_node

    @staticmethod
    def _router(state: MessagesState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    async def execute(self):
        """Test execution – remove in production."""
        if not self.graph:
            raise RuntimeError("Graph not initialized. Call initialize_graph() first.")

        cross_platform_prompt = (
            "Look at the open GitHub PRs. "
            "For failed build checks, identify related Jira issues "
            "and update their status to Blocker."
        )
        inputs = {"messages": [HumanMessage(content=cross_platform_prompt)]}
        logger.info("Starting LangGraph execution...")

        async for chunk in self.graph.astream(inputs):
            print(chunk)

        logger.info("LangGraph execution completed.")


async def main():
    try:
        logger.info("Application startup initiated.")

        import os
        from dotenv import load_dotenv
        load_dotenv()

        bedrock_client = BedrockClient(
            base_url=os.getenv("LLAMA_STACK_URL"),
            provider_data={
                "aws_bearer_token_bedrock": os.getenv("AWS_BEARER_TOKEN_BEDROCK")
            },
        )

        # Optional: register shields if needed
        # from common.llm.shields import BUSINESS_ANALYSIS_SHIELD
        # await bedrock_client.register_shields([BUSINESS_ANALYSIS_SHIELD])

        agent = LangGraphAgent(bedrock_client)
        await agent.initialize_graph()
        await agent.execute()

        logger.info("Application execution completed.")
    except KeyboardInterrupt:
        logger.warning("Application interrupted by user.")
    except Exception as error:
        logger.exception(f"Fatal application failure: {error}")


if __name__ == "__main__":
    asyncio.run(main())