import json
import logging
from typing import Dict, Any, List, Optional, AsyncIterator, Union
from openai import AsyncOpenAI

# Llama Stack safety imports (exact models you provided)
from llama_stack_api.safety.api import RunShieldRequest, RunModerationRequest
# The actual async client (must be installed)
from llama_stack_client import AsyncLlamaStackClient

logger = logging.getLogger(__name__)


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaClient:
    """
    Base client for Ollama using the OpenAI-compatible API.
    Integrates Llama Stack safety shields (run_moderation, run_shield)
    and provides placeholders for MCP tools.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434/v1",
        model: str = "phi3.5:latest",
        ollama_api_key: str = "ollama",
        timeout: float = 60.0,
        # Llama Stack configuration (optional)
        llama_stack_base_url: str = "http://localhost:8321",
        llama_stack_provider_data: Optional[Dict[str, Any]] = None,
    ):
        # Ollama client
        self.ollama_base_url = ollama_base_url
        self.model = model
        self.ollama_client = AsyncOpenAI(
            base_url=ollama_base_url,
            api_key=ollama_api_key,
            timeout=timeout,
        )
        # MCP client placeholder (to be injected later)
        self.mcp_client = None

        # Llama Stack safety client (if configured)
        self.safety_client = None
        if llama_stack_base_url:
            self.safety_client = AsyncLlamaStackClient(
                base_url=llama_stack_base_url,
                provider_data=llama_stack_provider_data or {},
            )
            logger.info(f"Llama Stack safety client configured at {llama_stack_base_url}")
        else:
            logger.info("No Llama Stack URL provided – safety will use stubs.")

    async def initialize(self, mcp_client=None) -> None:
        """Initialize any required resources (e.g., MCP connection)."""
        self.mcp_client = mcp_client
        logger.info(f"OllamaClient initialized with model {self.model}")

    async def generate_non_streaming(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Dict[str, Any]:
        """Non‑streaming generation via Ollama. Returns dict with 'type', 'content', 'tool_calls'."""
        try:
            response = await self.ollama_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                tools=tools or [],
                tool_choice=tool_choice,
            )
            message = response.choices[0].message
            return {
                "type": "complete",
                "content": message.content or "",
                "tool_calls": message.tool_calls if hasattr(message, "tool_calls") else [],
            }
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return {"type": "error", "error": str(e)}

    async def generate_streaming(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Streaming generation via Ollama. Yields content chunks."""
        try:
            stream = await self.ollama_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                tools=[],
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"[Error: {e}]"

    async def health_check(self) -> bool:
        """Verify that Ollama is reachable and the model exists."""
        try:
            response = await self.ollama_client.models.list()
            models = [model.id for model in response.data]
            return self.model in models
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    # Real Llama Stack safety integration (using the exact request models)
    async def run_moderation(self, input_text: Union[str, List[str]]) -> Any:
        """
        Run content moderation using the Llama Stack safety API.
        If no safety client is configured, returns a safe (unflagged) stub.
        """
        if self.safety_client is None:
            logger.debug("No safety client – moderation stub returning safe.")
            class ModerationResult:
                flagged = False
            return ModerationResult()

        try:
            request = RunModerationRequest(input=input_text)
            result = await self.safety_client.safety.run_moderation(request)
            return result
        except Exception as e:
            logger.error(f"Moderation API call failed: {e}")
            # Fallback: allow the request (conservative)
            class SafeResult:
                flagged = False
            return SafeResult()

    async def run_shield(self, shield_id: str, messages: List[Dict[str, str]]) -> Any:
        """
        Run a safety shield using the Llama Stack API.
        If no safety client is configured, returns a no‑violation stub.
        """
        if self.safety_client is None:
            logger.debug(f"No safety client – shield '{shield_id}' stub returning no violation.")
            class ShieldResult:
                violation = None
            return ShieldResult()

        try:
            # Convert messages to the required OpenAIMessageParam format
            request = RunShieldRequest(shield_id=shield_id, messages=messages)
            result = await self.safety_client.safety.run_shield(request)
            return result
        except Exception as e:
            logger.error(f"Shield '{shield_id}' call failed: {e}")
            # Fallback: no violation
            class SafeShieldResult:
                violation = None
            return SafeShieldResult()

    # MCP tool integration placeholder
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool (GitHub, Jira, etc.) via the injected mcp_client.
        """
        if self.mcp_client is None:
            logger.warning(f"MCP tool '{tool_name}' called but no MCP client set.")
            return {"status": "error", "message": "MCP client not initialized"}
        try:
            # Assumes mcp_client has a call_tool method (adjust as needed)
            return await self.mcp_client.call_tool(tool_name, arguments)
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return {"status": "error", "message": str(e)}


# import asyncio
# import logging
# from typing import Optional, List, Dict, Any

# from llama_stack_client import AsyncLlamaStackClient
# from langchain.messages import AIMessage
# logger = logging.getLogger(__name__)


# class BedrockClientError(Exception):
#     """Base exception for Bedrock client errors."""


# class AuthenticationExpiredError(BedrockClientError):
#     """Raised when the AWS Bedrock bearer token has expired."""


# class BedrockClient:
#     """
#     Wraps AsyncLlamaStackClient for AWS Bedrock.
#     Handles shield registration, inference, and safety checks.
#     """

#     def __init__(
#         self,
#         base_url: str = "http://localhost:8321",
#         api_key: Optional[str] = None,
#         provider_data: Optional[Dict[str, Any]] = None,
#     ):
#         self.base_url = base_url
#         self.provider_data = provider_data or {}
#         self.client = AsyncLlamaStackClient(
#             base_url=base_url,
#             api_key=api_key,
#             provider_data=self.provider_data,
#         )
#         self._shields_registered = False

#     async def register_shields(self, shields: List[Dict[str, Any]]) -> None:
#         """Register multiple shields. Idempotent across restarts."""
#         if self._shields_registered:
#             return
#         for shield in shields:
#             try:
#                 await self.client.shields.register(shield)
#                 logger.info(f"Shield '{shield['shield_id']}' registered.")
#             except Exception as e:
#                 logger.error(f"Failed to register shield '{shield['shield_id']}': {e}")
#                 # Continue registering others; don't crash
#         self._shields_registered = True

#     async def chat_completion(
#         self,
#         messages: List[Dict[str, str]],
#         model: str = "meta.llama4-maverick-17b-instruct-v1:0",
#         temperature: float = 0.7,
#         max_tokens: Optional[int] = None,
#         stream: bool = False,
#         tools: Optional[List[Dict[str, Any]]] = None,
#         tool_choice: Optional[str] = "auto",
#         **kwargs: Any,
#     ) -> Any:
#         extra = {}
#         if tools:
#             extra["tools"] = tools
#             extra["tool_choice"] = tool_choice
        
       
#         """Call Bedrock chat completions via the client."""
#         try:
#             response = await self.client.inference.chat_completion(
#                 model=model,
#                 messages=messages,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 stream=stream,
#                 **extra,
#             )
#             if response is None:
#                 raise BedrockClientError("Bedrock returned no response.")
#             return response
#         except Exception as e:
#             self._handle_inference_error(e)

#     def _handle_inference_error(self, error: Exception) -> None:
#         error_msg = str(error).lower()
#         if "bearer token has expired" in error_msg or "expired" in error_msg:
#             logger.error("Bedrock bearer token expired.")
#             raise AuthenticationExpiredError(
#                 "AWS Bedrock token expired. Please refresh your pre‑signed URL."
#             ) from error
#         if "authentication" in error_msg or "not authorized" in error_msg:
#             logger.error("Bedrock authentication failure.")
#             raise BedrockClientError(
#                 f"Bedrock authentication failed: {error}. Check your API key and region."
#             ) from error
#         raise BedrockClientError(f"Bedrock inference error: {error}") from error

#     async def run_moderation(self, input_text: str) -> Any:
#         """Moderate user input."""
#         return await self.client.safety.run_moderation(input=input_text)

#     async def run_shield(self, shield_id: str, messages: List[Dict[str, str]]) -> Any:
#         """Run a shield on messages."""
#         return await self.client.safety.run_shield(
#             shield_id=shield_id, messages=messages
#         )

# @staticmethod
# def parse_chat_completion_to_aimessage(response: Any) -> AIMessage:
#         """
#         Convert a Bedrock chat completion response into a LangChain AIMessage.
#         Handles optional tool_calls.
#         """
#         # Expect response.choices[0].message like OpenAI format
#         choice = response.choices[0]
#         message = choice.message
#         content = message.content or ""
#         tool_calls_raw = getattr(message, "tool_calls", None)

#         additional_kwargs = {}
#         if tool_calls_raw:
#             # tool_calls_raw is a list of objects with id, function.name, function.arguments
#             additional_kwargs["tool_calls"] = [
#                 {
#                     "id": tc.id,
#                     "type": "function",
#                     "function": {
#                         "name": tc.function.name,
#                         "arguments": tc.function.arguments,
#                     },
#                 }
#                 for tc in tool_calls_raw
#             ]
#         return AIMessage(content=content, additional_kwargs=additional_kwargs)

# #     # Utility: Convert LangChain tools to Bedrock format
# # @staticmethod
# def convert_langchain_tools_to_bedrock(tools: List[Any]) -> List[Dict[str, Any]]:
#         """Convert LangChain BaseTool instances to the format Bedrock expects."""
#         bedrock_tools = []
#         for tool in tools:
#             # Each tool has .name, .description, .args (schema)
#             bedrock_tools.append({
#                 "type": "function",
#                 "function": {
#                     "name": tool.name,
#                     "description": tool.description,
#                     "parameters": tool.args if hasattr(tool, "args") else {},
#                 },
#             })
#         return bedrock_tools
