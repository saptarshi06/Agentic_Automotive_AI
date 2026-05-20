import os
from google.adk.models.lite_llm import LiteLlm

def get_ollama_model(model_name: str) -> LiteLlm:
    """
    Initializes and returns a LiteLlm model instance configured for Ollama.
    It uses the 'openai/' prefix to bypass compatibility issues.
    """
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

    # The 'openai/' prefix tells LiteLlm to use the OpenAI-compatible endpoint
    # To use a specific Ollama model, e.g., 'openai/llama3'
    full_model_path = f"ollama/{model_name}"

    return LiteLlm(
        model=full_model_path,
        api_base=OLLAMA_BASE_URL,
    )




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
