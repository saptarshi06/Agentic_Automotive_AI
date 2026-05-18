import json
import logging
from typing import Dict, Any, AsyncIterator, Optional, Literal, List

from langgraph.graph import StateGraph, END, START
from common.llm.ollama_client import OllamaClient, OllamaClientError

logger = logging.getLogger(__name__)


# ============================================================================
# State definition for LangGraph
# ============================================================================
class BusinessAnalystState(Dict[str, Any]):
    """Typed state for the business analyst workflow."""
    messages: list
    user_input: str
    is_automotive: bool
    analysis_result: str
    final_answer: str
    shield_violation: Optional[str]
    error: Optional[str]

# For static type checking (optional)
BusinessAnalystState.update({
    "messages": [],
    "user_input": "",
    "is_automotive": False,
    "analysis_result": "",
    "final_answer": "",
    "shield_violation": None,
    "error": None,
})


# ============================================================================
# Prompt templates
# ============================================================================
DOMAIN_CHECK_SYSTEM = (
    "You are an expert in the automotive industry (manufacturing, supply chain, EVs, autonomous driving, after‑sales, etc.). "
    "Determine if the following user request is related to the automotive domain. "
    "Answer only with a single word: 'YES' if it is automotive‑related, otherwise 'NO'.\n\n"
)

DOMAIN_CHECK_USER = "User request: {user_input}\n\nAnswer (YES/NO):"

ANALYSIS_SYSTEM = (
    "You are a senior business analyst specialised in the automotive industry. "
    "Analyse the following request and provide a structured response. "
    "Include: 1) Key Findings, 2) Stakeholders Impacted, 3) Risks & Assumptions, "
    "4) Recommendations tailored to automotive manufacturing, supply chain, or retail. "
    "Format your answer as a JSON object with these four fields. Do not add extra commentary."
)

OUTSIDE_SKILLSET_MESSAGE = (
    "I'm sorry, but I can only assist with automotive‑related topics. "
    "My expertise covers vehicle manufacturing, supply chain optimisation, EV adoption, "
    "autonomous driving regulations, after‑sales services, and connected car technologies. "
    "Please rephrase your request to focus on the automotive industry."
)

FORMAT_TO_TEXT_SYSTEM = (
    "Convert the following JSON analysis into a clear, professional plain-text report for non-technical stakeholders. "
    "Use bullet points where appropriate and avoid any JSON syntax in the output.\n\n"
    "JSON:\n{json_data}\n\nPlain‑text report:"
)

# BusinessAnalystAgent – inherits from OllamaClient
class BusinessAnalystAgent(OllamaClient):
    """
    Automotive business analyst agent built on LangGraph.
    Only responds to automotive domain requests; politely rejects others.
    """

    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "phi3.5:latest"):
        super().__init__(ollama_base_url=base_url, model=model)
        self._graph = None
        self.tool_definitions = None

    async def initialize(self, mcp_client=None, tool_definitions=None) -> None:
        """Initialize base client and build the LangGraph workflow."""
        await super().initialize(mcp_client)
        self._graph = self._build_graph()
        self.tool_definitions = tool_definitions or []
        self._graph = self._build_graph()
        logger.info("BusinessAnalystAgent ready with {} tool definitions".format(len(self.tool_definitions)))

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph state machine."""
        workflow = StateGraph(BusinessAnalystState)

        # Add nodes (all async methods bound to self)
        workflow.add_node("moderation", self._moderation_node)
        workflow.add_node("domain_check", self._domain_check_node)
        workflow.add_node("analysis", self._analysis_node)
        workflow.add_node("formatting", self._formatting_node)
        workflow.add_node("shield", self._shield_node)
        workflow.add_node("final_response", self._final_response_node)

        # Conditional edges
        workflow.add_edge(START, "moderation")
        workflow.add_conditional_edges("moderation", self._route_after_moderation)
        workflow.add_conditional_edges("domain_check", self._route_after_domain_check)
        workflow.add_conditional_edges("analysis", self._route_after_analysis)
        workflow.add_conditional_edges("formatting", self._route_after_formatting)
        workflow.add_edge("shield", "final_response")
        workflow.add_edge("final_response", END)

        return workflow.compile()

    # LangGraph node implementations
    async def _moderation_node(self, state: BusinessAnalystState) -> dict:
        """Run input moderation (stub - can be replaced with actual shield)."""
        try:
            mod_result = await self.run_moderation(state["user_input"])
            if mod_result and getattr(mod_result, "flagged", False):
                return {"error": "Your input was flagged by the content safety system. Please rephrase."}
        except Exception as e:
            logger.warning(f"Moderation check failed (proceeding anyway): {e}")
        return {}

    async def _domain_check_node(self, state: BusinessAnalystState) -> dict:
        """Determine if the request is automotive‑related."""
        messages = [
            {"role": "system", "content": DOMAIN_CHECK_SYSTEM},
            {"role": "user", "content": DOMAIN_CHECK_USER.format(user_input=state["user_input"])}
        ]
        try:
            result = await self.generate_non_streaming(messages, temperature=0.0, max_tokens=10)
            if result.get("type") == "error":
                raise OllamaClientError(result["error"])
            answer = result.get("content", "").strip().upper()
            is_automotive = answer.startswith("YES")
            if not is_automotive:
                logger.info(f"Non-automotive request rejected: {state['user_input'][:100]}")
            return {"is_automotive": is_automotive}
        except Exception as e:
            logger.error(f"Domain check failed: {e}")
            return {"is_automotive": False, "error": f"Domain classification error: {e}"}

    async def _analysis_node(self, state: BusinessAnalystState) -> dict:
        """Generate structured analysis, optionally using tools."""
        if not state.get("is_automotive", False):
            return {}
        messages = [
            {"role": "system", "content": ANALYSIS_SYSTEM},
            {"role": "user", "content": state["user_input"]}
        ]
        try:
            # Use tool‑enabled execution if tool definitions exist
            if self.tool_definitions:
                # Need to pass tools to generate_non_streaming via a separate method
                # We'll call _execute_with_tools which handles the loop
                result = await self._execute_with_tools(messages, self.tool_definitions, temperature=0.3, max_tokens=2048)
            else:
                result = await self.generate_non_streaming(messages, temperature=0.3, max_tokens=2048)
            
            if result.get("type") == "error":
                raise OllamaClientError(result["error"])
            analysis_text = result.get("content", "")
            # Optional JSON validation
            try:
                json.loads(analysis_text)
            except json.JSONDecodeError:
                logger.warning("Analysis output is not valid JSON; will attempt to format anyway.")
            return {"analysis_result": analysis_text}
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": f"Analysis service error: {e}"}
    
    async def _execute_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Execute a chat completion that may involve tool calls.
        Recursively resolves tool calls using self.call_tool().
        Returns final result dict with 'type' and 'content'.
        """
        # Initial call (may return tool_calls)
        response = await self.generate_non_streaming(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice="auto",
        )
        if response.get("type") == "error":
            return response

        # If no tool calls, return the content directly
        tool_calls = response.get("tool_calls", [])
        if not tool_calls:
            return {"type": "complete", "content": response.get("content", "")}

        # Append assistant message with tool calls
        assistant_msg = {
            "role": "assistant",
            "content": response.get("content"),
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_msg)

        # Execute each tool call
        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            try:
                tool_result = await self.call_tool(tool_name, args)
            except Exception as e:
                tool_result = {"status": "error", "message": str(e)}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": json.dumps(tool_result),
            })

        # Final call with tool results (no tools needed now)
        final_response = await self.generate_non_streaming(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=[],   # empty to avoid further tool calls (or allow if you want multi‑step)
        )
        return final_response

    async def _formatting_node(self, state: BusinessAnalystState) -> dict:
        """Convert JSON analysis to plain text, or return polite rejection."""
        if state.get("error"):
            return {}
        if not state.get("is_automotive", False):
            return {"final_answer": OUTSIDE_SKILLSET_MESSAGE}

        raw = state.get("analysis_result", "")
        if not raw:
            return {"error": "No analysis output to format."}

        # If raw is valid JSON, convert using LLM; otherwise use raw as plain text
        try:
            data = json.loads(raw)
            format_prompt = FORMAT_TO_TEXT_SYSTEM.format(json_data=json.dumps(data, indent=2))
            fmt_result = await self.generate_non_streaming(
                [{"role": "system", "content": format_prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            if fmt_result.get("type") == "error":
                raise OllamaClientError(fmt_result["error"])
            final_text = fmt_result.get("content", raw)
        except json.JSONDecodeError:
            final_text = raw
        except Exception as e:
            logger.error(f"Formatting failed, falling back to raw output: {e}")
            final_text = raw

        return {"final_answer": final_text.strip()}

    async def _shield_node(self, state: BusinessAnalystState) -> dict:
        """Optional shield validation (stub)."""
        if state.get("error") or not state.get("is_automotive", False):
            return {}
        try:
            shield_result = await self.run_shield(
                shield_id="business_analyst_shield",
                messages=[{"role": "assistant", "content": state.get("final_answer", "")}],
            )
            if shield_result and getattr(shield_result, "violation", None):
                return {"shield_violation": str(shield_result.violation)}
        except Exception as e:
            logger.warning(f"Shield check failed (continuing anyway): {e}")
        return {}

    async def _final_response_node(self, state: BusinessAnalystState) -> dict:
        """Append shield warning if present."""
        if state.get("error"):
            return {"final_answer": f" Error: {state['error']}"}
        answer = state.get("final_answer", "No analysis produced.")
        if state.get("shield_violation"):
            answer = f" Quality Warning: {state['shield_violation']}\n\n{answer}"
        return {"final_answer": answer}

    # Routing helpers
    @staticmethod
    def _route_after_moderation(state: BusinessAnalystState) -> Literal["domain_check", "final_response"]:
        return "final_response" if state.get("error") else "domain_check"

    @staticmethod
    def _route_after_domain_check(state: BusinessAnalystState) -> Literal["analysis", "final_response"]:
        if state.get("error") or not state.get("is_automotive", False):
            return "final_response"
        return "analysis"

    @staticmethod
    def _route_after_analysis(state: BusinessAnalystState) -> Literal["formatting", "final_response"]:
        return "final_response" if state.get("error") else "formatting"

    @staticmethod
    def _route_after_formatting(state: BusinessAnalystState) -> Literal["shield", "final_response"]:
        return "final_response" if state.get("error") else "shield"

    @staticmethod
    def convert_langchain_tools_to_openai(tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert LangChain BaseTool instances to the OpenAI‑style tool definitions
        expected by generate_non_streaming().
        """
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args if hasattr(tool, "args") else {},
                },
            })
        return openai_tools
    
    # Public entry point
    async def generate(self, user_input: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Run the business analyst workflow on a single user input.
        Yields events (streaming) during processing.
        """
        if self._graph is None:
            await self.initialize()

        initial_state: BusinessAnalystState = {
            "messages": [],
            "user_input": user_input,
            "is_automotive": False,
            "analysis_result": "",
            "final_answer": "",
            "shield_violation": None,
            "error": None,
        }

        # Stream intermediate events (optional)
        async for event in self._graph.astream(initial_state):
            # Each event is a dict with node name as key and state updates as value
            for node_name, updates in event.items():
                if "final_answer" in updates:
                    yield {"type": "final", "content": updates["final_answer"]}
                elif "error" in updates:
                    yield {"type": "error", "error": updates["error"]}
                else:
                    # Progress update
                    yield {"type": "progress", "node": node_name, "state": updates}
        # After stream ends, the final state is available (not needed for yield)