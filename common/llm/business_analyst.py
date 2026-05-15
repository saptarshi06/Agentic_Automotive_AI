import json
import logging
from typing import TypedDict, List, Dict, Optional, Literal

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from common.llm.base_bedrock import BedrockClient, AuthenticationExpiredError, BedrockClientError

logger = logging.getLogger(__name__)


# ---- State Definition ----
class BusinessAnalysisState(TypedDict):
    messages: List[Dict[str, str]]       # conversation history
    user_input: str                      # original user request
    analysis_result: str                 # raw LLM output (could be JSON)
    final_answer: str                    # cleaned plain‑text answer
    shield_violation: Optional[str]      # if safety shield triggered
    error: Optional[str]                 # error message to return


# ---- Prompt Templates ----
ANALYSIS_SYSTEM_PROMPT = (
    "You are a senior business analyst specialised in the automotive industry. "
    "Analyse the following request and provide a structured response. "
    "Include: 1) Key Findings, 2) Stakeholders Impacted, 3) Risks & Assumptions, "
    "4) Recommendations tailored to automotive manufacturing, supply chain, or retail. "
    "Format your answer as a JSON object with these four fields. Do not add extra commentary."
)

FORMAT_TO_TEXT_PROMPT = (
    "Convert the following JSON analysis into a clear, professional plain‑text report for non‑technical stakeholders. "
    "Use bullet points where appropriate and avoid any JSON syntax in the output.\n\n"
    "JSON:\n{json_data}\n\nPlain‑text report:"
)


# ---- LangGraph Nodes ----
async def moderation_node(state: BusinessAnalysisState, client: BedrockClient) -> dict:
    """Run input moderation before processing."""
    try:
        mod_result = await client.run_moderation(state["user_input"])
        if mod_result and getattr(mod_result, "flagged", False):
            return {"error": "Your input was flagged by our content safety system. Please rephrase."}
    except Exception as e:
        logger.warning(f"Moderation check failed (proceeding anyway): {e}")
    return {}  # continue normally


async def analysis_node(state: BusinessAnalysisState, client: BedrockClient) -> dict:
    """Call Bedrock to produce the structured analysis (JSON)."""
    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": state["user_input"]},
    ]
    try:
        response = await client.chat_completion(
            messages=messages,
            temperature=0.3,  # more deterministic
            stream=False,
        )
        analysis_text = response.choices[0].message.content
        # Validate JSON (optional)
        try:
            json.loads(analysis_text)
        except json.JSONDecodeError:
            # If LLM didn't output valid JSON, we still keep the text and try to format later
            logger.warning("LLM output is not valid JSON, will attempt to format regardless.")
        return {"analysis_result": analysis_text}
    except AuthenticationExpiredError:
        return {"error": "AWS Bedrock token expired. Please refresh your credentials and try again."}
    except BedrockClientError as e:
        logger.error(f"Bedrock error during analysis: {e}")
        return {"error": f"Analysis failed due to a service error: {e}"}


async def formatting_node(state: BusinessAnalysisState, client: BedrockClient) -> dict:
    """Convert the JSON output (or semi‑structured text) into a user‑friendly plain text."""
    if state.get("error"):
        return {}  # don't try to format if we already have an error

    raw = state["analysis_result"]
    if not raw:
        return {"error": "No analysis output to format."}

    # If the raw text is already plain text (not JSON), we can optionally still polish it.
    # But we'll try to detect JSON and convert.
    try:
        # Check if it's valid JSON
        data = json.loads(raw)
        # Use LLM to convert to text
        format_messages = [
            {"role": "system", "content": FORMAT_TO_TEXT_PROMPT.format(json_data=json.dumps(data, indent=2))},
        ]
        fmt_response = await client.chat_completion(
            messages=format_messages,
            temperature=0.3,
            stream=False,
        )
        final_text = fmt_response.choices[0].message.content
    except json.JSONDecodeError:
        # Not JSON, use it directly (maybe with a fallback formatting)
        final_text = raw
    except Exception as e:
        logger.error(f"Formatting failed, falling back to raw output: {e}")
        final_text = raw

    return {"final_answer": final_text.strip()}


async def shield_node(state: BusinessAnalysisState, client: BedrockClient) -> dict:
    """Validate the final answer with the business analysis shield."""
    if state.get("error"):
        return {}  # skip shield if error

    try:
        shield_result = await client.run_shield(
            shield_id="business_analysis_shield",
            messages=[{"role": "assistant", "content": state["final_answer"]}],
        )
        if shield_result and getattr(shield_result, "violation", None):
            # We can either block the response or add a warning
            return {"shield_violation": str(shield_result.violation)}
    except Exception as e:
        logger.warning(f"Shield check failed (continuing anyway): {e}")
    return {}


async def final_response_node(state: BusinessAnalysisState) -> dict:
    """Compose the final message to return to the user."""
    if state.get("error"):
        return {"final_answer": f"❌ Error: {state['error']}"}
    answer = state.get("final_answer", "No analysis produced.")
    if state.get("shield_violation"):
        answer = f"⚠️ Quality Warning: {state['shield_violation']}\n\n{answer}"
    return {"final_answer": answer}


# ---- Graph Builder ----
def build_business_analyst_graph(client: BedrockClient) -> StateGraph:
    """Construct the LangGraph workflow for business analysis."""
    workflow = StateGraph(BusinessAnalysisState)

    # Add nodes
    workflow.add_node("moderation", lambda s: moderation_node(s, client))
    workflow.add_node("analysis", lambda s: analysis_node(s, client))
    workflow.add_node("formatting", lambda s: formatting_node(s, client))
    workflow.add_node("shield", lambda s: shield_node(s, client))
    workflow.add_node("final_response", final_response_node)

    # Edges: moderation -> (if no error) analysis -> formatting -> shield -> final
    # If error at any point, we skip to final.
    def route_after_moderation(state: BusinessAnalysisState) -> Literal["analysis", "final_response"]:
        return "final_response" if state.get("error") else "analysis"

    def route_after_analysis(state: BusinessAnalysisState) -> Literal["formatting", "final_response"]:
        return "final_response" if state.get("error") else "formatting"

    def route_after_formatting(state: BusinessAnalysisState) -> Literal["shield", "final_response"]:
        return "final_response" if state.get("error") else "shield"

    # (shield always goes to final_response, no conditional needed)

    workflow.add_edge(START, "moderation")
    workflow.add_conditional_edges("moderation", route_after_moderation)
    workflow.add_conditional_edges("analysis", route_after_analysis)
    workflow.add_conditional_edges("formatting", route_after_formatting)
    workflow.add_edge("shield", "final_response")
    workflow.add_edge("final_response", END)

    return workflow.compile()