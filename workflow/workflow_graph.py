# workflows/graph.py
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from agents.business_analyst import create_ba_agent

# Define the state structure
class AgentState(TypedDict):
    messages: List[Dict[str, str]]   # conversation history
    input: str                       # current user input
    output: str                      # agent's response

def create_workflow(agent):
    """Create a compiled LangGraph workflow using the provided ADK agent."""
    # Define the node that calls your ADK agent
    def ba_node(state: AgentState) -> AgentState:
        """Invoke the ADK business analyst agent."""
        # Prepare the conversation for ADK (adjust based on your agent's interface)
        # Assuming agent.invoke() accepts a list of messages
        adk_response = create_ba_agent.invoke({
            "messages": state["messages"] + [{"role": "user", "content": state["input"]}]
        })
        
        # Extract the last assistant message
        assistant_message = adk_response["messages"][-1]["content"]
        
        # Return updated state
        return {
            "messages": state["messages"] + [
                {"role": "user", "content": state["input"]},
                {"role": "assistant", "content": assistant_message}
            ],
            "input": state["input"],
            "output": assistant_message
        }

# 3. Build the graph
    workflow = StateGraph(AgentState)
    workflow.add_node("business_analyst", ba_node)
    workflow.set_entry_point("business_analyst")
    workflow.add_edge("business_analyst", END)

    # Compile the graph
    return workflow.compile()