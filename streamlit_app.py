import streamlit as st
import requests


# Configuration
API_BASE_URL = "http://localhost:8000"  # FastAPI server address


# Session state initialisation
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.email = ""
    st.session_state.messages = []  # chat history


# Helper functions to call FastAPI
def analyze_user_input(user_input: str) -> dict:
    """POST /analyze"""
    resp = requests.post(f"{API_BASE_URL}/analyze", json={"user_input": user_input})
    if resp.status_code == 200:
        return resp.json()
    else:
        return {"answer": f"Error: {resp.status_code} - {resp.text}"}

def get_mcp_status() -> dict:
    """GET /mcp/status"""
    resp = requests.get(f"{API_BASE_URL}/mcp/status")
    if resp.status_code == 200:
        return resp.json()
    return {"jira": "error", "github": "error"}

def connect_github(pat: str) -> dict:
    """POST /github/connect"""
    resp = requests.post(f"{API_BASE_URL}/github/connect", json={"pat": pat})
    return resp.json()  # may contain error message if not 200


# UI
st.set_page_config(page_title="Automotive Business Analyst", layout="wide")

# LOGIN PAGE (if not logged in)
if not st.session_state.logged_in:
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            # Fake authentication – just store values and proceed
            if username and email and password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.email = email
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Please fill in all fields.")
    st.stop()

# MAIN APP (after login)
st.title(f"🚗 Automotive Business Analyst - Welcome, {st.session_state.username}")

#  SIDEBAR 
with st.sidebar:
    st.header("MCP Servers")
    # Fetch status from backend
    mcp_status = get_mcp_status()

    # Jira status
    jira_status = mcp_status.get("jira", "disconnected")
    if jira_status == "connected":
        st.success("✅ Jira - Connected")
    else:
        st.warning("⚠️ Jira - Disconnected (set JIRA_URL, USERNAME, API_TOKEN)")

    # GitHub status
    github_status = mcp_status.get("github", "disconnected")
    if github_status.startswith("connected"):
        # Display connected with username
        username = github_status.split("(", 1)[1].rstrip(")")
        st.success(f"✅ GitHub - Connected to {username}")
    else:
        st.warning("⚠️ GitHub - Disconnected")
        # Allow user to paste PAT
    with st.expander("Connect GitHub"):
            pat = st.text_input("GitHub Personal Access Token", type="password")
            if st.button("Connect"):
                if pat:
                    try:
                        result = connect_github(pat)
                        if "username" in result:
                            st.success(f"Connected to {result['username']}!")
                            # Trigger full refresh to update status
                            st.rerun()
                        else:
                            st.error(result.get("message", "Connection failed"))
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Please enter a token.")

#  CHAT INTERFACE 
st.divider()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a business analysis question about the automotive sector..."):
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.spinner("Analysing..."):
        result = analyze_user_input(prompt)
    
    answer = result.get("answer", "No response generated.")
    if result.get("shield_alert"):
        answer = f"⚠️ Quality alert: {result['shield_alert']}\n\n{answer}"

    # Add assistant message to history and display
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)