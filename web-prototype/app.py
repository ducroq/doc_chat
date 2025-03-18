import streamlit as st
import httpx
import json
import os
import bcrypt
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

# Configuration
API_URL = os.getenv("API_URL", "http://api:8000")
ENABLE_CHAT_LOGGING = os.getenv("ENABLE_CHAT_LOGGING", "false").lower() == "true"

# Authentication functionality
def check_password(username, password):
    if username in st.secrets["passwords"]:
        # Verify password (stored as bcrypt hash)
        stored_hash = st.secrets["passwords"][username]
        return bcrypt.checkpw(password.encode(), stored_hash.encode())
    return False

def get_api_key():
    try:
        with open(os.environ.get("INTERNAL_API_KEY_FILE"), "r") as f:
            return f.read().strip()
    except Exception as e:
        st.error(f"Error reading API key: {str(e)}")
        return ""

def format_citation(source):
    """Format a source citation with metadata if available."""
    filename = source.get('filename', 'Unknown')
    chunk_id = source.get('chunkId', 'Unknown')
    
    if 'metadata' in source:
        metadata = source.get('metadata', {})
        
        # Build a rich citation with context
        if metadata and 'title' in metadata:
            citation = f"• {metadata['title']}"
            
            if metadata and 'itemType' in metadata:
                citation += f" [{metadata['itemType']}]"
                
            # Handle creators/authors
            if metadata and 'creators' in metadata and metadata['creators']:
                authors = []
                for creator in metadata['creators']:
                    if creator.get('creatorType') == 'author':
                        name = f"{creator.get('lastName', '')}, {creator.get('firstName', '')}"
                        authors.append(name.strip(', '))
                if authors:
                    citation += f" by {', '.join(authors[:2])}"
                    if len(authors) > 2:
                        citation += f" et al."
                        
            # Add date if available
            if metadata and 'date' in metadata:
                citation += f" ({metadata['date']})"
        else:
            citation = f"• {filename} (Chunk {chunk_id})"
            
        # Add section info if available
        if 'heading' in source:
            citation += f" - Section: {source['heading']}"
            
        # Add page number if available
        if 'page' in source:
            citation += f", Page {source['page']}"
            
        return citation
    else:
        # Basic citation without metadata
        return f"• {filename} (Chunk {chunk_id})"
    
def log_feedback(feedback_type, message_content, feedback_text=None):
    """Log feedback to API (simplified version)"""
    try:
        # Create a simple feedback object
        feedback_data = {
            "request_id": str(uuid.uuid4())[:8],
            "message_id": str(uuid.uuid4()),
            "rating": feedback_type,  # "positive" or "negative" 
            "feedback_text": feedback_text,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log to API if endpoint is ready
        try:
            response = httpx.post(
                f"{API_URL}/feedback",
                json=feedback_data,
                headers={"X-API-Key": get_api_key()},
                timeout=10.0
            )
            return True
        except Exception as e:
            # Just log locally if API endpoint isn't ready
            st.session_state.setdefault("feedback_history", []).append(feedback_data)
            return True
    except Exception as e:
        st.error(f"Error logging feedback: {str(e)}")
        return False    

def login_page():
    st.title("🇪🇺 Document Chat Login")
    
    # Create login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if check_password(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid username or password")

# Main app functionality
def main_app():
    # API URL
    API_URL = os.getenv("API_URL", "http://api:8000")

    st.title("🇪🇺 Document Chat")
    st.write("Ask questions about your documents stored in the system.")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                st.caption("Sources:")
                for source in message["sources"]:
                    st.caption(format_citation(source))
 
     # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = httpx.post(
                        f"{API_URL}/chat",
                        json={"question": prompt},
                        headers={"X-API-Key": get_api_key()},
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "Sorry, I couldn't generate a response.")
                        sources = data.get("sources", [])
                        
                        # Display the answer
                        st.markdown(answer)
                        
                        # Display sources if any
                        if sources:
                            st.caption("Sources:")
                            for source in sources:
                                st.caption(format_citation(source))

                        # Add feedback UI with expander
                        with st.expander("Rate this response"):
                            st.write("Was this response helpful?")
                            col1, col2 = st.columns(2)
                            
                            # Simple feedback without complex state management
                            with col1:
                                if st.button("👍 Helpful", key=f"helpful_{str(uuid.uuid4())[:8]}"):
                                    log_feedback("positive", answer)
                                    st.success("Thank you for your feedback!")
                                    
                            with col2:
                                if st.button("👎 Not Helpful", key=f"not_helpful_{str(uuid.uuid4())[:8]}"):
                                    # Set session state to show the detailed feedback form
                                    st.session_state["show_detail_feedback"] = True
                                    st.session_state["feedback_response"] = answer
                                    st.rerun()
                            
                            # Show detailed feedback form if negative feedback was given
                            if st.session_state.get("show_detail_feedback", False) and st.session_state.get("feedback_response") == answer:
                                with st.form(key=f"feedback_form_{str(uuid.uuid4())[:8]}"):
                                    st.write("Please tell us how we can improve:")
                                    feedback_text = st.text_area("What was wrong with this response?")
                                    feedback_categories = st.multiselect(
                                        "Select issues with the response:",
                                        ["Missing information", "Incorrect information", 
                                        "Didn't understand my question", "Irrelevant sources",
                                        "Other"]
                                    )
                                    
                                    submit_button = st.form_submit_button("Submit Feedback")
                                    if submit_button:
                                        # Log the detailed feedback
                                        log_feedback("negative", answer, feedback_text)
                                        st.success("Thank you for your detailed feedback!")
                                        # Reset the form visibility flag
                                        st.session_state["show_detail_feedback"] = False
                                        st.session_state["feedback_response"] = None

                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        error_msg = f"Error: {response.status_code} - {response.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg,
                            "id": str(uuid.uuid4())
                        })
                
                except Exception as e:
                    error_msg = f"Error connecting to API: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "id": str(uuid.uuid4())
                    })

    # Enhanced sidebar with custom controls
    with st.sidebar:
        # About section
        st.markdown("""
        This is an EU-compliant document chat system using:
        - Weaviate (Dutch) for vector database
        - Mistral AI (French) for LLM services
        - All data processing is GDPR compliant
        """)

        with st.expander("Privacy Notice",  expanded=False):
            st.markdown("""
            ## Chat Logging & Privacy
            
            When enabled, this system may log chat interactions for research and service improvement.
            
            **What we collect:**
            - Questions asked to the system
            - Responses provided
            - Document references used
            - Anonymized session identifiers
            - Feedback on response quality (when provided)
            
            **Data Protection:**
            - All identifiers are anonymized
            - Logs are automatically deleted after 30 days
            - Data is stored securely within the EU
            - You can request deletion of your data
            """)

            if st.button("View Full Privacy Notice", key="privacy_notice"):
                # Open privacy notice in new tab using JavaScript
                js = f"""<script>
                window.open('{API_URL}/privacy', '_blank').focus();
                </script>
                """
                st.components.v1.html(js, height=0)

        # Display logging status if enabled
        if ENABLE_CHAT_LOGGING:
            st.warning("⚠️ Chat logging is currently enabled for research purposes.")        

        # User info
        st.markdown(f"**Logged in as:** {st.session_state.get('username', 'User')}")
        user_col1, user_col2 = st.columns([2, 1])
        with user_col1:
            # log out button
            st.button("Logout", 
                    on_click=lambda: st.session_state.update({"authenticated": False}),
                    key="logout_button")        
        with user_col2:
            # Clear conversation option
            if st.button("🧹 Clear", key="clear_conversation"):
                st.session_state.messages = []
                st.rerun()                    
        
        # System Status
        st.subheader("System Status", divider="gray")
        
        # Check API connection
        try:
            status_response = httpx.get(f"{API_URL}/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                
                st.success("✅ API Service: Connected")
                
                weaviate_status = status_data.get("weaviate", "unknown")
                if weaviate_status == "connected":
                    st.success("✅ Vector Database: Connected")
                else:
                    st.error("❌ Vector Database: Disconnected")
                
                mistral_status = status_data.get("mistral_api", "unknown")
                if mistral_status == "configured":
                    st.success("✅ LLM Service: Configured")
                else:
                    st.error("❌ LLM Service: Not configured")
            else:
                st.error(f"❌ API Service: Error {status_response.status_code}")
        except Exception as e:
            st.error(f"❌ API Service: Error connecting ({str(e)})")
            
# Main entry point
def main():
    # Check if user is authenticated
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    if st.session_state["authenticated"]:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main()