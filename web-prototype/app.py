import streamlit as st
import httpx
import json
import os

# API URL
API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(
    page_title="EU-Compliant Document Chat",
    page_icon="📚",
    layout="wide"
)

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
                st.caption(f"• {source['filename']} (Chunk {source['chunkId']})")

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
                            st.caption(f"• {source['filename']} (Chunk {source['chunkId']})")
                    
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
                        "content": error_msg
                    })
            
            except Exception as e:
                error_msg = f"Error connecting to API: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This is a prototype for an EU-compliant document chat system using:
    - Weaviate (Dutch) for vector database
    - Mistral AI (French) for LLM services
    - All data processing is GDPR compliant
    """)
    
    st.header("System Status")
    
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