import streamlit as st
import os
from rag_core import process_documents, get_answer_from_llm, DOC_DIR

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("Document-Based Q&A Chatbot")
st.markdown("Upload your documents (PDF, DOCX, TXT) and ask questions about their content.")

# Sidebar for document management
with st.sidebar:
    st.header("Document Uploader")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            # Save files to a temporary directory
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DOC_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
            
            # Process the documents
            with st.spinner("Processing and indexing documents..."):
                message = process_documents(file_paths)
            st.success(message)
        else:
            st.warning("Please upload documents first.")

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["citations"]:
            st.markdown("---")
            st.markdown("**Sources:**")
            for citation in message["citations"]:
                st.markdown(f"- {citation}")

# React to user input
if query := st.chat_input("Ask a question about the documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(query)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query, "citations": []})
    
    # Get the answer from the RAG system
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, citations = get_answer_from_llm(query)
            st.markdown(answer)
            if citations:
                st.markdown("---")
                st.markdown("**Sources:**")
                for citation in citations:
                    st.markdown(f"- {citation}")
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})