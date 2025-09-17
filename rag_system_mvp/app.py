import streamlit as st
import os
from rag_core import process_documents, get_answer_from_llm, load_faiss_index_if_exists, DOC_DIR

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("Document-Based Q&A Chatbot")
st.markdown("Upload your documents (PDF, DOCX, TXT) and ask questions about their content.")

# Initialize FAISS database in session state
if 'faiss_db' not in st.session_state:
    st.session_state.faiss_db = load_faiss_index_if_exists()

# Sidebar for document management
with st.sidebar:
    st.header("Document Uploader")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    # Show uploaded documents (current session)
    st.subheader("Selected Documents (Current Session)")
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.markdown(f"- {uploaded_file.name}")
    else:
        st.markdown("_No documents selected in this session._")

    # Show previously uploaded documents (persisted in uploaded_docs)
    st.subheader("All Uploaded Documents")
    try:
        prev_files = os.listdir(DOC_DIR)
        if prev_files:
            for fname in prev_files:
                st.markdown(f"- {fname}")
        else:
            st.markdown("_No documents found in uploaded_docs._")
    except Exception as e:
        st.markdown(f"_Error reading uploaded_docs: {e}_")

    if st.button("Process Documents"):
        if uploaded_files:
            # Save files to a temporary directory
            file_paths = []
            for uploaded_file in uploaded_files:
                # Use a unique filename to prevent overwriting
                file_name = uploaded_file.name
                base, ext = os.path.splitext(file_name)
                counter = 1
                file_path = os.path.join(DOC_DIR, file_name)
                while os.path.exists(file_path):
                    file_name = f"{base}_{counter}{ext}"
                    file_path = os.path.join(DOC_DIR, file_name)
                    counter += 1
    
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
            
            # Process the documents
            with st.spinner("Processing and indexing documents..."):
                st.session_state.faiss_db, message = process_documents(file_paths, st.session_state.faiss_db)
            st.success(message)
        else:
            st.warning("Please upload documents first.")
    if st.button("Clear Chat History"):
       st.session_state.messages = []


# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
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
            answer, citations = get_answer_from_llm(query, st.session_state.faiss_db)
            st.markdown(answer)
            if citations:
                st.markdown("---")
                st.markdown("**Sources:**")
                for citation in citations:
                    st.markdown(f"- {citation}")
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})