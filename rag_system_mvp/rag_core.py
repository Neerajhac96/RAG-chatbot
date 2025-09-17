import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama # or from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
import os


# A dictionary to map file extensions to document loaders
LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}

# Define the directory to store uploaded documents
DOC_DIR = "uploaded_docs"
FAISS_INDEX_DIR = "faiss_index"

# Ensure directories exist
if not os.path.exists(DOC_DIR):
    os.makedirs(DOC_DIR)

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# A global variable to store the vector store
db = None

def get_text_from_documents(doc_paths):
    """
    Loads text from a list of document paths.
    """
    all_text = []
    for doc_path in doc_paths:
        ext = os.path.splitext(doc_path)[1].lower()
        if ext in LOADERS:
            loader = LOADERS[ext](doc_path)
            try:
               documents = loader.load()
               all_text.extend(documents)
            except Exception as e:
                print(f"Failed to load {doc_path}: {e}")
    return all_text

def process_documents(doc_paths):
    """
    Parse, split, create/update FAISS index, and save index to disk.
    """
    global db
    
    # 1. Load documents
    documents = get_text_from_documents(doc_paths)
    
    if not documents:
        return "No documents found to process."
    
    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    
    # 3. Create or update the vector store
    try:
        if db is None:
            db = FAISS.from_documents(docs, embeddings)
            print("New FAISS vector store created.")
        else:
            # A simple way to update - merge new docs into the existing FAISS index
            new_db = FAISS.from_documents(docs, embeddings)
            db.merge_from(new_db)
            print("FAISS vector store updated with new documents.")
    except Exception as e:
        return f"Error creating/updating FAISS index: {e}"

    # Save index to disk for persistence
    try:
        if not os.path.exists(FAISS_INDEX_DIR):
            os.makedirs(FAISS_INDEX_DIR)
        db.save_local(FAISS_INDEX_DIR)
        print(f"FAISS index saved to {FAISS_INDEX_DIR}")
    except Exception as e:
        print("Failed to save FAISS index:", e)

    return "Documents processed and indexed successfully."

def load_faiss_index_if_exists():
    """
    Try to load saved FAISS index from disk into global `db`.
    """
    global db
    try:
        if os.path.exists(FAISS_INDEX_DIR) and os.listdir(FAISS_INDEX_DIR):
            db = FAISS.load_local(FAISS_INDEX_DIR, embeddings)
            print("Loaded FAISS index from disk.")
            return True
    except Exception as e:
        print("Error loading FAISS index:", e)
        db = None
    return False

def initialize_vectorstore_on_import():
    """
    Called when module is imported. First try loading saved index.
    If not present, look for files in uploaded_docs and build index automatically.
    """
    loaded = load_faiss_index_if_exists()
    if loaded:
        return

    # If no saved index, try to process files in DOC_DIR (if present)
    try:
        files = [
            os.path.join(DOC_DIR, f)
            for f in os.listdir(DOC_DIR)
            if os.path.splitext(f)[1].lower() in LOADERS
        ]
    except Exception:
        files = []

    if files:
        print("Found files in uploaded_docs — building index now...")
        process_documents(files)
    else:
        print("No docs to process at startup.")

def get_answer_from_llm(query):
    """
    Retrieves relevant chunks and generates an answer using an LLM.
    """
    global db
    if db is None:
        return "Please upload documents first.", []
    
    # Define the LLM (using Ollama as an example)
    # try:
    #     # Replace model name if needed (ensure model exists locally)
    #     llm = Ollama(model="llama2") # or llm = OpenAI(api_key="YOUR_API_KEY")
    # except Exception as e:
    #     return f"LLM init error: {e}", []
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",     # ya koi bhi inference-supported model
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
        temperature=0.2
    )
    
    try:    
        # Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "stuff" chain type puts all retrieved docs in one prompt
            retriever=db.as_retriever(),
            return_source_documents=True,
        )
    
        # Get the response from the chain
        response = qa_chain({"query": query})
    except Exception as e:
        return f"LLM/query error: {e}", []
    
    # Extract the answer and citations
    answer = response("result") or response.get("answer") or str(response)
    source_docs = response.get("source_documents", [])
    
    citations = []
    for doc in source_docs:
        # Use a simple citation format
        citations.append(f"Source: {doc.metadata.get('source', 'N/A')}")
        
    return answer, citations

# Run initialization at import time
initialize_vectorstore_on_import()