import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama # or from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
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

def process_documents(doc_paths, existing_db=None):
    """
    Parse, split, create/update FAISS index, and save index to disk.
    This function now accepts and returns the FAISS DB object.
    """
    
    # 1. Load documents
    documents = get_text_from_documents(doc_paths)
    
    if not documents:
        return existing_db, "No documents found to process."
    
    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    
    # 3. Create or update the vector store
    try:
        if existing_db is None:
            db = FAISS.from_documents(docs, embeddings)
            # print("New FAISS vector store created.")
            message = "New FAISS vector store created."
        else:
            # A simple way to update - merge new docs into the existing FAISS index
            new_db = FAISS.from_documents(docs, embeddings)
            existing_db.merge_from(new_db)
            db = existing_db
            message = "FAISS vector store updated with new documents."
    except Exception as e:
        return existing_db, f"Error creating/updating FAISS index: {e}"

    # Save index to disk for persistence
    try:
        if not os.path.exists(FAISS_INDEX_DIR):
            os.makedirs(FAISS_INDEX_DIR)
        db.save_local(FAISS_INDEX_DIR)
        print(f"FAISS index saved to {FAISS_INDEX_DIR}")
    except Exception as e:
        print("Failed to save FAISS index:", e)

    # return "Documents processed and indexed successfully."
    return db, message

def load_faiss_index_if_exists():
    """
    Try to load saved FAISS index from disk into global `db`.
    """
    # global db
    try:
        if os.path.exists(FAISS_INDEX_DIR) and os.listdir(FAISS_INDEX_DIR):
            db = FAISS.load_local(
                FAISS_INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True # Add this line
            )
            print("Loaded FAISS index from disk.")
            return db
    except Exception as e:
        print("Error loading FAISS index:", e)

    # If no saved index, try to process files in DOC_DIR (if present)
    try:
        files = [
            os.path.join(DOC_DIR, f)
            for f in os.listdir(DOC_DIR)
            if os.path.splitext(f)[1].lower() in LOADERS
        ]

        if files:
            print("No saved index found. Building a new one from uploaded documents...")
            db, message = process_documents(files)
            print(message)
            return db
    except Exception as e:
        print("Error building index from existing documents:", e)

    return None

def get_answer_from_llm(query, db):
    """
    Retrieves relevant chunks and generates an answer using an LLM.
    """
    # global db
    if db is None:
        return "Please upload documents first.", []
    
    # Define the LLM (using Ollama as an example)
    # try:
    #     # Replace model name if needed (ensure model exists locally)
    #     llm = Ollama(model="llama2") # or llm = OpenAI(api_key="YOUR_API_KEY")
    # except Exception as e:
    #     return f"LLM init error: {e}", []
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",
        task="text2text-generation",     # ya koi bhi inference-supported model
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
        temperature = 0.2,
        max_new_tokens=200
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
        response = qa_chain.invoke({"query": query})
    except Exception as e:
        return f"LLM/query error: {e}", []
    
    # Extract the answer and citations
    answer = response.get("result") or response.get("answer") or str(response)
    source_docs = response.get("source_documents", [])
    
    citations = []
    for doc in source_docs:
        # Use a simple citation format
        citations.append(f"Source: {doc.metadata.get('source', 'N/A')}")
        
    return answer, citations

# Run initialization at import time
# initialize_vectorstore_on_import()