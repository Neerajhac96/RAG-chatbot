import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama # or from langchain_openai import OpenAI

# A dictionary to map file extensions to document loaders
LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}

# Define the directory to store uploaded documents
DOC_DIR = "uploaded_docs"
if not os.path.exists(DOC_DIR):
    os.makedirs(DOC_DIR)

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings()

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
            documents = loader.load()
            all_text.extend(documents)
    return all_text

def process_documents(doc_paths):
    """
    Parses documents, splits them into chunks, and creates a vector store.
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
    if db is None:
        db = FAISS.from_documents(docs, embeddings)
        print("New FAISS vector store created.")
    else:
        # A simple way to update - merge new docs into the existing FAISS index
        new_db = FAISS.from_documents(docs, embeddings)
        db.merge_from(new_db)
        print("FAISS vector store updated with new documents.")

    return "Documents processed and indexed successfully."

def get_answer_from_llm(query):
    """
    Retrieves relevant chunks and generates an answer using an LLM.
    """
    global db
    if db is None:
        return "Please upload documents first."
    
    # Define the LLM (using Ollama as an example)
    llm = Ollama(model="llama2") # or llm = OpenAI(api_key="YOUR_API_KEY")

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" chain type puts all retrieved docs in one prompt
        retriever=db.as_retriever(),
        return_source_documents=True,
    )
    
    # Get the response from the chain
    response = qa_chain({"query": query})
    
    # Extract the answer and citations
    answer = response['result']
    source_docs = response['source_documents']
    
    citations = []
    for doc in source_docs:
        # Use a simple citation format
        citations.append(f"Source: {doc.metadata.get('source', 'N/A')}")
        
    return answer, citations