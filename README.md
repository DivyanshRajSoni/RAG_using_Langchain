# LangChain RAG Engine with OpenAI GPT, ChromaDB, and Sentence Transformers

A complete **Retrieval-Augmented Generation (RAG)** pipeline implementation using LangChain, enabling intelligent document retrieval and LLM-powered question answering.

## 🎯 Project Overview

This project implements an end-to-end RAG system that:

1. **Loads & Processes Documents** - Extracts text from PDFs and text files
2. **Creates Embeddings** - Converts text chunks into semantic embeddings using SentenceTransformer
3. **Stores in Vector DB** - Persists embeddings in ChromaDB for efficient retrieval
4. **Semantic Search** - Retrieves relevant documents based on query similarity
5. **LLM Integration** - Augments retrieved context with OpenAI's GPT models for intelligent responses

## 📋 Architecture

```
Raw Data (PDFs, Text Files)
    ↓
Document Loading & Processing
    ↓
Text Chunking (Recursive Character Splitter)
    ↓
Embedding Generation (SentenceTransformer)
    ↓
Vector Store (ChromaDB)
    ↓
Semantic Retrieval
    ↓
LLM Context Augmentation (OpenAI GPT)
    ↓
Generated Answers
```

## 🚀 Features

- **Multi-format Document Loading**: Support for PDF and text files
- **Smart Chunking**: Recursive character splitting with configurable chunk size and overlap
- **Semantic Embeddings**: Uses `all-MiniLM-L6-v2` model for efficient embeddings (384 dimensions)
- **Persistent Vector Store**: ChromaDB with local persistence for scalability
- **Similarity-based Retrieval**: Configurable top-k retrieval with score thresholding
- **LLM Integration**: Ready for OpenAI GPT models integration
- **Metadata Tracking**: Preserves document metadata and content length information

## 📁 Project Structure

```
RAG_pipeline/
├── RAG_pipeline.ipynb          # Main Jupyter notebook with complete implementation
├── README.md                   # This file
├── Data/
│   ├── python.txt             # Sample text data
│   ├── pdf/                   # PDF documents folder
│   └── vector_store/          # ChromaDB persistence
│       ├── chroma.sqlite3     # Vector database
│       └── [uuid]/            # Collection data
```

## 📦 Dependencies

```
langchain>=0.1.0
langchain-core
langchain-community
langchain-openai
langchain-text-splitters
pypdf
pymupdf
sentence-transformers
chromadb
scikit-learn
```

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DivyanshRajSoni/RAG_using_Langchain.git
   cd RAG_pipeline
   ```

2. **Install dependencies**
   ```bash
   pip install langchain langchain-core langchain-community pypdf pymupdf sentence-transformers chromadb scikit-learn langchain-openai langchain-text-splitters
   ```

3. **Install Jupyter (if not already installed)**
   ```bash
   pip install jupyter
   ```

## 💡 Usage

### 1. Start Jupyter Notebook
```bash
jupyter notebook
```

### 2. Load Documents
```python
# Load all PDFs from the Data/pdf directory
from RAG_pipeline import load_all_pdfs
all_pdf_documents = load_all_pdfs()
```

### 3. Create Chunks
```python
# Split documents into manageable chunks
from RAG_pipeline import split_docs
chunks = split_docs(all_pdf_documents, chunk_size=500, chunk_overlap=50)
```

### 4. Generate Embeddings
```python
# Create embeddings using SentenceTransformer
from RAG_pipeline import EmbeddingManager
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in chunks])
```

### 5. Store in Vector Database
```python
# Initialize and populate ChromaDB
from RAG_pipeline import VectorStoreManager
vector_store = VectorStoreManager()
vector_store.add_documents(chunks, embeddings)
```

### 6. Retrieve Relevant Documents
```python
# Perform semantic search and retrieval
from RAG_pipeline import RAGRetriever
rag_retriever = RAGRetriever(embedding_manager, vector_store)
retrieved_docs = rag_retriever.retriever("What is RAG?", top_k=5)
```

### 7. Generate Answers with LLM
```python
# Integrate with OpenAI GPT for context-augmented responses
from langchain_openai import ChatOpenAI
from RAG_pipeline import generate_output

llm = ChatOpenAI(api_key="YOUR_OPENAI_API_KEY", model="gpt-4", temperature=0.1)
response = generate_output("Your query here", rag_retriever, llm, top_k=3)
```

## 🔑 Configuration

### EmbeddingManager
- **Default Model**: `all-MiniLM-L6-v2`
- **Embedding Dimension**: 384
- **Customizable**: Pass custom model name during initialization

### VectorStoreManager
- **Default Collection Name**: `pdf_documents`
- **Persistence Directory**: `Data/vector_store`
- **Vector DB**: ChromaDB with local SQLite backend

### RAGRetriever
- **Default Top-K**: 5 documents
- **Default Score Threshold**: 0.0 (no filtering)
- **Similarity Metric**: Cosine similarity (1 - distance)

## 📊 Key Components

### EmbeddingManager
Handles text embedding generation using SentenceTransformer models.

### VectorStoreManager
Manages ChromaDB collections with document storage, retrieval, and persistence.

### RAGRetriever
Performs semantic search using embeddings and returns ranked relevant documents.

## 🔐 Environment Setup

For OpenAI Integration:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or set directly in code:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(api_key="YOUR_OPENAI_API_KEY")
```

## 📈 Performance Tips

1. **Chunk Size**: Optimal range is 300-1000 tokens. Use 500 for balanced retrieval.
2. **Chunk Overlap**: 50 tokens overlap usually works well for context preservation.
3. **Top-K Retrieval**: Use 3-5 for most queries; increase for broader context needs.
4. **Score Threshold**: Set to 0.5+ for high-relevance filtering.
5. **Batch Processing**: Use batch embedding for large document sets.

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `chromadb.errors.InvalidCollectionException` | Delete `vector_store` directory and re-run initialization |
| `OutOfMemoryError` during embedding | Reduce chunk size or process in smaller batches |
| Missing OpenAI responses | Verify API key and model name (e.g., "gpt-4", "gpt-3.5-turbo") |
| Low retrieval accuracy | Increase `top_k`, decrease score threshold, or adjust chunk size |

## 📝 Notes

- Documents are automatically assigned unique UUIDs for tracking
- Metadata preservation includes source, page numbers, and content length
- Vector store persists across sessions in `Data/vector_store`
- Similarity scores range from 0 to 1 (1 = most similar)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or suggestions, please reach out through GitHub issues.

---

**Happy Retrieving! 🚀**
