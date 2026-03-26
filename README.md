# Learning RAG Pipeline

A project for learning and implementing Retrieval-Augmented Generation (RAG) pipelines using LangChain.

## Overview

This project demonstrates how to build a RAG pipeline that ingests documents, processes them, and uses them for retrieval-based question answering.

## Features

- Document ingestion from multiple sources (text files, PDFs)
- Document processing and chunking using LangChain text splitters
- Vector storage with ChromaDB and FAISS
- Sentence transformer embeddings for semantic search
- LangChain integration for RAG workflows

## Tech Stack

- **Python** 3.12+
- **LangChain** - LLM framework
- **LangChain Community** - Additional loaders and tools
- **ChromaDB** - Vector database
- **FAISS** - Facebook AI Similarity Search
- **Sentence Transformers** - Text embeddings
- **PyPDF** - PDF processing
- **PyMuPDF** - Alternative PDF processing

## Installation

```bash
uv sync
```

## Project Structure

```
learning-rag-pipeline/
├── data/              # Data files and documents
├── notebook/          # Jupyter notebooks for experimentation
├── main.py            # Main entry point
├── pyproject.toml     # Project configuration
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Getting Started

1. Activate the virtual environment
2. Run notebooks in `notebook/` directory to explore RAG concepts
3. Check `main.py` for the main application logic

## Development

- Python version: 3.12 (see `.python-version`)
- Dependencies are locked in `uv.lock` for reproducibility

<br/>
<br/>

---

---

<br/>
<br/>

# RAG Pipeline Learning Guide 🚀
## From Zero to AI Engineer

> **Target Audience**: Someone with basic Python knowledge who wants to understand RAG systems deeply

---

## Table of Contents
1. [What is RAG? (The Big Picture)](#1-what-is-rag-the-big-picture)
2. [The RAG Pipeline Components](#2-the-rag-pipeline-components)
3. [Step-by-Step Code Walkthrough](#3-step-by-step-code-walkthrough)
4. [Visual Diagrams](#4-visual-diagrams)
5. [Professional Tips](#5-professional-tips)

---

## 1. What is RAG? (The Big Picture)

### The Simple Explanation (Like You're 5)

Imagine you have a really smart robot friend who can answer questions. But this robot has a problem: it only knows things it learned in school (training data). If you ask about something new, it might make up answers!

**RAG solves this problem** by giving the robot a library of books. Now when you ask a question:
1. The robot searches the library for relevant books
2. Reads the relevant pages
3. Uses that information to answer your question accurately

### The Technical Explanation

**RAG = Retrieval-Augmented Generation**

- **Retrieval**: Finding relevant information from a knowledge base
- **Augmented**: Enhancing/improving something
- **Generation**: Creating a response using an AI model

**Why RAG matters:**
- LLMs (Large Language Models) have knowledge cutoff dates
- They can hallucinate (make up facts)
- RAG grounds responses in real documents
- You can update knowledge without retraining the model

---

## 2. The RAG Pipeline Components

### The 5 Core Steps

```
┌─────────────┐     ┌──────────┐     ┌────────────┐     ┌─────────────┐     ┌────────────┐
│   1. Load   │ --> │ 2. Chunk │ --> │ 3. Embed   │ --> │  4. Store   │ --> │ 5. Query   │
│  Documents  │     │   Text   │     │  Vectors   │     │  in Vector  │     │ & Retrieve │
└─────────────┘     └──────────┘     └────────────┘     └─────────────┘     └────────────┘
```

Let me explain each step:

### Step 1: Document Loading (Data Ingestion)

**What it does**: Reads files from your computer into a format Python can work with.

**Why we need it**: 
- Files come in different formats (PDF, TXT, DOCX)
- We need a standard way to handle them
- We want to keep metadata (source, author, date)

**Real-world analogy**: Like scanning physical books into a digital library system.

### Step 2: Chunking (Text Splitting)

**What it does**: Breaks large documents into smaller pieces.

**Why we need it**:
- AI models have token limits (can't process entire books at once)
- Smaller chunks = more precise retrieval
- Better matching between queries and content

**Real-world analogy**: Instead of giving someone an entire encyclopedia, you give them just the relevant paragraph.

**Key concepts**:
- `chunk_size`: How many characters per chunk (e.g., 1000)
- `chunk_overlap`: How many characters overlap between chunks (e.g., 200)
  - Overlap prevents losing context at boundaries

### Step 3: Embeddings (Converting Text to Numbers)

**What it does**: Converts text into vectors (lists of numbers).

**Why we need it**:
- Computers can't understand text directly
- Vectors capture semantic meaning
- Similar meanings = similar vectors

**Real-world analogy**: Like giving each book a unique barcode that also encodes what it's about.

**Example**:
```
"Python is great" → [0.2, 0.8, 0.1, ..., 0.5]  (384 numbers)
"Python is awesome" → [0.21, 0.79, 0.11, ..., 0.49]  (very similar!)
"C is fast" → [0.7, 0.3, 0.9, ..., 0.2]  (different!)
```

### Step 4: Vector Store (Database for Embeddings)

**What it does**: Stores embeddings in a specialized database.

**Why we need it**:
- Fast similarity search (find similar vectors quickly)
- Persistent storage (save for later use)
- Efficient retrieval at scale

**Real-world analogy**: Like a library catalog system that can find books by topic, not just title.

**Popular vector stores**:
- ChromaDB (what we use - simple, local)
- FAISS (Facebook's library - very fast)
- Pinecone (cloud-based)
- Weaviate (production-grade)

### Step 5: Retrieval (Finding Relevant Information)

**What it does**: 
1. Convert user query to embedding
2. Find most similar document embeddings
3. Return the actual text of those documents

**Why we need it**:
- This is the "R" in RAG
- Provides context for the AI to generate accurate answers

**Real-world analogy**: Like asking a librarian for books on a topic, and they bring you the 5 most relevant ones.

---

## 3. Step-by-Step Code Walkthrough

Now let's understand the actual code from your notebook!

### 3.1 Document Class (The Foundation)

```python
from langchain_core.documents import Document

doc = Document(
    page_content="This is a test document.",
    metadata={
        "source": "test_source.txt",
        "pages": 1,
        "author": "John Doe",
        "date_created": "2024-06-01"
    }
)
```

**What's happening**:
- `Document` is a data structure (like a container)
- `page_content`: The actual text
- `metadata`: Information ABOUT the document

**Why this matters**: Standardized format makes processing easier.

---

### 3.2 Creating Sample Data

```python
sample_texts = {
    "../data/text_files/python.txt": """Python is a high-level...""",
    "../data/text_files/c.txt": """C is a low-level...""",
}

for filepath, content in sample_texts.items():
    with open(filepath, "w") as f:
        f.write(content)
```

**What's happening**:
- Creates a dictionary with file paths as keys
- Loops through and writes each file
- `with open()` automatically closes files (good practice!)

**Python concept**: Dictionary iteration with `.items()`

---

### 3.3 Loading Documents

#### Single File Loader

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../data/text_files/python.txt", encoding="utf-8")
documents = loader.load()
```

**What's happening**:
1. Create a loader object
2. Specify encoding (UTF-8 = standard text encoding)
3. `.load()` returns a list of Document objects

**Output**:
```python
[Document(
    metadata={'source': '../data/text_files/python.txt'}, 
    page_content='Python is a high-level...'
)]
```

#### Directory Loader (Load Multiple Files)

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "../data/text_files",      # Directory path
    glob="*.txt",               # Pattern to match
    loader_cls=TextLoader,      # Which loader to use
    loader_kwargs={"encoding": "utf-8"},  # Arguments for loader
    show_progress=False
)
documents = loader.load()
```

**What's happening**:
- `glob="*.txt"`: Only load .txt files (* = wildcard)
- `loader_cls`: Tells it to use TextLoader for each file
- Returns list of ALL documents from directory

**Pro tip**: Use DirectoryLoader for batch processing!

---

### 3.4 Chunking (Text Splitting)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs
```

**What's happening**:

1. **RecursiveCharacterTextSplitter**: Smart splitter that tries different separators
2. **separators=["\n\n", "\n", " ", ""]**: Priority order
   - First tries to split on double newlines (paragraphs)
   - Then single newlines (lines)
   - Then spaces (words)
   - Finally characters (last resort)

**Why "Recursive"?**
- If a chunk is still too big after splitting on `\n\n`, it tries `\n`
- Keeps trying until chunks are small enough

**Visualization**:
```
Original Document (2000 chars)
        ↓
[Chunk 1: 0-1000]
[Chunk 2: 800-1800]  ← Notice overlap!
[Chunk 3: 1600-2000]
```

**The overlap (200 chars) ensures**:
- Context isn't lost at boundaries
- Sentences aren't cut in half

---

### 3.5 Embeddings (The Magic Part!)

```python
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, text: List[str]) -> np.ndarray:
        embeddings = self.model.encode(text, show_progress_bar=True)
        return embeddings
```

**What's happening**:

1. **SentenceTransformer**: Pre-trained model that converts text → vectors
2. **"all-MiniLM-L6-v2"**: Specific model name
   - "MiniLM" = Small, fast model
   - "L6" = 6 layers deep
   - "v2" = Version 2
   - Output: 384-dimensional vectors

3. **`.encode()`**: Does the actual conversion

**Under the hood**:
```
"Python is great"
    ↓ (Neural Network Processing)
[0.234, -0.123, 0.456, ..., 0.789]  (384 numbers)
```

**Why 384 dimensions?**
- Each dimension captures a different aspect of meaning
- More dimensions = more nuanced understanding
- But also slower and more storage

**Model comparison**:
- `all-MiniLM-L6-v2`: Fast, 384 dims, good for most tasks
- `all-mpnet-base-v2`: Slower, 768 dims, more accurate
- `text-embedding-ada-002` (OpenAI): 1536 dims, requires API

---

### 3.6 Vector Store (ChromaDB)

```python
import chromadb

class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", 
                 persist_directory: str = "../data/vector_store"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Collection of PDF document embeddings"}
        )
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        ids = []
        metadatas = []
        documents_text = []
        embedding_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            documents_text.append(doc.page_content)
            embedding_list.append(embedding.tolist())
        
        self.collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=documents_text,
            embeddings=embedding_list
        )
```

**What's happening**:

1. **PersistentClient**: Saves data to disk (not just RAM)
   - Data survives program restarts
   - Stored in `../data/vector_store` directory

2. **Collection**: Like a table in a database
   - Can have multiple collections for different projects
   - `get_or_create_collection`: Creates if doesn't exist, loads if it does

3. **add_documents method**:
   - Prepares data in ChromaDB format
   - `uuid.uuid4()`: Generates unique IDs
   - `zip(documents, embeddings)`: Pairs them together
   - `.tolist()`: Converts numpy array to Python list

**Data structure in ChromaDB**:
```
Collection: "pdf_documents"
├── Document 1
│   ├── id: "doc_a1b2c3d4_0"
│   ├── embedding: [0.2, 0.8, ..., 0.5]
│   ├── metadata: {source: "python.txt", doc_index: 0}
│   └── document: "Python is a high-level..."
├── Document 2
│   ├── id: "doc_e5f6g7h8_1"
│   └── ...
```

**Why we need IDs**:
- Unique identifier for each chunk
- Can update/delete specific documents later
- Track which document a result came from

---

### 3.7 RAG Retriever (The Search Engine)

```python
class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        # 1. Convert query to embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # 2. Search vector store
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
        
        # 3. Process results
        retrieved_docs = []
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        ids = results['ids'][0]
        
        for i, (doc_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            similarity_score = 1 - distance
            
            if similarity_score >= score_threshold:
                retrieved_docs.append({
                    "id": doc_id,
                    "content": document,
                    "metadata": metadata,
                    "similarity_score": similarity_score,
                    "distance": distance,
                    "rank": i + 1
                })
        
        return retrieved_docs
```

**What's happening**:

1. **Query Embedding**:
   ```python
   query = "What are the key features of Python?"
   query_embedding = [0.25, 0.82, ..., 0.51]  # 384 numbers
   ```

2. **Vector Search**:
   - ChromaDB compares query embedding to ALL stored embeddings
   - Uses cosine similarity (measures angle between vectors)
   - Returns `top_k` most similar documents

3. **Distance vs Similarity**:
   - **Distance**: How far apart vectors are (0 = identical, 2 = opposite)
   - **Similarity**: How similar they are (1 = identical, 0 = unrelated)
   - Formula: `similarity = 1 - distance`

**Visualization of Vector Search**:
```
Query Vector: "Python features?"
     ↓
[0.25, 0.82, 0.11, ...]
     ↓
Compare to all stored vectors:
     ↓
Doc 1: distance = 0.74 → similarity = 0.26 ✓ (Most similar!)
Doc 2: distance = 1.23 → similarity = -0.23 ✗ (Not similar)
     ↓
Return top_k results
```

**Parameters explained**:
- `top_k=5`: Return 5 most similar documents
- `score_threshold=0.0`: Minimum similarity score to include
  - 0.0 = include everything
  - 0.5 = only include if 50%+ similar
  - 0.8 = only include if 80%+ similar

---

### 3.8 Putting It All Together (The Full Pipeline)

```python
# Step 1: Load documents
loader = DirectoryLoader("../data/text_files", glob="*.txt", 
                         loader_cls=TextLoader, 
                         loader_kwargs={"encoding": "utf-8"})
documents = loader.load()

# Step 2: Chunk documents
chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)

# Step 3: Generate embeddings
embedding_manager = EmbeddingManager()
texts = [doc.page_content for doc in chunks]
embeddings = embedding_manager.generate_embeddings(texts)

# Step 4: Store in vector database
vector_store = VectorStore()
vector_store.add_documents(chunks, embeddings)

# Step 5: Query and retrieve
rag_retriever = RAGRetriever(vector_store, embedding_manager)
results = rag_retriever.retrieve("What are the key features of Python?")

# Print results
for result in results:
    print(f"Rank: {result['rank']}")
    print(f"Similarity: {result['similarity_score']:.2f}")
    print(f"Content: {result['content'][:100]}...")
    print("---")
```

**Output**:
```
Rank: 1
Similarity: 0.26
Content: Python is a high-level, interpreted programming language.
- Easy to learn and read with clean syntax...
---
```

**Why similarity is 0.26 (low)?**
- The query asks about "features"
- The document lists features but doesn't use the word "features" much
- Still the most relevant document!
- Higher similarity would need more exact wording match

---

## 4. Visual Diagrams

### 4.1 The Complete RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

INDEXING PHASE (Done Once):
┌──────────────┐
│  Documents   │  python.txt, c.txt
│  (Raw Files) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Loader     │  TextLoader / DirectoryLoader
│              │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Documents   │  [Document(page_content="...", metadata={...}), ...]
│  (Structured)│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Chunker    │  RecursiveCharacterTextSplitter
│              │  chunk_size=1000, overlap=200
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Chunks    │  [Chunk1, Chunk2, Chunk3, ...]
│              │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Embedding   │  SentenceTransformer
│    Model     │  "all-MiniLM-L6-v2"
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Embeddings  │  [[0.2, 0.8, ...], [0.3, 0.7, ...], ...]
│  (Vectors)   │  384 dimensions each
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Vector Store │  ChromaDB
│  (Database)  │  Persistent storage
└──────────────┘

RETRIEVAL PHASE (Every Query):
┌──────────────┐
│  User Query  │  "What are Python's features?"
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Embedding   │  Convert query to vector
│    Model     │  [0.25, 0.82, ...]
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Vector Store │  Search for similar vectors
│   Search     │  Cosine similarity
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Top-K      │  Return most relevant chunks
│  Documents   │  with similarity scores
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Results    │  [{"content": "...", "score": 0.85}, ...]
└──────────────┘
```

### 4.2 How Embeddings Work

```
TEXT → EMBEDDING MODEL → VECTOR

"Python is great"
       ↓
┌─────────────────┐
│ Neural Network  │  (Trained on billions of text examples)
│  - Tokenizer    │  Breaks text into pieces
│  - Transformer  │  Understands context
│  - Pooling      │  Combines into single vector
└─────────────────┘
       ↓
[0.234, -0.123, 0.456, 0.789, ..., 0.321]
 ↑       ↑       ↑       ↑           ↑
 |       |       |       |           |
Dim 1  Dim 2   Dim 3   Dim 4  ...  Dim 384

Each dimension captures different semantic aspects:
- Dim 1: Programming-related?
- Dim 2: Positive sentiment?
- Dim 3: Technical complexity?
- ... (384 total dimensions)
```

### 4.3 Vector Similarity Search

```
QUERY: "Python features"
EMBEDDING: [0.25, 0.82, 0.11, ...]

                    Vector Space (simplified to 2D)
                    
                    │
              Doc2  │  Query
                 ●  │   ★
                    │      ● Doc1
                    │     (Closest!)
        ────────────┼────────────
                    │
                    │  ● Doc3
                    │
                    │

Distance Calculation (Cosine Similarity):
- Doc1: 0.74 distance → 0.26 similarity ✓ Best match
- Doc2: 1.23 distance → -0.23 similarity
- Doc3: 1.45 distance → -0.45 similarity

Return: Doc1 (Python content)
```

### 4.4 Chunking Strategy

```
ORIGINAL DOCUMENT (2500 characters):

┌────────────────────────────────────────────────────────────┐
│ Python is a high-level programming language...             │
│ [paragraph 1 - 800 chars]                                  │
│                                                            │
│ Python has many features including...                      │
│ [paragraph 2 - 900 chars]                                  │
│                                                            │
│ Python is used in many domains...                          │
│ [paragraph 3 - 800 chars]                                  │
└────────────────────────────────────────────────────────────┘

AFTER CHUNKING (chunk_size=1000, overlap=200):

┌─────────────────────────────────────┐
│ CHUNK 1 (chars 0-1000)              │
│ Python is a high-level...           │
│ [paragraph 1 + part of paragraph 2] │
└─────────────────────────────────────┘
                    ↓ 200 char overlap
┌─────────────────────────────────────┐
│ CHUNK 2 (chars 800-1800)            │
│ ...features including...            │
│ [end of para 1 + para 2 + start 3]  │
└─────────────────────────────────────┘
                    ↓ 200 char overlap
┌─────────────────────────────────────┐
│ CHUNK 3 (chars 1600-2500)           │
│ ...used in many domains...          │
│ [end of paragraph 2 + paragraph 3]  │
└─────────────────────────────────────┘

WHY OVERLAP?
- Prevents losing context at boundaries
- Ensures complete sentences/ideas
- Better retrieval accuracy
```

---

## 5. Professional Tips

### 5.1 Choosing Chunk Size

**Small chunks (200-500 chars)**:
- ✅ More precise retrieval
- ✅ Less noise in results
- ❌ May lose context
- ❌ More chunks = slower indexing

**Medium chunks (500-1500 chars)**:
- ✅ Good balance
- ✅ Preserves context
- ✅ Works for most use cases
- **Recommended for beginners**

**Large chunks (1500-3000 chars)**:
- ✅ Maximum context
- ❌ Less precise
- ❌ May include irrelevant info

**Rule of thumb**: 
- Technical docs: 500-1000 chars
- Books/articles: 1000-2000 chars
- Code: 200-500 chars

### 5.2 Choosing Embedding Models

| Model | Dimensions | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| all-MiniLM-L6-v2 | 384 | ⚡⚡⚡ | ⭐⭐ | Development, testing |
| all-mpnet-base-v2 | 768 | ⚡⚡ | ⭐⭐⭐ | Production, better quality |
| text-embedding-ada-002 | 1536 | ⚡ | ⭐⭐⭐⭐ | Best quality, requires OpenAI API |

**Your notebook uses**: `all-MiniLM-L6-v2` (good choice for learning!)

### 5.3 Improving Retrieval Quality

**1. Adjust top_k**:
```python
# Too few: might miss relevant docs
results = retriever.retrieve(query, top_k=1)  

# Too many: includes irrelevant docs
results = retriever.retrieve(query, top_k=20)

# Sweet spot: 3-5 for most cases
results = retriever.retrieve(query, top_k=5)
```

**2. Use score threshold**:
```python
# Only return highly relevant results
results = retriever.retrieve(query, top_k=10, score_threshold=0.5)
```

**3. Better chunking**:
```python
# Experiment with different sizes
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Try different values
    chunk_overlap=150,   # 15-20% of chunk_size
    separators=["\n\n", "\n", ". ", " ", ""]  # Add ". " for sentences
)
```

**4. Metadata filtering**:
```python
# Search only specific sources
results = vector_store.collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"source": "../data/text_files/python.txt"}  # Filter by metadata
)
```

### 5.4 Common Pitfalls

**❌ Mistake 1: Not normalizing text**
```python
# Bad: Mixed case, extra spaces
text = "  Python  IS  great!  "

# Good: Clean text
text = text.strip().lower()  # "python is great!"
```

**❌ Mistake 2: Forgetting to persist**
```python
# Bad: Data lost when program ends
client = chromadb.Client()

# Good: Data saved to disk
client = chromadb.PersistentClient(path="./vector_store")
```

**❌ Mistake 3: Not handling errors**
```python
# Bad: Crashes on missing file
documents = loader.load()

# Good: Graceful error handling
try:
    documents = loader.load()
except Exception as e:
    print(f"Error loading documents: {e}")
    documents = []
```

**❌ Mistake 4: Ignoring metadata**
```python
# Bad: No source tracking
doc = Document(page_content="Python is great")

# Good: Track source
doc = Document(
    page_content="Python is great",
    metadata={"source": "python.txt", "date": "2024-01-01"}
)
```

### 5.5 Performance Optimization

**1. Batch processing**:
```python
# Bad: One at a time (slow)
for doc in documents:
    embedding = model.encode([doc.page_content])
    vector_store.add(embedding)

# Good: All at once (fast)
texts = [doc.page_content for doc in documents]
embeddings = model.encode(texts)  # Batch encoding
vector_store.add_documents(documents, embeddings)
```

**2. Caching embeddings**:
```python
# Save embeddings to avoid recomputing
np.save("embeddings.npy", embeddings)

# Load later
embeddings = np.load("embeddings.npy")
```

**3. Use GPU if available**:
```python
# Automatically uses GPU if available
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
```

### 5.6 Next Steps (Becoming a Pro)

**Level 1: You are here! ✓**
- Understand basic RAG pipeline
- Load, chunk, embed, store, retrieve

**Level 2: Add an LLM**
- Integrate OpenAI/Anthropic/Local LLM
- Generate answers using retrieved context
- Implement prompt engineering

**Level 3: Advanced Retrieval**
- Hybrid search (keyword + semantic)
- Re-ranking retrieved results
- Query expansion/rewriting

**Level 4: Production**
- API deployment (FastAPI)
- Monitoring and logging
- Scaling with cloud vector DBs
- A/B testing different strategies

**Level 5: Cutting Edge**
- Fine-tuning embedding models
- Multi-modal RAG (text + images)
- Agentic RAG (autonomous retrieval)
- Graph RAG (knowledge graphs)

---

## 6. Hands-On Exercises

### Exercise 1: Modify Chunk Size
Try different chunk sizes and see how it affects retrieval:
```python
# Test with small chunks
chunks_small = split_documents(documents, chunk_size=200, chunk_overlap=50)

# Test with large chunks
chunks_large = split_documents(documents, chunk_size=2000, chunk_overlap=400)

# Compare results
```

### Exercise 2: Add More Documents
Create a new text file about JavaScript and add it to the system:
```python
sample_texts["../data/text_files/javascript.txt"] = """
JavaScript is a high-level, interpreted programming language.
- Runs in web browsers and Node.js
- Event-driven and asynchronous
- Dynamic typing with prototype-based objects
...
"""
```

### Exercise 3: Experiment with Queries
Try different types of queries:
```python
queries = [
    "What is Python?",
    "memory management",
    "web development",
    "compiled vs interpreted"
]

for query in queries:
    results = rag_retriever.retrieve(query, top_k=2)
    print(f"Query: {query}")
    print(f"Top result: {results[0]['content'][:100]}...")
    print(f"Similarity: {results[0]['similarity_score']:.2f}\n")
```

### Exercise 4: Build a Simple Q&A System
Combine retrieval with a simple response:
```python
def answer_question(query):
    results = rag_retriever.retrieve(query, top_k=3)
    
    if not results:
        return "I don't have information about that."
    
    # Simple answer: return most relevant chunk
    best_match = results[0]
    return f"Based on {best_match['metadata']['source']}:\n{best_match['content']}"

# Test it
print(answer_question("What are Python's features?"))
```

---

## 7. Glossary

**Embedding**: A vector (list of numbers) representing text in a way that captures semantic meaning.

**Vector**: A list of numbers, like [0.2, 0.8, 0.1, ...]. In RAG, vectors represent text.

**Semantic**: Related to meaning. Semantic search finds documents by meaning, not just keywords.

**Cosine Similarity**: A measure of how similar two vectors are (0 = unrelated, 1 = identical).

**Chunk**: A piece of a larger document. Chunking breaks documents into smaller, manageable pieces.

**Metadata**: Data about data. For documents: source file, author, date, etc.

**Token**: A piece of text (word, subword, or character) that models process.

**Retrieval**: The process of finding relevant documents from a database.

**Augmented**: Enhanced or improved. In RAG, we augment LLM responses with retrieved information.

**Generation**: Creating new text. LLMs generate responses based on prompts and context.

**Vector Store/Database**: A specialized database optimized for storing and searching vectors.

**Top-K**: The K most similar/relevant results. K=5 means return the 5 best matches.

**Overlap**: In chunking, the number of characters shared between consecutive chunks.

---

## 8. Resources for Further Learning

### Documentation
- [LangChain Docs](https://python.langchain.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

### Tutorials
- LangChain RAG Tutorial (official)
- Building Production RAG Systems (YouTube)
- Advanced RAG Techniques (blog posts)

### Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (original RAG paper)
- "Dense Passage Retrieval for Open-Domain Question Answering"

### Communities
- LangChain Discord
- r/MachineLearning (Reddit)
- Hugging Face Forums

---

## 9. Summary

**What you learned**:
1. ✅ What RAG is and why it matters
2. ✅ The 5 core components: Load, Chunk, Embed, Store, Retrieve
3. ✅ How embeddings convert text to vectors
4. ✅ How vector databases enable semantic search
5. ✅ How to implement a complete RAG pipeline
6. ✅ Best practices and optimization techniques

**Key takeaways**:
- RAG grounds LLM responses in real documents
- Embeddings capture semantic meaning as vectors
- Chunking balances context and precision
- Vector stores enable fast similarity search
- The pipeline is: Document → Chunks → Embeddings → Vector Store → Retrieval

**You're now ready to**:
- Build your own RAG systems
- Experiment with different configurations
- Integrate LLMs for question answering
- Scale to production use cases

---

**Questions? Experiments? Keep learning! 🚀**

Remember: The best way to learn is by doing. Modify the code, break things, fix them, and understand why they work!
