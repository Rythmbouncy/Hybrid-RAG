# Hybrid RAG Pipeline: BM25 (Sparse) + Vector Search (Dense)

This project demonstrates a robust Retrieval-Augmented Generation (RAG) pipeline that combines keyword-based (BM25) and semantic (Vector) search.

## 1. Key Components

| **Dataset** | `zeroshot/arxiv-biology` | 
| **Embedding Model** | `all-MiniLM-L6-v2` | 
| **Vector Store** | Pinecone | 
| **LLM** | Google Gemini 2.5 Flash |

## 2. Setup and Installation

### 2.1. Prerequisites

* Python (3.9+)
* Pinecone API Key
* Gemini API Key

### 2.2. Installation Steps

1.  **Clone the project structure:**
    ```bash
    git clone <https://github.com/Rythmbouncy/Hybrid-RAG.git>
    cd hybrid_rag_project
    ```
2.  **Install dependencies:**
    Create a virtual environment and install requirements.
    ```bash
    pip install -r requirements.txt
    ```

### 2.3. Configuration (`.env`)

Create a `.env` file in the project root to store necessary credentials and settings:

```env
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

INDEX_NAME="hybrid-arxiv-bio"
EMBEDDING_MODEL="all-MiniLM-L6-v2"
EMBEDDING_DIMENSION=384

PINECONE_CLOUD="aws"              
PINECONE_REGION="us-east-1"