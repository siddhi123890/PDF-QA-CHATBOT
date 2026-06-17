# DocuQuery — PDF Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly from uploaded PDF documents. Built with LangChain, FAISS, and Streamlit.

## Features

- **PDF Upload & Indexing** — Upload any PDF and it gets chunked, embedded, and indexed automatically
- **Strict PDF-Only Answers** — Uses a multi-layered system to ensure answers come only from the document, not the LLM's training data
- **Source Citations** — Every answer shows the exact page numbers and text excerpts used
- **Conversational UI** — Clean chat interface with message history
- **Free to Run** — Uses Groq's free API tier for LLM inference and local embeddings

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Llama 3.1 8B (via Groq Cloud) |
| **Embeddings** | all-MiniLM-L6-v2 (local, HuggingFace) |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Framework** | LangChain |
| **Frontend** | Streamlit |
| **PDF Parser** | PyPDF |

## How It Works

```
PDF → Text Extraction → Chunking → Embeddings → FAISS Index
                                                      ↓
User Question → Embedding → Similarity Search → Top Chunks → LLM → Answer
```

1. **Ingest** — PDF is loaded and split into overlapping 1000-character chunks
2. **Embed** — Each chunk is converted to a vector using MiniLM-L6-v2
3. **Index** — Vectors are stored in a FAISS index for fast similarity search
4. **Retrieve** — User's question is embedded and matched against the index
5. **Generate** — Top matching chunks are passed to Llama 3.1 as context, which generates an answer

### Anti-Hallucination System

The chatbot uses three layers to prevent out-of-PDF answers:

- **Relevance Gate** — FAISS similarity scores are checked before calling the LLM. If no chunk is relevant enough, the question is rejected
- **Strict Prompt** — The LLM is instructed it has "zero general knowledge" and can only use the provided excerpts
- **Post-Processing Filter** — Output is scanned for hedging phrases that indicate outside knowledge

## Setup

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys)

### Installation

```bash
# Clone the repo
git clone https://github.com/siddhi123890/PDF-QA-CHATBOT.git
cd PDF-QA-CHATBOT

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key from [console.groq.com/keys](https://console.groq.com/keys).

### Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Project Structure

```
├── app.py              # Streamlit UI
├── ingest.py           # PDF loading and text chunking
├── vectorstore.py      # FAISS vector store creation
├── qa_chain.py         # RAG chain with anti-hallucination
├── requirements.txt    # Python dependencies
├── .env                # API keys (not tracked by git)
└── .gitignore
```

## Usage

1. Open the app in your browser
2. Upload a PDF using the sidebar
3. Wait for indexing to complete (you'll see page count and chunk count)
4. Ask questions in the chat input
5. Expand "Sources" under any answer to see which pages were referenced

## License

MIT
