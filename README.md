#  RAGstudio
An end-to-end, fully asynchronous Retrieval-Augmented Generation (RAG) pipeline designed for high-performance document QA. Built from scratch to run entirely on local hardware (Ollama / Local LLMs), focusing on memory safety, zero-hallucination prompt engineering, and a polished Apple-style UI.
## Key Engineering Highlights
Unlike standard tutorial-level RAG apps, this project tackles real-world systemic bottlenecks:
* **Advanced Semantic Chunking:** Replaced naive character splitters with a custom RegEx-powered ETL pipeline. It automatically heals broken PDF line-breaks, hyphenations, and cleans null bytes before embedding, ensuring high-fidelity semantic vectors.
* **High-Speed Vector Search:** Utilized **FAISS** with `IndexFlatIP` (Inner Product) combined with L2 Normalization (`faiss.normalize_L2`), transforming the vector search into a blazing-fast O(1) operation optimized for Cosine Similarity.
* **Fully Asynchronous Streaming Backend:** Built with **FastAPI** and `ollama.AsyncClient`. Solves the classic Event-Loop blocking issue, allowing the server to handle concurrent user requests without freezing, streaming tokens in real-time.
* **Bulletproof Frontend (Streamlit Hack):** Bypassed standard Streamlit UI limitations using custom injected CSS/JS.
    * **Anti-Lag Auto-Scroll:** Implemented a Javascript `MutationObserver` with a **Debounce algorithm (100ms)** to prevent CPU spikes during high-speed token streaming.
    * **Tag Leakage Prevention:** Custom Python buffer logic ensures XML/HTML tags (like `<SOURCES>`) are parsed safely in the background and never leak onto the UI.
* **Strict Hallucination Control:** Engineered an adaptive "Direct Strike" System Prompt that forces the LLM to perform **Inline Citations** (e.g., *Page 15*) and outright reject out-of-domain queries.

## Technology Stack

* **UI / Frontend:** Streamlit, Custom CSS (Apple Vibe / Be Vietnam Pro font), Vanilla JS
* **Backend / API:** FastAPI, Uvicorn, AsyncIO
* **AI / LLM:** Ollama (Qwen 1.5B / Llama 3), LangChain (Optional wrapping)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** Hugging Face (`intfloat/multilingual-e5-small` / `MiniLM`)

## System Architecture Flow

1.  **ETL Phase:** `load_pdf.py` -> `clean_pdf.py` (Noise Reduction) -> `chunk_pdf.py` (Semantic Splitting).
2.  **Indexing Phase:** `embedding.py` batches text chunks through HuggingFace models -> L2 Normalization -> FAISS Index storage.
3.  **Inference Phase (Async):** User Query -> `retrieval.py` fetches Top-K chunks -> FastAPI constructs strict Prompt -> Ollama streams response back to Frontend.
## RAG Pipeline Architecture

```mermaid
graph TD
    %% Colors & Styles
    classDef offline fill:#f8f9fa,stroke:#ced4da,stroke-width:2px;
    classDef online fill:#e7f5ff,stroke:#339af0,stroke-width:2px;
    classDef db fill:#ebfbee,stroke:#51cf66,stroke-width:2px;
    classDef llm fill:#fff0f6,stroke:#f06595,stroke-width:2px;

    subgraph Offline ETL Pipeline [Data Ingestion & Indexing]
        A[📄 Raw PDF Documents] --> B(PyPDF Loader)
        B --> C{Semantic Text Splitter}
        C -->|Text Chunks| D[Hugging Face Model<br>multilingual-e5-small]
        D -->|Vector Embeddings| E[(FAISS Vector Index)]
    end

    subgraph Online Inference Pipeline [Real-time QA]
        F[👤 User Query] --> G[💻 Streamlit Frontend]
        G --> H(⚡ FastAPI Backend)
        H --> I[Hugging Face Model]
        I -->|Query Vector| E
        E -->|Top-K Relevant Chunks| J(Strict Prompt Builder)
        J --> K[🧠 Ollama Local LLM<br>Qwen 1.5B / Llama 3]
        K -->|Streaming Tokens| G
    end

    %% Apply Styles
    class A,B,C,D offline;
    class F,G,H,I,J online;
    class E db;
    class K llm;
