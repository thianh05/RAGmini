# RAGstudio 
<p align="center">
  <img src="assets/demo.png" alt="RAG Philosophy Bot Preview" width="800">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/FAISS-Vector_Database-51cf66?style=for-the-badge" alt="FAISS">
</p>

---

## 🌟 Overview
**RAGstudio** is a sophisticated, **100% Offline** Retrieval-Augmented Generation (RAG) pipeline designed for complex academic domains like Philosophy. This project tackles the common pitfalls of Small Language Models (SLMs) by utilizing a Hybrid Retrieval strategy and strict prompt engineering to eliminate hallucinations and ensure high-fidelity responses.

## Key Engineering Highlights
*   **Hybrid Retrieval Engine:** Combines **FAISS (Dense Search)** for deep semantic understanding with **BM25 (Sparse Search)** for precise keyword matching. Results are fused using the **Reciprocal Rank Fusion (RRF)** algorithm.
*   **Fully Asynchronous Pipeline:** Architected with **FastAPI** and `ollama.AsyncClient` to support non-blocking concurrent requests and real-time token streaming.
*   **Hallucination Control:** Features a "Direct Strike" system prompt that mandates **Inline Citations** (e.g., *Page 15*) and forces the model to decline queries outside the provided knowledge base.
*   **Semantic Data ETL:** A custom ETL pipeline that handles PDF noise reduction, automatic line-break healing, and semantic-aware chunking for optimal embedding quality.

---

## System Architecture

```mermaid
graph TD
    subgraph Offline_ETL [Offline: Data Ingestion & Indexing]
        A[Raw PDF Documents] --> B(Clean & Semantic Chunking)
        B --> C[HuggingFace Embeddings]
        C --> D[(FAISS Vector Index)]
        B --> E[(BM25 Inverted Index)]
    end

    subgraph Online_Inference [Online: Real-time QA]
        F[User Query] --> G[Streamlit Frontend]
        G --> H(FastAPI Backend)
        H --> I{Hybrid Searcher}
        I -->|Semantic| D
        I -->|Keyword| E
        D & E --> J[RRF Fusion Re-ranking]
        J --> K(Strict Prompt Builder)
        K --> L[Ollama Local LLM]
        L -->|Streaming Tokens| G
    end
    
    style L fill:#f06595,stroke:#333,stroke-width:2px
    style D fill:#51cf66,stroke:#333,stroke-width:2px
    style E fill:#51cf66,stroke:#333,stroke-width:2px
