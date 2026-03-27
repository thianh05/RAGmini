import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# QUẢN LÍ ĐƯỜNG DẪN TỰ ĐỘNG
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# TẬP HỢP DỮ LIỆU
PDF_PATH = DATA_DIR / "triet.pdf"
CLEAN_JSON = DATA_DIR / "triet_clean.json"
CHUNK_JSON = DATA_DIR / "triet_chunks.json"
INDEX_PATH = DATA_DIR / "triet.index"
META_PATH = DATA_DIR / "triet_embeddings.json"

# CẤU HÌNH MODEL
EMBEDDING_MODEL = 'intfloat/multilingual-e5-small'
LLM_MODEL = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:1.5b")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")