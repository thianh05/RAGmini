import os
import sys
import time
import ollama
import logging
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from retrieval import HybridRetriever

# LOGIN - CAU HINH HE THONG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

class RAGConfig:
    """Quản lý hằng số và cấu hình hệ thống (Chuẩn Enterprise)"""
    MODEL = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:1.5b")
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INDEX_PATH = os.path.join(BASE_DIR, "data", "triet.index")
    META_PATH = os.path.join(BASE_DIR, "data", "triet_embeddings.json")
    
    TEMP = 0.15 
    TOP_K = 4

# RAG ENGINE - NÃO HỆ THỐNG
class PhilosophyRAG:
    def __init__(self):
        self.client = ollama.Client(host=RAGConfig.BASE_URL)
        try:
            # Load model trước - sau đó truyền vào hybrid retriever
            logger.info("Đang khởi tạo Embedding Model (E5)...")
            self.embed_model = SentenceTransformer('intfloat/multilingual-e5-small')
            
            logger.info("Đang nạp dữ liệu vào Hybrid Retriever...")
            self.retriever = HybridRetriever(RAGConfig.INDEX_PATH, RAGConfig.META_PATH, self.embed_model)
            
            logger.info(" Database initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            sys.exit(1)

        # SYSTEM PROMPT
        self.system_instructions = (
            "BẠN LÀ CHUYÊN GIA TRIẾT HỌC MÁC - LÊNIN CAO CẤP.\n"
            "NGUYÊN TẮC BẮT BUỘC:\n"
            "1. ĐỊNH NGHĨA CHUẨN: CSHT là Quan hệ sản xuất; KTTT là Tư tưởng & Thiết chế chính trị.\n"
            "2. QUY TRÌNH TƯ DUY: [Kiểm tra tài liệu] -> [Trích dẫn số trang] -> [Phân tích logic] -> [Kết luận].\n"
            "3. TRUNG THỰC: Nếu tài liệu không chứa câu trả lời, hãy nói: 'Dữ liệu giáo trình hiện tại không đủ để kết luận'.\n"
            "4. KHÔNG SUY DIỄN: Tuyệt đối không đưa kiến thức ngoài luồng vào nếu nó xung đột với giáo trình."
        )

    def _get_hyde_query(self, query: str) -> str:
        """Kỹ thuật HyDE (Hypothetical Document Embeddings) cấp cao"""
        try:
            prompt = f"Tóm tắt định nghĩa ngắn gọn về thuật ngữ này để hỗ trợ tìm kiếm: {query}"
            res = self.client.generate(model=RAGConfig.MODEL, prompt=prompt)
            return f"{query} {res['response']}"
        except Exception as e:
            logger.warning(f"HyDE failed: {e}")
            return query

    def generate_answer(self, user_query: str, chat_history: List[Dict]) -> Tuple[str, List]:
        # STEP 1: Query Enhancement
        enhanced_search = self._get_hyde_query(user_query)
        
        # STEP 2: Retrieval & Filtering
        logger.info(f"Retrieving context for: {user_query[:50]}...")
        filtered_docs = self.retriever.search(enhanced_search, top_k=RAGConfig.TOP_K)
        
        if not filtered_docs:
            return "Xin lỗi, tôi không tìm thấy nội dung tương ứng trong giáo trình để trả lời chính xác câu hỏi này.", []

        # STEP 3: Build Context & Prompt
        context_text = "\n\n".join([f"--- TRANG {d['page']} ---\n{d['text']}" for d in filtered_docs])
        
        messages = [
            {'role': 'system', 'content': self.system_instructions},
            *chat_history[-4:], 
            {'role': 'user', 'content': f"[NGỮ CẢNH GIÁO TRÌNH]:\n{context_text}\n\n[CÂU HỎI]: {user_query}"}
        ]

        # STEP 4: Streaming & Execution
        print(f"\n[Giáo sư AI - {RAGConfig.MODEL}]:\n" + "═"*50)
        
        full_response = ""
        try:
            stream = self.client.chat(
                model=RAGConfig.MODEL, 
                messages=messages, 
                stream=True,
                options={"temperature": RAGConfig.TEMP, "num_ctx": 4096}
            )
            
            for chunk in stream:
                token = chunk['message']['content']
                print(token, end="", flush=True)
                full_response += token
            
            print("\n" + "═"*50)
            
            # CAP NHAT LICH SU
            chat_history.append({'role': 'user', 'content': user_query})
            chat_history.append({'role': 'assistant', 'content': full_response})
            
            # METADATA OUTPUT
            pages = sorted(list(set([d['page'] for d in filtered_docs])))
            print(f"Nguồn trích dẫn: Trang {', '.join(map(str, pages))}")
            
            return full_response, filtered_docs

        except Exception as e:
            logger.error(f"Inference Error: {e}")
            return "Hệ thống gặp sự cố khi tạo phản hồi.", []

# 3. INTERFACE (THE MAIN ENTRY)

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    engine = PhilosophyRAG()
    history = []    
    print("\nTip: Nhập 'exit' để thoát, 'clear' để xóa lịch sử chat.")
    
    while True:
        try:
            prompt = input("\n Hãy hỏi gì đó: ").strip()
            
            if not prompt: continue
            if prompt.lower() in ['exit', 'quit']: break
            if prompt.lower() == 'clear':
                history = []
                print("Đã xóa lịch sử hội thoại.")
                continue
                
            engine.generate_answer(prompt, history)
            
        except KeyboardInterrupt:
            print("\nShutting down...")
            break