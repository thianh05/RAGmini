import os
import sys
import time
import ollama
import logging

# Typing cho code dễ đọc hơn
from typing import List, Dict, Any, Tuple

# Load biến môi trường từ file .env
from dotenv import load_dotenv

# Embedding model
from sentence_transformers import SentenceTransformer

# Hybrid Retriever tự viết
from retrieval import HybridRetriever


# =====================================================
# 1. LOGGING + ENV
# =====================================================

# Cấu hình log hệ thống
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Load biến môi trường từ .env
load_dotenv()


# =====================================================
# 2. CONFIG CLASS
# =====================================================
# Chứa toàn bộ cấu hình hệ thống
#
# Ưu điểm:
# - gom config 1 nơi
# - dễ chỉnh sửa
# - dễ scale project
class RAGConfig:

    # =================================================
    # OLLAMA CONFIG
    # =================================================

    # Tên model
    #
    # getenv:
    # -> đọc từ file .env
    # -> nếu không có thì dùng default
    MODEL = os.getenv(
        "OLLAMA_MODEL_NAME",
        "qwen2.5:1.5b"
    )

    # URL server Ollama
    BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    )

    # =================================================
    # PATH CONFIG
    # =================================================

    # Thư mục gốc project
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    # File FAISS index
    INDEX_PATH = os.path.join(
        BASE_DIR,
        "data",
        "triet.index"
    )

    # File metadata embedding
    META_PATH = os.path.join(
        BASE_DIR,
        "data",
        "triet_embeddings.json"
    )

    # =================================================
    # MODEL PARAMS
    # =================================================

    # Temperature thấp
    # -> output ổn định hơn
    TEMP = 0.15

    # Số document retrieve
    TOP_K = 4


# =====================================================
# 3. RAG ENGINE
# =====================================================
# Đây là "bộ não" của toàn hệ thống
#
# Chức năng:
# - retrieval
# - prompt building
# - gọi LLM
# - streaming
# - quản lý lịch sử chat
class PhilosophyRAG:

    # =================================================
    # INIT
    # =================================================
    def __init__(self):

        # Kết nối Ollama local
        self.client = ollama.Client(
            host=RAGConfig.BASE_URL
        )

        try:

            # =============================================
            # LOAD EMBEDDING MODEL
            # =============================================
            logger.info(
                "Đang khởi tạo Embedding Model (E5)..."
            )

            # multilingual-e5-small
            #
            # embedding model đa ngôn ngữ
            self.embed_model = SentenceTransformer(
                'intfloat/multilingual-e5-small'
            )

            # =============================================
            # LOAD HYBRID RETRIEVER
            # =============================================
            logger.info(
                "Đang nạp dữ liệu vào Hybrid Retriever..."
            )

            self.retriever = HybridRetriever(
                RAGConfig.INDEX_PATH,
                RAGConfig.META_PATH,
                self.embed_model
            )

            logger.info(
                "Database initialized successfully."
            )

        except Exception as e:

            logger.error(
                f"Failed to load database: {e}"
            )

            # Dừng chương trình nếu load fail
            sys.exit(1)

        # =================================================
        # SYSTEM PROMPT
        # =================================================
        # Prompt ép AI:
        # - đúng giáo trình
        # - không hallucination
        # - có logic suy luận
        self.system_instructions = (

            "BẠN LÀ CHUYÊN GIA TRIẾT HỌC MÁC - LÊNIN CAO CẤP.\n"

            "NGUYÊN TẮC BẮT BUỘC:\n"

            "1. ĐỊNH NGHĨA CHUẨN: "
            "CSHT là Quan hệ sản xuất; "
            "KTTT là Tư tưởng & Thiết chế chính trị.\n"

            "2. QUY TRÌNH TƯ DUY: "
            "[Kiểm tra tài liệu] -> "
            "[Trích dẫn số trang] -> "
            "[Phân tích logic] -> "
            "[Kết luận].\n"

            "3. TRUNG THỰC: "
            "Nếu tài liệu không chứa câu trả lời, "
            "hãy nói: "
            "'Dữ liệu giáo trình hiện tại không đủ để kết luận'.\n"

            "4. KHÔNG SUY DIỄN: "
            "Tuyệt đối không đưa kiến thức ngoài luồng "
            "vào nếu nó xung đột với giáo trình."
        )

    # =================================================
    # HYDE QUERY
    # =================================================
    # HyDE = Hypothetical Document Embeddings
    #
    # Ý tưởng:
    # - dùng LLM tạo "định nghĩa giả"
    # - ghép vào query
    # - retrieval semantic tốt hơn
    def _get_hyde_query(self, query: str) -> str:

        try:

            # Prompt cho LLM
            prompt = (
                "Tóm tắt định nghĩa ngắn gọn "
                f"về thuật ngữ này để hỗ trợ tìm kiếm: {query}"
            )

            # Generate text
            res = self.client.generate(
                model=RAGConfig.MODEL,
                prompt=prompt
            )

            # Query mở rộng
            return f"{query} {res['response']}"

        except Exception as e:

            logger.warning(f"HyDE failed: {e}")

            # Fail thì dùng query gốc
            return query

    # =================================================
    # GENERATE ANSWER
    # =================================================
    def generate_answer(
        self,
        user_query: str,
        chat_history: List[Dict]
    ) -> Tuple[str, List]:

        # =================================================
        # STEP 1: QUERY ENHANCEMENT
        # =================================================
        # Dùng HyDE để mở rộng query
        enhanced_search = self._get_hyde_query(
            user_query
        )

        # =================================================
        # STEP 2: RETRIEVAL
        # =================================================
        logger.info(
            f"Retrieving context for: {user_query[:50]}..."
        )

        # Search document liên quan
        filtered_docs = self.retriever.search(
            enhanced_search,
            top_k=RAGConfig.TOP_K
        )

        # Không tìm thấy document
        if not filtered_docs:

            return (
                "Xin lỗi, tôi không tìm thấy nội dung "
                "tương ứng trong giáo trình để trả lời "
                "chính xác câu hỏi này.",
                []
            )

        # =================================================
        # STEP 3: BUILD CONTEXT
        # =================================================

        # Ghép toàn bộ context
        context_text = "\n\n".join([

            f"--- TRANG {d['page']} ---\n{d['text']}"

            for d in filtered_docs
        ])

        # =================================================
        # BUILD CHAT MESSAGE
        # =================================================
        messages = [

            # System Prompt
            {
                'role': 'system',
                'content': self.system_instructions
            },

            # Lấy 4 message gần nhất
            *chat_history[-4:],

            # User Prompt cuối cùng
            {
                'role': 'user',
                'content':
                    f"[NGỮ CẢNH GIÁO TRÌNH]:\n"
                    f"{context_text}\n\n"
                    f"[CÂU HỎI]: {user_query}"
            }
        ]

        # =================================================
        # STEP 4: STREAMING
        # =================================================

        print(
            f"\n[Giáo sư AI - {RAGConfig.MODEL}]:\n"
            + "═" * 50
        )

        full_response = ""

        try:

            # Gọi Ollama Chat API
            stream = self.client.chat(

                model=RAGConfig.MODEL,

                messages=messages,

                # Stream realtime
                stream=True,

                options={

                    # Độ sáng tạo
                    "temperature": RAGConfig.TEMP,

                    # Context window
                    "num_ctx": 4096
                }
            )

            # =============================================
            # STREAM TOKEN
            # =============================================
            for chunk in stream:

                # Token mới sinh
                token = chunk['message']['content']

                # In realtime ra terminal
                print(token, end="", flush=True)

                # Lưu full response
                full_response += token

            print("\n" + "═" * 50)

            # =================================================
            # UPDATE CHAT HISTORY
            # =================================================

            # Lưu câu hỏi user
            chat_history.append({
                'role': 'user',
                'content': user_query
            })

            # Lưu câu trả lời assistant
            chat_history.append({
                'role': 'assistant',
                'content': full_response
            })

            # =================================================
            # METADATA OUTPUT
            # =================================================

            # Lấy danh sách page
            pages = sorted(list(set([
                d['page']
                for d in filtered_docs
            ])))

            # In source page
            print(
                f"Nguồn trích dẫn: "
                f"Trang {', '.join(map(str, pages))}"
            )

            return full_response, filtered_docs

        except Exception as e:

            logger.error(f"Inference Error: {e}")

            return (
                "Hệ thống gặp sự cố khi tạo phản hồi.",
                []
            )


# =====================================================
# 4. MAIN PROGRAM
# =====================================================
# Entry point của chương trình
if __name__ == "__main__":

    # =================================================
    # CLEAR TERMINAL
    # =================================================
    os.system(
        'cls' if os.name == 'nt'
        else 'clear'
    )

    # =================================================
    # INIT ENGINE
    # =================================================
    engine = PhilosophyRAG()

    # Lưu lịch sử chat
    history = []

    print(
        "\nTip: "
        "Nhập 'exit' để thoát, "
        "'clear' để xóa lịch sử chat."
    )

    # =================================================
    # CHAT LOOP
    # =================================================
    while True:

        try:

            # Input user
            prompt = input(
                "\n Hãy hỏi gì đó: "
            ).strip()

            # Empty input
            if not prompt:
                continue

            # Exit app
            if prompt.lower() in ['exit', 'quit']:
                break

            # Clear history
            if prompt.lower() == 'clear':

                history = []

                print(
                    "Đã xóa lịch sử hội thoại."
                )

                continue

            # Generate answer
            engine.generate_answer(
                prompt,
                history
            )

        except KeyboardInterrupt:

            print("\nShutting down...")

            break