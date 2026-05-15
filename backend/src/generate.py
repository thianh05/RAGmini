import os
import sys
import time
import ollama
import logging

from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from retrieval import HybridRetriever

# [1] Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# [1.1] Load biến môi trường từ file .env
load_dotenv()


# [2] Class quản lý toàn bộ config hệ thống
class RAGConfig:

    """
    Chứa toàn bộ cấu hình:

    - Ollama model
    - Path dữ liệu
    - Tham số generation
    """

    # [2.1] Tên model Ollama
    MODEL = os.getenv(
        "OLLAMA_MODEL_NAME",
        "qwen2.5:1.5b"
    )

    # [2.2] URL Ollama server
    BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    )

    # [2.3] Thư mục gốc project
    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    # [2.4] Đường dẫn FAISS index
    INDEX_PATH = os.path.join(
        BASE_DIR,
        "data",
        "triet.index"
    )

    # [2.5] Đường dẫn metadata
    META_PATH = os.path.join(
        BASE_DIR,
        "data",
        "triet_embeddings.json"
    )

    # [2.6] Độ sáng tạo của model
    TEMP = 0.15

    # [2.7] Số document retrieve
    TOP_K = 4


# [3] Core RAG Engine
class PhilosophyRAG:

    """
    Pipeline hệ thống RAG:

    1. Nhận câu hỏi user
    2. Query enhancement bằng HyDE
    3. Retrieval tài liệu liên quan
    4. Build context
    5. Gọi LLM generate
    6. Streaming phản hồi
    """

    # [3.1] Khởi tạo hệ thống
    def __init__(self):

        # [3.1.1] Kết nối Ollama server
        self.client = ollama.Client(
            host=RAGConfig.BASE_URL
        )

        try:

            # [3.1.2] Load embedding model
            logger.info(
                "Đang khởi tạo embedding model..."
            )

            self.embed_model = SentenceTransformer(
                'intfloat/multilingual-e5-small'
            )

            # [3.1.3] Load retrieval engine
            logger.info(
                "Đang nạp Hybrid Retriever..."
            )

            self.retriever = HybridRetriever(

                RAGConfig.INDEX_PATH,

                RAGConfig.META_PATH,

                self.embed_model
            )

            logger.info(
                "Database initialized successfully"
            )

        except Exception as e:

            logger.error(
                f"Không thể load database: {e}"
            )

            # Dừng chương trình nếu load fail
            sys.exit(1)

        # [3.1.4] System prompt điều khiển AI
        self.system_instructions = (

            "BẠN LÀ CHUYÊN GIA TRIẾT HỌC MÁC - LÊNIN.\n"
            "NGUYÊN TẮC:\n"
            "1. Trả lời theo giáo trình.\n"
            "2. Có trích dẫn số trang.\n"
            "3. Không suy diễn ngoài tài liệu.\n"
            "4. Nếu dữ liệu không đủ, "
            "hãy nói rõ không đủ thông tin."
        )

    # [3.2] Query Enhancement bằng HyDE
    def _get_hyde_query(
        self,
        query: str
    ) -> str:

        """
        HyDE:
        - Sinh định nghĩa giả bằng LLM
        - Ghép vào query
        - Giúp semantic retrieval tốt hơn
        """

        try:

            # [3.2.1] Prompt tạo định nghĩa giả
            prompt = (

                "Tóm tắt định nghĩa ngắn gọn "
                f"để hỗ trợ retrieval: {query}"
            )

            # [3.2.2] Generate text từ LLM
            res = self.client.generate(

                model=RAGConfig.MODEL,

                prompt=prompt
            )

            # [3.2.3] Ghép query mở rộng
            return (
                f"{query} "
                f"{res['response']}"
            )

        except Exception as e:

            logger.warning(
                f"HyDE failed: {e}"
            )

            # Fallback về query gốc
            return query

    # [3.3] Generate câu trả lời
    def generate_answer(
        self,
        user_query: str,
        chat_history: List[Dict]
    ) -> Tuple[str, List]:

        # [3.3.1] Mở rộng query bằng HyDE
        enhanced_search = self._get_hyde_query(
            user_query
        )

        # [3.3.2] Retrieval document liên quan
        logger.info(
            f"Retrieving context -> "
            f"{user_query[:50]}"
        )

        filtered_docs = self.retriever.search(

            enhanced_search,

            top_k=RAGConfig.TOP_K
        )

        # [3.3.3] Không tìm thấy tài liệu
        if not filtered_docs:

            return (
                "Không tìm thấy nội dung phù hợp "
                "trong giáo trình.",
                []
            )

        # [3.3.4] Build context cho LLM
        context_text = "\n\n".join([

            f"--- TRANG {d['page']} ---\n"
            f"{d['text']}"

            for d in filtered_docs
        ])

        # [3.3.5] Build message chat
        messages = [

            # System prompt
            {
                'role': 'system',
                'content': self.system_instructions
            },

            # Giữ 4 message gần nhất
            *chat_history[-4:],

            # User prompt
            {
                'role': 'user',

                'content':

                    f"[NGỮ CẢNH]\n"
                    f"{context_text}\n\n"

                    f"[CÂU HỎI]\n"
                    f"{user_query}"
            }
        ]

        # [3.3.6] Header terminal output
        print(
            f"\n[Giáo sư AI - {RAGConfig.MODEL}]\n"
            + "═" * 50
        )

        full_response = ""

        try:

            # [3.3.7] Gọi Ollama Chat API
            stream = self.client.chat(

                model=RAGConfig.MODEL,

                messages=messages,

                # Stream token realtime
                stream=True,

                options={

                    # Độ sáng tạo
                    "temperature": RAGConfig.TEMP,

                    # Context window
                    "num_ctx": 4096
                }
            )

            # [3.3.8] Stream token realtime
            for chunk in stream:

                token = chunk['message']['content']

                # In token realtime
                print(
                    token,
                    end="",
                    flush=True
                )

                # Lưu full response
                full_response += token

            print("\n" + "═" * 50)

            # [3.3.9] Cập nhật chat history
            chat_history.append({

                'role': 'user',

                'content': user_query
            })

            chat_history.append({

                'role': 'assistant',

                'content': full_response
            })

            # [3.3.10] Lấy source pages
            pages = sorted(list(set([

                d['page']

                for d in filtered_docs
            ])))

            # [3.3.11] Hiển thị source pages
            print(
                f"Nguồn tham khảo: "
                f"Trang {', '.join(map(str, pages))}"
            )

            return (
                full_response,
                filtered_docs
            )

        except Exception as e:

            logger.error(
                f"Inference error: {e}"
            )

            return (
                "Hệ thống gặp lỗi khi generate.",
                []
            )


# [4] Main Program
if __name__ == "__main__":

    # [4.1] Clear terminal
    os.system(
        'cls'
        if os.name == 'nt'
        else 'clear'
    )

    # [4.2] Khởi tạo RAG engine
    engine = PhilosophyRAG()

    # [4.3] Lưu lịch sử chat
    history = []

    print(
        "\nTip:\n"
        "- exit : thoát chương trình\n"
        "- clear : xóa lịch sử chat"
    )

    # [4.4] Chat loop
    while True:

        try:

            # [4.4.1] Nhận input user
            prompt = input(
                "\nHãy hỏi gì đó: "
            ).strip()

            # [4.4.2] Bỏ qua input rỗng
            if not prompt:
                continue

            # [4.4.3] Thoát chương trình
            if prompt.lower() in [

                'exit',
                'quit'
            ]:

                break

            # [4.4.4] Reset chat history
            if prompt.lower() == 'clear':

                history = []

                print(
                    "Đã xóa lịch sử hội thoại"
                )

                continue

            # [4.4.5] Generate answer
            engine.generate_answer(

                prompt,

                history
            )

        except KeyboardInterrupt:

            print("\nShutting down...")

            break