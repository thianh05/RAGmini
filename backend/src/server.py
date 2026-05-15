import os
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from ollama import AsyncClient 
from retrieval import HybridRetriever
from config import (
    INDEX_PATH, 
    META_PATH, 
    EMBEDDING_MODEL, 
    LLM_MODEL, 
    OLLAMA_URL
)

# =====================================================
# 1. LOGGING
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

logger = logging.getLogger(__name__)
retriever_engine = None
ollama_client = None


# =====================================================
# 2. KHỞI ĐỘNG & TẮT SERVER (LIFESPAN)
# =====================================================
# - Khi server start
# - Khi server shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):

    global retriever_engine, ollama_client

    logger.info("STARTING ENGINE: Đang load model vào RAM/VRAM...")

    try:
        # Load embedding model
        # asyncio.to_thread():
        # -> chạy tác vụ nặng ở thread riêng
        # -> tránh block event loop của FastAPI
        model = await asyncio.to_thread(
            SentenceTransformer,
            EMBEDDING_MODEL
        )

        # Khởi tạo hệ thống retrieval
        # Dùng để tìm dữ liệu liên quan trong vector DB
        retriever_engine = HybridRetriever(
            index_path=str(INDEX_PATH),
            meta_path=str(META_PATH),
            model=model
        )

        # Tạo client kết nối Ollama local
        ollama_client = AsyncClient(host=OLLAMA_URL)

        logger.info("ENGINE READY: Hệ thống đã sẵn sàng.")

        # yield = server bắt đầu chạy
        yield

    except Exception as e:
        logger.error(f"LỖI KHỞI ĐỘNG: {e}")
        raise e

    finally:
        logger.info("Shutting down server...")


# =====================================================
# 3. TẠO FASTAPI APP
# =====================================================
app = FastAPI(
    title="RAGstudio Philosophy",
    lifespan=lifespan
)

# =====================================================
# 4. CORS
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# 5. SYSTEM PROMPT
# =====================================================

SYSTEM_INSTRUCTIONS = """
#STRICT SYSTEM DIRECTIVE:
BẠN LÀ CHUYÊN GIA TRIẾT HỌC MÁC - LÊNIN
Nhiệm vụ :
- Chỉ trả lời dựa trên dữ liệu được cung cấp 
- Nghiêm cấm phép suy diễn, không tự ý bổ sung kiến thức ngoài dữ liệu 

QUY TẮC BẮT BUỘC :
1. NGÔN NGỮ : Chỉ sử dụng Tiếng Việt , Không sử dụng tiếng Anh và Tiếng Trung, Không suy diễn, không tự bổ sung kiến thức bên ngoài dữ liệu.
2. CHỐNG BỊA ĐẶT : Chỉ dùng thông tin có trong ngữ cảnh. Nếu ngữ cảnh không chứa đáp án, bắt buộc trả lời đúng 1 câu " Dữ liệu cung cấp không đủ để trả lời." và thực hiện bỏ qua FORMAT OUTPUT.
3. BẢO MẬT: Tuyệt đối không đề cập đến "Ollama", "Model", "AI", "RAG", "Context", "Retrieval", hoặc bất kỳ thuật ngữ kỹ thuật nào. Hãy đóng vai một giảng viên Triết học thuần túy.
4. PHONG CÁCH TRẢ LỜI : Trả lời trực tiếp vào trọng tâm, không lan man, lặp lại ý, không giải thích ngoài yêu cầu, không được sử dụng các lời dẫn như "Theo dữ liệu","Dựa vào thông tin","Tôi cho rằng","Có thể thấy rằng".
5. QUY TẮC CHO CÂU HỎI TRẮC NGHIỆM : Nếu chỉ là câu hỏi A/B/C/D Chỉ trả về đáp án đúng và chỉ giải thích khi người dùng yêu cầu.
6. QUY TẮC CHO CÂU HỎI TRỰC TIẾP : Trả lời trực tiếp nội dung chính, Không sử dụng format 3 phần, Trả lời đủ ý nhưng súc tích.
7. FORMAT( ĐỊNH DẠNG): Không thêm dấu mở đầu hoặc kết thúc, Không dụng các ký tự như #, *, _, ---, ```. 
8. ĐỘ DÀI CÂU HỎI : Nếu người dùng yêu cầu ngắn : Trả lời ngắn gọn nhưng đầy đủ ý; Nếu người dùng yêu cầu trả lời chi tiết: trả lời đầy đủ, rõ ràng và đúng trọng tâm.
9. ƯU TIÊN ĐỘ CHÍNH XÁC : Ưu tiên tính chính xác hơn văn phong, Không được phép suy luận vượt quá dữ liệu đã có.
10. ƯU TIÊN TỐC ĐỘ : Trả lời nhanh nhất có thể, không rườm rà, không thêm các thông tin không cần thiết.
"""

# =====================================================
# 6. API STREAMING
# =====================================================
# Endpoint:
# POST /ask_stream
#
# Client gửi:
# {
#    "query": "Câu hỏi ..."
# }
@app.post("/ask_stream")
async def ask_stream(query: str = Body(..., embed=True)):

    # Nếu model chưa load xong
    if not retriever_engine or not ollama_client:
        return StreamingResponse(
            iter(["Hệ thống chưa sẵn sàng."]),
            media_type="text/plain"
        )

    # =================================================
    # BƯỚC 1: RETRIEVAL
    # =================================================
    # Tìm top 4 đoạn liên quan nhất
    docs = retriever_engine.search(query, top_k=4)

    # Nếu không tìm thấy dữ liệu
    if not docs:
        return StreamingResponse(
            iter(["Dữ liệu giáo trình không có thông tin này."]),
            media_type="text/plain"
        )

    # =================================================
    # BƯỚC 2: LOẠI BỎ DOCUMENT TRÙNG
    # =================================================
    unique_docs = []
    seen_content = set()

    for d in docs:

        # Lấy 60 ký tự đầu để kiểm tra trùng
        content_snippet = d['text'][:60].strip()

        # Nếu chưa xuất hiện
        if content_snippet not in seen_content:

            unique_docs.append(d)

            # Đánh dấu đã thấy
            seen_content.add(content_snippet)

    # =================================================
    # BƯỚC 3: GHÉP CONTEXT
    # =================================================
    # Format context đưa vào LLM
    context_text = "\n\n".join([
        f"--- TRANG {d['page']} ---\n{d['text']}"
        for d in unique_docs
    ])

    # Lấy danh sách page nguồn
    pages = sorted(list(set([
        d['page']
        for d in unique_docs
    ])))

    # =================================================
    # BƯỚC 4: STREAM RESPONSE TỪ LLM
    # =================================================
    async def event_generator():

        # Prompt cuối cùng gửi cho model
        user_prompt = f"""
[NGỮ CẢNH GIÁO TRÌNH]:
{context_text}

[CÂU HỎI]:
{query}
"""

        try:

            # Gọi Ollama stream
            stream = await ollama_client.chat(

                model=LLM_MODEL,

                messages=[
                    {
                        'role': 'system',
                        'content': SYSTEM_INSTRUCTIONS.strip()
                    },
                    {
                        'role': 'user',
                        'content': user_prompt.strip()
                    }
                ],

                # Stream token realtime
                stream=True,

                # Tham số sinh text
                options={

                    # temperature thấp
                    # -> output ổn định hơn
                    "temperature": 0.0,

                    # Giới hạn randomness
                    "top_p": 0.1,

                    # Giảm lặp từ
                    "repeat_penalty": 1.15,

                    # Phạt việc lặp nội dung cũ
                    "presence_penalty": 0.5,

                    # Số token tối đa sinh ra
                    "num_predict": 400,

                    # Stop token
                    # gặp sẽ dừng generate
                    "stop": [
                        "<END>",
                        "User:",
                        "###",
                        "</s>",
                        "<|im_end|>"
                    ]
                }
            )

            # =============================================
            # STREAM TỪNG CHUNK VỀ CLIENT
            # =============================================
            async for chunk in stream:

                # Lấy text model vừa sinh
                content = chunk['message']['content']

                if content:

                    # Gửi text realtime về frontend
                    yield content

                    # Delay cực nhỏ
                    # giúp stream mượt hơn
                    await asyncio.sleep(0.001)

            # =============================================
            # GỬI SOURCE PAGE
            # =============================================
            if pages:
                yield f"\n\n<SOURCES>{','.join(map(str, pages))}</SOURCES>"

        except Exception as e:

            logger.error(f"Inference Error: {e}")

            yield f"\n\n[Lỗi kết nối mô hình: {str(e)}]"

    # Trả response dạng stream realtime
    return StreamingResponse(
        event_generator(),
        media_type="text/plain"
    )


# =====================================================
# 7. CHẠY SERVER
# =====================================================
if __name__ == "__main__":

    import uvicorn

    # Chạy FastAPI tại:
    # http://127.0.0.1:8000
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )

