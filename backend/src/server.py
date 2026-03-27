import os
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from ollama import AsyncClient # BẮT BUỘC DÙNG ASYNC CLIENT CHO FASTAPI

from retrieval import HybridRetriever
from config import INDEX_PATH, META_PATH, EMBEDDING_MODEL, LLM_MODEL, OLLAMA_URL

# ==========================================
# 1 - CẤU HÌNH LOGGING - BIẾN TOÀN CỤC 
# ==========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

retriever_engine = None
ollama_client = None

# ==========================================
# 2 - QUẢN LÝ TÀI NGUYÊN - LIFESPAN
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # CHỐNG MEMORY LEAK 
    global retriever_engine, ollama_client
    logger.info("ĐANG KHỞI ĐỘNG AI ENGINE - NẠP MODEL VÀO VRAM")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        retriever_engine = HybridRetriever(
            index_path=str(INDEX_PATH), 
            meta_path=str(META_PATH), 
            model=model
        )
        ollama_client = AsyncClient(host=OLLAMA_URL)
        logger.info("DATABASE - INFERENCE SẴN SÀNG")
        yield
    except Exception as e:
        logger.error(f"LỖI KHỞI ĐỘNG HỆ THỐNG: {e}")
        raise e
    finally:
        logger.info("TẮT SERVER - GIẢI PHÓNG TÀI NGUYÊN...")
        retriever_engine = None
        ollama_client = None

app = FastAPI(
    title="RAGstudio", 
    version="5.0", 
    description="Giáo trình Triết học Mác-Lênin",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 3. HỆ THỐNG PROMPT
# ==========================================
SYSTEM_INSTRUCTIONS = """
BẠN LÀ GIẢNG VIÊN CAO CẤP TRIẾT HỌC MÁC-LÊNIN. CHUYÊN NGHIỆP, SẮC SẢO, ĐI THẲNG VÀO VẤN ĐỀ.

[KỶ LUẬT THÉP - BẮT BUỘC TUÂN THỦ]:
1. TRẢ LỜI TRỰC TIẾP: Đi ngay vào đáp án. KHÔNG rào trước đón sau, KHÔNG dùng câu dẫn (Ví dụ cấm dùng: "Tôi xin giải thích...", "Theo giáo trình...", "Dựa vào ngữ cảnh...").
2. CẤM RÒ RỈ TƯ DUY: TUYỆT ĐỐI KHÔNG in ra các từ như "Ý định:", "Phân tích:", "Đáp án:", "Giải thích:", "Kết luận:". 
3. SỰ THẬT & TRÍCH DẪN: Lấy 100% kiến thức từ [NGỮ CẢNH]. Mọi ý chính phải gắn số trang (Trang X) ở cuối câu. Không có dữ liệu -> Từ chối trả lời.

[CẤU TRÚC TRÌNH BÀY MONG MUỐN]:
- Chia đoạn văn ngắn gọn, thoáng mắt.
- Dùng dấu gạch đầu dòng (-) hoặc Bullet points (•) để liệt kê đặc trưng/tính chất.
- **Bôi đậm** các thuật ngữ cốt lõi để sinh viên dễ nắm bắt.
"""

# ==========================================
# 4. API ENDPOINT (FULL ASYNC STREAMING)
# ==========================================
@app.get("/ask_stream")
async def ask_stream(query: str = Query(..., min_length=2)):
    if not retriever_engine or not ollama_client:
        return StreamingResponse(iter(["Hệ thống AI chưa sẵn sàng. Vui lòng thử lại sau."]), media_type="text/plain")
   
    logger.info(f"Processing Query: '{query}'") 

    # RANKING TOP 5 
    docs = retriever_engine.search(query, top_k=5)
    if not docs:
        return StreamingResponse(iter(["Không tìm thấy tài liệu liên quan trong giáo trình."]), media_type="text/plain")

    context_text = "\n\n".join([f"--- NGUỒN TRANG {d['page']} ---\n{d['text']}" for d in docs])
    pages = sorted(list(set([d['page'] for d in docs])))

    # USER PROMPT TỐI ƯU HÓA
    async def event_generator():
        user_prompt = f"""
[NGỮ CẢNH GIÁO TRÌNH]:
{context_text}

[CÂU HỎI CỦA NGƯỜI DÙNG]:
{query}
"""
        try:
            # SỬ DỤNG ASYNC CLIENT ĐỂ KHÔNG BLOCK FASTAPI
            stream = await ollama_client.chat(
                model=LLM_MODEL,
                messages=[
                    {'role': 'system', 'content': SYSTEM_INSTRUCTIONS.strip()},
                    {'role': 'user', 'content': user_prompt.strip()}
                ],
                stream=True,
                options={
                    "temperature": 0.05, # GIẢM HALLUCINATION 
                    "top_p": 0.85,       # CẮT TOKEN KHÔNG CẦN THIẾT
                    "num_ctx": 4096,
                    "repeat_penalty": 1.15 # NGĂN LẶP LẠI
                }
            )
            
            async for chunk in stream:
                content = chunk['message']['content']
                if content:
                    yield content
                    # EVENT STREAMING TẠM NGHỈ
                    await asyncio.sleep(0.005) 
                    
            if pages:
                yield f"\n\n<SOURCES>{','.join(map(str, pages))}</SOURCES>"
                
        except asyncio.TimeoutError:
            logger.error("LLM Timeout.")
            yield "\n\n[Lỗi: Mô hình phản hồi quá lâu.]"
        except Exception as e:
            logger.error(f"Inference Error: {e}")
            yield f"\n\n[Lỗi hệ thống trong quá trình sinh văn bản.]"

    return StreamingResponse(event_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    # DÙNG ĐỂ CHẠY SEVER FASTAPI VỚI TÍNH NĂNG RELOAD KHI CODE THAY ĐỔI
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)