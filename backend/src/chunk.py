import json
import logging
import re
import numpy as np
import torch
import hashlib
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import CLEAN_JSON, CHUNK_JSON, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class SemanticDocumentChunker:
    # SEMANTIC CHUNKER 
    def __init__(self, model_name: str = EMBEDDING_MODEL, breakpoint_percentile: int = 85):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Khởi tạo Semantic Engine trên phân vùng: {self.device.upper()}")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chunk_length = 1000

    def _split_into_sentences_with_meta(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # TÁCH CÂU 
        sentences_meta = []
        for page in pages:
            text = page.get("text", "")
            page_num = page.get("page", "Unknown")
            
            # TÁCH CÂU TIẾNG VIỆT
            raw_sentences = re.split(r'(?<=[.!?])\s+', text)
            # LỌC CÂU NGẮN VÀ GẮN NHÃN TRANG
            for s in raw_sentences:
                clean_s = s.strip()
                if len(clean_s) > 20:
                    sentences_meta.append({
                        "text": clean_s,
                        "page": page_num
                    })
        return sentences_meta
    # TẠO ID CHUNK DỰA TRÊN NỘI DUNG VÀ TRANG
    def _generate_chunk_id(self, text: str, pages_list: List[Any]) -> str:
        unique_string = f"pages_{'_'.join(map(str, set(pages_list)))}_{text}".encode('utf-8')
        return hashlib.md5(unique_string).hexdigest()

    # PHÂN TÍCH NGỮ NGHĨA VỚI BẢO TỒN TRANG
    def _semantic_split(self, sentences_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(sentences_meta) <= 2:
            return [{
                "text": " ".join([s["text"] for s in sentences_meta]),
                "pages": list(set([s["page"] for s in sentences_meta]))
            }]

        texts = [s["text"] for s in sentences_meta]
        embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=False)
        # TÍNH ĐỘ CHÊNH LỆCH NGỮ NGHĨA GIỮA CÁC CÂU LIÊN TIẾP
        distances = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            distances.append(1.0 - sim)

        threshold = np.percentile(distances, self.breakpoint_percentile)

        # BĂM 
        chunks_meta = []
        current_texts = [sentences_meta[0]["text"]]
        current_pages = [sentences_meta[0]["page"]]

        for i, distance in enumerate(distances):
            if distance > threshold:
                chunks_meta.append({
                    "text": " ".join(current_texts),
                    "pages": list(set(current_pages))
                }) # BẢO TỒN TẬP HỢP TRANG
                current_texts = [sentences_meta[i+1]["text"]]
                current_pages = [sentences_meta[i+1]["page"]]
            else: # CÙNG CHỦ ĐỀ > GOM NHÓM VĂN BẢN VÀ TẬP HỢP TRANG
                current_texts.append(sentences_meta[i+1]["text"])
                current_pages.append(sentences_meta[i+1]["page"])
        
        # ĐÓNG GÓI CHUNK CUỐI CÙNG
        if current_texts:
            chunks_meta.append({
                "text": " ".join(current_texts),
                "pages": list(set(current_pages))
            })

        return chunks_meta
    
    # KIỂM SOÁT ĐỘ DÀI CHUNK - THEO GIỚI HẠN HARDWARE
    def _apply_length_control(self, chunks_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        final_chunks = []
        for chunk_obj in chunks_meta:
            chunk_text = chunk_obj["text"]
            chunk_pages = chunk_obj["pages"]
            
            if len(chunk_text) <= self.max_chunk_length:
                final_chunks.append(chunk_obj)
            else:
                sub_sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
                
                temp_texts = []
                current_len = 0
                # GOM NHÓM CÂU CON THÀNH CHUNK MỚI THEO GIỚI HẠN ĐỘ DÀI
                for sentence in sub_sentences:
                    if current_len + len(sentence) > self.max_chunk_length:
                        final_chunks.append({
                            "text": " ".join(temp_texts),
                            "pages": chunk_pages
                        })
                        temp_texts = temp_texts[-1:]
                        current_len = sum(len(s) for s in temp_texts)

                    temp_texts.append(sentence)
                    current_len += len(sentence)

                if temp_texts:
                    final_chunks.append({
                        "text": " ".join(temp_texts),
                        "pages": chunk_pages
                    })

        return final_chunks

    # QUY TRÌNH XỬ LÝ TOÀN DIỆN
    def process_pipeline(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logging.info(f"BĂM NGỮ NGHĨA XUYÊN TRANG ({len(pages)} TRANG)...")
        # 1 - TÁCH CÂU - GẮN NHÃN
        sentences_meta = self._split_into_sentences_with_meta(pages)

        # 2 - BĂM CHUNK THEO NGỮ NGHĨA
        semantic_chunks = self._semantic_split(sentences_meta)

        # 3 - KIỂM SOÁT ĐỘ DÀI CHUNK THEO GIỚI HẠN PHẦN CỨNG
        controlled_chunks = self._apply_length_control(semantic_chunks)

        results = []
        for idx, chunk_obj in enumerate(controlled_chunks):
            primary_page = chunk_obj["pages"][0] if chunk_obj["pages"] else "Unknown"
            
            results.append({
                "chunk_id": self._generate_chunk_id(chunk_obj["text"], chunk_obj["pages"]),
                "page": primary_page, 
                "chunk_index": idx,
                "char_count": len(chunk_obj["text"]),
                "text": chunk_obj["text"]
            })

        logging.info(f"ĐÃ XUẤT {len(results)} CHUNKS CHUẨN HÓA.")
        return results

if __name__ == "__main__":
    try:
        logging.info(f"NẠP CẤU TRÚC DỮ LIỆU THÔ {CLEAN_JSON}")

        with open(CLEAN_JSON, "r", encoding="utf-8") as f:
            clean_pages = json.load(f)

        chunker = SemanticDocumentChunker(breakpoint_percentile=85)
        chunks = chunker.process_pipeline(clean_pages)

        with open(CHUNK_JSON, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        logging.info(f"ĐÃ GHI ĐÈ PHÂN MẢNH THÀNH CÔNG -{CHUNK_JSON}")

    except Exception as e:
        logging.critical(f"LỖI TIẾN TRÌNH: {str(e)}")