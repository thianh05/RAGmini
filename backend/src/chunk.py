import json
import logging
import re
import numpy as np
import torch
import hashlib

# Typing cho code rõ ràng hơn
from typing import List, Dict, Any

# Embedding model
from sentence_transformers import SentenceTransformer

# Tính cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

# Config project
from config import (
    CLEAN_JSON,
    CHUNK_JSON,
    EMBEDDING_MODEL
)


# =====================================================
# 1. LOGGING
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)


# =====================================================
# 2. SEMANTIC DOCUMENT CHUNKER
# =====================================================
# Class chịu trách nhiệm:
#
# - Tách câu
# - Phân tích ngữ nghĩa
# - Chia semantic chunk
# - Kiểm soát độ dài chunk
# - Xuất dữ liệu chuẩn hóa
class SemanticDocumentChunker:

    # =================================================
    # INIT
    # =================================================
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        breakpoint_percentile: int = 85
    ):

        # =============================================
        # CHỌN DEVICE
        # =============================================
        # Có CUDA:
        # -> dùng GPU
        #
        # Không có:
        # -> dùng CPU
        self.device = (
            'cuda'
            if torch.cuda.is_available()
            else 'cpu'
        )

        logging.info(
            f"Khởi tạo Semantic Engine "
            f"trên phân vùng: {self.device.upper()}"
        )

        # =============================================
        # LOAD EMBEDDING MODEL
        # =============================================
        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

        # =============================================
        # BREAKPOINT PERCENTILE
        # =============================================
        # Ngưỡng xác định:
        # "2 câu khác chủ đề"
        #
        # Ví dụ:
        # percentile = 85
        # -> chỉ cắt khi độ lệch semantic rất lớn
        self.breakpoint_percentile = breakpoint_percentile

        # =============================================
        # MAX CHUNK LENGTH
        # =============================================
        # Giới hạn ký tự mỗi chunk
        #
        # tránh:
        # - chunk quá dài
        # - vượt context
        # - tốn VRAM
        self.max_chunk_length = 1000

    # =================================================
    # TÁCH CÂU + GẮN METADATA
    # =================================================
    def _split_into_sentences_with_meta(
        self,
        pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        sentences_meta = []

        # =============================================
        # DUYỆT TỪNG TRANG
        # =============================================
        for page in pages:

            text = page.get("text", "")

            page_num = page.get(
                "page",
                "Unknown"
            )

            # =========================================
            # TÁCH CÂU TIẾNG VIỆT
            # =========================================
            # Regex:
            # tách sau dấu . ! ?
            raw_sentences = re.split(
                r'(?<=[.!?])\s+',
                text
            )

            # =========================================
            # LỌC CÂU NGẮN
            # =========================================
            for s in raw_sentences:

                clean_s = s.strip()

                # Bỏ câu quá ngắn
                if len(clean_s) > 20:

                    sentences_meta.append({

                        # Nội dung câu
                        "text": clean_s,

                        # Trang chứa câu
                        "page": page_num
                    })

        return sentences_meta

    # =================================================
    # TẠO ID CHUNK
    # =================================================
    # Dùng hash MD5:
    # - text
    # - danh sách page
    #
    # => tạo ID duy nhất
    def _generate_chunk_id(
        self,
        text: str,
        pages_list: List[Any]
    ) -> str:

        unique_string = (

            f"pages_"
            f"{'_'.join(map(str, set(pages_list)))}_"
            f"{text}"

        ).encode('utf-8')

        return hashlib.md5(
            unique_string
        ).hexdigest()

    # =================================================
    # SEMANTIC SPLIT
    # =================================================
    # Chia chunk dựa trên:
    # độ khác biệt ngữ nghĩa
    def _semantic_split(
        self,
        sentences_meta: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        # =============================================
        # TRƯỜNG HỢP QUÁ ÍT CÂU
        # =============================================
        if len(sentences_meta) <= 2:

            return [{
                "text": " ".join([
                    s["text"]
                    for s in sentences_meta
                ]),

                "pages": list(set([
                    s["page"]
                    for s in sentences_meta
                ]))
            }]

        # =================================================
        # LẤY TOÀN BỘ TEXT
        # =================================================
        texts = [
            s["text"]
            for s in sentences_meta
        ]

        # =================================================
        # ENCODE SENTENCE -> VECTOR
        # =================================================
        embeddings = self.model.encode(

            texts,

            batch_size=64,

            show_progress_bar=False
        )

        # =================================================
        # TÍNH KHOẢNG CÁCH NGỮ NGHĨA
        # =================================================
        # distance = 1 - cosine similarity
        #
        # similarity cao:
        # -> cùng chủ đề
        #
        # similarity thấp:
        # -> khác chủ đề
        distances = []

        for i in range(len(embeddings) - 1):

            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[i + 1]]
            )[0][0]

            distances.append(
                1.0 - sim
            )

        # =================================================
        # TÍNH NGƯỠNG BREAKPOINT
        # =================================================
        # Ví dụ:
        # percentile=85
        #
        # => chỉ cắt khi distance
        # nằm top 15% lớn nhất
        threshold = np.percentile(
            distances,
            self.breakpoint_percentile
        )

        # =================================================
        # BUILD SEMANTIC CHUNK
        # =================================================
        chunks_meta = []

        current_texts = [
            sentences_meta[0]["text"]
        ]

        current_pages = [
            sentences_meta[0]["page"]
        ]

        # =================================================
        # DUYỆT TỪNG DISTANCE
        # =================================================
        for i, distance in enumerate(distances):

            # =============================================
            # KHÁC CHỦ ĐỀ -> TẠO CHUNK MỚI
            # =============================================
            if distance > threshold:

                chunks_meta.append({

                    "text": " ".join(
                        current_texts
                    ),

                    # Lưu toàn bộ page liên quan
                    "pages": list(set(
                        current_pages
                    ))
                })

                # Reset chunk mới
                current_texts = [
                    sentences_meta[i + 1]["text"]
                ]

                current_pages = [
                    sentences_meta[i + 1]["page"]
                ]

            # =============================================
            # CÙNG CHỦ ĐỀ -> GỘP
            # =============================================
            else:

                current_texts.append(
                    sentences_meta[i + 1]["text"]
                )

                current_pages.append(
                    sentences_meta[i + 1]["page"]
                )

        # =================================================
        # ĐÓNG GÓI CHUNK CUỐI
        # =================================================
        if current_texts:

            chunks_meta.append({

                "text": " ".join(
                    current_texts
                ),

                "pages": list(set(
                    current_pages
                ))
            })

        return chunks_meta

    # =================================================
    # KIỂM SOÁT ĐỘ DÀI CHUNK
    # =================================================
    # Nếu chunk quá dài:
    # -> chia nhỏ tiếp
    def _apply_length_control(
        self,
        chunks_meta: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        final_chunks = []

        # =============================================
        # DUYỆT TỪNG CHUNK
        # =============================================
        for chunk_obj in chunks_meta:

            chunk_text = chunk_obj["text"]

            chunk_pages = chunk_obj["pages"]

            # =========================================
            # CHUNK HỢP LỆ
            # =========================================
            if len(chunk_text) <= self.max_chunk_length:

                final_chunks.append(
                    chunk_obj
                )

            # =========================================
            # CHUNK QUÁ DÀI
            # =========================================
            else:

                # Tách lại theo câu
                sub_sentences = re.split(
                    r'(?<=[.!?])\s+',
                    chunk_text
                )

                temp_texts = []

                current_len = 0

                # =====================================
                # GOM NHÓM THEO ĐỘ DÀI
                # =====================================
                for sentence in sub_sentences:

                    # Nếu vượt giới hạn
                    if (
                        current_len + len(sentence)
                        > self.max_chunk_length
                    ):

                        final_chunks.append({

                            "text": " ".join(
                                temp_texts
                            ),

                            "pages": chunk_pages
                        })

                        # overlap nhẹ
                        # giữ câu cuối
                        temp_texts = temp_texts[-1:]

                        current_len = sum(
                            len(s)
                            for s in temp_texts
                        )

                    temp_texts.append(sentence)

                    current_len += len(sentence)

                # =====================================
                # CHUNK CUỐI
                # =====================================
                if temp_texts:

                    final_chunks.append({

                        "text": " ".join(
                            temp_texts
                        ),

                        "pages": chunk_pages
                    })

        return final_chunks

    # =================================================
    # PIPELINE TOÀN BỘ
    # =================================================
    def process_pipeline(
        self,
        pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        logging.info(
            f"BĂM NGỮ NGHĨA XUYÊN TRANG "
            f"({len(pages)} TRANG)..."
        )

        # =================================================
        # STEP 1: TÁCH CÂU
        # =================================================
        sentences_meta = (
            self._split_into_sentences_with_meta(
                pages
            )
        )

        # =================================================
        # STEP 2: SEMANTIC CHUNKING
        # =================================================
        semantic_chunks = (
            self._semantic_split(
                sentences_meta
            )
        )

        # =================================================
        # STEP 3: LENGTH CONTROL
        # =================================================
        controlled_chunks = (
            self._apply_length_control(
                semantic_chunks
            )
        )

        # =================================================
        # BUILD FINAL OUTPUT
        # =================================================
        results = []

        for idx, chunk_obj in enumerate(controlled_chunks):

            # Lấy page chính
            primary_page = (

                chunk_obj["pages"][0]

                if chunk_obj["pages"]

                else "Unknown"
            )

            results.append({

                # ID unique
                "chunk_id": self._generate_chunk_id(
                    chunk_obj["text"],
                    chunk_obj["pages"]
                ),

                # Trang chính
                "page": primary_page,

                # Index chunk
                "chunk_index": idx,

                # Số ký tự
                "char_count": len(
                    chunk_obj["text"]
                ),

                # Nội dung chunk
                "text": chunk_obj["text"]
            })

        logging.info(
            f"ĐÃ XUẤT "
            f"{len(results)} "
            f"CHUNKS CHUẨN HÓA."
        )

        return results


# =====================================================
# 3. MAIN EXECUTION
# =====================================================
if __name__ == "__main__":

    try:

        # =================================================
        # LOAD CLEAN DATA
        # =================================================
        logging.info(
            f"NẠP CẤU TRÚC DỮ LIỆU THÔ "
            f"{CLEAN_JSON}"
        )

        with open(
            CLEAN_JSON,
            "r",
            encoding="utf-8"
        ) as f:

            clean_pages = json.load(f)

        # =================================================
        # INIT CHUNKER
        # =================================================
        chunker = SemanticDocumentChunker(
            breakpoint_percentile=85
        )

        # =================================================
        # PROCESS PIPELINE
        # =================================================
        chunks = chunker.process_pipeline(
            clean_pages
        )

        # =================================================
        # SAVE CHUNK JSON
        # =================================================
        with open(
            CHUNK_JSON,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                chunks,
                f,
                ensure_ascii=False,
                indent=2
            )

        logging.info(
            f"ĐÃ GHI ĐÈ PHÂN MẢNH THÀNH CÔNG - "
            f"{CHUNK_JSON}"
        )

    except Exception as e:

        logging.critical(
            f"LỖI TIẾN TRÌNH: {str(e)}"
        )