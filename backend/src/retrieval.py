import faiss
import numpy as np
import json
import os
import re
import logging

# Typing cho code dễ đọc hơn
from typing import List, Dict, Any

# BM25 keyword search
from rank_bm25 import BM25Okapi


# =====================================================
# 1. LOGGING
# =====================================================
# Hiển thị log ra terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)


# =====================================================
# 2. HYBRID RETRIEVER
# =====================================================
# Retriever kết hợp:
# - Dense Search (FAISS Vector)
# - Sparse Search (BM25 Keyword)
#
# => giúp tìm:
#    + đúng ngữ nghĩa
#    + đúng từ khóa
class HybridRetriever:

    # =================================================
    # KHỞI TẠO
    # =================================================
    def __init__(
        self,
        index_path: str,
        meta_path: str,
        model: Any
    ):

        # Kiểm tra file index và metadata tồn tại
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                "FAISS INDEX hoặc METADATA không tồn tại. Chạy embedding trước."
            )

        # =================================================
        # LOAD FAISS INDEX
        # =================================================
        logging.info(f"NẠP FAISS INDEX từ: {index_path}")

        # Load vector database
        self.index = faiss.read_index(str(index_path))

        # =================================================
        # LOAD METADATA
        # =================================================
        logging.info("NẠP METADATA...")

        with open(meta_path, 'r', encoding='utf-8') as f:

            # metadata chứa:
            # text, page, source,...
            self.metadata = json.load(f)

        # Embedding model
        self.model = model

        # =================================================
        # TẠO CORPUS
        # =================================================
        # Lấy toàn bộ text document
        self.corpus = [
            item.get('text', '')
            for item in self.metadata
        ]

        # =================================================
        # BM25 INDEX
        # =================================================
        logging.info("XÂY DỰNG INVERTED INDEX cho BM25...")

        # Tokenize từng document
        tokenized_corpus = [
            self._tokenize(doc)
            for doc in self.corpus
        ]

        # Xây BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)

        logging.info("HYBRID SEARCH READY.")

    # =================================================
    # TOKENIZER
    # =================================================
    # Chuẩn hóa text cho BM25
    def _tokenize(self, text: str) -> List[str]:

        # Chuyển về chữ thường
        text = text.lower()

        # Xóa ký tự đặc biệt
        text = re.sub(r'[^\w\s]', ' ', text)

        # Tách thành list token
        return text.split()

    # =================================================
    # SEARCH
    # =================================================
    def search(
        self,
        query: str,
        top_k: int = 4,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:

        """
        Hybrid Search Pipeline

        top_k:
            số document trả về cuối cùng

        alpha:
            trọng số ưu tiên vector search

            alpha = 0.7
            -> 70% dense search
            -> 30% BM25

        recall_size:
            lấy nhiều candidate hơn
            rồi mới rerank bằng RRF
        """

        # =================================================
        # RECALL SIZE
        # =================================================
        # Ví dụ:
        # top_k = 4
        # recall_size = 20
        #
        # => lấy 20 doc trước
        # => fusion + rerank
        # => chọn 4 doc cuối
        recall_size = top_k * 5

        # =================================================
        # PHASE 1: DENSE RETRIEVAL (VECTOR SEARCH)
        # =================================================

        # Prefix query:
        # nhiều embedding model cần format:
        # "query: ..."
        processed_query = f"query: {query}"

        # Encode query -> vector
        query_vec = self.model.encode(
            [processed_query],
            convert_to_numpy=True
        ).astype(np.float32)

        # Normalize vector
        # giúp cosine similarity chính xác hơn
        faiss.normalize_L2(query_vec)

        # Search trong FAISS
        #
        # vector_distances:
        #   độ tương đồng
        #
        # faiss_indices:
        #   index document
        vector_distances, faiss_indices = self.index.search(
            query_vec,
            recall_size
        )

        # =================================================
        # PHASE 2: SPARSE RETRIEVAL (BM25)
        # =================================================

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Tính điểm BM25
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Sắp xếp giảm dần
        # lấy top recall_size
        bm25_indices = np.argsort(
            bm25_scores
        )[::-1][:recall_size]

        # =================================================
        # PHASE 3: RRF FUSION
        # =================================================
        # Reciprocal Rank Fusion
        #
        # Kết hợp:
        # - vector ranking
        # - keyword ranking
        #
        # Công thức:
        # score = 1 / (k + rank)
        #
        # rank càng cao -> score càng lớn
        rrf_scores = {}

        # Hằng số giảm ảnh hưởng rank
        k_constant = 60

        # =================================================
        # VECTOR RANKING
        # =================================================
        for rank, doc_idx in enumerate(faiss_indices[0]):

            # -1 = không có kết quả
            if doc_idx == -1:
                continue

            # Cộng điểm RRF
            rrf_scores[doc_idx] = (
                rrf_scores.get(doc_idx, 0.0)
                +
                alpha * (
                    1.0 / (k_constant + rank + 1)
                )
            )

        # =================================================
        # KEYWORD RANKING
        # =================================================
        for rank, doc_idx in enumerate(bm25_indices):

            # Score <= 0
            # nghĩa là keyword không match
            if bm25_scores[doc_idx] <= 0:
                continue

            # Cộng điểm BM25 vào RRF
            rrf_scores[doc_idx] = (
                rrf_scores.get(doc_idx, 0.0)
                +
                (1.0 - alpha) * (
                    1.0 / (k_constant + rank + 1)
                )
            )

        # =================================================
        # PHASE 4: RERANK + FILTER
        # =================================================

        # Sort document theo điểm RRF giảm dần
        sorted_indices = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )

        results = []

        # =================================================
        # DUYỆT KẾT QUẢ
        # =================================================
        for idx in sorted_indices:

            # Lấy document metadata
            doc = self.metadata[idx]

            # =============================================
            # FILTER DOCUMENT QUÁ NGẮN
            # =============================================
            # < 80 ký tự:
            # thường thiếu context
            #
            # => dễ làm LLM hallucination
            if len(doc.get('text', '')) >= 80:

                # Copy document
                doc_copy = doc.copy()

                # Gắn thêm điểm RRF
                doc_copy['rrf_score'] = round(
                    rrf_scores[idx],
                    6
                )

                results.append(doc_copy)

            # Đủ top_k thì dừng
            if len(results) == top_k:
                break

        # =================================================
        # RETURN
        # =================================================
        return results