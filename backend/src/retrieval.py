import faiss
import numpy as np
import json
import os
import re
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class HybridRetriever:
    # HYBRID SEARCH: Kết hợp Vector Search (FAISS) + Keyword Search (BM25) với thuật toán RRF.
    def __init__(self, index_path: str, meta_path: str, model: Any):
        # KIỂM TRA FILE INDEX + METADATA
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("FAISS INDEX hoặc METADATA không tồn tại. Chạy embedding trước.")
        
        # 1 - LOAD FAISS INDEX
        logging.info(f"NẠP FAISS INDEX từ: {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        # 2 - LOAD METADATA
        logging.info("NẠP METADATA...")
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.model = model
        self.corpus = [item.get('text', '') for item in self.metadata]
        
        # 3 - KHỞI TẠO BM25
        logging.info("XÂY DỰNG INVERTED INDEX cho BM25...")
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logging.info("HYBRID SEARCH READY.")

    def _tokenize(self, text: str) -> List[str]:
        """Tiền xử lý văn bản: lowercase + remove ký tự đặc biệt"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def search(self, query: str, top_k: int = 4, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        TÌM KIẾM LAI:
        alpha = 0.5 (cân bằng), >0.5 ưu tiên VECTOR, <0.5 ưu tiên KEYWORD
        """
        recall_size = top_k * 3  # POOL ban đầu để fusion RRF

        # ==========================================
        # PHASE 1: DENSE RETRIEVAL (VECTOR)
        # ==========================================
        processed_query = f"query: {query}"
        query_vec = self.model.encode([processed_query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vec)  # CHUẨN HÓA L2
        vector_distances, faiss_indices = self.index.search(query_vec, recall_size)

        # ==========================================
        # PHASE 2: SPARSE RETRIEVAL (BM25)
        # ==========================================
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:recall_size]

        # ==========================================
        # PHASE 3: RRF FUSION
        # ==========================================
        rrf_scores = {}
        k_constant = 60  # Hằng số RRF chuẩn

        # VECTOR
        for rank, doc_idx in enumerate(faiss_indices[0]):
            if doc_idx == -1: continue
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + alpha * (1.0 / (k_constant + rank + 1))
        
        # KEYWORD
        for rank, doc_idx in enumerate(bm25_indices):
            if bm25_scores[doc_idx] <= 0: continue
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 - alpha) * (1.0 / (k_constant + rank + 1))

        # ==========================================
        # PHASE 4: RERANK + RETURN
        # ==========================================
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results = []

        for idx in sorted_indices:
            doc = self.metadata[idx]
            if len(doc.get('text', '')) >= 50:  # LOẠI BỎ RÁC
                doc_copy = doc.copy()
                doc_copy['rrf_score'] = round(rrf_scores[idx], 6)
                results.append(doc_copy)
            if len(results) == top_k: 
                break
                
        return results