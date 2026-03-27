import json
import logging
import os
import time
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import CHUNK_JSON, INDEX_PATH, META_PATH, EMBEDDING_MODEL


logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class VectorIndexer:

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"KHỞI ĐỘNG{self.device.upper()}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.metadata = []

    def build_database(self, input_json: str, index_out: str, meta_out: str):
        # 1 - KIỂM TRA FILE ĐẦU VÀO
        if not os.path.exists(input_json):
            logging.critical(f" ERROR : KHÔNG TÌM THẤY FILE {input_json} > CHẠY CHUNK TRƯỚC")
            return

        logging.info(f" NẠP DỮ LIỆU TỪ {input_json}")
        with open(input_json, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        if not self.metadata:
            logging.warning("ERROR : FILE JSON KHÔNG CÓ DỮ LIỆU")
            return
        texts = [f"passage: {chunk['text']}" for chunk in self.metadata]
        
        # 2 - TÍNH TOÁN
        logging.info(f"ÉP XUNG MÃ HÓA {len(texts)} CHUNKS - PHÂN BỔ BATCH SIZE")
        start_time = time.time()
        
        # 3 - VECTOR MATRIX - EMBEDDING 
        embeddings = self.model.encode(
            texts, 
            batch_size=64, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        
        # 4 - CONSINE SIMILARITY OPTIMIZATION > NORMALIZE VECTOR > FAISS INDEX
        faiss.normalize_L2(embeddings)
        
        encode_time = time.time() - start_time
        logging.info(f"TÍNH TOÁN TRONG{encode_time:.2f} GIÂY - CHIỀU VECTOR: {embeddings.shape[1]}")

        # 5 - KHỞI TẠO FAISS INDEX VỚI INNER PRODUCT
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension) 
        self.index.add(embeddings)
        
        # ĐẢM BẢO THƯ MỤC TỒN TẠI
        os.makedirs(os.path.dirname(index_out), exist_ok=True)
        
        # 6 - XUẤT INDEX
        faiss.write_index(self.index, str(index_out))
        
        with open(meta_out, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
        logging.info("=== CHÚC MỪNG ===")
        logging.info(" TRIỂN KHAI VECTOR DATABASE THÀNH CÔNG !")
        logging.info(f" VECTORS: {self.index.ntotal}")
        logging.info(f" FILE FAISS INDEX: {index_out}")
        logging.info(f" FILE METADATA: {meta_out}")

# THỰC THI 
if __name__ == "__main__":
    try:
        indexer = VectorIndexer()
        indexer.build_database(str(CHUNK_JSON), str(INDEX_PATH), str(META_PATH))
    except Exception as e:
        logging.critical(f"ERROR: LỖI FATAL{str(e)}")