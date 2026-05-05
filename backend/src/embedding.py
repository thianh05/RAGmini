import json
import logging
import os
import time

# PyTorch dùng để kiểm tra GPU CUDA
import torch

# FAISS dùng làm vector database
import faiss

# Xử lý vector
import numpy as np

# Embedding model
from sentence_transformers import SentenceTransformer

# Config project
from config import (
    CHUNK_JSON,
    INDEX_PATH,
    META_PATH,
    EMBEDDING_MODEL
)


# =====================================================
# 1. LOGGING
# =====================================================
# Hiển thị log hệ thống
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)


# =====================================================
# 2. VECTOR INDEXER
# =====================================================
# Class chịu trách nhiệm:
#
# - Load chunk dữ liệu
# - Encode text -> vector
# - Build FAISS index
# - Save vector database
class VectorIndexer:

    # =================================================
    # INIT
    # =================================================
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL
    ):

        # =============================================
        # KIỂM TRA GPU CUDA
        # =============================================
        # Nếu có GPU:
        # -> dùng cuda
        #
        # Không có:
        # -> dùng cpu
        self.device = (
            'cuda'
            if torch.cuda.is_available()
            else 'cpu'
        )

        logging.info(
            f"KHỞI ĐỘNG {self.device.upper()}"
        )

        # =============================================
        # LOAD EMBEDDING MODEL
        # =============================================
        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

        # FAISS index
        self.index = None

        # Metadata document
        self.metadata = []

    # =================================================
    # BUILD VECTOR DATABASE
    # =================================================
    def build_database(
        self,
        input_json: str,
        index_out: str,
        meta_out: str
    ):

        # =================================================
        # STEP 1: KIỂM TRA FILE INPUT
        # =================================================
        if not os.path.exists(input_json):

            logging.critical(
                f"ERROR : KHÔNG TÌM THẤY FILE "
                f"{input_json} > CHẠY CHUNK TRƯỚC"
            )

            return

        # =================================================
        # LOAD JSON DATA
        # =================================================
        logging.info(
            f"NẠP DỮ LIỆU TỪ {input_json}"
        )

        with open(
            input_json,
            'r',
            encoding='utf-8'
        ) as f:

            # metadata chứa:
            # text, page, source,...
            self.metadata = json.load(f)

        # =================================================
        # KIỂM TRA JSON RỖNG
        # =================================================
        if not self.metadata:

            logging.warning(
                "ERROR : FILE JSON KHÔNG CÓ DỮ LIỆU"
            )

            return

        # =================================================
        # PREPARE TEXT
        # =================================================
        # Format chuẩn cho model E5:
        # "passage: ..."
        texts = [

            f"passage: {chunk['text']}"

            for chunk in self.metadata
        ]

        # =================================================
        # STEP 2: ENCODING
        # =================================================
        logging.info(
            f"ÉP XUNG MÃ HÓA "
            f"{len(texts)} CHUNKS "
            f"- PHÂN BỔ BATCH SIZE"
        )

        start_time = time.time()

        # =================================================
        # STEP 3: TEXT -> VECTOR
        # =================================================
        embeddings = self.model.encode(

            # Danh sách text
            texts,

            # Batch size lớn
            # -> encode nhanh hơn
            batch_size=64,

            # Hiện progress bar
            show_progress_bar=True,

            # Output numpy array
            convert_to_numpy=True
        )

        # =================================================
        # STEP 4: NORMALIZE VECTOR
        # =================================================
        # Normalize L2:
        #
        # giúp:
        # cosine similarity chính xác hơn
        #
        # Sau normalize:
        # inner product ~= cosine similarity
        faiss.normalize_L2(embeddings)

        encode_time = time.time() - start_time

        logging.info(
            f"TÍNH TOÁN TRONG "
            f"{encode_time:.2f} GIÂY "
            f"- CHIỀU VECTOR: {embeddings.shape[1]}"
        )

        # =================================================
        # STEP 5: KHỞI TẠO FAISS INDEX
        # =================================================

        # Số chiều vector
        dimension = embeddings.shape[1]

        # =============================================
        # IndexFlatIP
        # =============================================
        # IP = Inner Product
        #
        # Vì vector đã normalize:
        # -> IP ≈ Cosine Similarity
        #
        # Ưu điểm:
        # - đơn giản
        # - chính xác
        # - phù hợp dataset vừa và nhỏ
        self.index = faiss.IndexFlatIP(
            dimension
        )

        # Add toàn bộ vector vào FAISS
        self.index.add(embeddings)

        # =================================================
        # TẠO THƯ MỤC OUTPUT
        # =================================================
        os.makedirs(
            os.path.dirname(index_out),
            exist_ok=True
        )

        # =================================================
        # STEP 6: SAVE DATABASE
        # =================================================

        # Save FAISS index
        faiss.write_index(
            self.index,
            str(index_out)
        )

        # Save metadata JSON
        with open(
            meta_out,
            'w',
            encoding='utf-8'
        ) as f:

            json.dump(
                self.metadata,
                f,
                ensure_ascii=False,
                indent=2
            )

        # =================================================
        # SUCCESS LOG
        # =================================================
        logging.info("======")

        logging.info(
            "TRIỂN KHAI VECTOR DATABASE THÀNH CÔNG !"
        )

        logging.info(
            f"VECTORS: {self.index.ntotal}"
        )

        logging.info(
            f"FILE FAISS INDEX: {index_out}"
        )

        logging.info(
            f"FILE METADATA: {meta_out}"
        )


# =====================================================
# 3. MAIN EXECUTION
# =====================================================
# File chạy trực tiếp:
#
# python embed.py
if __name__ == "__main__":

    try:

        # Khởi tạo indexer
        indexer = VectorIndexer()

        # Build vector database
        indexer.build_database(
            str(CHUNK_JSON),
            str(INDEX_PATH),
            str(META_PATH)
        )

    except Exception as e:

        logging.critical(
            f"ERROR: LỖI FATAL {str(e)}"
        )