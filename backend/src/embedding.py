import json
import logging
import os
import time

import torch
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_JSON,
    INDEX_PATH,
    META_PATH,
    EMBEDDING_MODEL
)


# [1] Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)


class VectorIndexer:

    """
    Pipeline vector database:

    1. Load chunk data
    2. Encode text thành vector
    3. Normalize embedding
    4. Build FAISS index
    5. Lưu vector database
    """

    # [2] Khởi tạo vector indexer
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL
    ):

        # [2.1] Chọn thiết bị xử lý
        self.device = (
            'cuda'
            if torch.cuda.is_available()
            else 'cpu'
        )

        logging.info(
            f"Khởi động trên "
            f"{self.device.upper()}"
        )

        # [2.2] Load embedding model
        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

        # [2.3] Khởi tạo FAISS index
        self.index = None

        # [2.4] Metadata document
        self.metadata = []

    # [3] Build vector database
    def build_database(
        self,
        input_json: str,
        index_out: str,
        meta_out: str
    ):

        # [3.1] Kiểm tra file input
        if not os.path.exists(input_json):

            logging.critical(
                f"Không tìm thấy file "
                f"{input_json}"
            )

            return

        # [3.2] Load chunk metadata
        logging.info(
            f"Nạp dữ liệu từ {input_json}"
        )

        with open(
            input_json,
            'r',
            encoding='utf-8'
        ) as f:

            self.metadata = json.load(f)

        # [3.3] Kiểm tra dữ liệu rỗng
        if not self.metadata:

            logging.warning(
                "File JSON không có dữ liệu"
            )

            return

        # [3.4] Chuẩn bị text cho embedding model
        texts = [

            f"passage: {chunk['text']}"

            for chunk in self.metadata
        ]

        # [3.5] Encode text thành vector
        logging.info(
            f"Đang encode "
            f"{len(texts)} chunks"
        )

        start_time = time.time()

        embeddings = self.model.encode(

            # Danh sách text input
            texts,

            # Batch encode để tăng tốc
            batch_size=64,

            # Hiển thị progress bar
            show_progress_bar=True,

            # Output numpy array
            convert_to_numpy=True
        )

        # [3.6] Normalize vector embedding
        # Sau normalize:
        # Inner Product ≈ Cosine Similarity
        faiss.normalize_L2(
            embeddings
        )

        encode_time = time.time() - start_time

        logging.info(
            f"Encode hoàn tất trong "
            f"{encode_time:.2f}s"
        )

        logging.info(
            f"Vector dimension: "
            f"{embeddings.shape[1]}"
        )

        # [3.7] Khởi tạo FAISS index
        dimension = embeddings.shape[1]

        # IndexFlatIP:
        # dùng inner product search
        self.index = faiss.IndexFlatIP(
            dimension
        )

        # [3.8] Add vector vào index
        self.index.add(
            embeddings
        )

        # [3.9] Tạo thư mục output
        os.makedirs(
            os.path.dirname(index_out),
            exist_ok=True
        )

        # [3.10] Lưu FAISS index
        faiss.write_index(
            self.index,
            str(index_out)
        )

        # [3.11] Lưu metadata
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

        # [3.12] Log kết quả
        logging.info(
            "Build vector database thành công"
        )

        logging.info(
            f"Tổng vectors: "
            f"{self.index.ntotal}"
        )

        logging.info(
            f"FAISS index -> "
            f"{index_out}"
        )

        logging.info(
            f"Metadata -> "
            f"{meta_out}"
        )


# [4] Main execution
if __name__ == "__main__":

    try:

        # [4.1] Khởi tạo indexer
        indexer = VectorIndexer()

        # [4.2] Build vector database
        indexer.build_database(

            str(CHUNK_JSON),

            str(INDEX_PATH),

            str(META_PATH)
        )

    except Exception as e:

        # [4.3] Xử lý lỗi hệ thống
        logging.critical(
            f"Lỗi pipeline -> {str(e)}"
        )