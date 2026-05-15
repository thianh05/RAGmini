from asyncio import WindowsSelectorEventLoopPolicy
import json
import logging
import re
import hashlib
import torch
import numpy as np

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    CLEAN_JSON,
    CHUNK_JSON,
    EMBEDDING_MODEL,
)

# 1. Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

class SemanticDocumentChunker:

    # Tách câu > Semantic Chunking > Kiểm soát độ dài chunk
    def __init__(self, model_name: str = EMBEDDING_MODEL, break_percentile: int = 85):

        # Chọn device
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Semantic Engine -> {self.device.upper()}")

        # Load Embedding Model
        self.model = SentenceTransformer(model_name, device=self.device)

        self.break_percentile = break_percentile
        self.max_chunk_length = 1000

    # 2. Tách câu + Metadata
    def _split_into_sentences(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        sentences_meta = []

        for page in pages:

            text = page.get("text", "")
            page_num = page.get("page", "Unknown")

            # Tách câu
            raw_sentences = re.split(r'(?<=[.!?])\s+', text)

            for sentence in raw_sentences:

                clean_sentence = sentence.strip()

                # Bỏ câu quá ngắn
                if len(clean_sentence) > 20:

                    sentences_meta.append({
                        "text": clean_sentence,
                        "page": page_num,
                    })

        return sentences_meta

    # 3. Tạo Chunk ID
    def _generate_chunk_id(self, text: str, pages_list: List[Any]) -> str:

        unique_string = (
            f"pages_{'_'.join(map(str, set(pages_list)))}_{text}"
        ).encode('utf-8')

        return hashlib.md5(unique_string).hexdigest()

    # 4. Semantic Chunking
    def _semantic_split(self, sentences_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        # Nếu quá ít câu -> gom luôn
        if len(sentences_meta) <= 2:

            return [{
                "text": " ".join(s["text"] for s in sentences_meta),
                "pages": list(set(s["page"] for s in sentences_meta))
            }]

        # Lấy text
        texts = [s["text"] for s in sentences_meta]

        # Encode sentence -> vector
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False
        )

        distances = []

        # Tính semantic distance
        for i in range(len(embeddings) - 1):

            similarity = cosine_similarity(
                [embeddings[i]],
                [embeddings[i + 1]]
            )[0][0]

            distances.append(1.0 - similarity)

        # Threshold semantic split
        threshold = np.percentile(
            distances,
            self.break_percentile
        )

        chunks_meta = []

        current_texts = [sentences_meta[0]["text"]]
        current_pages = [sentences_meta[0]["page"]]

        # Build semantic chunks
        for i, distance in enumerate(distances):

            # Khác chủ đề -> tạo chunk mới
            if distance > threshold:

                chunks_meta.append({
                    "text": " ".join(current_texts),
                    "pages": list(set(current_pages))
                })

                current_texts = [sentences_meta[i + 1]["text"]]
                current_pages = [sentences_meta[i + 1]["page"]]

            # Cùng chủ đề -> gộp chunk
            else:

                current_texts.append(sentences_meta[i + 1]["text"])
                current_pages.append(sentences_meta[i + 1]["page"])

        # Thêm chunk cuối
        if current_texts:

            chunks_meta.append({
                "text": " ".join(current_texts),
                "pages": list(set(current_pages))
            })

        return chunks_meta

    # 5. Kiểm soát độ dài chunk
    def _apply_length_control(self, chunks_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        final_chunks = []

        for chunk_obj in chunks_meta:

            chunk_text = chunk_obj["text"]
            chunk_pages = chunk_obj["pages"]

            # Chunk hợp lệ
            if len(chunk_text) <= self.max_chunk_length:

                final_chunks.append(chunk_obj)

            # Chunk quá dài
            else:

                sub_sentences = re.split(r'(?<=[.!?])\s+', chunk_text)

                temp_texts = []
                current_len = 0

                for sentence in sub_sentences:

                    # Nếu vượt giới hạn
                    if (current_len + len(sentence) > self.max_chunk_length):

                        final_chunks.append({
                            "text": " ".join(temp_texts),
                            "pages": chunk_pages
                        })

                        # Overlap
                        temp_texts = temp_texts[-1:]
                        current_len = sum(len(s) for s in temp_texts)

                    temp_texts.append(sentence)
                    current_len += len(sentence)

                # Chunk cuối
                if temp_texts:

                    final_chunks.append({
                        "text": " ".join(temp_texts),
                        "pages": chunk_pages
                    })

        return final_chunks

    # 6. Full Pipeline
    def process_pipeline(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        logging.info(f"Processing {len(pages)} pages...")

        # Tách câu
        sentences_meta = self._split_into_sentences(pages)

        # Semantic Chunking
        semantic_chunks = self._semantic_split(sentences_meta)

        # Length Control
        controlled_chunks = self._apply_length_control(semantic_chunks)

        # Build Output
        results = []

        for idx, chunk_obj in enumerate(controlled_chunks):

            primary_page = (
                chunk_obj["pages"][0]
                if chunk_obj["pages"]
                else "Unknown"
            )

            results.append({

                "chunk_id": self._generate_chunk_id(
                    chunk_obj["text"],
                    chunk_obj["pages"]
                ),

                "page": primary_page,
                "chunk_index": idx,
                "char_count": len(chunk_obj["text"]),
                "text": chunk_obj["text"],
            })

        logging.info(f"Generated {len(results)} chunks")

        return results


# Main Execution
if __name__ == "__main__":

    try:

        # Load clean document
        logging.info(f"Loading -> {CLEAN_JSON}")

        with open(CLEAN_JSON, "r", encoding="utf-8") as f:
            clean_pages = json.load(f)

        # Initialize Chunker
        chunker = SemanticDocumentChunker(
            break_percentile=85
        )

        # Run Pipeline
        chunks = chunker.process_pipeline(clean_pages)

        # Save chunk output
        with open(CHUNK_JSON, "w", encoding="utf-8") as f:

            json.dump(
                chunks,
                f,
                ensure_ascii=False,
                indent=2
            )

        logging.info(f"Saved -> {CHUNK_JSON}")

    except Exception as e:

        logging.critical(
            f"Pipeline Error -> {str(e)}"
        )