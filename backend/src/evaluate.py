import csv
import time
import logging

from datetime import datetime
from typing import List, Dict, Callable, Any

import concurrent.futures


# [1] Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)


class RAGEvaluator:

    """
    Pipeline đánh giá hệ thống RAG:

    1. Chạy truy vấn test
    2. Thu thập kết quả retrieval
    3. Tính metrics đánh giá
    4. Tổng hợp báo cáo
    5. Xuất CSV report
    """

    # [2] Khởi tạo evaluator
    def __init__(
        self,
        retriever_func: Callable,
        top_k: int = 4
    ):

        # [2.1] Hàm retrieval engine
        self.retriever = retriever_func

        # [2.2] Số lượng kết quả retrieve
        self.top_k = top_k

        # [2.3] Danh sách kết quả đánh giá
        self.results: List[Dict[str, Any]] = []

    # [3] Đánh giá một query
    def _evaluate_single_query(
        self,
        test_case: Dict[str, Any]
    ) -> Dict[str, Any]:

        query = test_case["query"]

        expected_pages = set(
            test_case["expected_pages"]
        )

        start_time = time.time()

        # [3.1] Gọi retrieval engine
        try:

            retrieved_chunks = self.retriever(
                query,
                top_k=self.top_k
            )

            retrieved_pages = [

                chunk.get("page", -1)

                for chunk in retrieved_chunks
            ]

        except Exception as e:

            logging.error(
                f"Lỗi retrieval query "
                f"'{query}': {e}"
            )

            retrieved_chunks = []
            retrieved_pages = []

        # [3.2] Tính latency query
        latency = time.time() - start_time

        # [3.3] Khởi tạo metrics
        hit = False

        rank = 0

        precision_hits = 0

        # [3.4] So sánh kết quả retrieval
        for i, page in enumerate(retrieved_pages):

            # Query hit ground-truth page
            if page in expected_pages:

                # Ghi nhận hit đầu tiên
                if not hit:

                    hit = True

                    rank = i + 1

                precision_hits += 1

        # [3.5] Tính Precision@K
        precision_at_k = (

            precision_hits / self.top_k

            if self.top_k > 0

            else 0
        )

        # [3.6] Trả kết quả đánh giá
        return {

            "query": query,

            "expected_pages": list(
                expected_pages
            ),

            "retrieved_pages": retrieved_pages,

            "status": (
                "PASS"
                if hit
                else "FAIL"
            ),

            "hit_rank": (
                rank
                if hit
                else 0
            ),

            "precision_at_k": precision_at_k,

            "latency_sec": round(
                latency,
                4
            ),

            # Preview kết quả top-1
            "top_1_snippet": (

                retrieved_chunks[0]['text'][:100] + "..."

                if retrieved_chunks

                else "N/A"
            )
        }

    # [4] Chạy toàn bộ testbench
    def run_testbench(
        self,
        test_data: List[Dict[str, Any]],
        max_workers: int = 4
    ):

        logging.info(
            f"Chạy testbench với "
            f"{len(test_data)} test cases"
        )

        start_time = time.time()

        # [4.1] Chạy song song nhiều queries
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:

            futures = [

                executor.submit(
                    self._evaluate_single_query,
                    case
                )

                for case in test_data
            ]

            # [4.2] Thu thập kết quả
            for future in concurrent.futures.as_completed(
                futures
            ):

                self.results.append(
                    future.result()
                )

        total_time = time.time() - start_time

        logging.info(
            f"Hoàn thành testbench trong "
            f"{total_time:.2f}s"
        )

        # [4.3] Sinh báo cáo đánh giá
        self._generate_reports()

    # [5] Tổng hợp metrics và xuất báo cáo
    def _generate_reports(self):

        total_cases = len(
            self.results
        )

        # [5.1] Kiểm tra dữ liệu đánh giá
        if total_cases == 0:

            logging.warning(
                "Không có dữ liệu đánh giá"
            )

            return

        # [5.2] Tính Hit Rate
        hits = sum(

            1

            for r in self.results

            if r["status"] == "PASS"
        )

        # [5.3] Tính Mean Reciprocal Rank
        mrr_sum = sum(

            1.0 / r["hit_rank"]

            for r in self.results

            if r["hit_rank"] > 0
        )

        # [5.4] Tính Average Latency
        avg_latency = (

            sum(
                r["latency_sec"]
                for r in self.results
            )

            / total_cases
        )

        # [5.5] Tính Average Precision@K
        avg_precision = (

            sum(
                r["precision_at_k"]
                for r in self.results
            )

            / total_cases
        )

        # [5.6] Final metrics
        hit_rate = (
            hits / total_cases
        ) * 100

        mrr = (
            mrr_sum / total_cases
        )

        # [5.7] In báo cáo console
        print("\n" + "=" * 50)

        print(
            " TỔNG KẾT ĐÁNH GIÁ HỆ THỐNG RAG"
        )

        print("=" * 50)

        print(
            f"Tổng số Test Cases  : "
            f"{total_cases}"
        )

        print(
            f"Top-K               : "
            f"{self.top_k}"
        )

        print(
            f"Accuracy (Hit Rate) : "
            f"{hit_rate:.2f}%"
        )

        print(
            f"MRR                 : "
            f"{mrr:.4f}"
        )

        print(
            f"Precision@{self.top_k} : "
            f"{avg_precision:.4f}"
        )

        print(
            f"Avg Latency         : "
            f"{avg_latency:.4f}s"
        )

        print("-" * 50)

        # [5.8] Tạo tên file report
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        csv_file = (
            f"eval_report_{timestamp}.csv"
        )

        # [5.9] Xuất CSV report
        with open(
            csv_file,
            mode="w",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            fieldnames = [

                "query",
                "expected_pages",
                "retrieved_pages",
                "status",
                "hit_rank",
                "precision_at_k",
                "latency_sec",
                "top_1_snippet"
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                self.results
            )

        logging.info(
            f"Đã xuất report -> "
            f"{csv_file}"
        )


# [6] Main execution
if __name__ == "__main__":

    from retrieval import HybridRetriever
    from sentence_transformers import SentenceTransformer

    from config import (
        INDEX_PATH,
        META_PATH,
        EMBEDDING_MODEL
    )

    # [6.1] Ground-truth test data
    TEST_DATA = [

        {
            "query": "Vật chất là gì theo Lênin?",
            "expected_pages": [68, 69, 70]
        },

        {
            "query": "Mối quan hệ giữa cơ sở hạ tầng và kiến trúc thượng tầng?",
            "expected_pages": [155, 156]
        },

        {
            "query": "Quy luật lượng chất phát biểu như thế nào?",
            "expected_pages": [119, 121, 122]
        },

        {
            "query": "Định nghĩa thực tiễn",
            "expected_pages": [102, 103]
        }
    ]

    try:

        # [6.2] Load embedding model
        logging.info(
            "Đang nạp model và FAISS index"
        )

        model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # [6.3] Khởi tạo retrieval engine
        retriever_engine = HybridRetriever(

            index_path=str(INDEX_PATH),
            meta_path=str(META_PATH),
            model=model
        )

        # [6.4] Wrapper cho retrieval function
        def search_wrapper(
            query: str,
            top_k: int
        ):

            return retriever_engine.search(
                query,
                top_k=top_k
            )

        # [6.5] Khởi tạo evaluator
        evaluator = RAGEvaluator(

            retriever_func=search_wrapper,

            top_k=4
        )

        # [6.6] Chạy benchmark đánh giá
        evaluator.run_testbench(

            TEST_DATA,

            max_workers=4
        )

    except Exception as e:

        # [6.7] Xử lý lỗi hệ thống
        logging.critical(
            f"Lỗi khởi chạy testbench -> {e}"
        )