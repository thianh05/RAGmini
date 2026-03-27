import csv
import time
import logging
from datetime import datetime
from typing import List, Dict, Callable, Any
import concurrent.futures

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class RAGEvaluator:
    """
    Hệ thống Đánh giá Tự động (Automated Evaluation Framework) cho RAG.
    Tích hợp xử lý song song để tăng tốc độ chạy Testbench.
    """
    
    def __init__(self, retriever_func: Callable, top_k: int = 4):
        self.retriever = retriever_func
        self.top_k = top_k
        self.results: List[Dict[str, Any]] = []
        
    def _evaluate_single_query(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Đánh giá độc lập 1 câu hỏi (Worker function cho ThreadPool)"""
        query = test_case["query"]
        expected_pages = set(test_case["expected_pages"])
        
        start_time = time.time()
        
        # Gọi engine tìm kiếm
        try:
            retrieved_chunks = self.retriever(query, top_k=self.top_k)
            retrieved_pages = [chunk.get("page", -1) for chunk in retrieved_chunks]
        except Exception as e:
            logging.error(f"Lỗi khi truy xuất query '{query}': {e}")
            retrieved_chunks, retrieved_pages = [], []
            
        latency = time.time() - start_time
        
        # Tính toán Logic các Metrics
        hit = False
        rank = 0
        precision_hits = 0
        
        for i, page in enumerate(retrieved_pages):
            if page in expected_pages:
                if not hit: # Chỉ lấy rank của kết quả đúng ĐẦU TIÊN để tính MRR
                    hit = True
                    rank = i + 1
                precision_hits += 1 # Đếm tổng số kết quả đúng trong top_k
                
        # Tính Precision@K (Độ chuẩn xác trong K kết quả trả về)
        precision_at_k = precision_hits / self.top_k if self.top_k > 0 else 0
        
        return {
            "query": query,
            "expected_pages": list(expected_pages),
            "retrieved_pages": retrieved_pages,
            "status": "PASS" if hit else "FAIL",
            "hit_rank": rank if hit else 0,
            "precision_at_k": precision_at_k,
            "latency_sec": round(latency, 4),
            "top_1_snippet": retrieved_chunks[0]['text'][:100] + "..." if retrieved_chunks else "N/A"
        }

    def run_testbench(self, test_data: List[Dict[str, Any]], max_workers: int = 4):
        """
        Khởi chạy Testbench song song. 
        Tận dụng Multi-threading để vượt qua giới hạn I/O bound khi query model.
        """
        logging.info(f"Bắt đầu chạy Testbench với {len(test_data)} test cases. Threads: {max_workers}")
        start_time = time.time()
        
        # Xử lý bất đồng bộ bằng ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Gửi task vào pool
            futures = [executor.submit(self._evaluate_single_query, case) for case in test_data]
            
            # Thu thập kết quả khi hoàn thành
            for future in concurrent.futures.as_completed(futures):
                self.results.append(future.result())
                
        total_time = time.time() - start_time
        logging.info(f"Hoàn thành Testbench trong {total_time:.2f} giây.")
        
        self._generate_reports()

    def _generate_reports(self):
        """Tính toán tổng thể và xuất báo cáo chuẩn Enterprise"""
        total_cases = len(self.results)
        if total_cases == 0:
            logging.warning("Không có dữ liệu để xuất báo cáo.")
            return

        hits = sum(1 for r in self.results if r["status"] == "PASS")
        mrr_sum = sum(1.0 / r["hit_rank"] for r in self.results if r["hit_rank"] > 0)
        avg_latency = sum(r["latency_sec"] for r in self.results) / total_cases
        avg_precision = sum(r["precision_at_k"] for r in self.results) / total_cases

        hit_rate = (hits / total_cases) * 100
        mrr = mrr_sum / total_cases

        # Xuất Console
        print("\n" + "=" * 50)
        print(" TỔNG KẾT ĐÁNH GIÁ HỆ THỐNG RAG (CI/CD REPORT)")
        print("=" * 50)
        print(f"Tổng số Test Cases  : {total_cases}")
        print(f"Tham số Top-K       : {self.top_k}")
        print(f"Accuracy (Hit Rate) : {hit_rate:.2f}%")
        print(f"MRR                 : {mrr:.4f}")
        print(f"Average Precision@{self.top_k} : {avg_precision:.4f}")
        print(f"Avg Latency/Query   : {avg_latency:.4f} giây")
        print("-" * 50)

        # Xuất File CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"eval_report_{timestamp}.csv"
        
        with open(csv_file, mode="w", encoding="utf-8-sig", newline="") as f:
            fieldnames = ["query", "expected_pages", "retrieved_pages", "status", "hit_rank", "precision_at_k", "latency_sec", "top_1_snippet"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
            
        logging.info(f"Đã xuất file báo cáo chi tiết: {csv_file}")


# ==========================================
# GIAO DIỆN THỰC THI (MÔ PHỎNG)
# ==========================================
if __name__ == "__main__":
    from retrieval import HybridRetriever
    from sentence_transformers import SentenceTransformer
    import os
    from config import INDEX_PATH, META_PATH, EMBEDDING_MODEL

    # Dữ liệu Ground Truth
    TEST_DATA = [
        {"query": "Vật chất là gì theo Lênin?", "expected_pages": [68, 69, 70]},
        {"query": "Mối quan hệ giữa cơ sở hạ tầng và kiến trúc thượng tầng?", "expected_pages": [155, 156]},
        {"query": "Quy luật lượng chất phát biểu như thế nào?", "expected_pages": [119, 121, 122]},
        {"query": "Định nghĩa thực tiễn", "expected_pages": [102, 103]}
    ]

    try:
        logging.info("Nạp Model & Index để bắt đầu kiểm thử...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        retriever_engine = HybridRetriever(
            index_path=str(INDEX_PATH), 
            meta_path=str(META_PATH), 
            model=model
        )

        # Wrapper function để tương thích với Evaluator
        def search_wrapper(query: str, top_k: int):
            return retriever_engine.search(query, top_k=top_k)

        # Khởi chạy Evaluator với 4 luồng chạy song song
        evaluator = RAGEvaluator(retriever_func=search_wrapper, top_k=4)
        evaluator.run_testbench(TEST_DATA, max_workers=4)

    except Exception as e:
        logging.critical(f"Lỗi khởi chạy Testbench: {e}")