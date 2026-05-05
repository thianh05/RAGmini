import re
import json
import logging
from typing import List, Dict, Any
from load_pdf import load_pdf 
from config import PDF_PATH, CLEAN_JSON

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class DocumentCleaner:
    def __init__(self, custom_headers_to_remove: List[str] = None):
        
        self.page_number_pattern = re.compile(r"(Trang\s+\d+|^\s*\d+\s*$|^\s*-\s*\d+\s*-\s*$)", re.MULTILINE)
        self.hyphenation_pattern = re.compile(r"-\n\s*") 
        self.single_newline_pattern = re.compile(r"(?<!\n)\n(?!\n)") 
        self.multiple_spaces_pattern = re.compile(r"[ \t]+") 
        self.headers = custom_headers_to_remove or [
        # TIÊU ĐỀ CHUNG
        "Giáo trình Triết học Mác - Lênin",
        "Giáo trình Triết học Mác – Lênin",
        "TRIẾT HỌC MÁC - LÊNIN",
        "TRIẾT HỌC MÁC – LÊNIN",
        "Tài liệu lưu hành nội bộ",
        "Lưu hành nội bộ",
        "Không phát hành",
        "Dành cho sinh viên",
        "Dành cho đào tạo",
        "Nhà xuất bản Chính trị quốc gia",
        "Nhà xuất bản Chính trị quốc gia Sự thật",
        "NXB Chính trị quốc gia",
        "NXB Chính trị quốc gia Sự thật",
        "Chương",
        "CHƯƠNG",
        "Bài",
        "BÀI",
        "Mục",
        "MỤC",
        "Trang",
        "TRANG",
        "Trang 1", "Trang 2", "Trang 3", "Trang 4", "Trang 5",
        "Trang 6", "Trang 7", "Trang 8", "Trang 9", "Trang 10",
        "©",
        "(C)",
        "Bản quyền",
        "All rights reserved",
        "...",
        "……",
        "---",
        "___"
    ]

    # XÓA HEADER, FOOTER, SỐ TRANG
    def _remove_headers(self, text: str) -> str:
        for header in self.headers:
            text = text.replace(header, "")
        return text
    # XÓA NGĂN CÁCH TỪ VỰNG DO NGẮT DÒNG
    def clean_text(self, raw_text: str) -> str:
        if not raw_text:
            return "" 
              
        # 1 - XÓA HEADER, FOOTER, SỐ TRANG
        text = self._remove_headers(raw_text)
        text = self.page_number_pattern.sub("", text)
        # 2 - XÓA NGĂN CÁCH TỪ VỰNG DO NGẮT DÒNG
        text = self.hyphenation_pattern.sub("", text)
        # 3 - GIỮ NGUYÊN CÁC NGĂN CÁCH DÒNG 
        text = self.single_newline_pattern.sub(" ", text)
        # 4 - GIẢM NHIỀU KHOẢNG TRẮNG THÀNH 1 KHOẢNG TRẮNG
        text = self.multiple_spaces_pattern.sub(" ", text)
        text = text.replace("\x00", "") 
        return text.strip()

    def process_pipeline(self, raw_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
       
        logging.info(f"BẮT ĐẦU LÀM SẠCH {len(raw_pages)} TRANG TÀI LIỆU.")
        cleaned_pages = []
        # LOOP - CLEAN
        for page in raw_pages:
            page_num = page.get("page", 0)
            raw_text = page.get("text", "")
            
            cleaned_text = self.clean_text(raw_text)
            # 30 KÝ TỰ TRỞ LÊN THÌ GIỮ 
            if len(cleaned_text) > 30: 
                cleaned_pages.append({
                    "page": page_num,
                    "text": cleaned_text
                })

        logging.info(f"HOÀN THÀNH LÀM SẠCH {len(cleaned_pages)} TRANG")
        return cleaned_pages

# THỰC THI 
if __name__ == "__main__":
    try:
        # 1 - FILE GỐC
        logging.info(f" ĐANG LẤY FILE GỐC {PDF_PATH}")
        raw_data = load_pdf(str(PDF_PATH))
        
        # 2 - KHỞI TẠO BỘ LỌC 
        cleaner = DocumentCleaner(
            custom_headers_to_remove=[
                "Giáo trình Triết học Mác - Lênin", 
                "Bộ Giáo dục và Đào tạo"
            ]
        )
        # 3 - XỬ LÝ LÀM SẠCH
        clean_data = cleaner.process_pipeline(raw_data)
        # 4 - XUẤT FILE 
        with open(CLEAN_JSON, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=2)
            
        logging.info(f"DỮ LIỆU AN TOÀN ĐÃ ĐƯỢC XUẤT RA {CLEAN_JSON}")
        
    except Exception as e:
        logging.critical(f"TIẾN TRÌNH BỊ GIÁN ĐOẠN {e}")