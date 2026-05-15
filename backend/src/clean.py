import re
import json
import logging

from typing import List, Dict, Any
from load_pdf import load_pdf
from config import PDF_PATH, CLEAN_JSON


# [1] Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)


class DocumentCleaner:

    # [2] Khởi tạo các rule làm sạch dữ liệu
    def __init__(
        self,
        custom_headers_to_remove: List[str] = None
    ):

        # [2.1] Regex nhận diện số trang
        self.page_number_pattern = re.compile(
            r"(Trang\s+\d+|^\s*\d+\s*$|^\s*-\s*\d+\s*-\s*$)",
            re.MULTILINE
        )

        # [2.2] Xóa từ bị ngắt dòng bằng dấu "-"
        self.hyphenation_pattern = re.compile(
            r"-\n\s*"
        )

        # [2.3] Chuyển xuống dòng đơn thành khoảng trắng
        self.single_newline_pattern = re.compile(
            r"(?<!\n)\n(?!\n)"
        )

        # [2.4] Gộp nhiều khoảng trắng thành 1
        self.multiple_spaces_pattern = re.compile(
            r"[ \t]+"
        )

        # [2.5] Danh sách header/footer cần loại bỏ
        self.headers = custom_headers_to_remove or [

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

            "Trang 1", "Trang 2", "Trang 3",
            "Trang 4", "Trang 5", "Trang 6",
            "Trang 7", "Trang 8", "Trang 9",
            "Trang 10",

            "©",
            "(C)",
            "Bản quyền",
            "All rights reserved",

            "...",
            "……",
            "---",
            "___"
        ]

    # [3] Xóa header / footer lặp lại
    def _remove_headers(
        self,
        text: str
    ) -> str:

        for header in self.headers:
            text = text.replace(header, "")

        return text

    # [4] Làm sạch nội dung văn bản
    def clean_text(
        self,
        raw_text: str
    ) -> str:

        # [4.1] Kiểm tra dữ liệu rỗng
        if not raw_text:
            return ""

        # [4.2] Xóa header và số trang
        text = self._remove_headers(raw_text)

        text = self.page_number_pattern.sub(
            "",
            text
        )

        # [4.3] Ghép lại các từ bị xuống dòng
        text = self.hyphenation_pattern.sub(
            "",
            text
        )

        # [4.4] Chuyển xuống dòng đơn thành khoảng trắng
        text = self.single_newline_pattern.sub(
            " ",
            text
        )

        # [4.5] Chuẩn hóa khoảng trắng
        text = self.multiple_spaces_pattern.sub(
            " ",
            text
        )

        # [4.6] Xóa ký tự null
        text = text.replace(
            "\x00",
            ""
        )

        return text.strip()

    # [5] Chạy pipeline làm sạch dữ liệu
    def process_pipeline(
        self,
        raw_pages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        logging.info(
            f"Đang làm sạch "
            f"{len(raw_pages)} trang"
        )

        cleaned_pages = []

        # [5.1] Duyệt từng trang
        for page in raw_pages:

            page_num = page.get(
                "page",
                0
            )

            raw_text = page.get(
                "text",
                ""
            )

            cleaned_text = self.clean_text(
                raw_text
            )

            # [5.2] Giữ lại trang hợp lệ
            if len(cleaned_text) > 30:

                cleaned_pages.append({

                    "page": page_num,
                    "text": cleaned_text
                })

        logging.info(
            f"Làm sạch thành công "
            f"{len(cleaned_pages)} trang"
        )

        return cleaned_pages


# [6] Điểm bắt đầu chương trình
if __name__ == "__main__":

    try:

        # [6.1] Đọc dữ liệu PDF gốc
        logging.info(
            f"Đang tải file PDF -> {PDF_PATH}"
        )

        raw_data = load_pdf(
            str(PDF_PATH)
        )

        # [6.2] Khởi tạo bộ làm sạch
        cleaner = DocumentCleaner(

            custom_headers_to_remove=[

                "Giáo trình Triết học Mác - Lênin",
                "Bộ Giáo dục và Đào tạo"
            ]
        )

        # [6.3] Chạy pipeline làm sạch
        clean_data = cleaner.process_pipeline(
            raw_data
        )

        # [6.4] Lưu dữ liệu đã làm sạch
        with open(
            CLEAN_JSON,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                clean_data,
                f,
                ensure_ascii=False,
                indent=2
            )

        logging.info(
            f"Đã lưu dữ liệu sạch -> "
            f"{CLEAN_JSON}"
        )

    except Exception as e:

        # [6.5] Xử lý lỗi pipeline
        logging.critical(
            f"Pipeline thất bại -> {e}"
        )