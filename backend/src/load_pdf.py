import pdfplumber

FILE_PATH = "../data/triet.pdf"
def load_pdf(file_path):

    pages = []
    
    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                pages.append({
                    "page": page_number + 1,
                    "text": page_text
                })
    return pages
if __name__ == "__main__":
    pages = load_pdf(FILE_PATH)
    print("Total pages:", len(pages))
    print("\nSample text:\n")
    print(pages[0]["text"][:300])