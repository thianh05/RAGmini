import streamlit as st
import requests
import re
import os
import logging

# Cấu hình Logging cho Frontend
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [FRONTEND] - %(message)s')

# ==========================================
# 1. CẤU HÌNH TRANG RAGstudio
# ==========================================
st.set_page_config(
    page_title="RAGstudio", 
    layout="centered", # 'centered' giúp khung chat nhìn gọn và cao cấp hơn 'wide'
    page_icon="🏛️"
)

def load_assets():
    """Nạp file tĩnh với cơ chế cảnh báo an toàn"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, "assets", "style.css")
    js_path = os.path.join(base_dir, "assets", "scripts.js")
    
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        if os.path.exists(js_path):
            with open(js_path, "r", encoding="utf-8") as f:
                st.markdown(f"<script>{f.read()}</script>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Chạy ở chế độ an toàn: Không tìm thấy thư mục 'assets/style.css'.")

load_assets()

# ==========================================
# 2. UI HEADER & STATE MANAGEMENT
# ==========================================
st.markdown('<h1 style="text-align: center; font-weight: 800; letter-spacing: -1px;">RAGstudio</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle" style="text-align: center; color: #666; margin-bottom: 2rem;">Giáo trình Triết học Mác-Lênin AI</p>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 3. RENDER LỊCH SỬ CHAT (CÓ INLINE CSS BACKUP)
# ==========================================
for message in st.session_state.messages:
    if message["role"] == "user":
        # iMessage Blue Style
        st.markdown(f'<div style="background: #007AFF; color: white; padding: 12px 18px; border-radius: 20px 20px 4px 20px; max-width: 80%; margin-left: auto; margin-bottom: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">{message["content"]}</div>', unsafe_allow_html=True)
    elif message["role"] == "ai":
        # Apple Light Gray Style
        st.markdown(f'<div style="background: #F2F2F7; color: black; padding: 12px 18px; border-radius: 20px 20px 20px 4px; max-width: 85%; margin-right: auto; margin-bottom: 0.5rem; line-height: 1.6;">{message["content"]}</div>', unsafe_allow_html=True)
        
        if message.get("pages"):
            pills = "".join([f'<span style="display: inline-block; background: #E5E5EA; color: #333; font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 12px; margin-right: 6px; margin-bottom: 1rem;">📄 Trang {p}</span>' for p in message["pages"]])
            st.markdown(f'<div>{pills}</div>', unsafe_allow_html=True)

# ==========================================
# 4. XỬ LÝ CÂU HỎI MỚI (BULLETPROOF STREAMING)
# ==========================================
if query := st.chat_input("Hỏi gì đó..."):
    
    # 1. Hiển thị câu hỏi của User
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f'<div class="user-bubble">{query}</div>', unsafe_allow_html=True)

    # 2. Khởi tạo Placeholder thay vì dùng with st.empty() (An toàn hơn)
    chat_placeholder = st.empty()
    chat_placeholder.markdown('<div class="ai-bubble" style="color:#86868b !important;">Đang tra cứu giáo trình... 🔍</div>', unsafe_allow_html=True)
        
    try:
        API_URL = "http://127.0.0.1:8000/ask_stream"
        response = requests.get(API_URL, params={"query": query}, stream=True, timeout=120)
        response.raise_for_status() 
        
        full_response = ""
        pages = []
        
        # 3. Đọc dữ liệu thô (Raw Stream)
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                full_response += chunk
                
                # Ẩn thẻ <SOURCES> ra khỏi màn hình trong lúc đang stream chữ
                display_text = re.sub(r'<SOURCES>.*', '', full_response, flags=re.DOTALL)
                chat_placeholder.markdown(f'<div class="ai-bubble">{display_text}▌</div>', unsafe_allow_html=True)
        
        # 4. Bóc tách số trang (Pills) khi stream hoàn tất
        if "<SOURCES>" in full_response:
            match = re.search(r'<SOURCES>(.*?)</SOURCES>', full_response)
            if match:
                raw_pages = match.group(1).split(',')
                pages = [p.strip() for p in raw_pages if p.strip()]
            
            # Xóa sạch thẻ rác để lấy nội dung text tinh khiết
            full_response = re.sub(r'<SOURCES>.*?</SOURCES>', '', full_response).strip()

        # 5. Lưu bộ nhớ và Render kết quả chốt hạ
        st.session_state.messages.append({"role": "ai", "content": full_response, "pages": pages})
        
        final_html = f'<div class="ai-bubble">{full_response}</div>'
        if pages:
            pills = "".join([f'<span class="source-pill">Trang {p}</span>' for p in pages])
            final_html += f'<div class="source-pills-container">{pills}</div>'
            
        chat_placeholder.markdown(final_html, unsafe_allow_html=True)
        
    except requests.exceptions.ConnectionError:
        chat_placeholder.error("LỖI : CHƯA BẬT SERVER")
    except Exception as e:
        chat_placeholder.error(f"LỖI HỆ THỐNG {e}")