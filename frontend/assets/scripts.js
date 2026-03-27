/* =========================================================
   RAGstudio - Smart Auto-Scroll Engine (Enterprise Grade)
   ========================================================= */

// Đặt trong khối try-catch để ngăn ứng dụng crash nếu Streamlit 
// thay đổi chính sách bảo mật iframe trên các trình duyệt khác nhau
try {
    // Tìm kiếm khung nhập liệu để đảm bảo DOM đã load xong
    const chatInputContainer = parent.document.querySelector('div[data-testid="stChatInput"]');

    if (chatInputContainer) {
        let isUserScrollingUp = false;
        let scrollTimeout; // Biến dùng để chống lag (Debounce)

        // ---------------------------------------------------------
        // 1. NHẬN DIỆN HÀNH VI NGƯỜI DÙNG (USER INTENT)
        // ---------------------------------------------------------
        parent.window.addEventListener('scroll', () => {
            const scrollPosition = parent.window.scrollY + parent.window.innerHeight;
            const totalHeight = parent.document.body.scrollHeight;
            
            // Nếu người dùng cuộn ngược lên trên hơn 150px so với đáy trang,
            // hệ thống sẽ ngầm hiểu: "À, sếp đang muốn đọc lại lịch sử cũ".
            // Lúc này, auto-scroll sẽ bị TẠM KHÓA để không làm phiền sếp đọc bài.
            isUserScrollingUp = (totalHeight - scrollPosition > 150);
        });

        // ---------------------------------------------------------
        // 2. HÀM CUỘN TRANG MƯỢT MÀ (SMOOTH SCROLL)
        // ---------------------------------------------------------
        const smoothScrollToBottom = () => {
            if (!isUserScrollingUp) {
                parent.window.scrollTo({
                    top: parent.document.body.scrollHeight,
                    behavior: 'smooth' // Tạo hiệu ứng trượt êm ái chuẩn Apple
                });
            }
        };

        // ---------------------------------------------------------
        // 3. MẮT THẦN THEO DÕI DOM (MUTATION OBSERVER)
        // ---------------------------------------------------------
        const observer = new MutationObserver((mutations) => {
            let shouldScroll = false;
            
            mutations.forEach((mutation) => {
                // Bắt sự kiện: Có bong bóng chat mới xuất hiện HOẶC AI đang gõ từng chữ
                if (mutation.addedNodes.length > 0 || mutation.type === 'characterData') {
                    shouldScroll = true;
                }
            });

            // 🚀 BÍ QUYẾT CHỐNG LAG (DEBOUNCE ALGORITHM): 
            // Thay vì cuộn trang 50 lần/giây (làm cháy CPU máy tính),
            // Hệ thống sẽ đợi 100ms. Trong 100ms đó, các chữ AI nhả ra sẽ được gom lại,
            // sau đó trình duyệt mới cuộn mượt xuống 1 lần.
            if (shouldScroll) {
                clearTimeout(scrollTimeout);
                scrollTimeout = setTimeout(smoothScrollToBottom, 100); 
            }
        });
        
        // ---------------------------------------------------------
        // 4. KÍCH HOẠT OBSERVER VÀO VÙNG CHỨA CHAT
        // ---------------------------------------------------------
        const chatBlock = parent.document.querySelector('div[data-testid="stMainBlockContainer"]');
        if (chatBlock) {
            observer.observe(chatBlock, {
                childList: true,      // Theo dõi số lượng tin nhắn tăng lên
                subtree: true,        // Theo dõi sâu vào các thẻ div con bên trong
                characterData: true   // Bắt buộc phải có: Theo dõi sự biến đổi của dòng chữ đang stream
            });
        }
    }
} catch (error) {
    console.warn("RAGstudio Engine: Không thể can thiệp vào DOM gốc do hạn chế của Iframe.", error);
}