# TÀI LIỆU BÀN GIAO XÂY DỰNG ỨNG DỤNG WEB (HANDOFF SPECIFICATION)
**Dự án:** Hệ thống Phân tích Phản hồi Sinh viên Việt Nam & Đề xuất Cải thiện Học phần (VSFC Feedback Analyzer)  
**Phiên bản tài liệu:** 1.0  
**Thư mục gốc dự án:** `d:\VSFC-Feedback-Analyzer\`  
**Đối tượng nhận bàn giao:** Nhóm phát triển Web App (Backend API & Frontend UI/UX)

---

## 1. TỔNG QUAN HỆ THỐNG & PHÂN VAI MÔ HÌNH AI

### 1.1. Mục tiêu của ứng dụng Web
Xây dựng ứng dụng Web tương tác thời gian thực với **2 chế độ người dùng (chuyển đổi nhanh trên thanh điều hướng, không cần đăng nhập phức tạp)** phục vụ báo cáo đồ án và triển khai thực tế:
1. **Chế độ Sinh viên (Student Mode):** Sinh viên chọn môn học, đánh giá mức độ hài lòng (thang điểm Likert 1–5 sao) và nhập ý kiến đóng góp tự do. Hệ thống tự động **ẩn danh hóa thông tin cá nhân (PII Masking)** ngay lập tức, phân tích ngầm bằng AI và đẩy trực tiếp sang danh sách dữ liệu của Giảng viên theo thời gian thực.
2. **Chế độ Giảng viên / Nhà trường (Lecturer Mode):** Giảng viên xem thống kê phản hồi thời gian thực (bao gồm các góp ý sinh viên vừa gửi và dữ liệu từ file khảo sát tải lên), tải lên bất kỳ file Excel/CSV khảo sát "bẩn" nào để hệ thống tự động nhận diện cấu trúc, xem giải thích từ khóa **SHAP (Explainable AI)**, duyệt các câu mâu thuẫn (`needs_review`) và nhận **bảng Gợi ý hành động cải thiện môn học (Actionable Insights)**.

---

### 1.2. Phân biệt rõ: Mô hình Chính (PhoBERT) vs. Mô hình Cơ sở (TF-IDF Baseline)

Để nhóm làm App và nhóm thuyết trình thống nhất 100%, hệ thống AI bàn giao gồm **2 nhóm mô hình** với vai trò rõ ràng như sau:

| Phân loại | Tên mô hình | Độ chính xác trên tập Test (`3,166` câu) | Vai trò trong Đồ án & trên Web App |
| :--- | :--- | :--- | :--- |
| **MÔ HÌNH CHÍNH (Primary Model)** | **`PhoBERT Multi-Task`** (`vinai/phobert-base-v2` với 1 bộ mã hóa dùng chung + 2 nhánh phân loại Cảm xúc & Chủ đề) | • **Cảm xúc:** Macro F1 = **`0.7928`**, Accuracy = **`91.66%`**<br>• **Chủ đề:** Macro F1 = **`0.7775`**, Accuracy = **`86.51%`** | • **Là bộ não chính mặc định của toàn hệ thống.**<br>• Dùng để dự đoán đồng thời Cảm xúc + Chủ đề + Độ tự tin (Confidence) cho mọi phản hồi sinh viên gửi trực tiếp và các file Excel tải lên.<br>• Hiểu sâu ngữ cảnh tiếng Việt, từ ghép và các câu có cấu trúc đảo ngữ/nhượng bộ (*"thầy dạy nhiệt tình nhưng giọng hơi nhỏ"*). |
| **MÔ HÌNH CƠ SỞ (Baseline Models)** | **Bộ 3 mô hình `TF-IDF`**:<br>1. `ComplementNB`<br>2. `LogisticRegression`<br>3. `LinearSVC` *(Baseline tốt nhất)* | • **Cảm xúc (`LinearSVC`):** Macro F1 = **`0.7380`**, Accuracy = **`89.13%`**<br>• **Chủ đề (`LinearSVC`):** Macro F1 = **`0.7447`**, Accuracy = **`84.74%`** | • **Làm mốc so sánh khoa học (Benchmark):** Chứng minh việc dùng Deep Learning (`PhoBERT Multi-Task`) giúp tăng **+5.48% Macro F1 Cảm xúc** và **+3.28% Macro F1 Chủ đề** so với phương pháp học máy truyền thống.<br>• **Chế độ Siêu tốc (Fast Mode):** Cho phép giảng viên bật khi muốn xử lý nhanh các file Excel lớn hàng ngàn dòng trên máy tính chỉ có CPU yếu (< 0.2 giây / 3,000 dòng).<br>• **Trích xuất trọng số từ khóa SHAP tức thì:** Mô hình `lr_sentiment.pkl` + `tfidf_vectorizer.pkl` kết hợp với `LinearExplainer` giúp tô màu mức độ đóng góp SHAP của từng từ trong câu trong thời gian `< 20ms` mà không làm treo giao diện. |

---

## 2. DANH MỤC CÁC FILE BÀN GIAO & ĐƯỜNG DẪN CHI TIẾT

Nhóm AI/Data đã huấn luyện, kiểm định và xuất sẵn toàn bộ tài nguyên trong thư mục `d:\VSFC-Feedback-Analyzer\`. Bên làm App chỉ cần đọc đúng các đường dẫn dưới đây:

### 2.1. Thư mục Mô hình (`d:\VSFC-Feedback-Analyzer\models\`)

| Đường dẫn File | Dung lượng | Ý nghĩa & Cách sử dụng trên App |
| :--- | :---: | :--- |
| `models/phobert_multitask/phobert_multitask.pt` | ~540 MB | Trọng số (`state_dict`) của **Mô hình Chính `PhoBERT Multi-Task`** (đã huấn luyện trên Google Colab GPU T4). Đầu ra gồm 2 vector logits: `logits_sentiment` (3 chiều) và `logits_topic` (4 chiều). |
| `models/phobert_multitask/vocab.txt`<br>`models/phobert_multitask/bpe.codes`<br>`models/phobert_multitask/tokenizer_config.json`<br>`models/phobert_multitask/added_tokens.json` | ~2.0 MB | Bộ từ điển BPE và cấu hình Tokenizer của `vinai/phobert-base-v2` đi kèm với `phobert_multitask.pt`. |
| `models/tfidf_vectorizer.pkl` | ~425 KB | Bộ biến đổi văn bản thành vector TF-IDF (`ngram_range=(1,2)`, `max_features=12000`, chỉ fit trên tập Train). Dùng cho chế độ Baseline và tính SHAP. |
| `models/svc_sentiment.pkl` | ~290 KB | Mô hình Baseline tốt nhất cho bài toán Cảm xúc (`LinearSVC` đã tinh chỉnh tham số). |
| `models/svc_topic.pkl` | ~385 KB | Mô hình Baseline tốt nhất cho bài toán Chủ đề (`LinearSVC` đã tinh chỉnh tham số). |
| `models/lr_sentiment.pkl` | ~290 KB | Mô hình `LogisticRegression` cho bài toán Cảm xúc — dùng để xuất xác suất Baseline và **khởi tạo bộ giải thích từ khóa `shap.LinearExplainer`**. |
| `models/lr_topic.pkl` | ~385 KB | Mô hình `LogisticRegression` cho bài toán Chủ đề. |
| `models/model_metadata.json` | ~1 KB | File JSON lưu sẵn các chỉ số đánh giá chính thức trên tập Test (`3,166` câu) của cả `PhoBERT Multi-Task` và `TF-IDF Baseline` để hiển thị trực tiếp lên **Tab 4 (Đối chiếu hiệu năng Model)** trên giao diện. |

### 2.2. Thư mục Dữ liệu Đã xử lý & Kết quả Mẫu (`d:\VSFC-Feedback-Analyzer\data\processed\`)

| Đường dẫn File | Số dòng | Ý nghĩa & Cách sử dụng trên App |
| :--- | :---: | :--- |
| `data/processed/test_pred_phobert.csv` | `3,166` dòng | **Dữ liệu mặc định để nạp sẵn lên Dashboard Giảng viên khi vừa mở App.** Chứa 3,166 câu phản hồi tập Test kèm nhãn thật, nhãn dự đoán từ `PhoBERT Multi-Task` (`pred_sentiment`, `pred_topic`), độ tự tin (`sentiment_conf`, `topic_conf`) và cờ cảnh báo (`needs_review`). |
| `data/processed/test_pred_baseline.csv` | `3,166` dòng | Kết quả dự đoán trên 3,166 câu tập Test của mô hình `TF-IDF Baseline` (để đối chiếu trực tiếp từng câu với PhoBERT khi cần). |
| `data/processed/actionable_insights.csv` | `8` nhóm vấn đề | Bảng tổng hợp các nhóm vấn đề tiêu cực, số lượng phản hồi, từ khóa đại diện, câu trích dẫn mẫu và **Khuyến nghị hành động cụ thể cho Giảng viên/Nhà trường**. Dùng để hiển thị ngay trên **Tab 1** của Dashboard. |
| `data/processed/sample_excel_predicted.xlsx` | `120` dòng | File Excel mẫu xuất ra sau khi chạy qua toàn bộ pipeline (ẩn danh hóa, dự đoán Cảm xúc/Chủ đề, gắn cờ `needs_review`). Dùng làm mẫu định dạng cho nút **"Xuất báo cáo Excel"** trên App. |
| `data/processed/train_clean.csv` | `11,426` dòng | Tập dữ liệu huấn luyện đã làm sạch (dùng lấy 100 mẫu ngẫu nhiên làm tập nền `background` khi khởi tạo `shap.LinearExplainer`). |

### 2.3. Thư mục 10 File Excel "Bẩn" để Test tính năng Upload (`d:\VSFC-Feedback-Analyzer\data\samples_messy\`)
Bên làm App sử dụng trực tiếp 10 file trong thư mục này để kiểm thử tính năng **"Tải lên & Tự động nhận diện file khảo sát bất kỳ"**:
1. `01_clean.xlsx`: File Excel chuẩn, dòng tiêu đề ở dòng đầu tiên (`header_row = 0`).
2. `02_title_rows.xlsx`: Có 5 dòng tiêu đề biểu mẫu trường đại học ở đầu (`header_row = 5`) + có cột điểm số **Likert (1–5)**.
3. `03_english_cols.xlsx`: Tên cột bằng tiếng Anh (`Student_Feedback`, `Course_Code`, `Rating`).
4. `04_multi_sheet.xlsx`: File có nhiều sheet (`HuongDan`, `ThongKe`, `DuLieu`), dữ liệu thực nằm ở sheet `DuLieu`.
5. `05_legacy.xls`: Định dạng Excel 97–2003 cũ (`.xls`).
6. `06_semicolon_bom.csv`: File CSV xuất từ Excel châu Âu/Việt Nam dùng dấu chấm phẩy `;` và mã hóa `UTF-8-SIG`.
7. `07_utf16_tab.csv`: File CSV/TSV mã hóa `UTF-16` phân tách bằng phím Tab (`\t`).
8. `08_two_text_cols.xlsx`: File khảo sát tách làm **2 cột văn bản riêng biệt** (`Ưu điểm` và `Nhược điểm`).
9. `09_long_paragraphs.xlsx`: File có dòng tiêu đề phụ lặp lại giữa bảng và các đoạn nhận xét dài nhiều câu.
10. `10_html_as_xls.xls`: File bảng HTML được hệ thống giáo vụ trường đại học đổi đuôi thành `.xls`.

### 2.4. Thư mục Biểu đồ Thống kê (`d:\VSFC-Feedback-Analyzer\reports\figures\`)
Các hình ảnh chất lượng cao có thể nhúng vào phần Giới thiệu dữ liệu hoặc Tab Đối chiếu Mô hình:
- `eda_sentiment_dist.png`: Phân bố 3 lớp Cảm xúc.
- `eda_topic_dist.png`: Phân bố 4 lớp Chủ đề.
- `eda_word_count_hist.png`: Phân bố độ dài câu phản hồi.
- `eda_topic_sentiment_heatmap.png`: Tương quan giữa Chủ đề và Cảm xúc.
- `cm_sentiment_test.png` & `cm_topic_test.png`: Ma trận nhầm lẫn (Confusion Matrix) trên tập Test.
- `shap_top_negative_words.png`: Top 15 từ khóa đóng góp mạnh nhất vào dự đoán Tiêu cực theo SHAP.

---

## 3. BẢNG QUY ĐỔI NHÃN CHUẨN (LABEL MAPPING)

Mọi mô hình, file CSV và API trong hệ thống **bắt buộc** tuân thủ bảng mã hóa nhãn sau:

### 3.1. Nhãn Cảm xúc (`sentiment` / `pred_sentiment`)
| Mã số (ID) | Mã định danh (Slug) | Tên hiển thị Tiếng Việt | Màu sắc UI khuyến nghị |
| :---: | :--- | :--- | :--- |
| **`0`** | `negative` | **Tiêu cực (Cần cải thiện)** | Đỏ (`#EF4444` / Nền `#FEF2F2`) |
| **`1`** | `neutral` | **Trung lập (Bình thường)** | Vàng cam hoặc Xám (`#F59E0B` / Nền `#FFFBEB`) |
| **`2`** | `positive` | **Tích cực (Khen ngợi)** | Xanh lá (`#10B981` / Nền `#ECFDF5`) |

### 3.2. Nhãn Chủ đề (`topic` / `pred_topic`)
| Mã số (ID) | Mã định danh (Slug) | Tên hiển thị Tiếng Việt | Biểu tượng (Icon) gợi ý |
| :---: | :--- | :--- | :--- |
| **`0`** | `lecturer` | **Giảng viên** *(Phương pháp dạy, tác phong, nhiệt tình)* | 👨‍🏫 Giảng viên |
| **`1`** | `training_program` | **Chương trình đào tạo** *(Giáo trình, slide, bài tập, đề thi)* | 📚 Chương trình & Tài liệu |
| **`2`** | `facility` | **Cơ sở vật chất** *(Phòng học, máy chiếu, điều hòa, phòng máy)* | 🏫 Cơ sở vật chất |
| **`3`** | `others` | **Khác** *(Nhận xét chung không thuộc 3 nhóm trên)* | 💬 Khác |

---

## 4. ĐẶC TẢ LUỒNG NGHIỆP VỤ (BUSINESS WORKFLOWS)

### 4.1. Luồng 1: Tiền xử lý văn bản & Bảo vệ danh tính Sinh viên (PII Anonymization)
Mọi câu phản hồi (dù nhập từ form Sinh viên hay đọc từ file Excel của Giảng viên) đều phải đi qua 6 bước chuẩn hóa theo đúng thứ tự đã thiết kế trong Notebook `01`:
1. **Chuẩn hóa bảng mã Unicode:** Đưa toàn bộ chuỗi về chuẩn dựng sẵn **`NFC`** và chuyển về chữ thường (`lowercase`).
2. **Thu gọn ký tự lặp kéo dài:** Các từ kéo dài cảm xúc từ 3 ký tự giống nhau trở lên (ví dụ: `"buồnnnn"`, `"hayyyy"`, `"lunnnn"`) được rút gọn về tối đa 2 ký tự.
3. **Chuẩn hóa từ viết tắt / Teencode học đường:** Thay thế các từ viết tắt phổ biến của sinh viên Việt Nam về từ chuẩn:
   - `k`, `ko`, `hok`, `hong`, `kh` $\rightarrow$ `không`
   - `dc`, `đc` $\rightarrow$ `được`
   - `gv` $\rightarrow$ `giảng viên` ; `sv` $\rightarrow$ `sinh viên`
   - `vs` $\rightarrow$ `với` ; `cx` $\rightarrow$ `cũng` ; `r` $\rightarrow$ `rồi`
   - `bt` $\rightarrow$ `bình thường` ; `j` $\rightarrow$ `gì` ; `z`, `v` $\rightarrow$ `vậy`
4. **Ẩn danh hóa Thông tin Cá nhân (PII Masking — Bắt buộc thực hiện trước khi lưu DB):**
   - **Email:** Mọi chuỗi dạng email $\rightarrow$ thay bằng `[EMAIL]`.
   - **Số điện thoại:** Chuỗi số điện thoại 10–11 số Việt Nam $\rightarrow$ thay bằng `[PHONE]`.
   - **Mã số sinh viên (MSSV):** Chuỗi 7–10 chữ số liền nhau hoặc mã dạng `B20DCCN...`, `2052xxxx`, `SE16xxxx` $\rightarrow$ thay bằng `[STUDENT_ID]`.
   - **Tên riêng đi sau danh xưng:** Cụm `"thầy/cô + Tên riêng viết hoa"` (ví dụ: `"thầy Nguyễn Văn A"`, `"cô Lan"`) $\rightarrow$ thay bằng `"thầy/cô [gv]"` để báo cáo tập trung vào chất lượng môn học một cách khách quan.
5. **Kiểm tra câu không mang nội dung (`is_no_content`):**
   - Nếu câu rỗng, chỉ chứa dấu câu/ký tự đặc biệt, hoặc khớp với các mẫu câu bỏ trống khảo sát như: *"không có"*, *"không có ý kiến"*, *"không ý kiến"*, *"ko có ý kiến"*, *"không ạ"*, *"bình thường không có gì"* $\rightarrow$ đánh dấu `is_no_content = True`.
   - Các câu có `is_no_content = True` được gán nhãn mặc định là `Trung lập (1)` & `Khác (3)`, và **loại khỏi thống kê tỷ lệ khen/chê** để không làm nhiễu biểu đồ của giảng viên.
6. **Ghép từ tiếng Việt (`sentence_seg`):**
   - Nối các từ ghép học đường quan trọng bằng dấu gạch dưới `_` (ví dụ: `giảng_viên`, `sinh_viên`, `bài_tập`, `kiến_thức`, `thực_hành`, `lý_thuyết`, `phòng_học`, `máy_chiếu`, `điều_hòa`, `máy_lạnh`, `phòng_máy`, `cơ_sở_vật_chất`, `chương_trình`, `giáo_trình`, `tài_liệu`, `nhiệt_tình`, `dễ_hiểu`, `khó_hiểu`) trước khi đưa vào `PhoBERT Multi-Task`.

---

### 4.2. Luồng 2: Sinh viên gửi phản hồi trực tiếp & Đồng bộ Thời gian thực sang Giảng viên
1. Người dùng đang ở **"Chế độ Sinh viên"** trên thanh điều hướng:
   - Chọn **Học phần / Môn học** từ danh sách (ví dụ: *Nhập môn Trí tuệ Nhân tạo - IT001*, *Cấu trúc Dữ liệu & Giải thuật - IT002*, *Cơ sở Dữ liệu - IT003*, *Mạng Máy tính - IT004*).
   - Chọn **Mức độ hài lòng chung (Điểm Likert từ 1 đến 5 sao)**.
   - Nhập **Ý kiến đóng góp tự do** vào ô văn bản (có nút bấm thử nhanh các câu mẫu Khen / Chê / Có chứa SĐT-MSSV để test lúc thuyết trình).
2. Khi Sinh viên bấm nút **"Gửi phản hồi ẩn danh"**:
   - Backend chạy ngay **Luồng 1** để che PII (`sentence_masked`) và chạy **Mô hình Chính (`PhoBERT Multi-Task`)** để lấy nhãn Cảm xúc, Chủ đề, Độ tự tin và kiểm tra `needs_review`.
   - Lưu bản ghi mới vào đầu danh sách phản hồi chung của hệ thống (SQLite hoặc bộ nhớ phiên làm việc).
   - **Trên màn hình Sinh viên:** Hiển thị thông báo *"Đã gửi phản hồi ẩn danh thành công!"* kèm **Hộp minh bạch bảo mật** cho sinh viên thấy câu của mình đã được hệ thống tự động xóa sạch MSSV/SĐT/Email như thế nào trước khi gửi tới giảng viên.
3. Khi người thuyết trình chuyển công tắc trên thanh điều hướng sang **"Chế độ Giảng viên"**:
   - Bản ghi sinh viên vừa gửi **xuất hiện ngay ở dòng đầu tiên** của bảng dữ liệu Giảng viên, các con số KPI tổng quan và biểu đồ tự động cập nhật tăng thêm 1 phản hồi mới.

---

### 4.3. Luồng 3: Tự động nhận diện & Phân tích File Excel/CSV "Bẩn" bất kỳ
Khi Giảng viên tải lên một file khảo sát (`.xlsx`, `.xls`, `.csv`, `.tsv`):
1. **Bước 1 — Đọc đa định dạng (Multi-format Reader):**
   - Kiểm tra xem file `.xls` có phải là bảng HTML đội lốt không (kiểm tra chữ `<html` hoặc `<table` ở đầu file). Nếu đúng, đọc bằng bộ phân tích bảng HTML và giữ nguyên dòng tiêu đề `<th>`.
   - Nếu là `.xlsx` hoặc `.xls` thật: đọc danh sách tất cả các Sheet.
   - Nếu là `.csv` / `.tsv`: tự động thử lần lượt các bảng mã (`utf-8-sig`, `utf-16`, `utf-8`, `cp1258`, `latin1`) và tự động dò dấu phân cách (`,`, `;`, `\t`).
2. **Bước 2 — Tự động chọn Sheet chứa dữ liệu (`pick_best_sheet`) & Dò dòng tiêu đề (`detect_header_row`):**
   - Bỏ qua các sheet hướng dẫn/trống, chọn sheet có nhiều dòng và cột có nghĩa nhất.
   - Quét 15 dòng đầu tiên của sheet để tìm dòng có mật độ chuỗi văn bản ngắn không trùng lặp cao nhất làm `header_row` (giúp bỏ qua các dòng quốc hiệu, tiêu đề trường/khoa ở đầu file Excel).
   - Loại bỏ các dòng bị lặp lại y hệt dòng tiêu đề ở giữa bảng (lỗi xuất trang in thường gặp).
3. **Bước 3 — Chấm điểm tự động tìm Cột chứa Ý kiến phản hồi (`detect_text_columns`):**
   - Loại bỏ ngay các cột có tên trùng với từ khóa định danh cá nhân (`mssv`, `mã sv`, `họ tên`, `email`, `số điện thoại`, `id`, `stt`, `timestamp`).
   - Chấm điểm từng cột còn lại dựa trên: độ dài văn bản trung bình (`>= 12` ký tự), tỷ lệ câu không trùng lặp, tỷ lệ chứa dấu tiếng Việt và điểm cộng nếu tên cột chứa từ khóa (`ý kiến`, `góp ý`, `nhận xét`, `phản hồi`, `feedback`, `comment`, `ưu điểm`, `nhược điểm`).
   - Hỗ trợ trường hợp file có **2 cột văn bản riêng biệt** (ví dụ: cột `Ưu điểm` và cột `Nhược điểm`): hệ thống nhận diện cả 2 cột để phân tích từng ý kiến.
   - Đồng thời tự động dò **cột điểm số Likert (1–5)** nếu có trong file (cột kiểu số có giá trị nằm trong tập `{1, 2, 3, 4, 5}`).
4. **Bước 4 — Màn hình Xác nhận thông minh & Chạy hàng loạt:**
   - Hiển thị cho Giảng viên thấy kết quả tự động dò: *"Đã nhận diện Sheet: `DuLieu` | Dòng tiêu đề: `5` | Cột ý kiến: `Góp ý` | Đã tự động loại bỏ 2 cột thông tin cá nhân (`MSSV`, `Họ tên`)"* kèm 5 dòng xem trước.
   - Giảng viên bấm **"Bắt đầu phân tích"** (hoặc đổi cột nếu muốn) $\rightarrow$ Hệ thống chạy phân tích toàn bộ các dòng, cập nhật lên Dashboard và cho phép tải về file Excel kết quả đầy đủ.

---

### 4.4. Luồng 4: Cơ chế Gắn cờ Cần kiểm tra lại (`needs_review`)
Một câu phản hồi được hệ thống tự động bật cờ `needs_review = True` để nhắc Giảng viên xem xét thủ công khi rơi vào **1 trong 3 trường hợp**:
1. **Độ tự tin của mô hình thấp:** Xác suất dự đoán Cảm xúc (`sentiment_conf < 0.65`) hoặc Chủ đề (`topic_conf < 0.65`).
2. **Mâu thuẫn giữa Điểm chấm sao (Likert 1–5) và Nội dung chữ viết:**
   - Sinh viên chấm **1–2 sao** (rất tệ) nhưng AI dự đoán câu chữ là **Tích cực (`2`)**.
   - Hoặc sinh viên chấm **4–5 sao** (rất tốt) nhưng AI dự đoán câu chữ là **Tiêu cực (`0`)** *(thường gặp khi sinh viên bấm nhầm cột sao hoặc viết câu khen trước chê sau)*.
3. **Câu chứa từ nối tương phản đa chiều:** Câu vừa có từ khen vừa có từ liên từ đối lập (`"nhưng"`, `"tuy nhiên"`, `"mặc dù"`, `"chỉ có điều"`).

---

### 4.5. Luồng 5: Giải thích từ khóa SHAP & Đề xuất Hành động (`Actionable Insights`)
1. **Giải thích cục bộ từng câu (Local SHAP Explanation):**
   - Khi Giảng viên bấm vào bất kỳ câu phản hồi nào trên bảng, hệ thống trả về danh sách các từ/cụm từ trong câu kèm giá trị đóng góp SHAP (`shap_value`).
   - Giao diện tô màu nền trực tiếp lên từng từ trong câu:
     - **Màu Đỏ đậm/nhạt:** Các từ đẩy dự đoán về phía **Tiêu cực** (ví dụ: `khó_hiểu`, `nhanh`, `đi_trễ`, `nóng`, `mờ`).
     - **Màu Xanh lá đậm/nhạt:** Các từ đẩy dự đoán về phía **Tích cực** (ví dụ: `nhiệt_tình`, `dễ_hiểu`, `tận_tâm`, `chi_tiết`).
2. **Gợi ý hành động tự động theo khía cạnh (Actionable Course Improvement Insights):**
   - Toàn bộ các câu được dự đoán là **Tiêu cực (`0`)** được gom cụm theo Chủ đề + Nhóm từ khóa vấn đề (8 nhóm chuẩn trong Notebook `05`) và xếp hạng từ vấn đề bị phàn nàn nhiều nhất xuống ít nhất, kèm **Khuyến nghị hành động cụ thể**:
     1. `[Giảng viên]` *Phương pháp giảng dạy & truyền đạt* (`khó hiểu, nhanh, buồn ngủ, nhàm chán`) $\rightarrow$ **Khuyến nghị:** Rà soát tốc độ giảng, bổ sung ví dụ minh họa thực tế và dành 5–10 phút hỏi đáp cuối buổi.
     2. `[Chương trình đào tạo]` *Khối lượng bài tập & độ khó đề thi* (`bài tập, quá nhiều, nặng, đề thi`) $\rightarrow$ **Khuyến nghị:** Cân đối lại khối lượng bài tập về nhà giữa kỳ và công bố đề cương/cấu trúc đề thi rõ ràng.
     3. `[Chương trình đào tạo]` *Giáo trình, slide & tài liệu học tập* (`tài liệu, slide, giáo trình, cũ`) $\rightarrow$ **Khuyến nghị:** Cập nhật lại slide bài giảng và cung cấp tài liệu tham khảo tiếng Việt trước buổi học.
     4. `[Cơ sở vật chất]` *Thiết bị trình chiếu & âm thanh* (`máy chiếu, mic, loa, mờ`) $\rightarrow$ **Khuyến nghị:** Báo bộ phận quản trị thiết bị kiểm tra/thay mới bóng đèn máy chiếu và micro phòng học.
     5. `[Giảng viên]` *Tác phong & giờ giấc giảng viên* (`đi trễ, nghỉ, báo nghỉ, trễ giờ`) $\rightarrow$ **Khuyến nghị:** Đảm bảo lịch lên lớp đúng giờ và thông báo sớm cho lớp khi có lịch dạy bù.
     6. `[Cơ sở vật chất]` *Cấu hình máy tính phòng thực hành* (`phòng máy, máy tính, chậm, lag`) $\rightarrow$ **Khuyến nghị:** Nâng cấp cấu hình hoặc cài đặt đồng bộ phần mềm tại phòng máy thực hành trước kỳ học.
     7. `[Cơ sở vật chất]` *Nhiệt độ phòng học & điều hòa* (`nóng, máy lạnh, điều hòa, quạt`) $\rightarrow$ **Khuyến nghị:** Kiểm tra hệ thống điều hòa/quạt thông gió tại phòng học trước giờ lên lớp.
     8. `[Chương trình đào tạo]` *Thời lượng thực hành & lý thuyết* (`thực hành, phòng lab, ít tiết, thời lượng`) $\rightarrow$ **Khuyến nghị:** Tăng thời lượng hướng dẫn thực hành trực tiếp và chữa bài mẫu trên lớp.

---

## 5. ĐẶC TẢ GIAO DIỆN MÀN HÌNH (UI/UX SCREEN SPECIFICATIONS)

### 5.1. Thanh Điều hướng Trên cùng (Top Navigation Bar — Luôn hiển thị)
- **Bên trái:** Logo & Tên hệ thống: **"VSFC-Insight | Hệ thống Phân tích Phản hồi Sinh viên & Cải thiện Học phần"**.
- **Ở giữa (Nút chuyển đổi vai trò nhanh - Role Switcher Pill):**
  - Nút 1: `🎓 Chế độ Sinh viên (Gửi phản hồi ẩn danh)`
  - Nút 2: `📊 Chế độ Giảng viên (Dashboard Phân tích & XAI)` *(Có hiển thị chấm thông báo số lượng phản hồi mới vừa gửi từ Chế độ Sinh viên)*.
- **Bên phải (Khi ở Chế độ Giảng viên):**
  - Công tắc chọn Engine phân tích:
    - `🧠 Chính xác cao (PhoBERT Multi-Task - Mặc định)`
    - `⚡ Siêu tốc (TF-IDF Baseline - Dành cho file lớn)`

---

### 5.2. Màn hình 1: Chế độ Sinh viên (`🎓 Student Feedback Portal`)
Thiết kế gọn gàng, thân thiện như một cổng khảo sát học phần hiện đại:
1. **Khối Biểu mẫu Gửi góp ý:**
   - **Dropdown chọn Môn học:** Ví dụ `IT001 - Nhập môn Trí tuệ Nhân tạo`, `IT002 - Cấu trúc Dữ liệu & Giải thuật`, `IT003 - Hệ Cơ sở Dữ liệu`, `IT004 - Mạng Máy tính`.
   - **Bộ chọn Điểm hài lòng (1 đến 5 sao Likert):** Có nhãn mô tả từ *1 sao (Rất không hài lòng)* đến *5 sao (Rất hài lòng)*.
   - **Ô nhập nhận xét (`Textarea`):** Cho phép sinh viên nhập tiếng Việt tự nhiên hoặc teencode.
   - **Thanh "Câu mẫu thử nhanh" (Quick Demo Chips):** 3 nút nhỏ để khi thuyết trình bấm 1 phát điền sẵn câu mẫu:
     - *Mẫu 1 (Khen):* `"Thầy dạy cực kỳ nhiệt tình, dễ hiểu, bài tập sát thực tế."`
     - *Mẫu 2 (Chê + Teencode):* `"Phòng máy thực hành máy tính quá chậm và lag, gv giảng hơi nhanh sv ko theo kịp."`
     - *Mẫu 3 (Kiểm tra Ẩn danh PII + Mâu thuẫn sao):* `"Em là MSSV 21520123, sđt 0909123456, thầy Minh dạy hơi nhanh và slide chữ nhỏ khó nhìn ạ."`
   - **Nút bấm chính:** `"🔒 Gửi phản hồi ẩn danh"`.
2. **Khối Xác nhận & Minh bạch Bảo mật (Hiện ra ngay sau khi bấm Gửi):**
   - Thông báo xanh: *"Đã ghi nhận ý kiến của bạn vào hệ thống ẩn danh!"*
   - Hiển thị đối chiếu **Câu gốc bạn nhập** vs **Câu sau khi Hệ thống Che danh tính (PII Masked)** (làm nổi bật các thẻ `[STUDENT_ID]`, `[PHONE]`, `[EMAIL]`, `thầy/cô [gv]`) để chứng minh hệ thống không lưu thông tin cá nhân của sinh viên.
   - Nút bấm tắt: *"👉 Chuyển sang Chế độ Giảng viên để xem câu này trên Dashboard"*.

---

### 5.3. Màn hình 2: Chế độ Giảng viên (`📊 Lecturer Analytics Dashboard`)
Mặc định khi vừa mở lên sẽ hiển thị dữ liệu từ file `data/processed/test_pred_phobert.csv` (`3,166` câu) cộng với các câu mới do Sinh viên vừa gửi trực tiếp. Giao diện chia làm **4 Tab chức năng**:

#### **TAB 1: Tổng quan & Gợi ý Cải thiện Học phần (Overview & Actionable Insights)**
1. **Hàng 4 Thẻ Chỉ số KPI (KPI Cards) ở trên cùng:**
   - **Tổng số phản hồi hợp lệ:** Tổng số câu (kèm chú thích số câu không có nội dung `is_no_content` đã được lọc bỏ).
   - **Tỷ lệ Tích cực (%):** Phần trăm câu Khen ngợi (màu xanh lá).
   - **Tỷ lệ Tiêu cực (%):** Phần trăm câu Phàn nàn cần xử lý (màu đỏ).
   - **Số câu Cần kiểm tra lại (`needs_review`):** Số câu độ tự tin thấp hoặc lệch điểm sao Likert (màu cam, bấm vào nhảy sang Tab 3 lọc riêng các câu này).
2. **Hàng Biểu đồ Trực quan (2 biểu đồ cạnh nhau):**
   - **Biểu đồ 1 (Donut / Bar Chart):** Tỷ lệ phân bố 3 sắc thái Cảm xúc (`Tiêu cực`, `Trung lập`, `Tích cực`).
   - **Biểu đồ 2 (Stacked Bar Chart theo Chủ đề):** Phân bố Cảm xúc trong từng Chủ đề (`Giảng viên`, `Chương trình đào tạo`, `Cơ sở vật chất`, `Khác`) — giúp giảng viên nhìn ngay chủ đề nào đang bị tỷ lệ đỏ (tiêu cực) cao nhất.
3. **Bảng Khuyến nghị Hành động Cải thiện Môn học (Actionable Insights Table — Điểm nhấn quan trọng nhất của Tab 1):**
   - Đọc từ logic gom cụm hoặc file `data/processed/actionable_insights.csv`.
   - Hiển thị danh sách các vấn đề tiêu cực nổi cộm nhất, gồm các cột:
     - **Mức độ ưu tiên & Chủ đề** (Tag màu)
     - **Nhóm vấn đề phát hiện** (ví dụ: *Phương pháp giảng dạy & truyền đạt*)
     - **Số lượng phản hồi tiêu cực** (ví dụ: `82 phản hồi`)
     - **Từ khóa SHAP nổi bật** (`khó hiểu`, `nhanh`, `buồn ngủ`)
     - **Trích dẫn phản hồi thực tế của sinh viên** (2 câu tiêu biểu)
     - **Đề xuất hành động cải thiện cho Giảng viên / Nhà trường** (in đậm, rõ ràng).

#### **TAB 2: Tải lên & Tự động Nhận diện File Excel/CSV Khảo sát (Universal Excel Ingestion)**
1. **Khu vực Kéo-Thả File (Drag & Drop Upload Zone):**
   - Hỗ trợ `.xlsx`, `.xls`, `.csv`, `.tsv`.
   - Có nút chọn nhanh 1 trong **10 file khảo sát bẩn mẫu** từ thư mục `data/samples_messy/` để demo trực tiếp trên lớp mà không mất công tìm file trong máy.
2. **Hộp Kết quả Tự động Nhận diện Cấu trúc File (Auto-Detection Report Card):**
   - Ngay khi chọn file, hiển thị rõ các thông số hệ thống tự dò được:
     - **Sheet được chọn tự động:** (ví dụ `DuLieu`)
     - **Vị trí dòng tiêu đề (`header_row`):** (ví dụ `Dòng 5 - Đã tự động bỏ qua 5 dòng tiêu đề biểu mẫu ở đầu file`)
     - **Cột văn bản phản hồi phát hiện được:** (ví dụ `Góp ý` hoặc cả 2 cột `Ưu điểm, Nhược điểm`)
     - **Cột điểm Likert phát hiện được (nếu có):** (ví dụ `Điểm đánh giá (1-5)`)
     - **Các cột định danh cá nhân đã tự động loại bỏ để bảo mật:** (ví dụ `MSSV`, `Họ và tên`, `Email`)
   - Hiển thị bảng **Xem trước 5 dòng đầu tiên** và cho phép giảng viên bấm nút **"🚀 Chạy Phân tích Toàn bộ File"**.
3. **Kết quả sau khi chạy xong File Excel:**
   - Cập nhật toàn bộ Dashboard theo dữ liệu của file vừa tải lên (hoặc gộp vào dữ liệu hiện tại).
   - Hiển thị nút **"📥 Xuất kết quả ra File Excel (`.xlsx`)"** (file tải về có thêm các cột `Câu đã ẩn danh`, `Cảm xúc dự đoán`, `Độ tự tin Cảm xúc`, `Chủ đề dự đoán`, `Độ tự tin Chủ đề`, `Cần kiểm tra lại (needs_review)` giống mẫu `data/processed/sample_excel_predicted.xlsx`).

#### **TAB 3: Tra cứu Chi tiết, Giải thích Từ khóa SHAP & Duyệt câu Nghi ngờ (XAI Explorer & Active Review)**
1. **Thanh Bộ lọc Đa chiều (Filter Bar):**
   - Lọc theo **Cảm xúc** (`Tất cả`, `Tiêu cực`, `Trung lập`, `Tích cực`).
   - Lọc theo **Chủ đề** (`Tất cả`, `Giảng viên`, `Chương trình đào tạo`, `Cơ sở vật chất`, `Khác`).
   - Nút gạt **`⚠️ Chỉ hiện câu Cần kiểm tra lại (needs_review)`**.
   - Ô tìm kiếm từ khóa tự do.
2. **Bảng Danh sách Câu phản hồi:**
   - Hiển thị: Câu phản hồi (đã che PII), Nhãn Cảm xúc + Thanh phần trăm độ tự tin, Nhãn Chủ đề + Thanh phần trăm độ tự tin, Cờ `needs_review` (kèm lý do: *Độ tự tin thấp* hoặc *Lệch điểm sao Likert*).
   - Cho phép Giảng viên sửa lại nhãn đúng nếu câu đó bị gắn cờ `needs_review`.
3. **Khung Giải thích AI (SHAP Word Attribution Panel — Mở ra khi bấm vào 1 dòng bất kỳ hoặc nhập câu bất kỳ vào ô Test nhanh):**
   - **Tô màu trực tiếp lên từng từ trong câu (Highlighted Text):** Từ mang trọng số SHAP âm (kéo về Tiêu cực) tô nền Đỏ; từ mang trọng số SHAP dương (kéo về Tích cực) tô nền Xanh lá; độ đậm nhạt tỉ lệ thuận với trị tuyệt đối của giá trị SHAP.
   - **Biểu đồ thanh ngang (Horizontal Bar Chart):** Hiển thị Top 5–8 từ trong câu đó có ảnh hưởng mạnh nhất đến quyết định của mô hình kèm con số SHAP cụ thể (ví dụ: `nhiệt_tình (+0.64)`, `dễ_hiểu (+0.51)`, `nhanh (-0.48)`).
   - Hiển thị thêm biểu đồ tổng hợp **Top 15 từ khóa SHAP gây phản hồi tiêu cực mạnh nhất toàn bộ tập dữ liệu** (`reports/figures/shap_top_negative_words.png`).

#### **TAB 4: Đối chiếu Hiệu năng Mô hình (Model Benchmark: PhoBERT vs. TF-IDF Baseline)**
1. **Bảng So sánh Chỉ số Thực nghiệm trên 3,166 câu Test (đọc từ `models/model_metadata.json`):**
   - Thể hiện rõ `PhoBERT Multi-Task` (Mô hình chính) vượt trội hơn `TF-IDF + LinearSVC` (Mô hình cơ sở) ở cả 2 bài toán Cảm xúc (`Macro F1: 0.7928 vs 0.7380`) và Chủ đề (`Macro F1: 0.7775 vs 0.7447`).
2. **Khu vực So sánh Trực tiếp trên Cùng 1 Câu (Side-by-Side Model Comparison):**
   - Có 1 ô nhập câu bất kỳ (ví dụ câu phức có từ `"nhưng"`): bên trái hiện kết quả của **PhoBERT Multi-Task**, bên phải hiện kết quả của **TF-IDF Baseline** để chứng minh trước hội đồng chấm đồ án những trường hợp câu khó mà PhoBERT bắt đúng ngữ cảnh còn TF-IDF bị nhầm.
3. **Hiển thị Ma trận Nhầm lẫn (Confusion Matrices):**
   - Nhúng các biểu đồ từ `reports/figures/` (`cm_sentiment_test.png`, `cm_topic_test.png`, `eda_topic_sentiment_heatmap.png`).

---

## 6. ĐẶC TẢ CẤU TRÚC API & JSON (REST API CONTRACT)

Nếu nhóm làm App tách riêng Backend (FastAPI / Flask) và Frontend (React / Vue / HTML-JS), dưới đây là quy chuẩn các Endpoint và cấu trúc JSON giao tiếp giữa 2 bên:

### 6.1. `POST /api/v1/predict` — Phân tích 1 câu phản hồi & Giải thích từ khóa SHAP
*(Dùng cho cả lúc Sinh viên bấm Gửi phản hồi VÀ lúc Giảng viên bấm vào 1 câu để xem giải thích SHAP)*

- **Request JSON:**
```json
{
  "text": "Em là MSSV 21520999, thầy Minh dạy rất nhiệt tình nhưng phòng học máy chiếu hơi mờ.",
  "course_code": "IT001",
  "likert_score": 4,
  "model_mode": "phobert",
  "include_shap": true,
  "save_to_live_feed": true
}
```
- **Response JSON (`200 OK`):**
```json
{
  "id": "fb_20261028_001",
  "created_at": "2026-10-28T21:35:00+07:00",
  "course_code": "IT001",
  "likert_score": 4,
  "original_preview_masked": "em là mssv [STUDENT_ID], thầy/cô [gv] dạy rất nhiệt tình nhưng phòng học máy chiếu hơi mờ.",
  "sentence_seg": "em là mssv [STUDENT_ID] , thầy/cô [gv] dạy rất nhiệt_tình nhưng phòng_học máy_chiếu hơi mờ .",
  "pii_detected": ["STUDENT_ID", "LECTURER_NAME"],
  "is_no_content": false,
  "model_used": "phobert_multitask",
  "sentiment": {
    "label_id": 0,
    "label_slug": "negative",
    "label_name_vi": "Tiêu cực",
    "confidence": 0.7842,
    "probabilities": {
      "negative": 0.7842,
      "neutral": 0.0615,
      "positive": 0.1543
    }
  },
  "topic": {
    "label_id": 2,
    "label_slug": "facility",
    "label_name_vi": "Cơ sở vật chất",
    "confidence": 0.8410,
    "probabilities": {
      "lecturer": 0.1120,
      "training_program": 0.0250,
      "facility": 0.8410,
      "others": 0.0220
    }
  },
  "needs_review": true,
  "review_reasons": [
    "Điểm đánh giá Likert cao (4 sao) nhưng nội dung dự đoán là Tiêu cực",
    "Câu chứa liên từ tương phản ('nhưng') mang nhiều sắc thái"
  ],
  "shap_explanation": [
    {"token": "máy_chiếu", "shap_value": -0.421, "direction": "negative"},
    {"token": "mờ", "shap_value": -0.385, "direction": "negative"},
    {"token": "nhiệt_tình", "shap_value": 0.312, "direction": "positive"},
    {"token": "phòng_học", "shap_value": -0.140, "direction": "negative"}
  ]
}
```

---

### 6.2. `POST /api/v1/upload/inspect` — Tải file Excel/CSV lên & Tự động dò cấu trúc
- **Request:** `multipart/form-data` chứa `file` (`.xlsx`, `.xls`, `.csv`, `.tsv`) hoặc tên file mẫu trong `data/samples_messy/`.
- **Response JSON (`200 OK`):**
```json
{
  "file_id": "upload_8821",
  "filename": "02_title_rows.xlsx",
  "detected_sheet": "Sheet1",
  "available_sheets": ["Sheet1"],
  "detected_header_row": 5,
  "detected_text_columns": ["Góp ý"],
  "detected_likert_column": "Điểm hài lòng (1-5)",
  "dropped_pii_columns": ["MSSV", "Họ tên SV"],
  "total_rows": 120,
  "preview_rows": [
    {"STT": 1, "Điểm hài lòng (1-5)": 2, "Góp ý": "Máy chiếu phòng học bị mờ và nóng quá."},
    {"STT": 2, "Điểm hài lòng (1-5)": 5, "Góp ý": "Giảng viên dạy rất dễ hiểu và nhiệt tình."}
  ]
}
```

---

### 6.3. `POST /api/v1/upload/analyze` — Chạy phân tích hàng loạt cho File đã tải lên
- **Request JSON:**
```json
{
  "file_id": "upload_8821",
  "sheet_name": "Sheet1",
  "header_row": 5,
  "text_columns": ["Góp ý"],
  "likert_column": "Điểm hài lòng (1-5)",
  "model_mode": "phobert"
}
```
- **Response JSON (`200 OK`):**
```json
{
  "batch_id": "batch_8821",
  "total_processed": 120,
  "no_content_filtered": 4,
  "valid_feedback_count": 116,
  "needs_review_count": 11,
  "summary": {
    "sentiment_counts": {"negative": 52, "neutral": 8, "positive": 56},
    "topic_counts": {"lecturer": 74, "training_program": 22, "facility": 15, "others": 5}
  },
  "actionable_insights": [
    {
      "topic_label": "lecturer",
      "topic_name_vi": "Giảng viên",
      "issue_group": "Phương pháp giảng dạy & truyền đạt",
      "negative_count": 21,
      "matched_keywords": "khó hiểu, nhanh, buồn ngủ",
      "sample_feedback": [
        "tốc độ dạy của giảng viên nhanh , sinh viên không theo kịp .",
        "giảng hơi khó hiểu ở phần bài tập lớn ."
      ],
      "recommended_action": "Rà soát tốc độ giảng, bổ sung ví dụ minh họa và hỏi đáp cuối buổi"
    }
  ],
  "download_excel_url": "/api/v1/upload/export/batch_8821"
}
```

---

### 6.4. `GET /api/v1/dashboard/initial` — Lấy dữ liệu khởi tạo Dashboard & Thông số Model
- **Response JSON (`200 OK`):**
  - Trả về thống kê tổng hợp từ `data/processed/test_pred_phobert.csv` (cộng các phản hồi mới trong phiên), bảng `data/processed/actionable_insights.csv`, danh sách 10 file mẫu trong `data/samples_messy/` và nội dung `models/model_metadata.json`.

---

## 7. LƯU Ý KỸ THUẬT QUAN TRỌNG CHO BÊN LÀM APP (GOTCHAS & TIPS)

1. **Cách nạp `phobert_multitask.pt` không bị lỗi trên máy không có GPU:**
   - File `models/phobert_multitask/phobert_multitask.pt` được huấn luyện trên GPU T4 của Google Colab. Khi nạp trên máy tính cá nhân chỉ có CPU, bắt buộc phải truyền tham số `map_location="cpu"` (hoặc `map_location=device`).
   - Kiến trúc lớp của mô hình gồm: `encoder = AutoModel.from_pretrained("vinai/phobert-base-v2")`, `drop = nn.Dropout(0.3)`, `sent_head = nn.Linear(768, 3)`, `topic_head = nn.Linear(768, 4)`, lấy vector đặc trưng tại vị trí token đầu tiên `last_hidden_state[:, 0, :]`. Toàn bộ định nghĩa này đã có sẵn trong Notebook `04_phobert_multitask.ipynb` (Mục 2).
2. **Chỉ nạp Model 1 lần duy nhất khi khởi động Server (Singleton / Caching):**
   - Tuyệt đối không nạp lại file `phobert_multitask.pt` (~540 MB) ở mỗi lần gọi API hoặc mỗi lần bấm nút trên giao diện. Hãy nạp 1 lần vào bộ nhớ RAM khi khởi động App.
3. **Tối ưu tốc độ khi phân tích File Excel nhiều dòng trên CPU:**
   - Nếu chạy bằng `PhoBERT Multi-Task` trên CPU, hãy gom theo lô (`batch_size = 32` hoặc `64`) thay vì chạy vòng lặp từng câu một.
   - Nếu người dùng chọn chế độ **Siêu tốc (`TF-IDF Baseline`)**, dùng `tfidf_vectorizer.pkl` + `svc_sentiment.pkl` + `svc_topic.pkl` sẽ xử lý xong 3,000 dòng trong chưa đầy `0.2 giây`.
4. **Logic tự dò file Excel bẩn và tính SHAP đã có sẵn 100% trong Notebook `05` và `06`:**
   - Bên làm App có thể mở trực tiếp `notebooks/05_shap_and_actionable_insights.ipynb` và `notebooks/06_excel_auto_detector.ipynb` để tham khảo logic chi tiết đã chạy đạt **10/10 file bẩn**.

---

## 8. DANH SÁCH NGHIỆM THU BÀN GIAO (ACCEPTANCE CHECKLIST)

Bên làm App được coi là hoàn thành 100% yêu cầu khi vượt qua 6 bài kiểm thử sau:
- [ ] **Test 1 (Chuyển đổi vai trò & Thời gian thực):** Gửi 1 câu phản hồi có chứa MSSV + SĐT ở *Chế độ Sinh viên* $\rightarrow$ Thấy câu đã được che thành `[STUDENT_ID]`, `[PHONE]` $\rightarrow$ Bấm sang *Chế độ Giảng viên* thấy ngay câu đó nằm ở đầu bảng dữ liệu.
- [ ] **Test 2 (Gắn cờ `needs_review`):** Ở *Chế độ Sinh viên*, chọn **5 sao** nhưng nhập câu chê *"Phòng học quá nóng và máy chiếu bị mờ"* $\rightarrow$ Sang *Chế độ Giảng viên* thấy câu này được dự đoán là `Tiêu cực` - `Cơ sở vật chất` và tự động bật cờ cảnh báo màu cam `needs_review`.
- [ ] **Test 3 (Tự động nhận diện 10 file Excel bẩn):** Thử tải lên các file trong `data/samples_messy/` (đặc biệt là `02_title_rows.xlsx`, `04_multi_sheet.xlsx`, `08_two_text_cols.xlsx`, `10_html_as_xls.xls`) $\rightarrow$ Hệ thống nhận diện đúng Sheet, đúng dòng tiêu đề, đúng cột ý kiến và cho phép tải về file `.xlsx` đã gán nhãn.
- [ ] **Test 4 (Giải thích từ khóa SHAP):** Bấm vào 1 câu bất kỳ ở Tab 3 $\rightarrow$ Hiển thị tô màu Đỏ/Xanh trực tiếp trên từng từ khóa và biểu đồ thanh ngang giá trị SHAP.
- [ ] **Test 5 (Bảng Gợi ý hành động - Actionable Insights):** Tab 1 hiển thị rõ bảng tổng hợp các nhóm vấn đề tiêu cực kèm số lượng câu, từ khóa đại diện và khuyến nghị hành động cụ thể cho Giảng viên.
- [ ] **Test 6 (Đối chiếu PhoBERT vs. Baseline):** Tab 4 hiển thị bảng so sánh chỉ số Test (`0.7928` vs `0.7380` ở Cảm xúc; `0.7775` vs `0.7447` ở Chủ đề) và cho phép thử nghiệm so sánh kết quả dự đoán của 2 mô hình.
