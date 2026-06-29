# 📓 NotebookLM-style RAG (Local & Private)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black.svg)](https://ollama.com/)
[![UI](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)](https://streamlit.io/)

> Ứng dụng hỏi-đáp tài liệu kiểu **NotebookLM**, chạy **100% cục bộ**: mỗi
> *notebook* sở hữu nguồn PDF và các cuộc hội thoại riêng, cách ly hoàn toàn với
> nhau. Trả lời chỉ dựa trên tài liệu của chính notebook đó (giảm bịa đặt), kèm
> trích dẫn tên file + số trang. Không gửi dữ liệu ra ngoài máy.

---

## 🌟 Tổng quan

- **Riêng tư & miễn phí (Local LLM):** dùng **Ollama** chạy `llama3` +
  `nomic-embed-text` ngay trên máy — không tốn phí, không rò rỉ dữ liệu.
- **RAG có phản tỉnh (Agentic RAG):** truy xuất → tự chấm điểm độ liên quan →
  viết lại truy vấn nếu chưa đủ → mới sinh câu trả lời.
- **Cách ly theo notebook:** mỗi notebook là một collection Chroma riêng; lịch sử
  chat lưu trong SQLite nên **vẫn còn sau khi refresh/khởi động lại**.

## 🛠️ Tech Stack

- **Ngôn ngữ:** Python 3.10+
- **UI:** Streamlit (một tiến trình duy nhất, gọi thẳng package `rag`)
- **LLM Engine:** Ollama — `llama3` (chat) + `nomic-embed-text` (embeddings)
- **Vector DB:** ChromaDB · **Metadata DB:** SQLite (stdlib)
- **RAG framework:** LangChain

---

## 🏗️ Kiến trúc & Cách hoạt động

Ứng dụng là **một tiến trình Streamlit duy nhất**, gọi trực tiếp các service trong
package `rag` — **không có backend riêng**.

### Sơ đồ thành phần

```
┌──────────────────────────────────────────────────────────────┐
│                  apps/streamlit_app.py (UI)                    │
│   Sidebar: Notebooks · Sources · Conversations                │
│   Main:    Khung chat + lịch sử + trích dẫn                    │
└───────────────┬───────────────────────────┬──────────────────┘
                │ gọi trực tiếp (in-process) │
        ┌───────▼────────┐          ┌────────▼─────────────────┐
        │ rag.db.store   │          │ rag.services             │
        │ (SQLite)       │          │  ├─ ingestion.py         │
        │ notebooks,     │          │  └─ agentic_rag.py       │
        │ sources,       │          └────────┬─────────────────┘
        │ conversations, │                   │
        │ messages       │          ┌────────▼─────────────────┐
        └────────────────┘          │ rag.core                 │
                                     │  ├─ vectorstore (Chroma) │
                                     │  └─ llm (Ollama)         │
                                     └────────┬─────────────────┘
                                              │
                                     ┌────────▼─────────┐
                                     │ Ollama (local)   │
                                     │ llama3 +         │
                                     │ nomic-embed-text │
                                     └──────────────────┘
```

### Hai tầng lưu trữ

| Tầng | Lưu gì | Module |
|------|--------|--------|
| **SQLite** (`data/app.db`) | Metadata: notebooks, sources, conversations, messages (kèm trích dẫn dạng JSON) | `rag.db.store` |
| **ChromaDB** (`chroma_db/`) | Vector embeddings của các đoạn (chunk) tài liệu | `rag.core.vectorstore` |

> **Cách ly theo notebook:** mỗi notebook ánh xạ tới **một Chroma collection riêng**
> (`nb_<uuid>`). Nhờ vậy tìm kiếm chỉ chạy trong tài liệu của chính notebook đó,
> không rò rỉ dữ liệu giữa các notebook.

### Luồng 1 — Nạp tài liệu (Ingestion)

Khi upload PDF và bấm **⚡ Ingest** trong sidebar:

1. Lưu file vào `data/uploads/<notebook_id>/` và tạo bản ghi `sources` (status `ingesting`).
2. `ingestion.ingest_files()` đọc PDF bằng `PyPDFLoader`, cắt thành chunk bằng
   `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `overlap=200`).
3. Mỗi chunk được embed bằng `nomic-embed-text` rồi đẩy vào Chroma theo **batch**,
   với id xác định `"<source_id>:<index>"` để sau này xoá chính xác.
4. Cập nhật lại `sources` (số chunk, số trang, status `done`/`error`). Thanh
   progress trong UI phản ánh tiến độ qua `progress_cb`.

### Luồng 2 — Hỏi đáp (Agentic RAG + reflection loop)

`agentic_rag.answer_question()` chạy một **vòng lặp phản tỉnh** cục bộ — không web:

```
retrieve(query)
     │
     ▼
grade(câu hỏi, ngữ cảnh)  ── đủ? ──► YES ─┐
     │                                     │
     │ NO (và còn lượt)                     │
     ▼                                      │
rewrite_query (thêm từ khoá/đồng nghĩa)     │
     │   ───── lặp lại retrieve ─────►      │
     ▼                                      ▼
                                       generate(câu trả lời + trích dẫn)
```

1. **Retrieve** — `similarity_search(query, k=5)` trên collection của notebook.
2. **Grade** — LLM chấm YES/NO xem ngữ cảnh đã đủ trả lời chưa.
3. **Rewrite** — nếu chưa đủ và còn lượt (`max_reflection_iters=2`), LLM viết lại
   truy vấn rồi tìm lại; kết quả các vòng được gộp & khử trùng lặp.
4. **Generate** — LLM soạn câu trả lời **chỉ dựa trên ngữ cảnh đã truy xuất**, trả
   lời đúng ngôn ngữ câu hỏi và kèm trích dẫn (tên file + số trang).

Câu hỏi/câu trả lời (kèm `sources`) được ghi vào bảng `messages`, nên lịch sử vẫn
còn sau khi refresh. Các bước xử lý hiển thị realtime qua `st.status`.

### Nguyên tắc thiết kế

- **Không lưu DB row trong `session_state`** — chỉ giữ `id`; mỗi lần rerun đều đọc
  lại từ SQLite để dữ liệu luôn nhất quán.
- **`store.py` không import Streamlit** — tầng dữ liệu tách rời UI; mỗi thao tác mở
  một kết nối SQLite ngắn (an toàn với cơ chế rerun của Streamlit).
- **Dọn dẹp đồng bộ 3 nơi** khi xoá notebook/source: bản ghi SQLite (cascade),
  collection/vector trong Chroma, và file PDF trên đĩa.

---

## 📁 Cấu trúc dự án

```
rag/
├── apps/
│   └── streamlit_app.py    # Toàn bộ giao diện web (UI)
├── src/rag/                # Package lõi
│   ├── config.py           # Cấu hình tập trung (pydantic-settings, đọc .env)
│   ├── core/
│   │   ├── vectorstore.py  # Embeddings + Chroma (1 collection / notebook)
│   │   └── llm.py          # Chat LLM (Ollama)
│   ├── services/
│   │   ├── ingestion.py    # Đọc & cắt PDF, nạp vào vector store
│   │   └── agentic_rag.py  # Vòng lặp phản tỉnh: retrieve→grade→rewrite→generate
│   └── db/
│       ├── store.py        # CRUD SQLite (UI-agnostic)
│       └── schema.sql      # notebooks / sources / conversations / messages
├── data/                   # app.db (SQLite) + uploads/ (tự sinh, gitignored)
├── chroma_db/              # Vector DB (tự sinh, gitignored)
├── .env.example
├── pyproject.toml
└── requirements.txt
```

---

## 🚀 Cài đặt & Chạy

### 1. Khởi động Ollama (bộ não cục bộ)

Tải tại [ollama.com](https://ollama.com/), rồi pull model:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Thiết lập môi trường Python

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e .                # cài package rag + dependencies
cp .env.example .env            # (tuỳ chọn) chỉnh model / đường dẫn
```

### 3. Chạy ứng dụng

```bash
streamlit run apps/streamlit_app.py
```

Sau đó trong giao diện: **tạo notebook → upload PDF → bấm ⚡ Ingest → đặt câu hỏi**.
Toàn bộ chạy **100% cục bộ**, không gọi dịch vụ ngoài.

---

## ⚙️ Cấu hình (`.env`)

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `model_llm` | `llama3` | Model chat |
| `model_embeddings` | `nomic-embed-text` | Model embeddings |
| `ollama_base_url` | `http://localhost:11434` | Endpoint Ollama |
| `chunk_size` / `chunk_overlap` | `1000` / `200` | Kích thước cắt chunk |
| `search_k` | `5` | Số chunk truy xuất mỗi vòng |
| `max_reflection_iters` | `2` | Số lần viết lại truy vấn tối đa |
| `llm_num_ctx` | `8192` | Cửa sổ ngữ cảnh của LLM |
