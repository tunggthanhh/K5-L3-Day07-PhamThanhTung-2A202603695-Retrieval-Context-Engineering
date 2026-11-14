# Chạy Day 7

Yêu cầu: Python 3.11+, Internet lần đầu để tải embedding model, và API key cho
một chat model tương thích OpenAI khi chạy agent. Chạy các lệnh từ thư mục gốc
của repository.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Sửa `.env` và đặt `LLM_API_KEY`, `LLM_MODEL`, tùy chọn `LLM_BASE_URL`. Không
commit file này.

## 1. Tạo dữ liệu và dense index

```powershell
python -m src.build_data
python -m src.index
```

Hai lệnh tạo `artifacts/data/{chunks,questions}.json` và
`artifacts/qdrant/`. Chúng dùng corpus Day 6 đã chép vào `data/day6/`.

## 2. Chạy retrieval trên dev

```powershell
python -m src.evaluate --split dev --output artifacts/runs/dev-baseline
```

Lệnh so sánh cùng dữ liệu và nhãn cho dense, TF-IDF, hybrid RRF. Kết quả ở
`retrieval_summary.json` và trace từng query ở `retrieval_traces.json`.

## 3. Chạy agent thật

```powershell
python -m src.evaluate --split dev --agent --output artifacts/runs/dev-agent
```

Agent dùng hybrid RRF, có tối đa 3 lượt tìm và context budget 3,500 ký tự. Lệnh
ghi `agent_traces.json`, `agent_cases.json`, và `reviews_TEMPLATE.json`. Đọc
từng answer/citation, rồi thay các nhãn mặc định trong template bằng đánh giá
thủ công trước khi tổng hợp metrics:

```powershell
python starter/agent_metrics.py --cases artifacts/runs/dev-agent/agent_cases.json --traces artifacts/runs/dev-agent/agent_traces.json --reviews artifacts/runs/dev-agent/reviews_TEMPLATE.json --split dev --output artifacts/runs/dev-agent/agent_summary.json
```

Thử ít nhất hai thay đổi trên **dev** (ví dụ `--budget` khác, RRF/rerank bổ sung,
hoặc context selector mới), lưu output mỗi lần. Chỉ sau khi chọn cấu hình, chạy
test với các lệnh tương tự thay `--split test` và một thư mục output mới.

## Kiểm tra mã

```powershell
python -m unittest discover -s tests -v
```

Một test starter có thể lỗi trên Windows khi Python mặc định dùng cp1252 để đọc
trace UTF-8; đây là lỗi encoding của test fixture, không dùng kết quả mock đó để
đánh giá hệ thống.

## Chatbot cục bộ

Sau khi đã tạo data/index và điền `.env`, chạy:

```powershell
python -m src.chat_server
```

Mở `http://127.0.0.1:8000`. API key chỉ được đọc bởi Python server từ `.env`,
không xuất hiện trong HTML hoặc trình duyệt.
