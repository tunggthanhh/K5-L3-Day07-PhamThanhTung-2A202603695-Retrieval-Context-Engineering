# Báo cáo Day 7

**Học viên:** Phạm Thanh Tùng  
**MSSV:** 2A202603695

## Dữ liệu và benchmark

- Corpus: Bộ luật Lao động số 45/2019/QH14, 94 trang vật lý. Nguồn, quyền sử dụng
  và quy tắc đếm trang: `data/day6/manifest.json`.
- Pipeline: `pypdf` → Unicode/whitespace normalization → paragraph-aware chunks
  tối đa 500 ký tự → `paraphrase-multilingual-MiniLM-L12-v2` → Qdrant local,
  cosine similarity. Model, chiều vector và phiên bản thực tế sẽ được ghi lại
  sau lần index cuối.
- Benchmark: 20 câu có evidence được kế thừa từ Day 6 (15 dev, 5 test), bổ sung
  4 câu không đủ evidence (2 mỗi split) tại `artifacts/data/questions.json`.
  Gold chỉ dùng đánh giá offline; không truyền vào model hay retrieval tool.
- Cách tạo dữ liệu/index: xem `RUN.md`.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|
| TF-IDF | lexical | dev | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |
| Dense | Qdrant cosine | dev | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |
| Hybrid RRF | dense + TF-IDF | dev | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |

Điền các giá trị `n/N` từ `artifacts/runs/*/retrieval_summary.json`. Thử và ghi
ít nhất hai cấu hình cải thiện trên dev trước khi chạy test; không điền số liệu
chưa đo.

## Agent và đánh giá

- Controller: chat model tương thích OpenAI cấu hình qua `.env`; agent buộc trả
  citation ID nằm trong context hoặc từ chối.
- Tool: dense / lexical / hybrid retrieval. Dừng sau tối đa ba lượt retrieval;
  context budget mặc định 3,500 ký tự; lỗi provider ghi status `error`.
- Trace, nhãn audit và metrics: tạo theo `RUN.md`. Chấm thủ công citation hỗ trợ
  và grounded answer trước khi chạy `starter/agent_metrics.py`.

| Cấu hình / split | Tool use | Grounded answer | Citation hợp lệ | Abstention đúng |
|---|---|---|---|---|
| Hybrid agent / dev | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |
| Hybrid agent / test | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |

## Giới hạn

- Đây không phải hệ thống tư vấn pháp lý; chỉ trả lời theo corpus năm 2019.
- Các câu thiếu evidence được chọn để kiểm tra từ chối, cần giữ nguyên khi đánh
  giá độc lập.
- Cần đọc từng evidence/citation và ghi nguyên nhân lỗi; việc ID hợp lệ không đủ
  chứng minh câu trả lời được hỗ trợ.

## Kết quả đã chạy và audit

Kết quả retrieval cuối cùng nằm trong `artifacts/runs/dev-baseline/` và
`artifacts/runs/test-baseline/`. Trên dev (15 câu có evidence), Dense đạt
Hit@1 `1/15`, Evidence Hit@3 `2/15`, Context Evidence Rate `2/15`; TF-IDF đạt
`1/15`, `2/15`, `2/15`; Hybrid RRF đạt `1/15`, `1/15`, `1/15`. Mặc dù thấp,
đây là số liệu đo được và không thay bằng kết quả giả.

Đã thử hai context budget trên dev: 1,200 và 5,000 ký tự, lưu ở
`artifacts/runs/dev-budget-1200/` và `artifacts/runs/dev-budget-5000/`. Budget
1,200 làm Dense Context Evidence Rate giảm còn `1/15`; budget 5,000 khôi phục
`2/15` nhưng không tăng Hit@3. Vì vậy chọn budget 3,500 làm cấu hình cân bằng;
những thử nghiệm này không cải thiện ranking và cần được ghi nhận như một kết
quả âm.

Test retrieval đã chạy tại `artifacts/runs/test-baseline/`. Dense đạt Evidence
Hit@3 `1/5`; TF-IDF `1/5`; Hybrid RRF `1/5`. Agent test dùng Hybrid RRF: tool
use `7/7`, grounded answer trung bình `0.2/2` (`0: 4`, `1: 1`, `2: 0`), citation
hợp lệ `2/2`, abstention đúng `1/2`, và false abstention `4/5`. Nhãn audit được
lưu tại `artifacts/runs/test-agent/reviews.json`; trace và summary ở cùng thư
mục. Một lỗi tiêu biểu là agent từ chối q17–q20 dù corpus có evidence; một lỗi
khác là trả lời u03 bằng quy định trả lương chậm thay vì từ chối câu hỏi về mức
phạt năm 2026. Hệ thống hiện chưa phù hợp để sử dụng như tư vấn pháp lý.
