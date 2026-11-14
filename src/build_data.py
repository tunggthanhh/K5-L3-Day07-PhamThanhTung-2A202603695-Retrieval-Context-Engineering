"""Build auditable Day 7 chunks and benchmark inputs from the Day 6 corpus."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .day6.chunking import by_paragraph
from .day6.ingestion import load_pdf

ROOT = Path(__file__).resolve().parents[1]
DAY6_DATA = ROOT / "data" / "day6"

UNANSWERABLE: list[dict[str, Any]] = [
    {"id": "u01", "split": "dev", "question": "Mức lương tối thiểu vùng I năm 2026 là bao nhiêu?", "answerable": False,
     "expected_behavior": "search_and_abstain", "absence_reason": "Bộ luật Lao động 2019 không ấn định mức lương tối thiểu vùng hoặc số liệu cho năm 2026."},
    {"id": "u02", "split": "dev", "question": "Năm 2026 người lao động được nghỉ Tết Âm lịch chính xác những ngày nào?", "answerable": False,
     "expected_behavior": "search_and_abstain", "absence_reason": "Bộ luật chỉ nêu số ngày nghỉ Tết Âm lịch; lịch ngày cụ thể từng năm do thông báo khác quyết định."},
    {"id": "u03", "split": "test", "question": "Mức phạt tiền cụ thể năm 2026 khi doanh nghiệp trả lương chậm là bao nhiêu?", "answerable": False,
     "expected_behavior": "search_and_abstain", "absence_reason": "Corpus là Bộ luật Lao động 2019, không phải nghị định xử phạt và không chứa mức phạt năm 2026."},
    {"id": "u04", "split": "test", "question": "Tên và địa chỉ của cơ quan bảo hiểm xã hội quận Hoàn Kiếm là gì?", "answerable": False,
     "expected_behavior": "search_and_abstain", "absence_reason": "Thông tin địa chỉ cơ quan địa phương không thuộc corpus Bộ luật Lao động 2019."},
]


def _chapter(text: str, previous: str | None) -> str | None:
    match = re.search(r"CHƯƠNG\s+([IVXLCDM]+)", text, re.I)
    return f"Chương {match.group(1).upper()}" if match else previous


def _tokens(text: str) -> set[str]:
    """Robust tokens for PDF text where a few Vietnamese words are spaced apart."""
    text = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    return {part for part in re.findall(r"[a-z0-9]+", text) if len(part) > 1}


def build(output: Path) -> tuple[Path, Path]:
    pdf = DAY6_DATA / "bo_luat_lao_dong.pdf"
    pages = load_pdf(pdf, "bo-luat-lao-dong-2019", "data/day6/bo_luat_lao_dong.pdf")
    chapter: str | None = None
    chunks: list[dict[str, Any]] = []
    for page in pages:
        chapter = _chapter(page.text, chapter)
        page.metadata.update({"document_type": "Bộ luật", "year": 2019, "chapter": chapter})
        for chunk in by_paragraph(page, max_size=500):
            chunks.append({"id": chunk.chunk_id, "text": chunk.text, "source_id": chunk.document_id,
                           "source": chunk.source, "page": chunk.page, "metadata": chunk.metadata})
    raw = json.loads((DAY6_DATA / "questions.json").read_text(encoding="utf-8"))
    questions: list[dict[str, Any]] = []
    for q in raw:
        # Preserve an exact, inspectable chunk excerpt as gold evidence.  The
        # original Day 6 quote is human-written while pypdf inserts line breaks
        # and occasional spaces inside words, so direct substring equality is
        # not reliable.  This deterministic source-side alignment is performed
        # only while preparing labels; no gold enters the runtime pipeline.
        candidates = [c for c in chunks if c["source_id"] == q["gold_source_id"] and c["page"] == q["gold_page"]]
        if not candidates:  # q19 used a printed-page number in Day 6.
            candidates = [c for c in chunks if c["source_id"] == q["gold_source_id"]]
        quote_terms = _tokens(q["evidence_quote"])
        evidence_chunk = max(candidates, key=lambda c: len(quote_terms & _tokens(c["text"])))
        gold_page = evidence_chunk["page"]
        query_filter = q.get("filter")
        if query_filter and query_filter.get("chapter"):
            query_filter = {**query_filter, "chapter": query_filter["chapter"].split(" - ")[0]}
        questions.append({"id": q["id"], "split": q["split"], "question": q["question"],
                          "answerable": True, "expected_answer": q["expected_answer"], "filter": query_filter,
                          "gold": [{"source_id": q["gold_source_id"], "page": gold_page,
                                    "evidence": evidence_chunk["text"]}]})
    questions.extend(UNANSWERABLE)
    output.mkdir(parents=True, exist_ok=True)
    chunk_path, question_path = output / "chunks.json", output / "questions.json"
    chunk_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    question_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return chunk_path, question_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "data")
    args = parser.parse_args()
    chunks, questions = build(args.output)
    print(f"Wrote {chunks} and {questions}")


if __name__ == "__main__":
    main()
