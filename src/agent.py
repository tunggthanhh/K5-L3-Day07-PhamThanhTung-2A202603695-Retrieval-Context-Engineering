"""A real-model, evidence-only agent using an OpenAI-compatible chat endpoint."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .retrieval import Retriever

SYSTEM = """Bạn là tác tử hỏi đáp dựa trên bằng chứng. Bạn chỉ biết trả lời bằng context tool trả về.
Mỗi lượt phải trả JSON hợp lệ, không markdown.
- Nếu cần tìm, trả {\"action\":\"search\",\"query\":\"...\"}.
- Nếu context đủ, trả {\"action\":\"answer\",\"answer\":\"... [chunk-id]\",\"citations\":[\"chunk-id\"]}.
- Chỉ trả {\"action\":\"abstain\"} sau khi đã có context và context thật sự không chứa câu trả lời.
Khi context có điều khoản trả lời trực tiếp, bắt buộc trả action answer và citation ID của đoạn đó; không từ chối vì thiếu tự tin.
Không được viện dẫn ID không có trong context. Không suy đoán."""


def pack_context(rows: list[dict[str, Any]], budget: int) -> tuple[str, list[str]]:
    blocks: list[str] = []; ids: list[str] = []; seen: set[str] = set()
    for row in rows:
        if row["id"] in seen: continue
        seen.add(row["id"])
        block = f"[{row['id']} | {row['source']} | page {row['page']}]\n{row['text']}"
        if len("\n\n".join([*blocks, block])) <= budget:
            blocks.append(block); ids.append(row["id"])
    return "\n\n".join(blocks), ids


@dataclass
class OpenAIController:
    """Model controller; API errors deliberately propagate to the runner."""
    model: str
    api_key: str
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> "OpenAIController":
        key = os.getenv("LLM_API_KEY")
        model = os.getenv("LLM_MODEL")
        if not key or not model:
            raise RuntimeError("Set LLM_API_KEY and LLM_MODEL in .env before running an agent benchmark.")
        return cls(model=model, api_key=key, base_url=os.getenv("LLM_BASE_URL") or None)

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is not installed; activate the project virtual environment and install requirements.txt") from exc
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        user = {"question": question, "tool_history": history}
        response = client.chat.completions.create(model=self.model, temperature=0,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}])
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, re.S)
        if not match: raise ValueError(f"Model did not return JSON: {content[:200]}")
        try: return json.loads(match.group())
        except json.JSONDecodeError as exc: raise ValueError(f"Invalid model JSON: {content[:200]}") from exc

    def answer_from_context(self, question: str, context: str, context_ids: list[str]) -> dict[str, Any]:
        """One final, citation-constrained answer attempt for an existing context."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai is not installed") from exc
        prompt = ("Trả lời câu hỏi CHỈ bằng context dưới đây. Nếu context có câu trả lời trực tiếp, "
                  "bắt buộc trả lời ngắn gọn và trích ít nhất một ID đúng theo dạng [chunk-id]. "
                  "Nếu không có, trả đúng câu: Không đủ bằng chứng để trả lời.\n\n"
                  f"Câu hỏi: {question}\n\nContext:\n{context}")
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(model=self.model, temperature=0,
            messages=[{"role": "system", "content": "Bạn trả lời có căn cứ, không suy đoán."},
                      {"role": "user", "content": prompt}])
        answer = (response.choices[0].message.content or "").strip()
        citations = [cid for cid in context_ids if f"[{cid}]" in answer]
        if answer and citations:
            return {"status": "answered", "answer": answer, "citations": citations}
        return {"status": "abstain", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}


def run_agent(question: str, controller: OpenAIController, retriever: Retriever, *, filters: dict[str, Any] | None,
              max_searches: int = 3, top_k: int = 5, budget: int = 3500) -> dict[str, Any]:
    if not question.strip(): raise ValueError("question is required")
    events: list[dict[str, Any]] = []; candidates: list[dict[str, Any]] = []; context_ids: list[str] = []
    history: list[dict[str, Any]] = []
    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        kind = action.get("action")
        if kind == "search":
            if len([e for e in events if e["event"] == "search"]) >= max_searches: break
            query = action.get("query")
            if not isinstance(query, str) or not query.strip(): raise ValueError("Model returned an empty search query")
            results = retriever.search(query, filters=filters, top_k=top_k)
            candidates.extend(results); context, context_ids = pack_context(candidates, budget)
            events.append({"event": "search", "tool": retriever.name, "query": query, "filter": filters,
                           "results": results, "context": context, "context_ids": context_ids})
            history = [{"tool": "retrieval", "queries": [e["query"] for e in events if e["event"] == "search"],
                        "context": context, "context_ids": context_ids}]
        elif kind == "answer":
            answer, citations = action.get("answer"), action.get("citations")
            ok = isinstance(answer, str) and isinstance(citations, list) and citations and all(
                isinstance(c, str) and c in context_ids and f"[{c}]" in answer for c in citations)
            if ok: return {"status": "answered", "answer": answer, "citations": citations, "events": events + [{"event": "answer"}]}
            return {"status": "abstain", "answer": "Không đủ bằng chứng để trả lời.", "citations": [], "events": events + [{"event": "invalid_answer"}]}
        elif kind == "abstain":
            return {"status": "abstain", "answer": "Không đủ bằng chứng để trả lời.", "citations": [], "events": events + [{"event": "abstain"}]}
        else: raise ValueError("Model action must be search, answer, or abstain")
    return {"status": "abstain", "answer": "Không đủ bằng chứng để trả lời.", "citations": [], "events": events + [{"event": "search_limit"}]}
