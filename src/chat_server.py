"""Local browser chatbot for the Day 7 Agentic RAG pipeline.

The OpenAI key remains in .env and is never exposed to browser JavaScript.
"""
from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from .agent import OpenAIController, run_agent
from .retrieval import DenseRetriever, HybridRetriever, LexicalRetriever

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "ui" / "dist"


class App:
    def __init__(self) -> None:
        load_dotenv(ROOT / ".env")
        chunks = json.loads((ROOT / "artifacts" / "data" / "chunks.json").read_text(encoding="utf-8"))
        model = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        # The dense store is rebuilt separately; lexical retrieval immediately
        # benefits from the clean PyMuPDF chunks in artifacts/data.
        self.dense = DenseRetriever(ROOT / "artifacts" / "qdrant", model)
        self.retriever = HybridRetriever(self.dense, LexicalRetriever(chunks))
        self.controller = OpenAIController.from_env()

    def ask(self, question: str) -> dict:
        # Chat needs a broader candidate set than the benchmark's fixed top-3:
        # the model still sees a bounded, traceable context and must cite it.
        result = run_agent(question, self.controller, self.retriever, filters=None,
                           max_searches=3, top_k=12, budget=7000)
        if result["status"] == "abstain":
            searches = [event for event in result["events"] if event.get("event") == "search"]
            if searches and searches[-1].get("context"):
                retry = self.controller.answer_from_context(question, searches[-1]["context"], searches[-1]["context_ids"])
                if retry["status"] == "answered": result = {**result, **retry}
        return {"answer": result["answer"], "status": result["status"], "citations": result["citations"],
                "searches": sum(event.get("event") == "search" for event in result["events"])}


APP: App | None = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            question = payload.get("question", "")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Hãy nhập câu hỏi.")
            global APP
            if APP is None: APP = App()
            response = APP.ask(question.strip())
            body = json.dumps(response, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.BAD_REQUEST); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


def main() -> None:
    port = int(os.getenv("CHAT_PORT", "8000"))
    print(f"Day 7 chatbot: http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__": main()
