"""Run reproducible retrieval and optional live-agent benchmarks."""
from __future__ import annotations

import argparse, json, os, statistics, time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from .agent import OpenAIController, pack_context, run_agent
from .retrieval import DenseRetriever, HybridRetriever, LexicalRetriever

ROOT = Path(__file__).resolve().parents[1]

def hit(row: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    return any(row["source_id"] == g["source_id"] and row["page"] == g["page"] and g["evidence"] in row["text"] for g in gold)

def rate(n: int, N: int) -> dict[str, Any]: return {"n": n, "N": N, "rate": n / N if N else None}

def evaluate_retrieval(name: str, retriever: Any, questions: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for q in questions:
        if not q["answerable"]: continue
        started = time.perf_counter(); results = retriever.search(q["question"], filters=q.get("filter"), top_k=3)
        context, ids = pack_context(results, budget); marked = [{**row, "evidence_hit": hit(row, q["gold"])} for row in results]
        traces.append({"query_id": q["id"], "split": q["split"], "strategy": name, "question": q["question"],
                       "filters": q.get("filter"), "results": marked, "context": context, "context_ids": ids,
                       "elapsed_ms": (time.perf_counter() - started) * 1000,
                       "hit1": bool(marked and marked[0]["evidence_hit"]), "hit3": any(x["evidence_hit"] for x in marked),
                       "context_evidence": any(x["evidence_hit"] and x["id"] in ids for x in marked)})
    N = len(traces)
    return traces, {"strategy": name, "answerable_questions": N, "hit_at_1": rate(sum(x["hit1"] for x in traces), N),
                    "evidence_hit_at_3": rate(sum(x["hit3"] for x in traces), N),
                    "context_evidence_rate": rate(sum(x["context_evidence"] for x in traces), N),
                    "median_ms": statistics.median(x["elapsed_ms"] for x in traces) if traces else None, "service_cost_usd": 0}

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--chunks", type=Path, default=ROOT / "artifacts/data/chunks.json"); parser.add_argument("--questions", type=Path, default=ROOT / "artifacts/data/questions.json")
    parser.add_argument("--store", type=Path, default=ROOT / "artifacts/qdrant"); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent", action="store_true"); parser.add_argument("--budget", type=int, default=3500); args = parser.parse_args()
    load_dotenv(ROOT / ".env"); chunks = json.loads(args.chunks.read_text(encoding="utf-8")); all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = [x for x in all_questions if x["split"] == args.split]; model = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    dense = DenseRetriever(args.store, model); lexical = LexicalRetriever(chunks); retrievers = {"dense": dense, "tfidf": lexical, "hybrid_rrf": HybridRetriever(dense, lexical)}
    try:
        all_traces: list[dict[str, Any]] = []; summaries: list[dict[str, Any]] = []
        for name, retriever in retrievers.items():
            traces, summary = evaluate_retrieval(name, retriever, questions, args.budget); all_traces.extend(traces); summaries.append(summary)
        args.output.mkdir(parents=True, exist_ok=True); (args.output / "retrieval_traces.json").write_text(json.dumps(all_traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output / "retrieval_summary.json").write_text(json.dumps({"split": args.split, "configs": summaries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.agent:
            controller = OpenAIController.from_env(); selected = retrievers["hybrid_rrf"]; agent_traces: list[dict[str, Any]] = []
            for q in questions:
                started = time.perf_counter()
                try: row = run_agent(q["question"], controller, selected, filters=q.get("filter"))
                except Exception as exc: row = {"status": "error", "answer": "", "citations": [], "events": [{"event": "error", "message": str(exc)}]}
                agent_traces.append({"query_id": q["id"], "split": args.split, "elapsed_ms": (time.perf_counter()-started)*1000, **row})
            cases = [{"id": q["id"], "split": args.split, "answerable": q["answerable"]} for q in questions]
            reviews = [{"query_id": x["query_id"], "citation_support": [False] * len(x["citations"]),
                        **({"answer_grade": 0} if next(q for q in questions if q["id"] == x["query_id"])["answerable"] else {"abstention_correct": False})} for x in agent_traces if x["status"] != "error"]
            for name, value in (("agent_cases.json", cases), ("agent_traces.json", agent_traces), ("reviews_TEMPLATE.json", reviews)):
                (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally: dense.close()

if __name__ == "__main__": main()
