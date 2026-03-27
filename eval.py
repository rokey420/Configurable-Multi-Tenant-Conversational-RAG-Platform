import os
import json
import time
import requests
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer, util


# ----------------------------
# Config (env overridable)
# ----------------------------
API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
USER_ID = int(os.getenv("USER_ID", "1"))
TOPIC_ID = int(os.getenv("TOPIC_ID", "1"))
EVAL_SET_PATH = os.getenv("EVAL_SET", "eval_set.json")
OUT_PATH = os.getenv("EVAL_OUT", "eval_report.json")

TIMEOUT = int(os.getenv("TIMEOUT", "60"))

# ✅ NEW: allow retrieval-only eval (no /query calls)
SKIP_QUERY = os.getenv("SKIP_QUERY", "0") == "1"

# ✅ pacing delays to avoid 429 quota/rate-limit errors
RETRIEVAL_DELAY_SEC = float(os.getenv("RETRIEVAL_DELAY_SEC", "0.2"))  # pause after /debug_retrieve
ANSWER_DELAY_SEC = float(os.getenv("ANSWER_DELAY_SEC", "2.0"))        # pause after /query

# ✅ retries for flaky / slow LLM calls
MAX_QUERY_RETRIES = int(os.getenv("MAX_QUERY_RETRIES", "2"))           # how many times to retry /query
RETRY_BACKOFF_SEC = float(os.getenv("RETRY_BACKOFF_SEC", "5.0"))       # base backoff for retries (grows)

# This is the exact prefix your backend uses in fallback
FALLBACK_PREFIX = "I don't know this based on the provided documents. But here is what I know about it:"


# ----------------------------
# Embedder for semantic similarity
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ----------------------------
# Metrics
# ----------------------------
def recall_at_k(retrieved_sources: List[str], expected_sources: List[str], k: int) -> float:
    topk = retrieved_sources[:k]
    return 1.0 if any(s in topk for s in expected_sources) else 0.0


def mrr(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    for i, s in enumerate(retrieved_sources, start=1):
        if s in expected_sources:
            return 1.0 / i
    return 0.0


def semantic_sim(a: str, b: str) -> Optional[float]:
    if not a or not b:
        return None
    ea = embedder.encode(a, convert_to_tensor=True)
    eb = embedder.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(ea, eb)[0][0])


def strip_fallback_prefix(ans: str) -> str:
    if not ans:
        return ans
    if ans.startswith(FALLBACK_PREFIX):
        return ans[len(FALLBACK_PREFIX):].lstrip("\n ").strip()
    return ans.strip()


def classify_answer_mode(ans: str) -> str:
    a = (ans or "").strip()
    if not a:
        return "empty"
    if a.startswith("I don't know this based on the provided documents."):
        return "fallback"
    if a == "I don't know based on the provided documents.":
        return "rag_unknown"
    return "rag_or_other"


# ----------------------------
# API calls
# ----------------------------
def post_json(path: str, payload: dict, timeout: Optional[int] = None) -> requests.Response:
    return requests.post(
        f"{API}{path}",
        json=payload,
        timeout=timeout if timeout is not None else TIMEOUT,
    )


def post_query_with_retries(payload: dict) -> requests.Response:
    """
    Calls /query with retries for timeouts/5xx.
    Uses exponential-ish backoff: RETRY_BACKOFF_SEC * attempt.
    """
    last_exc = None
    for attempt in range(MAX_QUERY_RETRIES + 1):
        try:
            # allow query to take longer than overall TIMEOUT
            # (still controlled by env TIMEOUT)
            return post_json("/query", payload, timeout=TIMEOUT)
        except requests.exceptions.ReadTimeout as e:
            last_exc = e
        except requests.exceptions.ConnectionError as e:
            last_exc = e
        except requests.exceptions.HTTPError as e:
            # If it's a 5xx, retry; if it's 4xx, don't retry
            resp = getattr(e, "response", None)
            if resp is not None and 500 <= resp.status_code <= 599:
                last_exc = e
            else:
                raise

        # retry delay
        if attempt < MAX_QUERY_RETRIES:
            sleep_s = RETRY_BACKOFF_SEC * (attempt + 1)
            time.sleep(sleep_s)

    # If all retries failed, raise last exception
    if last_exc:
        raise last_exc
    raise RuntimeError("Query failed without an exception (unexpected).")


def run():
    data = json.load(open(EVAL_SET_PATH, "r", encoding="utf-8"))
    rows: List[Dict[str, Any]] = []

    for ex in data:
        qid = ex.get("qid")
        q = ex["question"]
        expected_sources = ex.get("expected_sources", [])
        expected_answer = ex.get("expected_answer", "")

        # ---------- Retrieval ----------
        t0 = time.time()
        r = post_json("/debug_retrieve", {"user_id": USER_ID, "topic_id": TOPIC_ID, "text": q})
        r.raise_for_status()
        dbg = r.json()
        retrieval_time = time.time() - t0

        if RETRIEVAL_DELAY_SEC > 0:
            time.sleep(RETRIEVAL_DELAY_SEC)

        retrieved = dbg.get("retrieved", [])
        retrieved_sources = [x.get("source") for x in retrieved if isinstance(x, dict) and x.get("source")]

        r5 = recall_at_k(retrieved_sources, expected_sources, 5)
        r10 = recall_at_k(retrieved_sources, expected_sources, 10)
        rr = mrr(retrieved_sources, expected_sources)

        # defaults for retrieval-only mode
        ans = ""
        citations: List[Any] = []
        cited_sources: List[str] = []
        citation_hit: Optional[float] = None
        sim: Optional[float] = None
        answer_time: Optional[float] = None
        mode = "retrieval_only" if SKIP_QUERY else "unknown"

        # ---------- Answer ----------
        if not SKIP_QUERY:
            t1 = time.time()
            a = post_query_with_retries({"user_id": USER_ID, "topic_id": TOPIC_ID, "text": q})
            a.raise_for_status()
            out = a.json()
            answer_time = time.time() - t1

            if ANSWER_DELAY_SEC > 0:
                time.sleep(ANSWER_DELAY_SEC)

            ans = out.get("answer", "")
            citations = out.get("citations", [])

            cited_sources = [c.get("source") for c in citations if isinstance(c, dict) and c.get("source")]
            citation_hit = 1.0 if any(s in cited_sources for s in expected_sources) else 0.0

            cleaned_ans_for_sim = strip_fallback_prefix(ans)
            sim = semantic_sim(cleaned_ans_for_sim, expected_answer) if expected_answer else None

            mode = classify_answer_mode(ans)

        rows.append({
            "qid": qid,
            "question": q,
            "topic_id": TOPIC_ID,
            "expected_sources": expected_sources,
            "expected_answer": expected_answer,

            "recall@5": r5,
            "recall@10": r10,
            "mrr": rr,
            "citation_hit": citation_hit,
            "semantic_sim": sim,

            "retrieval_sec": retrieval_time,
            "answer_sec": answer_time,
            "answer_mode": mode,

            "answer": ans,
            "citations": citations,
            "top_sources": retrieved_sources[:10],
        })

        if SKIP_QUERY:
            print(f"[{qid}] r@5={r5} r@10={r10} mrr={rr:.3f} mode={mode}")
        else:
            print(
                f"[{qid}] r@5={r5} r@10={r10} mrr={rr:.3f} "
                f"cite={citation_hit} sim={None if sim is None else round(sim,3)} mode={mode}"
            )

    # ---------- Summary ----------
    def avg(key: str) -> Optional[float]:
        vals = [x[key] for x in rows if x.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "api": API,
        "user_id": USER_ID,
        "topic_id": TOPIC_ID,
        "eval_set": EVAL_SET_PATH,
        "n": len(rows),

        "avg_recall@5": avg("recall@5"),
        "avg_recall@10": avg("recall@10"),
        "avg_mrr": avg("mrr"),
        "avg_citation_hit": avg("citation_hit"),
        "avg_semantic_sim": avg("semantic_sim"),
        "avg_retrieval_sec": avg("retrieval_sec"),
        "avg_answer_sec": avg("answer_sec"),

        "fallback_rate": (
            sum(1 for x in rows if x.get("answer_mode") == "fallback") / len(rows)
            if rows else None
        ),

        "skip_query": SKIP_QUERY,
        "retrieval_delay_sec": RETRIEVAL_DELAY_SEC,
        "answer_delay_sec": ANSWER_DELAY_SEC,
        "timeout_sec": TIMEOUT,
        "max_query_retries": MAX_QUERY_RETRIES,
        "retry_backoff_sec": RETRY_BACKOFF_SEC,
    }

    json.dump({"summary": summary, "rows": rows}, open(OUT_PATH, "w", encoding="utf-8"), indent=2)
    print("\nSUMMARY:", summary)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    run()