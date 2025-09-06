
import os, json
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Body, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Make local package importable
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from copilot.core import config
from copilot.core.storage import init_db, insert_call, insert_utterances, insert_segments, list_calls, all_segments, get_segments_for_call, utterances_by_call
from copilot.core.parser import parse_file
from copilot.core.chunker import chunk_utterances
from copilot.core.tags import tag_text
from copilot.core.sentiment import compound_score
from copilot.core.index_lex import BM25Index
from copilot.core.index_dense import DenseIndex
from copilot.core.retriever import Retriever
from copilot.core.composer import build_answer

app = FastAPI(title="Sales Call Copilot API")

# Static UI
WEB_DIR = BASE_DIR / "web"
WEB_DIR.mkdir(exist_ok=True, parents=True)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

@app.on_event("startup")
def _maybe_auto_ingest():
    # Auto-ingest sample transcripts if DB or index is empty and AUTO_INGEST enabled
    if not config.AUTO_INGEST:
        return
    try:
        # Check if BM25 index exists and if we have any calls
        bm25_exists = (BASE_DIR / "index" / "bm25.pkl").exists()
        rows = list_calls()
        has_calls = len(rows) > 0
        if bm25_exists and has_calls:
            return  # nothing to do

        # Ingest sample data if present
        init_db()
        data_dir = BASE_DIR / "data"
        files = sorted(list(data_dir.glob("*.txt")))
        if not files:
            return

        total_segs = 0
        total_utts = 0
        for f in files:
            meta, utts = parse_file(str(f))
            insert_call(meta)
            insert_utterances(utts)
            segs = chunk_utterances(utts)
            for s in segs:
                s.labels = json.dumps(tag_text(s.text))
                s.sentiment = compound_score(s.text)
            insert_segments(segs)
            total_utts += len(utts)
            total_segs += len(segs)

        # rebuild bm25
        seg_rows = all_segments()
        bm25 = BM25Index(config.INDEX_DIR)
        bm25.build([(r[0], r[4]) for r in seg_rows])
        bm25.save()

        # dense optional
        if config.EMBEDDING_BACKEND == "sbert":
            try:
                import numpy as np
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                texts = [r[4] for r in seg_rows]
                embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
                dense = DenseIndex(config.INDEX_DIR)
                dense.build([r[0] for r in seg_rows], embs)
                dense.save()
            except Exception:
                pass
    except Exception:
        # Best-effort; avoid blocking server startup
        pass

@app.get("/", response_class=HTMLResponse)
def index():
    idx = WEB_DIR / "index.html"
    return HTMLResponse(idx.read_text(encoding="utf-8"))

def _load_indices():
    bm25 = BM25Index(config.INDEX_DIR)
    if not bm25.load():
        raise RuntimeError("BM25 index not found. Ingest first.")
    dense = DenseIndex(config.INDEX_DIR)
    has_dense = dense.load()
    return bm25, (dense if has_dense else None)

def _segment_dict(seg_row) -> Dict[str, Any]:
    seg_id, call_id, start, end, text, labels, sentiment = seg_row
    return {
        "seg_id": seg_id, "call_id": call_id, "start_sec": start, "end_sec": end,
        "text": text, "labels": labels, "sentiment": sentiment
    }

@app.post("/api/ingest-sample")
def ingest_sample():
    init_db()
    data_dir = BASE_DIR / "data"
    files = sorted(list(data_dir.glob("*.txt")))
    if not files:
        return {"ok": False, "message": "No .txt transcripts in ./data"}

    total_segs = 0
    total_utts = 0
    for f in files:
        meta, utts = parse_file(str(f))
        insert_call(meta)
        insert_utterances(utts)
        segs = chunk_utterances(utts)
        for s in segs:
            s.labels = json.dumps(tag_text(s.text))
            s.sentiment = compound_score(s.text)
        seg_ids = insert_segments(segs)
        total_utts += len(utts)
        total_segs += len(seg_ids)

    # rebuild bm25
    seg_rows = all_segments()
    bm25 = BM25Index(config.INDEX_DIR)
    bm25.build([(r[0], r[4]) for r in seg_rows])
    bm25.save()

    # dense embeddings optional
    built_dense = False
    if config.EMBEDDING_BACKEND == "sbert":
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            texts = [r[4] for r in seg_rows]
            embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            dense = DenseIndex(config.INDEX_DIR)
            dense.build([r[0] for r in seg_rows], embs)
            dense.save()
            built_dense = True
        except Exception as e:
            built_dense = False

    return {"ok": True, "utterances": total_utts, "segments": total_segs, "dense": built_dense}

@app.get("/api/calls")
def calls():
    init_db()  # NEW: creates tables if missing
    rows = list_calls()
    return [{"call_id": cid, "filename": fn} for cid, fn in rows]


@app.post("/api/ask")
def ask(payload: Dict[str, Any] = Body(...)):
    query = payload.get("query") or ""
    include_topics = payload.get("include_topics")
    exclude_topics = payload.get("exclude_topics")
    min_sent = float(payload.get("min_sentiment", -1.0))
    max_sent = float(payload.get("max_sentiment", 1.0))

    bm25, dense = _load_indices()
    retr = Retriever(bm25, dense)

    # Optional dense qvec
    qvec = None
    if dense is not None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            qvec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        except Exception:
            qvec = None

    merged = retr.retrieve(query, qvec=qvec, topk=30)

    seg_rows_map = {r[0]: r for r in all_segments()}
    contexts = []
    for sid, score, parts in merged:
        if sid in seg_rows_map:
            contexts.append(_segment_dict(seg_rows_map[sid]))

    # topic + sentiment filters
    def _to_set(csv):
        if not csv:
            return None
        return set([x.strip().lower() for x in csv.split(",") if x.strip()])

    inc = _to_set(include_topics)
    exc = _to_set(exclude_topics)

    def _has_topic(labels_json, target_set):
        if target_set is None:
            return True
        try:
            labs = set([x.lower() for x in json.loads(labels_json or "[]")])
        except Exception:
            labs = set()
        return any(t in labs for t in target_set)

    filtered = []
    for it in contexts:
        ok = True
        if inc and not _has_topic(it.get("labels","[]"), inc):
            ok = False
        if ok and exc and _has_topic(it.get("labels","[]"), exc):
            ok = False
        if ok and not (min_sent <= float(it.get("sentiment", 0.0)) <= max_sent):
            ok = False
        if ok:
            filtered.append(it)

    # Cross-encoder rerank
    try:
        from copilot.core.rerank import cross_encoder_rerank
        pairs = [(it['seg_id'], it['text']) for it in filtered]
        scores = cross_encoder_rerank(query, pairs)
        if scores:
            s_map = {sid: sc for sid, sc in scores}
            filtered.sort(key=lambda x: s_map.get(x['seg_id'], 0.0), reverse=True)
    except Exception:
        pass

    contexts = filtered[:12] if filtered else contexts[:12]
    answer = build_answer(query, contexts)
    return {"ok": True, "answer": answer, "contexts": contexts}

@app.post("/api/summarise")
def summarise(payload: Dict[str, Any] = Body(...)):
    call_id = payload.get("call_id", "last")
    rows = list_calls()
    if not rows:
        return {"ok": False, "message": "No calls ingested"}
    target = rows[0][0] if call_id == "last" else call_id

    segs = get_segments_for_call(target)
    if not segs:
        return {"ok": False, "message": f"No segments for {target}"}
    contexts = [dict(seg_id=sid, call_id=target, start_sec=st, end_sec=en, text=txt) for sid, st, en, txt, labels, sent in segs[:6]]
    question = f"Summarise call {target}: key points, risks, next steps."
    answer = build_answer(question, contexts)
    return {"ok": True, "answer": answer, "call_id": target, "contexts": contexts}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    #Accepts .txt transcript files, saves to ./data, ingests, rebuilds indices.
    init_db()
    saved = []
    BASE = Path(__file__).resolve().parent.parent
    data_dir = BASE / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for uf in files:
        name = uf.filename
        if not name.lower().endswith(".txt"):
            continue
        target = data_dir / name
        content = await uf.read()
        target.write_bytes(content)
        saved.append(str(target.name))

    # Ingest the newly saved files (reuse logic from ingest-sample)
    total_segs = 0
    total_utts = 0
    for fname in saved:
        f = (BASE / "data" / fname)
        meta, utts = parse_file(str(f))
        insert_call(meta)
        insert_utterances(utts)
        segs = chunk_utterances(utts)
        for s in segs:
            s.labels = json.dumps(tag_text(s.text))
            s.sentiment = compound_score(s.text)
        seg_ids = insert_segments(segs)
        total_utts += len(utts)
        total_segs += len(seg_ids)

    # rebuild indices
    seg_rows = all_segments()
    bm25 = BM25Index(config.INDEX_DIR)
    bm25.build([(r[0], r[4]) for r in seg_rows])
    bm25.save()

    # dense embeddings optional
    built_dense = False
    if config.EMBEDDING_BACKEND == "sbert":
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            texts = [r[4] for r in seg_rows]
            embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            dense = DenseIndex(config.INDEX_DIR)
            dense.build([r[0] for r in seg_rows], embs)
            dense.save()
            built_dense = True
        except Exception:
            built_dense = False

    return {"ok": True, "saved": saved, "utterances": total_utts, "segments": total_segs, "dense": built_dense}


@app.get("/api/transcript/{call_id}")
def transcript(call_id: str):
    rows = utterances_by_call(call_id)
    if not rows:
        return {"ok": False, "message": f"No utterances for {call_id}"}
    def to_mmss(sec: int):
        m = sec // 60
        s = sec % 60
        return f"{m:02d}:{s:02d}"
    out = []
    for (st,en,speaker,role,text) in rows:
        out.append({
            "start_sec": st, "end_sec": en,
            "start": to_mmss(st), "end": to_mmss(en),
            "speaker": speaker, "role": role, "text": text
        })
    return {"ok": True, "call_id": call_id, "utterances": out}
