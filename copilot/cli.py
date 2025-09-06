import os, sys, json, pickle
from pathlib import Path
from typing import List, Dict, Tuple
import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from .core import config
from .core.storage import init_db, insert_call, insert_utterances, insert_segments, list_calls, all_segments, get_segments_for_call
from .core.parser import parse_file
from .core.chunker import chunk_utterances
from .core.tags import tag_text
from .core.sentiment import compound_score
from .core.index_lex import BM25Index
from .core.index_dense import DenseIndex
from .core.retriever import Retriever
from .core.composer import build_answer

app = typer.Typer(help="Sales Call Copilot CLI")
console = Console()

INDEX_DIR = config.INDEX_DIR

def _format_mmss(sec: int) -> str:
    m = sec // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"

@app.command()
def ingest(paths: List[str] = typer.Argument(..., help="Glob(s) for transcript files")):
    init_db()
    files: List[Path] = []
    for p in paths:
        files += [Path(x) for x in sorted(Path().glob(p))]
    if not files:
        console.print("[red]No files matched[/red]")
        raise typer.Exit(1)

    for f in files:
        meta, utts = parse_file(str(f))
        insert_call(meta)
        insert_utterances(utts)
        segs = chunk_utterances(utts)
        # annotate
        for s in segs:
            s.labels = json.dumps(tag_text(s.text))
            s.sentiment = compound_score(s.text)
        seg_ids = insert_segments(segs)
        console.print(f"Ingested {f.name}: {len(utts)} utterances → {len(seg_ids)} segments (with topics & sentiment)")

    # rebuild indices
    seg_rows = all_segments()  # (seg_id, call_id, start, end, text)
    # BM25
    from .core.index_lex import BM25Index
    bm25 = BM25Index(INDEX_DIR)
    bm25.build([(r[0], r[4]) for r in seg_rows])
    bm25.save()

    # Dense (optional)
    if config.EMBEDDING_BACKEND == 'sbert':
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            texts = [r[4] for r in seg_rows]
            embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
            dense = DenseIndex(INDEX_DIR)
            dense.build([r[0] for r in seg_rows], embs)
            dense.save()
            console.print("[green]Saved dense embeddings[/green]")
        except Exception as e:
            console.print(f"[yellow]Dense embeddings disabled: {e}[/yellow]")
    else:
        console.print("[yellow]Skipping dense embeddings (EMBEDDING_BACKEND != 'sbert')[/yellow]")

    console.print("[bold green]Rebuilt indices[/bold green]")

@app.command("list-calls")
def list_calls_cmd():
    rows = list_calls()
    if not rows:
        console.print("No calls ingested yet.")
        raise typer.Exit(0)
    table = Table(title="Calls")
    table.add_column("Call ID", style="cyan")
    table.add_column("Filename", style="magenta")
    for cid, fn in rows:
        table.add_row(cid, fn)
    console.print(table)

def _load_indices():
    from .core.index_lex import BM25Index
    bm25 = BM25Index(INDEX_DIR)
    if not bm25.load():
        console.print("[red]BM25 index not found. Run 'ingest' first.[/red]")
        raise typer.Exit(1)
    dense = DenseIndex(INDEX_DIR)
    has_dense = dense.load()
    return bm25, (dense if has_dense else None)

def _topic_set(csv: str):
    if not csv:
        return None
    return set([x.strip().lower() for x in csv.split(',') if x.strip()])

def _segment_dict(seg_row) -> Dict:
    seg_id, call_id, start, end, text, labels, sentiment = seg_row
    return {"seg_id": seg_id, "call_id": call_id, "start_sec": start, "end_sec": end, "text": text, "labels": labels, "sentiment": sentiment}

@app.command()
def ask(query: str,
        include_topics: str = typer.Option(None, help='Comma-separated topics to include'),
        exclude_topics: str = typer.Option(None, help='Comma-separated topics to exclude'),
        min_sentiment: float = typer.Option(-1.0, help='VADER compound min'),
        max_sentiment: float = typer.Option(1.0, help='VADER compound max')
    ):
    bm25, dense = _load_indices()
    from .core.retriever import Retriever
    retr = Retriever(bm25, dense)

    # Optional dense query vec
    qvec = None
    if dense is not None:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer('all-MiniLM-L6-v2')
            qvec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        except Exception:
            qvec = None

    merged = retr.retrieve(query, qvec=qvec, topk=20)

    # Fetch all segments and build map
    seg_rows_map = {r[0]: r for r in all_segments()}

    # Initial contexts (top N)
    contexts = []
    for sid, score, parts in merged:
        if sid in seg_rows_map:
            item = _segment_dict(seg_rows_map[sid])
            contexts.append(item)

    # Topic & sentiment filtering
    inc = _topic_set(include_topics)
    exc = _topic_set(exclude_topics)
    def has_topic(labels_json, target):
        try:
            labs = set([x.lower() for x in json.loads(labels_json or '[]')])
        except Exception:
            labs = set()
        return any(t in labs for t in target)

    filtered = []
    for it in contexts:
        ok = True
        if inc:
            ok = has_topic(it.get("labels","[]"), inc)
        if ok and exc:
            if has_topic(it.get("labels","[]"), exc):
                ok = False
        if ok and not (min_sentiment <= float(it.get("sentiment", 0.0)) <= max_sentiment):
            ok = False
        if ok:
            filtered.append(it)

    # Cross-encoder rerank (if available)
    try:
        from .core.rerank import cross_encoder_rerank
        pairs = [(it['seg_id'], it['text']) for it in filtered]
        scores = cross_encoder_rerank(query, pairs)
        if scores:
            score_map = {sid: sc for sid, sc in scores}
            filtered.sort(key=lambda x: score_map.get(x['seg_id'], 0.0), reverse=True)
    except Exception as e:
        pass

    # Limit contexts for LLM
    contexts = filtered[:12] if filtered else contexts[:12]

    answer = build_answer(query, contexts)
    console.print(answer)

@app.command()
def summarise(call_id: str = typer.Argument("last")):
    rows = list_calls()
    if not rows:
        console.print("No calls ingested.")
        raise typer.Exit(1)
    target = rows[0][0] if call_id == "last" else call_id

    segs = get_segments_for_call(target)
    if not segs:
        console.print(f"No segments for call {target}")
        raise typer.Exit(1)

    # Heuristic summary: use first few segments as context for LLM (or offline extractive)
    contexts = [dict(seg_id=sid, call_id=target, start_sec=st, end_sec=en, text=txt) for sid, st, en, txt in segs[:6]]
    question = f"Summarise call {target}: key points, risks, next steps."
    answer = build_answer(question, contexts)
    console.print(answer)

if __name__ == "__main__":
    app()
