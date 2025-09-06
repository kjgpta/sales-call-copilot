import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
from .models import Utterance, Segment, CallMeta
from .config import DB_PATH

DDL = [
    '''CREATE TABLE IF NOT EXISTS calls(
        call_id TEXT PRIMARY KEY,
        filename TEXT,
        sha256 TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''',
    '''CREATE TABLE IF NOT EXISTS utterances(
        utt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_id TEXT,
        start_sec INT,
        end_sec INT,
        speaker TEXT,
        role TEXT,
        text TEXT,
        FOREIGN KEY(call_id) REFERENCES calls(call_id)
    )''',
    '''CREATE TABLE IF NOT EXISTS segments(
        seg_id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_id TEXT,
        start_sec INT,
        end_sec INT,
        text TEXT,
        labels TEXT DEFAULT '[]',
        sentiment REAL DEFAULT 0.0,
        FOREIGN KEY(call_id) REFERENCES calls(call_id)
    )'''
]

def connect():
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))

def ensure_migrations(con):
    # Ensure segments has labels & sentiment
    cur = con.cursor()
    cur.execute("PRAGMA table_info(segments)")
    cols = [r[1] for r in cur.fetchall()]
    if "labels" not in cols:
        cur.execute("ALTER TABLE segments ADD COLUMN labels TEXT DEFAULT '[]'")
    if "sentiment" not in cols:
        cur.execute("ALTER TABLE segments ADD COLUMN sentiment REAL DEFAULT 0.0")
    con.commit()

def init_db():
    con = connect()
    try:
        cur = con.cursor()
        for stmt in DDL:
            cur.execute(stmt)
        ensure_migrations(con)
        con.commit()
    finally:
        con.close()

def insert_call(meta: CallMeta):
    con = connect()
    try:
        con.execute("INSERT OR REPLACE INTO calls(call_id, filename, sha256) VALUES(?,?,?)",
                    (meta.call_id, meta.filename, meta.sha256))
        con.commit()
    finally:
        con.close()

def insert_utterances(utts: Iterable[Utterance]):
    con = connect()
    try:
        con.executemany(            "INSERT INTO utterances(call_id,start_sec,end_sec,speaker,role,text) VALUES(?,?,?,?,?,?)",
            [(u.call_id, u.start_sec, u.end_sec, u.speaker, u.role, u.text) for u in utts])
        con.commit()
    finally:
        con.close()

def insert_segments(segs: Iterable[Segment]) -> List[int]:
    con = connect()
    ids = []
    try:
        cur = con.cursor()
        for s in segs:
            cur.execute("INSERT INTO segments(call_id,start_sec,end_sec,text,labels,sentiment) VALUES(?,?,?,?,?,?)",
                        (s.call_id, s.start_sec, s.end_sec, s.text, s.labels, s.sentiment))
            ids.append(cur.lastrowid)
        con.commit()
        return ids
    finally:
        con.close()

def list_calls() -> List[Tuple[str,str]]:
    con = connect()
    try:
        rows = con.execute("SELECT call_id, filename FROM calls ORDER BY created_at DESC").fetchall()
        return rows
    finally:
        con.close()

def get_segments_for_call(call_id: str) -> List[Tuple[int,int,int,str,str,float]]:
    con = connect()
    try:
        rows = con.execute("SELECT seg_id,start_sec,end_sec,text,labels,sentiment FROM segments WHERE call_id=? ORDER BY start_sec", (call_id,)).fetchall()
        return rows
    finally:
        con.close()

def all_segments() -> List[Tuple[int,str,int,int,str,str,float]]:
    con = connect()
    try:
        rows = con.execute("SELECT seg_id,call_id,start_sec,end_sec,text,labels,sentiment FROM segments").fetchall()
        return rows
    finally:
        con.close()


def utterances_by_call(call_id: str):
    con = connect()
    try:
        rows = con.execute("SELECT start_sec,end_sec,speaker,role,text FROM utterances WHERE call_id=? ORDER BY start_sec", (call_id,)).fetchall()
        return rows
    finally:
        con.close()
