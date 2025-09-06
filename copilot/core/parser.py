import re
import hashlib
from pathlib import Path
from typing import List
from .models import Utterance, CallMeta

LINE_RE = re.compile(r"^\[(\d{2}):(\d{2})\]\s([^:]+):\s(.*)$")  # [mm:ss] Speaker: text

def parse_timestamp(mm: str, ss: str) -> int:
    return int(mm) * 60 + int(ss)

def parse_file(path: str) -> (CallMeta, List[Utterance]):
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    sha256 = hashlib.sha256(text.encode('utf-8')).hexdigest()
    call_id = p.stem

    lines = text.splitlines()
    utts: List[Utterance] = []
    last_start = None
    last_idx = None

    parsed = []
    for i, line in enumerate(lines):
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        mm, ss, speaker, content = m.groups()
        start = parse_timestamp(mm, ss)

        # Infer role from speaker token if available in parentheses, else keep as speaker
        role = speaker
        r = re.search(r"\(([^)]+)\)", speaker)
        if r:
            role = r.group(1)

        parsed.append((start, speaker, role, content))

    # infer end times
    for i, (start, speaker, role, content) in enumerate(parsed):
        end = parsed[i+1][0] if i+1 < len(parsed) else start + 5  # fallback small window
        utts.append(Utterance(call_id=call_id, start_sec=start, end_sec=end, speaker=speaker, role=role, text=content))

    meta = CallMeta(call_id=call_id, filename=str(p.name), sha256=sha256)
    return meta, utts
