from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Utterance:
    call_id: str
    start_sec: int
    end_sec: int
    speaker: str
    role: str
    text: str

@dataclass
class Segment:
    seg_id: Optional[int]
    call_id: str
    start_sec: int
    end_sec: int
    text: str
    labels: str = "[]"  # JSON list of topics
    sentiment: float = 0.0

@dataclass
class CallMeta:
    call_id: str
    filename: str
    sha256: str
