from typing import List, Tuple
from .models import Utterance, Segment
from .config import MAX_CHUNK_TOKENS, CHUNK_OVERLAP_RATIO

def _count_tokens_simple(s: str) -> int:
    # crude proxy: word count
    return max(1, len(s.split()))

def chunk_utterances(utts: List[Utterance]) -> List[Segment]:
    # Greedy pack utterances until ~MAX_CHUNK_TOKENS words, with overlap
    segments: List[Segment] = []
    i = 0
    N = len(utts)

    while i < N:
        start_i = i
        words = 0
        texts = []
        start_sec = utts[i].start_sec
        end_sec = utts[i].end_sec

        while i < N and words < MAX_CHUNK_TOKENS:
            u = utts[i]
            w = _count_tokens_simple(u.text)
            words += w
            texts.append(f"{u.speaker}: {u.text}")
            end_sec = u.end_sec
            i += 1

        text = "\n".join(texts)
        segments.append(Segment(seg_id=None, call_id=utts[0].call_id, start_sec=start_sec, end_sec=end_sec, text=text))

        # Overlap: step back by overlap ratio of the collected utterances
        span = i - start_i
        step_back = int(span * CHUNK_OVERLAP_RATIO)
        i = max(i - step_back, i) if step_back == 0 else i - step_back

    return segments
