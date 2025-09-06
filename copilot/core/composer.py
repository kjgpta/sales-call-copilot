from typing import List, Dict
from .llm import answer_query

def _coalesce(items: List[Dict], max_gap_sec: int = 20) -> List[Dict]:
    # Merge adjacent segments from same call if they are close
    if not items:
        return items
    items = sorted(items, key=lambda x: (x['call_id'], x['start_sec']))
    out: List[Dict] = []
    cur = dict(items[0])
    for it in items[1:]:
        if it['call_id'] == cur['call_id'] and it['start_sec'] <= cur['end_sec'] + max_gap_sec:
            # merge
            cur['end_sec'] = max(cur['end_sec'], it['end_sec'])
            cur['text'] = cur['text'] + "\n" + it['text']
        else:
            out.append(cur)
            cur = dict(it)
    out.append(cur)
    return out

def build_answer(query: str, seg_rows: List[Dict]) -> str:
    # seg_rows: list of dicts {call_id, start_sec, end_sec, text}
    coalesced = _coalesce(seg_rows, max_gap_sec=20)
    # Keep top 5 windows by length to avoid overloading context
    coalesced.sort(key=lambda x: (x['end_sec']-x['start_sec']), reverse=True)
    coalesced = coalesced[:5]
    return answer_query(query, coalesced)
