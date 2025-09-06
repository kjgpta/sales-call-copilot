from typing import List, Dict, Tuple
from .config import LLM_PROVIDER, OPENAI_API_KEY
from .prompts import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE

def _format_context(items: List[Dict]) -> str:
    # items: dicts with call_id, start_sec, end_sec, text
    def to_mmss(sec: int):
        m = sec // 60
        s = sec % 60
        return f"{m:02d}:{s:02d}"
    blocks = []
    for it in items:
        blocks.append(f"[CALL_ID={it['call_id']} START={to_mmss(it['start_sec'])} END={to_mmss(it['end_sec'])}]\n{it['text']}")
    return "\n\n".join(blocks)

def _sources_block(items: List[Dict]) -> str:
    # Build a conservative sources list (first 3), quoting a short slice
    def to_mmss(sec: int):
        m = sec // 60
        s = sec % 60
        return f"{m:02d}:{s:02d}"
    lines = []
    for it in items[:5]:
        quote = it['text'].split('\n')[0]
        quote = (quote[:120] + '…') if len(quote) > 120 else quote
        lines.append(f"• {it['call_id']} @ [{to_mmss(it['start_sec'])}–{to_mmss(it['end_sec'])}]: \"{quote}\"")
    return "\n".join(lines)

def answer_query(query: str, contexts: List[Dict]) -> str:
    if LLM_PROVIDER == 'openai' and OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = QA_USER_TEMPLATE.format(query=query, context=_format_context(contexts))
            resp = client.responses.create(
                model="o4-mini-2025-04-16",
                input=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT + "\nReturn an explicit 'Sources:' block at the end."},
                    {"role": "user", "content": prompt},
                ]
            )
            text = resp.output_text or ""
            if "Sources:" not in text:
                # enforce sources
                text = text.rstrip() + "\n\nSources:\n" + _sources_block(contexts)
            return text
        except Exception as e:
            print(f"LLM call failed: {e}")
            return f"[LLM call failed, falling back to extractive]\n\n" + _extractive_answer(query, contexts)
    # Offline: extractive baseline
    return _extractive_answer(query, contexts)

def _extractive_answer(query: str, contexts: List[Dict]) -> str:
    # Very simple: echo top sentences + sources
    body = "Here are the most relevant segments I found:\n\n"
    for it in contexts[:3]:
        snippet = it['text']
        if len(snippet) > 400:
            snippet = snippet[:400] + '…'
        body += f"— {it['call_id']} [{it['start_sec']}–{it['end_sec']}]\n{snippet}\n\n"
    body += "Sources:\n" + _sources_block(contexts)
    return body
