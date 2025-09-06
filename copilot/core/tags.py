from typing import List, Dict

TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "pricing": ["price", "pricing", "list", "quote", "cost", "per seat", "per user", "sku", "minutes", "unlimited", "discount"],
    "discount": ["discount", "concession", "promo", "deal", "match", "counter", "negotiate"],
    "security": ["soc2", "iso", "dpa", "gdpr", "residency", "retention", "sso", "saml", "oauth", "scim", "okta", "encryption"],
    "legal": ["msa", "order form", "mfn", "indemnity", "liability", "arbitration", "governing law"],
    "sla": ["sla", "uptime", "downtime", "credit", "response time", "rto", "rpo"],
    "competitor": ["competitor", "alternative", "vendor", "quote from", "pricing from"],
    "objection": ["concern", "risk", "pushback", "blocker", "issue", "problem"],
    "negotiation": ["counter", "offer", "propose", "close", "signature", "approval", "procurement"]
}

def tag_text(text: str) -> List[str]:
    t = text.lower()
    out = set()
    for topic, kws in TOPIC_KEYWORDS.items():
        for k in kws:
            if k in t:
                out.add(topic)
                break
    return sorted(out)
