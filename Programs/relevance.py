import re
import csv
import json
import sys
from typing import List, Tuple


# Load config file for stock-specific settings
def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

CONFIG = None
TARGET_ALIASES = []
DIRECT_REGEX = None
INDIRECT_REGEX = None
LOW_SIGNAL_PHRASES = None

def set_config(config):
    global CONFIG, TARGET_ALIASES, DIRECT_REGEX, INDIRECT_REGEX, LOW_SIGNAL_PHRASES
    CONFIG = config
    TARGET_ALIASES = config.get('aliases', [])
    # Load direct keywords and products
    direct_keywords = config.get('direct_keywords', [])
    product_keywords = config.get('products', [])
    all_direct_keywords = direct_keywords + product_keywords
    if all_direct_keywords:
        direct_pattern = r"|".join([re.escape(k) for k in all_direct_keywords])
        DIRECT_REGEX = re.compile(direct_pattern, re.IGNORECASE)
    else:
        DIRECT_REGEX = re.compile(r".")
    # Load indirect keywords
    indirect_keywords = config.get('indirect_keywords', [])
    if indirect_keywords:
        indirect_pattern = r"|".join([re.escape(k) for k in indirect_keywords])
        INDIRECT_REGEX = re.compile(indirect_pattern, re.IGNORECASE)
    else:
        INDIRECT_REGEX = re.compile(r".")
    # Load low signal phrases
    low_signal_phrases = config.get('low_signal_phrases', [])
    if low_signal_phrases:
        low_signal_pattern = r"|".join([re.escape(k) for k in low_signal_phrases])
        LOW_SIGNAL_PHRASES = re.compile(low_signal_pattern, re.IGNORECASE)
    else:
        LOW_SIGNAL_PHRASES = re.compile(r".")


QUOTE_PATTERN = re.compile(r"[\"']([^\"']{10,180})[\"']")


def text_fields(row: dict) -> str:
    return " ".join([row.get('headline',''), row.get('source',''), row.get('summary','')])


def mentions_target(txt: str) -> bool:
    return any(re.search(rf"\b{re.escape(alias)}\b", txt, re.IGNORECASE) for alias in TARGET_ALIASES)


def category_and_score(row: dict) -> Tuple[str, float, List[str]]:
    txt = text_fields(row)
    is_target = mentions_target(txt)

    snippets: List[str] = []

    for field in ['headline', 'summary']:
        if field in row and row[field]:
            for m in QUOTE_PATTERN.finditer(row[field]):
                snippets.append(m.group(0))
                if len(snippets) >= 3:
                    break
        if len(snippets) >= 3:
            break

    def add_snippet_from(field: str, pattern: re.Pattern, max_len: int = 160):
        nonlocal snippets
        if len(snippets) >= 3 or not row.get(field):
            return
        m = pattern.search(row[field])
        if m:
            span_text = row[field][max(0, m.start()-40): m.end()+40]
            span_text = span_text.strip()
            if len(span_text) > max_len:
                span_text = span_text[:max_len].rstrip() + '…'
            snippets.append('"' + span_text.replace('"', '""') + '"')

    # Use config aliases for indirect patterns
    indirect_pattern = re.compile("|".join([re.escape(alias) for alias in TARGET_ALIASES]), re.IGNORECASE) if TARGET_ALIASES else re.compile(r".")

    if is_target and DIRECT_REGEX.search(txt):
        category = "Directly Related"
        if re.search(r"\b(FDA|approval|recall|verdict|acquisition|merger|spin[- ]?off|CEO|CFO)\b", txt, re.IGNORECASE):
            score = 0.92
        elif re.search(r"\b(earnings|EPS|guidance|price target|upgrade|downgrade)\b", txt, re.IGNORECASE):
            score = 0.86
        else:
            score = 0.75
        add_snippet_from('headline', DIRECT_REGEX)
        add_snippet_from('summary', DIRECT_REGEX)
    elif is_target:
        category = "Indirectly Related"
        score = 0.34 if LOW_SIGNAL_PHRASES.search(txt) else 0.48
        add_snippet_from('headline', indirect_pattern)
        add_snippet_from('summary', indirect_pattern)
    else:
        if INDIRECT_REGEX.search(txt):
            category = "Indirectly Related"
            score = 0.36 if LOW_SIGNAL_PHRASES.search(txt) else 0.44
            add_snippet_from('headline', INDIRECT_REGEX)
            add_snippet_from('summary', INDIRECT_REGEX)
        else:
            category = "Unrelated"
            score = 0.06 if LOW_SIGNAL_PHRASES.search(txt) else 0.12
            add_snippet_from('headline', re.compile(r".+"))

    def clamp(score: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, score))

    if category == "Directly Related":
        score = clamp(score, 0.70, 1.00)
    elif category == "Indirectly Related":
        score = clamp(score, 0.30, 0.69)
    else:
        score = clamp(score, 0.00, 0.29)

    if not snippets:
        for field in ['headline', 'summary']:
            if row.get(field):
                s = row[field].strip()
                if s:
                    if len(s) > 160:
                        s = s[:160].rstrip() + '…'
                    snippets.append('"' + s.replace('"', '""') + '"')
            if len(snippets) >= 1:
                break

    return category, round(score, 2), snippets[:3]


def rationale_for(category: str, row: dict) -> str:
    # Use config aliases for rationale
    ticker = TARGET_ALIASES[0] if TARGET_ALIASES else "the target stock"
    if category == "Directly Related":
        return f"Article contains company-specific, near-term catalyst (earnings/regulatory/transaction) likely to move {ticker} next week."
    elif category == "Indirectly Related":
        return f"Mentions {ticker} or sector/peer factors with plausible but weaker near-term impact on {ticker}."
    else:
        return f"General market or unrelated content with no expected near-term effect on {ticker}."


def process(input_path: str, output_path: str, config_path: str):
    config = load_config(config_path)
    set_config(config)

    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        required_cols = ['date', 'headline', 'source', 'summary']
        for c in required_cols:
            if c not in reader.fieldnames:
                raise ValueError(f"Missing required column: {c}")

        rows = list(reader)

    out_fields = ['date', 'relevance_score', 'category', 'rationale', 'evidence_spans']
    with open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for row in rows:
            date = row.get('date', '').strip()
            category, score, snippets = category_and_score(row)
            rationale = rationale_for(category, row)

            ev_list = []
            for s in snippets:
                if s.startswith('"') and s.endswith('"'):
                    ev_list.append(s[1:-1])
                else:
                    ev_list.append(s)
            evidence = json.dumps(ev_list, ensure_ascii=False)

            writer.writerow({
                'date': date,
                'relevance_score': f"{score:.2f}",
                'category': category,
                'rationale': rationale,
                'evidence_spans': evidence
            })

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python relevance.py <input_csv> <output_csv> <config_json>")
        sys.exit(1)
    process(sys.argv[1], sys.argv[2], sys.argv[3])