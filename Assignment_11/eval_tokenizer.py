# eval_tokenizer.py
from pathlib import Path
import json, unicodedata, re
import fitz  # PyMuPDF
from tokenizers import Tokenizer

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for ch in ["\u200c", "\u200d", "\u00ad", "\ufeff", "\u2060"]:
        text = text.replace(ch, "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def looks_devanagari(s, thresh=0.20):
    import regex as reg
    dev = len(reg.findall(r"\p{InDevanagari}", s))
    letters = len(reg.findall(r"\p{Letter}", s))
    return (letters and dev/letters >= thresh)

def ocr_page(page, lang="hin+eng"):
    import pytesseract
    from PIL import Image
    pix = page.get_pixmap(dpi=400, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img, lang=lang)

def extract_hybrid(pdf_path, start_page, end_page):
    doc = fitz.open(pdf_path)
    s, e = max(0, start_page-1), min(end_page-1, doc.page_count-1)
    chunks = []
    for i in range(s, e+1):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        if len(text.strip()) < 50 or not looks_devanagari(text):
            try:
                text = ocr_page(page, lang="hin+eng")
            except Exception:
                pass
        chunks.append(text or "")
    return "\n".join(chunks)

def compression_ratio(tok: Tokenizer, text: str):
    chars = len(text.replace("\n", " "))
    n_tok, n_unk = 0, 0
    for ln in text.splitlines():
        if ln.strip():
            enc = tok.encode(ln)
            n_tok += len(enc.ids)
            n_unk += sum(1 for t in enc.tokens if t == "[UNK]")
    cr = (chars/n_tok) if n_tok else 0.0
    unk_rate = (n_unk/max(1, n_tok))
    return cr, unk_rate, n_tok

def split_awadhi_hindi(text: str):
    awadhi, hindi = [], []
    for ln in text.splitlines():
        s = ln.strip()
        if not s: continue
        if ("॥" in s or "।" in s) and len(s) <= 40:
            awadhi.append(s)
        else:
            hindi.append(s)
    return "\n".join(awadhi), "\n".join(hindi)

if __name__ == "__main__":
    pdf = r".\shri_ramcharitramanas.pdf"
    tok = Tokenizer.from_file("bpe_awadhi_hindi/run1/tokenizer.json")  # your saved tokenizer

    for (a,b) in [(200,220), (300,320)]:   # pick far-away ranges
        raw = extract_hybrid(pdf, a, b)
        txt = normalize(raw)
        aw, hi = split_awadhi_hindi(txt)
        for name, part in [("Awadhi-ish", aw), ("Hindi gloss", hi)]:
            cr, unk, n_tok = compression_ratio(tok, part)
            print(f"[EVAL:{name}] CR={cr:.3f} | UNK={unk*100:.2f}% | tokens={n_tok}")
