import unicodedata, re, regex as reg, sys, os
from pathlib import Path
import fitz  # PyMuPDF

# Optional: set tesseract path on Windows if not on PATH
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DEV_RANGE = r"\p{InDevanagari}"
DEV_DANDA = "\u0964"   # ।
DEV_DDANDA = "\u0965"  # ॥

def looks_devanagari(s: str, thresh=0.20) -> bool:
    """Return True if >= thresh fraction of letters are Devanagari."""
    if not s:
        return False
    # Count letters that are Devanagari
    dev = len(reg.findall(DEV_RANGE, s))
    letters = len(reg.findall(r"\p{Letter}", s))
    if letters == 0:
        # Fallback: check codepoints directly
        return dev >= max(10, int(0.1 * len(s)))
    return (dev / max(1, letters)) >= thresh

def ocr_page(page, lang="hin"):
    import pytesseract
    from PIL import Image
    pix = page.get_pixmap(dpi=400, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(img, lang=lang)  # try "hin+eng" if needed

def extract_hybrid(pdf_path: str, start_page: int, end_page: int, dev_thresh=0.20) -> str:
    doc = fitz.open(pdf_path)
    s = max(0, start_page - 1)
    e = min(end_page - 1, doc.page_count - 1)
    chunks = []
    for i in range(s, e + 1):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        # If very little text, or looks non-Devanagari, OCR the page
        if len(text.strip()) < 50 or not looks_devanagari(text, dev_thresh):
            try:
                text = ocr_page(page, lang="hin")
            except Exception as ex:
                print(f"[WARN] OCR failed on page {i+1}: {ex}", file=sys.stderr)
                text = text  # keep whatever we had (maybe empty)
        chunks.append(text or "")
    return "\n".join(chunks)

def clean_devanagari_relaxed(text: str) -> str:
    # Normalize and remove invisible controls
    text = unicodedata.normalize("NFKC", text)
    for ch in ["\u200c", "\u200d", "\u00ad", "\ufeff", "\u2060"]:
        text = text.replace(ch, "")
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Keep lines that have Devanagari letters or danda or any letters/numbers
    kept = []
    for line in text.splitlines():
        ln = line.strip()
        if not ln:
            continue
        if reg.search(fr"{DEV_RANGE}|{DEV_DANDA}|{DEV_DDANDA}|\p{{Letter}}|\p{{Number}}", ln):
            kept.append(ln)
    return "\n".join(kept)

if __name__ == "__main__":
    pdf = r".\shri_ramcharitramanas.pdf"
    start, end = 22, 350

    raw = extract_hybrid(pdf, start, end, dev_thresh=0.20)
    print("RAW CHARS:", len(raw))
    cleaned = clean_devanagari_relaxed(raw)
    print("CLEAN CHARS:", len(cleaned))
    # Quick quality peek
    preview = "\n".join(cleaned.splitlines()[:20])
    print(preview)
    # Optional: save
    out = Path("bpe_awadhi_hindi")
    out.mkdir(exist_ok=True)
    (out / "sample_22_25.txt").write_text(cleaned, encoding="utf-8")
