import os, sys, unicodedata, re
from pathlib import Path
from typing import Tuple
import regex as reg
import fitz  # PyMuPDF
import concurrent.futures as cf

DEV_RANGE = r"\p{InDevanagari}"
DEV_DANDA, DEV_DDANDA = "\u0964", "\u0965"

def looks_devanagari(s: str, thresh=0.20) -> bool:
    if not s: 
        return False
    dev = len(reg.findall(DEV_RANGE, s))
    letters = len(reg.findall(r"\p{Letter}", s))
    if letters == 0:
        return dev >= max(10, int(0.1 * len(s)))
    return (dev / max(1, letters)) >= thresh

def clean_devanagari_relaxed(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for ch in ["\u200c","\u200d","\u00ad","\ufeff","\u2060"]:
        text = text.replace(ch, "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    kept = []
    for line in text.splitlines():
        ln = line.strip()
        if ln and reg.search(fr"{DEV_RANGE}|{DEV_DANDA}|{DEV_DDANDA}|\p{{Letter}}|\p{{Number}}", ln):
            kept.append(ln)
    return "\n".join(kept)

def _extract_one_task(args) -> Tuple[int, str]:
    """
    Runs in a separate process. Must re-open the PDF here.
    Returns (page_index, text)
    """
    (pdf_path, page_idx, dev_thresh, tesseract_cmd, ocr_lang, dpi) = args
    # set pytesseract path inside worker (Windows)
    if tesseract_cmd:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        except Exception:
            pass

    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_idx)
        text = page.get_text("text") or ""
        # if the text layer looks fine, skip OCR
        if len(text.strip()) >= 50 and looks_devanagari(text, dev_thresh):
            return (page_idx, text)

        # OCR fallback
        try:
            from PIL import Image
            import pytesseract
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang=ocr_lang) or ""
        except Exception as ex:
            sys.stderr.write(f"[WARN] OCR failed on page {page_idx+1}: {ex}\n")
            # keep whatever text we had (maybe empty)
        finally:
            doc.close()
        return (page_idx, text)
    except Exception as ex:
        sys.stderr.write(f"[ERR] Worker failed on page {page_idx+1}: {ex}\n")
        return (page_idx, "")

def extract_hybrid_parallel(pdf_path: str,
                            start_page: int,
                            end_page: int,
                            dev_thresh: float = 0.20,
                            tesseract_cmd: str | None = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                            ocr_lang: str = "hin+eng",
                            dpi: int = 400,
                            max_workers: int | None = None,
                            chunksize: int = 2) -> str:
    # page indices are 0-based for fitz
    with fitz.open(pdf_path) as d:
        n = d.page_count
    s = max(0, start_page - 1)
    e = min(end_page - 1, n - 1)
    pages = list(range(s, e + 1))

    if max_workers is None:
        cores = os.cpu_count() or 4
        max_workers = min(max(1, cores // 2), 8)

    tasks = [(pdf_path, i, dev_thresh, tesseract_cmd, ocr_lang, dpi) for i in pages]

    results = [None] * len(pages)
    # Use map to preserve order and avoid the as_completed unpacking bug
    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        # small progress indicator
        for k, (idx, text) in enumerate(ex.map(_extract_one_task, tasks, chunksize=chunksize)):
            results[idx - s] = text
            if (k + 1) % 10 == 0:
                print(f"[PROG] {k+1}/{len(tasks)} pages done...", flush=True)

    return "\n".join(results)

if __name__ == "__main__":
    pdf = r".\shri_ramcharitramanas.pdf"
    start, end = 20, 350

    try:
        raw = extract_hybrid_parallel(
            pdf_path=pdf,
            start_page=start,
            end_page=end,
            dev_thresh=0.20,
            tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe",  # or None if on PATH
            ocr_lang="hin+eng",
            dpi=400,
            max_workers=None,   # auto: ~half cores, capped at 8
            chunksize=2
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopped by user.", flush=True)
        sys.exit(1)

    print("RAW CHARS:", len(raw))
    cleaned = clean_devanagari_relaxed(raw)
    print("CLEAN CHARS:", len(cleaned))
    outdir = Path("bpe_awadhi_hindi"); outdir.mkdir(exist_ok=True)
    (outdir / "sample_22_350.txt").write_text(cleaned, encoding="utf-8")
    print("[DONE] Saved:", outdir / "sample_22_350.txt")
