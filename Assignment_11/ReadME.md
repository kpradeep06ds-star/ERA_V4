
---

## 🪶 Awadhi–Hindi BPE Tokenizer (8k)

A custom **Byte Pair Encoding (BPE)** tokenizer trained from scratch on *Shri Ramcharitmanas* (Gita Press edition), combining **Awadhi dohas** (poetic verses) and **Hindi glosses**.
It uses the Devanagari script and achieves a **compression ratio ≈ 3.7×** with **8 000 tokens**.

---

### 📖 Dataset Construction

1. **Source PDF**:
   *Shri Ramcharitmanas – Gita Press (Hindi)* (`shri_ramcharitramanas.pdf`)

2. **Extraction Logic**:
   Using `PyMuPDF` (`fitz`) to read textual layers directly from PDF pages.
   If a page had little or no embedded text, or wasn’t recognized as Devanagari,
   fallback OCR was triggered with Tesseract (`lang="hin+eng"`).

   ```python
   text = page.get_text("text") or ""
   if len(text.strip()) < 50 or not looks_devanagari(text):
       text = ocr_page(page, lang="hin")
   ```

   The hybrid extractor is defined in [`extract_parallel.py`](extract_parallel.py),
   which parallelizes the process for hundreds of pages using `concurrent.futures`.
   Normalization steps remove invisible Unicode controls, clean spacing, and keep only
   lines with Devanagari, danda (`।`), or double danda (`॥`) marks.

3. **Train / Validation Splits**:

   * Pages 22 – 350  → training corpus (`train.txt`)
   * Pages 351 – 380  → validation corpus (`valid.txt`)

   Total clean text ≈ 500 000 characters for training.

---

### ⚙️ Tokenizer Training

`train_bpe_devanagari.py` builds a BPE model with the 🤗 `tokenizers` library:

```python
tokenizer.train(files=["train.txt"], vocab_size=8000, min_frequency=2)
```

* Vocabulary size = 8 000
* Compression ratio = 3.776× on validation set
* Character coverage ≈ 99.9 %

Sample tokenization:

```
गुरु पद रज मृदु मंजुल अंजन । नयन अमिअ दूग दोष बिभंजन ॥
→ ['गुरु', 'पद', 'रज', 'मृदु', 'मंजुल', 'अं', 'जन', '।', 'नयन', 'अमिअ', 'दू', 'ग', 'दोष', 'बि', 'भंजन', '॥']
```

---

### 🧪 Evaluation

[`eval_tokenizer.py`](eval_tokenizer.py) checks held-out pages to compute:

* **Compression ratio (chars/token)**
* **UNK rate**
* **Token counts**

Typical results:

```
[EVAL:Awadhi-ish]  CR=3.347 | UNK=0.00% | tokens=652
[EVAL:Hindi gloss] CR=4.076 | UNK=0.00% | tokens=8179
```



---

### 🚀 Deployment

The trained tokenizer (`tokenizer.json`) is hosted at:
🔗 **[justpradeep/awadhi-hindi-bpe-8k](https://huggingface.co/justpradeep/awadhi-hindi-bpe-8k/tree/main)**

A demo Gradio app is available here:
🔗 **[justpradeep/awadhi-hindi-bpe-demo](https://huggingface.co/spaces/justpradeep/hindi-awadhi-bpe-tokenizer)**
It loads the tokenizer via `hf_hub_download` and visualizes tokens and IDs.

---

### 🧵 Pipeline Overview

| Stage              | Script                                                                    | Description                                                              |
| ------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| PDF Extraction     | [`extract.py`](extract.py) / [`extract_parallel.py`](extract_parallel.py) | Hybrid text + OCR extraction with Devanagari detection and normalization |
| Tokenizer Training | [`train_bpe_devanagari.py`](train_bpe_devanagari.py)                      | Builds and saves an 8 k vocab BPE tokenizer                              |
| Evaluation         | [`eval_tokenizer.py`](eval_tokenizer.py)                                  | Computes compression ratio and UNK rate on unseen pages                  |
| Deployment         | [`app.py`](app.py)                                                        | Gradio UI for interactive testing on Hugging Face Spaces                 |

---

### 🧩 Tech Stack

* Python 3.10
* PyMuPDF / Tesseract / Pillow
* 🤗 tokenizers 0.20
* huggingface_hub 0.24
* Gradio 5.x

---
