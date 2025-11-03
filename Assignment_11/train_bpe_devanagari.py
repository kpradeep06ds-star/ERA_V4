import os, random, unicodedata, re
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence as NormalizerSequence, NFKC



def dev_initial_alphabet():
    chars = [chr(cp) for cp in range(0x0900, 0x0980)]  # Devanagari block
    chars += list(" \t\n\r.,;:?!()[]{}\"'“”‘’—–-_/|+*=॥।०१२३४५६७८९0123456789")
    # de-dup while preserving order
    seen, out = set(), []
    for c in chars:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def train_bpe(train_path: str, vocab_size=12000, min_freq=2):  # ← 12k
    tok = Tokenizer(BPE(unk_token="[UNK]", byte_fallback=True))  # ← fallback
    tok.normalizer = NormalizerSequence([NFKC()])
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["[PAD]","[UNK]","[BOS]","[EOS]"],
        initial_alphabet=dev_initial_alphabet()                 # ← seed chars
    )
    tok.train([train_path], trainer=trainer)
    return tok


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for ch in ["\u200c", "\u200d", "\u00ad", "\ufeff", "\u2060"]:
        text = text.replace(ch, "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def write_split(corpus_path: str, out_dir: str, valid_ratio=0.10):
    txt = Path(corpus_path).read_text(encoding="utf-8")
    txt = normalize(txt)
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    random.shuffle(lines)
    n_valid = max(1, int(len(lines) * valid_ratio))
    valid = lines[:n_valid]
    train = lines[n_valid:]
    out = Path(out_dir); out.mkdir(exist_ok=True)
    (out/"train.txt").write_text("\n".join(train), encoding="utf-8")
    (out/"valid.txt").write_text("\n".join(valid), encoding="utf-8")
    return str(out/"train.txt"), str(out/"valid.txt")


def compression_ratio(tok: Tokenizer, text: str) -> float:
    chars = len(text.replace("\n", " "))
    token_count = 0
    for line in text.splitlines():
        if line.strip():
            token_count += len(tok.encode(line).ids)
    return (chars / token_count) if token_count else 0.0

if __name__ == "__main__":
    corpus = "bpe_awadhi_hindi/sample_22_350.txt"
    out_dir = "bpe_awadhi_hindi/run1"
    train_path, valid_path = write_split(corpus, out_dir, valid_ratio=0.10)

    print("[CHECK] train chars:", len(Path(train_path).read_text(encoding="utf-8")))
    print("[CHECK] valid chars:", len(Path(valid_path).read_text(encoding="utf-8")))

    vocab_size = 8000  # >= 5000 requirement
    tok = train_bpe(train_path, vocab_size=vocab_size, min_freq=2)
    Path(out_dir, "tokenizer.json").write_text(tok.to_str(), encoding="utf-8")  # single file is enough

    valid_text = Path(valid_path).read_text(encoding="utf-8")
    cr = compression_ratio(tok, valid_text)
    print(f"[RESULT] Vocab size: {vocab_size}")
    print(f"[RESULT] Compression ratio (chars/token) on held-out: {cr:.3f}")

    # Tiny sample
    sample = "गुरु पद रज मृदु मंजुल अंजन । नयन अमिअ दूग दोष बिभंजन ॥"
    enc = tok.encode(sample)
    print("SAMPLE:", sample)
    print("TOKENS:", enc.tokens)
    print("N_TOKENS:", len(enc.ids))
