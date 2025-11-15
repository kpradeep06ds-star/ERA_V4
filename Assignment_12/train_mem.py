# train_mem.py
# Train-only (or optional tiny val) GPT-2 small (124M) on input.txt and save resumable checkpoints + logs.
import os, math, time, json, argparse, random, logging, csv
from dataclasses import asdict
import tiktoken
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
torch.set_float32_matmul_precision("high")  # TF32 -> big GEMM speedup on Ampere+


# import your model as-is
from transformer import GPT, GPTConfig  

# ---------------------------
# Small dataset loader
# ---------------------------
class TextChunks(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size
        # number of starting positions that have a full block+next token
        self.n = self.tokens.size(0) - block_size - 1

    def __len__(self): return max(self.n, 0)

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + 1 + self.block_size]
        return x, y

def make_indices(num_tokens, block_size, stride=None):
    if stride is None:
        stride = block_size  # non-overlapping
    starts = list(range(0, num_tokens - block_size, stride))
    return starts

def build_dataloaders(input_path, tokenizer, block_size, val_ratio, batch_size):
    text = open(input_path, "r", encoding="utf-8").read()
    tokens = tokenizer.encode(text)
    n = len(tokens)
    split = int(n * (1 - val_ratio))
    train_ids, val_ids = tokens[:split], tokens[split:]

    stride = block_size  # or block_size // 2 for overlap
    train_starts = make_indices(len(train_ids), block_size, stride)
    val_starts = make_indices(len(val_ids), block_size, stride)

    # Create datasets
    train_dataset = TextDataset(train_ids, train_starts, block_size)
    val_dataset = TextDataset(val_ids, val_starts, block_size)

    # ✅ Use the batch_size argument, not args.batch_size
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    total_tokens = len(tokens)
    return train_dl, val_dl, total_tokens


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, starts, block_size):
        self.tokens = tokens
        self.starts = starts
        self.block_size = block_size

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = self.starts[idx]
        x = torch.tensor(self.tokens[i:i+self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[i+1:i+self.block_size+1], dtype=torch.long)
        return x, y

# ---------------------------
# Logging helpers (append mode)
# ---------------------------
def setup_logging(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)

    # CSV history (append)
    csv_path = os.path.join(out_dir, "history.csv")
    new_file = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "step","epoch","split","loss","ppl","lr","tok_per_s"
    ])
    if new_file:
        csv_writer.writeheader()

    return logger, csv_file, csv_writer

# ---------------------------
# Training step
# ---------------------------
def step(model, batch, device, scaler, grad_accum):
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device=="cuda")):
        logits, loss = model(x, y)  # assumes your GPT forward returns (logits, loss)
        loss = loss / grad_accum
    scaler.scale(loss).backward()
    return loss.detach().item() * grad_accum  # return unscaled loss for logging

@torch.no_grad()
def evaluate(model, dl, device, max_batches=None):
    model.eval()
    losses, n = 0.0, 0
    with torch.no_grad():
        for bidx, (x, y) in enumerate(dl):
            if max_batches is not None and bidx >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses += loss.item()
            n += 1
    return losses / max(1, n)

# ---------------------------
# Save / Load
# ---------------------------
def save_ckpt(path, model, optim, scaler, step, epoch, cfg_obj, extra=None):
    obj = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": asdict(cfg_obj) if hasattr(cfg_obj, "__dict__") else cfg_obj.__dict__,
        "extra": extra or {},
    }
    torch.save(obj, path)

def save_hf_bundle(out_dir, model, cfg_obj):
    # For HF model repo (simple bundle)
    torch.save({
        "model": model.state_dict(),
        "config": asdict(cfg_obj) if hasattr(cfg_obj, "__dict__") else cfg_obj.__dict__,
    }, os.path.join(out_dir, "model.pt"))

def load_ckpt(path, model, optim, scaler, map_location="cpu"):
    obj = torch.load(path, map_location=map_location)
    model.load_state_dict(obj["model"], strict=True)
    if optim is not None and "optim" in obj:
        optim.load_state_dict(obj["optim"])
    if scaler is not None and "scaler" in obj:
        scaler.load_state_dict(obj["scaler"])
    step = obj.get("step", 0); epoch = obj.get("epoch", 0)
    return step, epoch, obj

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    # data & IO
    parser.add_argument("--input", type=str, default="input.txt")
    parser.add_argument("--out",   type=str, default="out_124m")
    parser.add_argument("--append_name", type=str, default="")
    # model size (GPT-2 small defaults)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head",  type=int, default=12)
    parser.add_argument("--n_embd",  type=int, default=768)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--dropout", type=float, default=0.0)  # keep 0.0 for pure memorization
    # train hyperparams
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=0)  # 0 => ignore, use max_steps
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--val_ratio", type=float, default=0.0)  # 0.0 => no val
    parser.add_argument("--eval_every", type=int, default=1000)  # used only if val_ratio>0
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num_workers", type=int, default=0)
    # memory / speed
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--eval_batches", type=int, default=200)

    args = parser.parse_args()

    if args.append_name:
        args.out = f"{args.out}_{args.append_name}"

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # logging
    logger, csv_file, csv_writer = setup_logging(args.out)
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    # tokenizer

    enc = tiktoken.get_encoding("gpt2")   # 50257 vocab, GPT-2 bytes BPE
    args.vocab_size = enc.n_vocab  # 50257

    # data
    train_dl, val_dl, total_tokens = build_dataloaders(
        args.input, enc, args.block_size, args.val_ratio, args.batch_size
    )

    logger.info(f"Total tokens: {total_tokens:,}. Train batches/epoch: {len(train_dl)}"
                + (f", Val batches: {len(val_dl)}" if val_dl else ""))

    # model
    cfg = GPTConfig(
        block_size=args.block_size, vocab_size=args.vocab_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd
    )
    model = GPT(cfg).to(device)

    # optional grad ckpt flag per block (best-effort)
    if args.grad_ckpt:
        for m in model.modules():
            if hasattr(m, "grad_ckpt") and isinstance(getattr(m, "grad_ckpt"), (bool, int)):
                setattr(m, "grad_ckpt", True)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params/1e6:.2f}M")

    # optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1,args.beta2),
                              weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device=="cuda"))

    # simple cosine schedule w/ warmup
    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step+1)/max(1,args.warmup_steps)
        progress = (step - args.warmup_steps)/max(1, (args.max_steps - args.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * args.lr * (1.0 + math.cos(math.pi * progress))

    # resume if last.pt exists
    last_ckpt = os.path.join(args.out, "last.pt")
    step0, epoch0 = 0, 0
    if os.path.exists(last_ckpt):
        step0, epoch0, _obj = load_ckpt(last_ckpt, model, optim, scaler, map_location=device)
        logger.info(f"Resumed from step={step0}, epoch={epoch0}")

    if args.compile and device=="cuda":
        try:
            model = torch.compile(model)
            logger.info("torch.compile: enabled")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    # training loop
    model.train()
    global_step = step0
    start_time = time.time()
    best_val = float("inf")
    grad_accum = max(1, args.accum_steps)

    # we can cycle over epochs until max_steps
    epoch = epoch0
    while True:
        epoch += 1
        tokens_this_epoch = 0
        loss_running = 0.0
        t0 = time.time()

        optim.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_dl):
            # set LR by schedule
            for pg in optim.param_groups:
                pg["lr"] = lr_at(global_step)

            loss_val = step(model, batch, device, scaler, grad_accum)
            loss_running += loss_val
            tokens_this_epoch += args.batch_size * args.block_size

            if (it + 1) % grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            # log & save
            if (global_step + 1) % 100 == 0:
                tok_per_s = tokens_this_epoch / max(1e-6, (time.time() - t0))
                avg_loss = loss_running / (it + 1)
                logger.info(f"{global_step+1} | loss {avg_loss:.4f} | lr {optim.param_groups[0]['lr']:.2e} | tok/s {tok_per_s:,.0f}")
                csv_writer.writerow({
                    "step": global_step+1, "epoch": epoch, "split":"train",
                    "loss": f"{avg_loss:.6f}",
                    "ppl": f"{math.exp(avg_loss):.6f}",
                    "lr": f"{optim.param_groups[0]['lr']:.6e}",
                    "tok_per_s": f"{tok_per_s:.2f}",
                }); csv_file.flush()

            if args.save_every and (global_step + 1) % args.save_every == 0:
                save_ckpt(os.path.join(args.out, "last.pt"), model, optim, scaler, global_step+1, epoch, cfg)
                save_hf_bundle(args.out, model, cfg)  # for quick HF upload
                logger.info("Saved checkpoint: last.pt & model.pt")

            if args.val_ratio > 0.0 and args.eval_every and (global_step + 1) % args.eval_every == 0:
                vl = evaluate(model, val_dl, device, max_batches=args.eval_batches)
                if vl is not None:
                    logger.info(f"[eval] step {global_step+1} | val {vl:.4f} | ppl {math.exp(vl):.3f}")
                    csv_writer.writerow({
                        "step": global_step+1, "epoch": epoch, "split":"val",
                        "loss": f"{vl:.6f}", "ppl": f"{math.exp(vl):.6f}",
                        "lr": f"{optim.param_groups[0]['lr']:.6e}",
                        "tok_per_s": ""
                    }); csv_file.flush()
                    if vl < best_val:
                        best_val = vl
                        save_ckpt(os.path.join(args.out, "best.pt"), model, optim, scaler, global_step+1, epoch, cfg,
                                  extra={"best_val": best_val})
                        logger.info(f"↳ saved best (val {best_val:.4f})")

            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                break

        # end epoch
        elapsed = time.time() - t0
        avg_loss = loss_running / max(1, len(train_dl))
        logger.info(f"Epoch {epoch} done | loss {avg_loss:.4f} | time {elapsed:.1f}s")

        if args.epochs and epoch >= args.epochs:
            break
        if args.max_steps and global_step >= args.max_steps:
            break

    # final save
    save_ckpt(os.path.join(args.out, "last.pt"), model, optim, scaler, global_step, epoch, cfg)
    save_hf_bundle(args.out, model, cfg)
    logger.info(f"Training complete in {(time.time()-start_time)/3600:.2f}h. Final step={global_step}.")

    csv_file.close()

if __name__ == "__main__":
    main()
