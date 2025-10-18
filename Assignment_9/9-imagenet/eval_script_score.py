import os, glob, torch, argparse
from torch.utils.data import DataLoader
from torch.amp import autocast
from torchvision.models import resnet50
from torchvision import transforms
from torch.optim.swa_utils import AveragedModel
import webdataset as wds
from pathlib import Path

# --- CUDA allocator tuning to reduce fragmentation on long evals / Windows ---
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128"
)

# --- Top-level transforms & helpers (Windows spawn needs picklable callables) ---
VAL_TFMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def cast_label(y):
    if isinstance(y, (bytes, bytearray)):
        y = y.decode("utf-8")
    if hasattr(y, "item"):
        y = y.item()
    return int(y)

def val_transform(img):
    return VAL_TFMS(img)

def strip_module_prefix(sd_dict):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd_dict.items()}

def load_model_from_ckpt(ckpt_path, device):
    # Build base model
    model = resnet50(weights=None, num_classes=1000).to(device).eval()

    # Load checkpoint dict (supports raw dict or {"state_dict": ...} or {"swa_state_dict": ...})
    state = torch.load(ckpt_path, map_location="cpu")  # keep weights_only=False for broad compat
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict) and "swa_state_dict" in state:
        sd = state["swa_state_dict"]
    else:
        sd = state

    # Handle SWA AveragedModel (presence of "n_averaged") + possible DP/DDP "module." prefix
    if "n_averaged" in sd:
        tmp_base = resnet50(weights=None, num_classes=1000)
        swa = AveragedModel(tmp_base)
        if any(k.startswith("module.") for k in sd.keys()):
            sd = strip_module_prefix(sd)
        swa.load_state_dict(sd, strict=False)  # ignore tiny buffer diffs
        model.load_state_dict(swa.module.state_dict(), strict=True)
    else:
        if any(k.startswith("module.") for k in sd.keys()):
            sd = strip_module_prefix(sd)
        model.load_state_dict(sd, strict=True)

    return model

def build_val_loader(wds_dir, batch_size=256, num_workers=0):
    # 1) Find shards
    pat = os.path.join(wds_dir, "imagenet1k-validation-*.tar")
    raw_shards = sorted(glob.glob(pat))
    assert raw_shards, f"No val shards found in {wds_dir}"

    # 2) Convert to file://-style URIs that gopen understands on Windows
    shards = ["file:" + Path(p).as_posix() for p in raw_shards]

    ds = (
        wds.WebDataset(shards, shardshuffle=False)
        .decode("pil")
        .to_tuple("jpg;png;jpeg", "cls")
        .map_tuple(val_transform, cast_label)
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
    )

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.item())
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds_dir", required=True, help="Local dir with WDS val shards")
    ap.add_argument("--ckpt", required=True, help="Path to model .pth (supports SWA)")
    ap.add_argument("--batch_size", type=int, default=128)  # safer default for 4090 eval
    ap.add_argument("--workers", type=int, default=0)       # Windows-friendly default
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--channels_last", action="store_true", default=True,
                    help="Use NHWC (channels_last) memory format for lower activation memory")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load model AFTER args exist
    model = load_model_from_ckpt(args.ckpt, device)
    model.eval()

    # Optional channels_last (helps memory on conv nets)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Data
    val_loader = build_val_loader(
        args.wds_dir, batch_size=args.batch_size, num_workers=args.workers
    )

    # Eval
    total, top1_sum, top5_sum, loss_sum = 0, 0.0, 0.0, 0.0
    criterion = torch.nn.CrossEntropyLoss().to(device)
    dtype = "cuda" if device.type == "cuda" else "cpu"

    # Stronger than no_grad: blocks autograd graph allocations entirely
    with torch.inference_mode():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            if args.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            target = target.to(device, non_blocking=True)
            with autocast(enabled=args.amp, device_type=dtype):
                out = model(images)
                loss = criterion(out, target)
            c1, c5 = accuracy(out, target, topk=(1, 5))
            bsz = images.size(0)
            total += bsz
            top1_sum += c1
            top5_sum += c5
            loss_sum += loss.item() * bsz
            if (i + 1) % 50 == 0:
                print(
                    f"[{i+1}] processed={total}  "
                    f"acc1={(top1_sum/total)*100:.2f}  acc5={(top5_sum/total)*100:.2f}"
                )

    print("\n=== FINAL ===")
    print(
        f"val_loss: {loss_sum/total:.4f}  "
        f"acc@1: {(top1_sum/total)*100:.2f}  acc@5: {(top5_sum/total)*100:.2f}"
    )

if __name__ == "__main__":
    main()


## Call
'''
python .\eval_script_score.py `  --wds_dir "D:\ERV_V4\ImageNet_v4-aoc\ImageNet_v4-aoc\final_codes_checkpoints\valid" `  --ckpt "D:\ERV_V4\ImageNet_v4-aoc\ImageNet_v4-aoc\final_codes_checkpoints\outputs\imagenet1k_resnet50\model_swa.pth" 

'''