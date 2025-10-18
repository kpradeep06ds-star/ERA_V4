# config.py
config = {
    # --- paths / data format ---
    "data_format": "wds",                      # "wds" for webdataset shards (was folder)
    "data_dir": "/mnt/data/imagenet-wds",      # directory where *.tar live
    "train_pattern": "imagenet1k-train-*.tar",
    "val_pattern": "imagenet1k-validation-*.tar",

    # --- core training (keep your choices) ---
    "num_classes": 1000,
    "batch_size": 256,                         # tune by GPU memory (e.g. 64 on 8GB)
    "num_workers": 6,
    "pin_memory": True,
    "prefetch_factor": 4,

    "use_amp": True,
    "optimizer": "SGD",
    "base_lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 1e-4,

    "epochs": 90,
    "warmup_epochs": 5,
    "cosine_min_lr": 1e-6,

    "use_randaugment": True,
    "mixup_alpha": 0.2,
    "label_smoothing": 0.1,

    # --- logging / ckpts (add these to match your trainer) ---
    "checkpoint_dir": "./outputs/imagenet1k_resnet50/",
    "checkpoint_frequency": 5,
    "log_interval": 50,

    # --- extras you already use / optional ---
    "use_swa": True,
    "swa_last_epochs": 10,
    "swa_lr_mult": 0.5,
    "swa_anneal_epochs": 1,
    "grad_clip_value": None,

    # --------------------------------------------------------------
    "samples_per_epoch": 1281167,   # ImageNet-1k train
    "val_samples": 50000,           # ImageNet-1k val

    "wds_shardshuffle": 1000,       # quell the WebDataset warning
}
