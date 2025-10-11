config = {
    # --- dataset ---
    "data_dir": "./archive/",   # point this to your ImageNet-100 root (train/ and val/ inside)
    "num_classes": 100,               # <-- set 100 for IN-100; switch to 1000 for full ImageNet

    # --- core training ---
    "epochs": 60,                     # 60 is plenty for IN-100; use 100–120 for full 1K
    "batch_size": 128,
    "num_workers": 8,
    "pin_memory": True,
    "prefetch_factor": 4,

    # --- optimizer & LR schedule ---
    "optimizer": "sgd",
    "base_lr": 0.1,                   # scaled internally by (batch/256)
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,               # you'll sweep this (see Section 3)
    "cosine_min_lr": 1e-6,

    # --- regularization ---
    "label_smoothing": 0.1,
    "use_mixup": True,
    "mixup_alpha": 0.1,               # consider 0.1–0.2 for IN-100
    "use_randaugment": True,
    "random_erasing_p": 0.1,          # a bit helps on IN-100

    # --- SWA (late only) ---
    "use_swa": True,
    "swa_last_epochs": 8,
    "swa_anneal_epochs": 1,
    "swa_lr_mult": 0.5,

    # --- AMP / grad ---
    "use_amp": True,
    "grad_clip_value": 1.0,
    "gradient_accumulation_steps": 1,

    # --- ckpts / logging ---
    "checkpoint_dir": "./checkpoint",
    "checkpoint_frequency": 5,
    "early_stopping_patience": 0,
    "early_stopping_delta": 0.0,
    "log_interval": 50,
    "checkpoint_dir": "./checkpoint"
}
