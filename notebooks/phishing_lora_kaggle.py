# %% [markdown]
# # Sentinel, Module 3: fine-tune DeBERTa-v3-small with LoRA for phishing detection
#
# **Before you press Run All** (right-hand panel, *Session options*):
# 1. **Accelerator:** GPU T4 x2 (or GPU P100).
# 2. **Internet:** On (needed to download the base model and pinned libraries).
# 3. **Input:** add your dataset *sentinel-phishing-emails* (Add Input -> Your Work -> Datasets).
#
# Runtime is about 30-60 minutes. When it finishes, download
# `phishing_lora_output.zip` from the **Output** section (right-hand panel) and put it in
# `C:\Users\DELL\sentinel\models\phishing\` on your laptop.

# %%
import os
import subprocess
import sys

# SENTINEL_SMOKE=1 runs a tiny CPU version of this notebook locally, as a test.
SMOKE = os.environ.get("SENTINEL_SMOKE") == "1"
PINNED = ["transformers==5.17.0", "peft==0.21.1", "accelerate==1.15.0", "sentencepiece", "protobuf"]
if not SMOKE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINNED], check=True)
    # Kaggle preinstalls torchao 0.10; PEFT refuses any torchao older than 0.16 even when
    # it is not used. Nothing here needs torchao, so remove it.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

# %%
import glob
import json
import platform
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CONFIG = {
    "model_name": "microsoft/deberta-v3-small",
    "max_len": 256,
    "epochs": 2,
    "lr": 3e-4,
    "batch_size": 32,
    "eval_batch_size": 64,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["query_proj", "key_proj", "value_proj"],
    "seed": 42,
    # True trains every weight instead of LoRA adapters (report's optional comparison).
    # Output is then ~550 MB instead of ~10 MB.
    "full_finetune": False,
}
if SMOKE:
    CONFIG.update(max_len=64, epochs=1, batch_size=8, eval_batch_size=16)
print(json.dumps(CONFIG, indent=2))


def find_data() -> Path:
    if os.environ.get("SENTINEL_DATA_DIR"):
        return Path(os.environ["SENTINEL_DATA_DIR"])
    hits = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
    if not hits:
        raise FileNotFoundError(
            "Dataset not found. Add 'sentinel-phishing-emails' as input: "
            "right-hand panel -> Add Input -> Your Work -> Datasets."
        )
    return Path(hits[0]).parent


DATA = find_data()
OUT = Path(os.environ.get("SENTINEL_OUT_DIR", "/kaggle/working")) / "phishing_lora_output"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

splits = {s: pd.read_parquet(DATA / f"{s}.parquet") for s in ("train", "val", "test")}
if SMOKE:
    splits = {
        s: pd.concat(
            [
                df[df["label"] == c].sample(min((df["label"] == c).sum(), n // 2), random_state=0)
                for c in (0, 1)
            ]
        )
        for (s, df), n in zip(splits.items(), (200, 100, 100), strict=True)
    }
for s, df in splits.items():
    print(f"{s:5s} {len(df):7,} emails, {df['label'].mean():.1%} malicious")
print(
    "CUDA:",
    torch.cuda.is_available(),
    [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
)

# %%
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

set_seed(CONFIG["seed"])
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])


class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.enc = tokenizer(df["text"].tolist(), truncation=True, max_length=CONFIG["max_len"])
        self.labels = df["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


t0 = time.time()
datasets = {s: EmailDataset(df) for s, df in splits.items()}
lengths = [len(x) for x in datasets["train"].enc["input_ids"]]
print(
    f"tokenised in {time.time() - t0:.0f}s; {np.mean(np.array(lengths) >= CONFIG['max_len']):.0%} "
    f"of training emails fill the {CONFIG['max_len']}-token window"
)

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    # The checkpoint is stored in float16 and transformers 5 keeps the stored dtype by
    # default. Mixed-precision training needs float32 master weights (fp16 weights fail
    # with "Attempting to unscale FP16 gradients"), and fp16 on CPU is ~100x slower.
    dtype=torch.float32,
    num_labels=2,
    id2label={0: "legitimate", 1: "malicious"},
    label2id={"legitimate": 0, "malicious": 1},
)
if not CONFIG["full_finetune"]:
    from peft import LoraConfig, TaskType, get_peft_model

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["target_modules"],
        # The classification head and pooler start untrained, so they train in full.
        modules_to_save=["classifier", "pooler"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

# %%
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def softmax_pos(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    out = {"precision": float(p), "recall": float(r), "f1": float(f1)}
    if 0 < y.mean() < 1:
        out["pr_auc"] = float(average_precision_score(y, prob))
        out["roc_auc"] = float(roc_auc_score(y, prob))
    return out


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    return binary_metrics(np.asarray(labels), softmax_pos(np.asarray(logits)))


args = TrainingArguments(
    output_dir=str(OUT / "checkpoints"),
    num_train_epochs=CONFIG["epochs"],
    learning_rate=CONFIG["lr"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["eval_batch_size"],
    weight_decay=CONFIG["weight_decay"],
    warmup_steps=CONFIG["warmup_ratio"],  # transformers 5: a float < 1 is a ratio of total steps
    lr_scheduler_type="linear",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    logging_steps=10 if SMOKE else 100,
    report_to="none",
    seed=CONFIG["seed"],
    dataloader_num_workers=0 if SMOKE else 2,
    use_cpu=not torch.cuda.is_available(),
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["val"],
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)
t0 = time.time()
trainer.train()
train_minutes = (time.time() - t0) / 60
print(f"training took {train_minutes:.1f} min")

# %%
results = {}
for s in ("val", "test"):
    pred = trainer.predict(datasets[s])
    df = splits[s][["id", "label", "kind", "source"]].copy()
    df["prob"] = softmax_pos(pred.predictions)
    df.to_csv(OUT / f"{s}_predictions.csv", index=False)
    results[s] = {"overall": binary_metrics(df["label"].to_numpy(), df["prob"].to_numpy())}
    flagged = df["prob"] >= 0.5
    results[s]["flagged_by_kind"] = {
        k: float(flagged[df["kind"] == k].mean()) for k in sorted(df["kind"].unique())
    }
    results[s]["f1_by_source"] = {
        src: binary_metrics(g["label"].to_numpy(), g["prob"].to_numpy())["f1"]
        for src, g in df.groupby("source")
        if g["label"].nunique() == 2
    }

print(json.dumps(results["test"], indent=2))
print("\nShare flagged as malicious on TEST, by kind (legitimate = false-alarm rate):")
for k, v in results["test"]["flagged_by_kind"].items():
    print(f"  {k:11s} {v:.2%}")

# %%
save_dir = OUT / "adapter" if not CONFIG["full_finetune"] else OUT / "model"
trainer.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
run_info = {
    "config": CONFIG,
    "results": results,
    "train_minutes": train_minutes,
    "log_history": trainer.state.log_history,
    "environment": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "packages": PINNED,
    },
    "smoke_test": SMOKE,
}
(OUT / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))
shutil.rmtree(OUT / "checkpoints", ignore_errors=True)

zip_path = OUT.parent / "phishing_lora_output.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in OUT.rglob("*"):
        if f.is_file():
            zf.write(f, f.relative_to(OUT.parent))
print(
    f"\nDONE. Download {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB) "
    "from the Output panel."
)
