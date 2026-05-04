import argparse
import importlib
import inspect
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# Compatibility shim for older PyTorch builds where TRL expects FSDPModule.
try:
    import torch.distributed.fsdp as _fsdp

    if not hasattr(_fsdp, "FSDPModule") and hasattr(_fsdp, "FullyShardedDataParallel"):
        _fsdp.FSDPModule = _fsdp.FullyShardedDataParallel
except Exception:
    pass

# TRL 1.x imports is_liger_kernel_available from transformers.utils; older transformers
# (e.g. 4.44.x) define it only inside TRL. Patch the module before loading DPOTrainer.
try:
    import importlib

    import transformers as _transformers
    import transformers.utils as _transformers_utils

    # Avoid hasattr/getattr on lazy modules here: some builds raise non-AttributeError
    # during attribute resolution, which would skip the shim entirely.
    if "is_trackio_available" not in vars(_transformers):
        # Older/newer version skew: some TRL builds import this symbol from
        # top-level transformers, but many transformers releases do not expose it.
        def _is_trackio_available() -> bool:
            return False

        _transformers.is_trackio_available = _is_trackio_available

    if "is_liger_kernel_available" not in vars(_transformers_utils):
        _trl_iu = importlib.import_module("trl.import_utils")
        _transformers_utils.is_liger_kernel_available = _trl_iu.is_liger_kernel_available
except Exception:
    pass

_dpo_trainer_import_error = None
try:
    from trl import DPOTrainer
except Exception as e:
    DPOTrainer = None
    _dpo_trainer_import_error = e

try:
    from trl import DPOConfig  # newer TRL versions
except Exception:
    DPOConfig = None



def supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


def load_pairs(path: str):
    ds = load_dataset("json", data_files=path)["train"]
    cols = set(ds.column_names)
    if "prompt" not in cols and "context" in cols:
        # Preference-pair datasets may use `context`; normalize for TRL APIs.
        ds = ds.map(lambda ex: {"prompt": ex.get("context", "")})
        cols = set(ds.column_names)

    required = {"prompt", "chosen", "rejected"}
    if not required.issubset(cols):
        raise ValueError(
            "Dataset must include columns {'prompt'|'context', 'chosen', 'rejected'}; "
            f"got {ds.column_names}"
        )

    def _valid(example: Dict[str, Any]) -> bool:
        return bool(str(example.get("prompt", "")).strip()) and bool(str(example.get("chosen", "")).strip()) and bool(
            str(example.get("rejected", "")).strip()
        )

    ds = ds.filter(_valid)
    return ds


def build_training_args(output_dir: str, bf16_flag: bool):
    common_kwargs = dict(
        output_dir=output_dir,
        learning_rate=5e-6,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        bf16=bf16_flag,
    )

    if DPOConfig is not None:
        # TRL has changed DPOConfig field names across versions.
        # Pass only parameters supported by the installed version.
        dpo_kwargs = dict(
            **common_kwargs,
            max_length=1024,
            max_prompt_length=512,
            beta=0.1,
        )
        sig = inspect.signature(DPOConfig.__init__)
        supported = {k: v for k, v in dpo_kwargs.items() if k in sig.parameters}
        return DPOConfig(**supported)
    return TrainingArguments(**common_kwargs)


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    training_args,
    peft_config,
):
    if DPOTrainer is None:
        # Some TRL builds don't export DPOTrainer at the top level.
        trainer_cls = None
        import_paths = ["trl.trainer", "trl.trainer.dpo_trainer"]
        for module_name in import_paths:
            try:
                mod = importlib.import_module(module_name)
                trainer_cls = getattr(mod, "DPOTrainer", None)
                if trainer_cls is not None:
                    break
            except Exception:
                continue
        if trainer_cls is None:
            raise ImportError(
                "DPOTrainer import failed. Your `trl` install may be incompatible or partially broken. "
                f"Original import error: {_dpo_trainer_import_error!r}"
            )
    else:
        trainer_cls = DPOTrainer
    kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_dataset,
    }

    sig = inspect.signature(trainer_cls.__init__)
    if "beta" in sig.parameters and DPOConfig is None:
        kwargs["beta"] = 0.1
    if "max_length" in sig.parameters and DPOConfig is None:
        kwargs["max_length"] = 1024
    if "max_prompt_length" in sig.parameters and DPOConfig is None:
        kwargs["max_prompt_length"] = 512
    if "tokenizer" in sig.parameters:
        kwargs["tokenizer"] = tokenizer
    if "processing_class" in sig.parameters:
        kwargs["processing_class"] = tokenizer
    if "peft_config" in sig.parameters:
        kwargs["peft_config"] = peft_config
    return trainer_cls(**kwargs)


def extract_loss_curve(log_history: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for row in log_history:
        if "loss" not in row:
            continue
        step = int(row.get("step", len(out)))
        try:
            loss = float(row["loss"])
        except (TypeError, ValueError):
            continue
        out.append((step, loss))
    return out


def save_loss_curve(loss_points: List[Tuple[int, float]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "training_loss_curve.json")
    png_path = os.path.join(out_dir, "training_loss_curve.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{"step": s, "loss": l} for s, l in loss_points], f, ensure_ascii=False, indent=2)

    if not loss_points:
        return
    xs = [p[0] for p in loss_points]
    ys = [p[1] for p in loss_points]
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=1.5)
    plt.xlabel("step")
    plt.ylabel("training loss")
    plt.title("DPO training loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--pairs_path", default="data/dpo/training_pairs.jsonl")
    parser.add_argument("--output_dir", default="models/dpo_lora_llama8b")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bf16_flag = supports_bf16()

    train_dataset = load_pairs(args.pairs_path)
    print(f"[OK] Loaded {len(train_dataset)} DPO pairs")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if bf16_flag else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = build_training_args(args.output_dir, bf16_flag=bf16_flag)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        training_args=training_args,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    losses = extract_loss_curve(trainer.state.log_history)
    save_loss_curve(losses, args.output_dir)

    print(f"[OK] Train result: {train_result}")
    print(f"[OK] Saved LoRA adapter to {args.output_dir}")
    print(f"[OK] BF16 enabled: {bf16_flag}")
    print(f"[OK] Saved loss logs in {args.output_dir}")


if __name__ == "__main__":
    main()
