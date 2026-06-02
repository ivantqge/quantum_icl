"""LoRA SFT on Qwen-instruct using the simulator-verified circuit dataset.

A 7B model in bf16 with LoRA fits comfortably on a 40 GB A100 (and easily on
80 GB). Saves the LoRA adapter only; load it at eval time with LocalHFLLM
adapter_path.

Usage:
    python -m quantum_icl.sft.train \
        --model Qwen/Qwen2.5-7B-Instruct \
        --data /pscratch/sd/i/ivang/qicl_sft/data \
        --out /pscratch/sd/i/ivang/qicl_sft/runs/qwen7b_v1
"""

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", required=True, help="directory containing train/val jsonl")
    ap.add_argument("--out", required=True, help="output directory for adapter + logs")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--assistant-only-loss", action="store_true", default=True,
                    help="train only on assistant tokens (mask prompt loss)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    print(f"[load] model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = load_dataset(
        "json", data_files=os.path.join(args.data, "train.jsonl"), split="train",
    )
    val_ds = load_dataset(
        "json", data_files=os.path.join(args.data, "val.jsonl"), split="train",
    )
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    # With assistant_only_loss=True the trainer applies the chat template
    # itself and masks non-assistant tokens; we just need the `messages` field.
    fmt = None
    if not args.assistant_only_loss:
        def fmt(example):
            return tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False,
            )

    cfg_kwargs = dict(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        max_length=args.max_seq_len,
        report_to="none",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
    )
    # Only train loss on the assistant turn -- otherwise the shared system+user
    # boilerplate dominates the loss and the model memorizes a fixed output.
    if args.assistant_only_loss:
        cfg_kwargs["assistant_only_loss"] = True
    cfg = SFTConfig(**cfg_kwargs)

    trainer_kwargs = dict(
        model=model, args=cfg, processing_class=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds,
    )
    if fmt is not None:
        trainer_kwargs["formatting_func"] = fmt
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    adapter_dir = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[save] adapter -> {adapter_dir}")


if __name__ == "__main__":
    main()
