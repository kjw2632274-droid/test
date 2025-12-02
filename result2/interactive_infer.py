"""Interactive inference script for the MultiHeadKoBART multitask model.

Usage example:

    C:/Users/sexim/AppData/Local/Programs/Python/Python314/python.exe interactive_infer.py \
        --ckpt_path checkpoint/checkpoint_143247.pt \
        --pretrained_model gogamza/kobart-base-v2 \
        --tokenizer_path gogamza/kobart-base-v2

During the session you can pick one of the supported tasks and enter
prompts to generate model outputs. Press Ctrl+C or enter an empty line
to exit.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import torch
from transformers import PreTrainedTokenizerFast

from new_multitask.model import MultiHeadKoBART, TASKS


DEFAULT_MAX_LENGTH = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive multitask inference")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--pretrained_model", type=str, default="gogamza/kobart-base-v2", help="Base pretrained model name or path")
    parser.add_argument("--tokenizer_path", type=str, default="gogamza/kobart-base-v2", help="Tokenizer name or path")
    parser.add_argument("--task", type=str, choices=TASKS, default=None, help="Lock to a single task (optional)")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature coefficient used inside the custom decoder loop")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty (>1 discourages repeats)")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2, help="Block repeated n-grams of this size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    parser.add_argument("--show_tokens", action="store_true", help="Print tokenized ids for debug")
    return parser


def format_input(task: str, user_text: str) -> str:
    text = user_text.strip()
    if not text:
        return text

    lowered = text.lower().lstrip()
    if lowered.startswith("[qa]") or lowered.startswith("[role_generation]") or lowered.startswith("[style_transfer]") or lowered.startswith("[dialogue_summarization]"):
        return text

    if task == "role_generation":
        return text

    return f"[{task}] {text}"


def load_model(args: argparse.Namespace) -> tuple[MultiHeadKoBART, PreTrainedTokenizerFast]:
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)

    print(f"Loading base model {args.pretrained_model}...")
    model = MultiHeadKoBART.from_pretrained(args.pretrained_model)

    print(f"Loading checkpoint from {args.ckpt_path}...")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)

    model.to(args.device)
    model.eval()
    return model, tokenizer


def generate(
    model: MultiHeadKoBART,
    tokenizer: PreTrainedTokenizerFast,
    task: str,
    prompt: str,
    args: argparse.Namespace,
) -> str:
    formatted = format_input(task, prompt)
    if not formatted:
        return ""

    encoded = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    if args.show_tokens:
        print("input_ids:", input_ids.tolist())

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_name=task,
            max_length=args.max_length,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def pick_task(fixed_task: Optional[str]) -> str:
    if fixed_task:
        return fixed_task

    prompt = "Select task (" + ", ".join(TASKS) + ") > "
    while True:
        choice = input(prompt).strip()
        if choice in TASKS:
            return choice
        print(f"Unknown task '{choice}'. Supported: {', '.join(TASKS)}")


def repl(model: MultiHeadKoBART, tokenizer: PreTrainedTokenizerFast, args: argparse.Namespace) -> None:
    print("\nInteractive session started. Enter an empty line to quit.")
    if args.task:
        print(f"Locked task: {args.task}")
    else:
        print(f"Available tasks: {', '.join(TASKS)}")

    try:
        while True:
            task = pick_task(args.task)
            user_text = input(f"[{task}] prompt > ")
            if not user_text.strip():
                print("Received empty input. Exiting.")
                break

            output = generate(model, tokenizer, task, user_text, args)
            print("--- Generated ---")
            print(output)
            print("-----------------\n")
    except KeyboardInterrupt:
        print("\nSession terminated by user.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model, tokenizer = load_model(args)
    repl(model, tokenizer, args)


if __name__ == "__main__":
    encoding = getattr(sys.stdout, "encoding", None)
    if encoding and encoding.lower() != "utf-8":
        # Ensure Unicode output is handled on Windows terminals.
        sys.stdout.reconfigure(encoding="utf-8")
    main()
