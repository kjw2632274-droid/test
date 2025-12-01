"""
Multi-task data loader for KoBART multi-head model.

- Loads 4 tasks from `data/processed`:
  - style_transfer: style_transfer_formal_informal_tagged.jsonl (formal -> informal only)
  - dialogue_summarization: dialogue_summarization.jsonl
  - role_generation: role_generation.jsonl
  - qa: qa_generation.jsonl

- Formats inputs with bracket task tags (aligned to new_multitask.model TASKS):
  - [style_transfer] {formal}
  - [dialogue_summarization] {dialogue}
  - [role_generation] 역할: {role} | 대화: {context}
  - [qa] 질문: {question}[ | 지문: {context}]

- Tokenizes, pads/truncates, and batches examples ensuring one-task-per-batch.

Usage:
	from transformers import PreTrainedTokenizerFast
	from new_multitask.dataset_loader import (
		MultiTaskDataset, TaskBatchSampler, collate_fn, build_dataloader
	)

	tok = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
	ds = MultiTaskDataset(data_dir='./data/processed', tokenizer=tok)
	sampler = TaskBatchSampler(ds, batch_size=8, shuffle=True)
	dl = build_dataloader(ds, sampler=sampler, batch_size=8)

CLI preview:
	python -m new_multitask.dataset_loader --tokenizer_path .\output\kobart_qa 
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Sampler


# Model task keys expected by new_multitask.model.MultiHeadKoBART
TASKS = (
	'qa',
	'role_generation',
	'style_transfer',
	'dialogue_summarization',
)


TASK_TO_FILE = {
	# Use only informal->formal (반말->존댓말) slice
	'style_transfer': 'style_transfer_informal_to_formal.jsonl',
	'dialogue_summarization': 'dialogue_summarization.jsonl',
	'role_generation': 'role_generation.jsonl',
	# Prefer qa_generation_no_doc.jsonl; fall back to korquad_train.jsonl, qa_train.jsonl, qa_generation.jsonl
	'qa': 'qa_generation_no_doc.jsonl',
}


class MultiTaskDataset(Dataset):
	"""Unified multi-task dataset with on-the-fly formatting and tokenization."""

	def __init__(
		self,
		data_dir: str,
		tokenizer=None,
		max_input_length: int = 256,
		max_target_length: int = 128,
		include_tasks: Optional[List[str]] = None,
		task_sampling: str = 'balanced',  # currently informational
		drop_empty_target: bool = False,
		task_max_samples: Optional[Dict[str, int]] = None,
		task_skip_samples: Optional[Dict[str, int]] = None,
	) -> None:
		self.data_dir = str(data_dir)
		self.tokenizer = tokenizer
		self.max_input_length = max_input_length
		self.max_target_length = max_target_length
		self.task_sampling = task_sampling
		self.drop_empty_target = drop_empty_target
		self.task_max_samples = task_max_samples or {}
		self.task_skip_samples = task_skip_samples or {}

		if include_tasks is None:
			include_tasks = list(TASKS)
		self.include_tasks = tuple(include_tasks)

		# Load data per task
		self.data: Dict[str, List[Dict]] = {t: [] for t in self.include_tasks}
		root = Path(self.data_dir)

		for task in self.include_tasks:
			fname = TASK_TO_FILE.get(task)
			if not fname:
				continue
			fpath = root / fname

			# For QA, try fallback to legacy filenames if primary is missing
			tried_paths = []
			if task == 'qa' and not fpath.exists():
				fallback1 = root / 'korquad_train.jsonl'
				fallback2 = root / 'qa_train.jsonl'
				fallback3 = root / 'qa_generation.jsonl'
				tried_paths = [str(fpath), str(fallback1), str(fallback2), str(fallback3)]
				if fallback1.exists():
					fpath = fallback1
				elif fallback2.exists():
					fpath = fallback2
				elif fallback3.exists():
					fpath = fallback3

			if not fpath.exists():
				if tried_paths:
					print(f"Warning: none of QA files found: {tried_paths}. Skipping task 'qa'.")
				else:
					print(f"Warning: {fpath} not found. Skipping task '{task}'.")
				continue

			# Load with optional max_samples limit and skip_samples offset
			max_samples = self.task_max_samples.get(task, None)
			skip_samples = self.task_skip_samples.get(task, None)
			self.data[task] = self._load_jsonl(fpath, max_samples=max_samples, skip_samples=skip_samples)
			
			if skip_samples:
				print(f"Loaded {len(self.data[task])} samples for task '{task}' from {fpath.name} (skipped first {skip_samples})")
			elif max_samples and max_samples < len(self.data[task]):
				print(f"Loaded {len(self.data[task])} samples for task '{task}' from {fpath.name} (limited to {max_samples})")
			else:
				print(f"Loaded {len(self.data[task])} samples for task '{task}' from {fpath.name}")

		# Flatten examples with task key
		self.examples: List[Dict] = []
		for task, items in self.data.items():
			for it in items:
				if self.drop_empty_target:
					src, tgt = self._format(task, it)
					if not str(tgt).strip():
						continue
				self.examples.append({'task': task, 'data': it})

		random.shuffle(self.examples)
		self._print_distribution()

	# ---- IO helpers ----
	def _load_jsonl(self, path: Path, max_samples: Optional[int] = None, skip_samples: Optional[int] = None) -> List[Dict]:
		out: List[Dict] = []
		# Use utf-8-sig to handle BOM if present
		with open(path, 'r', encoding='utf-8-sig') as f:
			for i, line in enumerate(f):
				# Skip initial samples if specified
				if skip_samples and i < skip_samples:
					continue
				if max_samples and len(out) >= max_samples:
					break
				line = line.strip()
				if not line:
					continue
				out.append(json.loads(line))
		return out

	def _print_distribution(self) -> None:
		total = len(self.examples)
		if total == 0:
			print("No examples loaded.")
			return
		print("\nTask distribution:")
		counts = {}
		for ex in self.examples:
			t = ex['task']
			counts[t] = counts.get(t, 0) + 1
		for k in sorted(counts):
			pct = 100.0 * counts[k] / total
			print(f"  {k}: {counts[k]} ({pct:.1f}%)")

	# ---- Formatting ----
	def _fmt_style_transfer(self, rec: Dict) -> Tuple[str, str]:
		# Expect pre-tagged 'formal' like: "[style_transfer] ..."; avoid double-tagging
		src = str(rec.get('formal') or rec.get('input') or '')
		if not src.lstrip().startswith('[style_transfer]'):
			src = f"[style_transfer] {src}"
		tgt = str(rec.get('casual') or rec.get('target') or rec.get('output') or '')
		return src, tgt

	def _fmt_dialogue_summ(self, rec: Dict) -> Tuple[str, str]:
		dialogue = str(rec.get('dialogue') or rec.get('input') or '')
		# Prevent double label: only add if not already present
		if dialogue.lstrip().startswith('[dialogue_summarization]'):
			src = dialogue
		else:
			src = f"[dialogue_summarization] {dialogue}"

		# Try multiple keys for summary/target
		summary = rec.get('summary')
		if summary is None or not str(summary).strip():
			summary = rec.get('output')
		if summary is None or not str(summary).strip():
			summary = rec.get('target')
		if summary is None:
			summary = ''
		return src, str(summary)

	def _fmt_role_gen(self, rec: Dict) -> Tuple[str, str]:
		# Use input/target fields as-is (turn-by-turn dialogue)
		src = str(rec.get('input') or '')
		tgt = str(rec.get('target') or rec.get('output') or '')
		return src, tgt

	def _fmt_qa(self, rec: Dict) -> Tuple[str, str]:
		# Check if already formatted (korquad format: input/target with [qa] prefix)
		if 'input' in rec and 'target' in rec:
			src = str(rec.get('input') or '')
			tgt = str(rec.get('target') or '')
			return src, tgt
		
		# Legacy format: question/answer/context
		q = str(rec.get('question') or '')
		a = str(rec.get('answer') or rec.get('output') or '')
		ctx = str(rec.get('context') or '')
		# Prepend task label only; no "질문:" prefix
		if ctx:
			src = f"[qa] {q} | 지문: {ctx}"
		else:
			src = f"[qa] {q}"
		return src, a

	def _format(self, task: str, rec: Dict) -> Tuple[str, str]:
		if task == 'style_transfer':
			return self._fmt_style_transfer(rec)
		if task == 'dialogue_summarization':
			return self._fmt_dialogue_summ(rec)
		if task == 'role_generation':
			return self._fmt_role_gen(rec)
		if task == 'qa':
			return self._fmt_qa(rec)
		# Fallback
		src = str(rec.get('input') or '')
		tgt = str(rec.get('output') or '')
		return src, tgt

	# ---- Dataset protocol ----
	def __len__(self) -> int:  # noqa: D401
		return len(self.examples)

	def __getitem__(self, idx: int) -> Dict:
		b = self.examples[idx]
		task = b['task']
		rec = b['data']
		src, tgt = self._format(task, rec)

		if self.tokenizer is None:
			return {'task': task, 'input_text': src, 'target_text': tgt}

		enc = self.tokenizer(
			src,
			max_length=self.max_input_length,
			padding='max_length',
			truncation=True,
			return_tensors='pt',
		)
		dec = self.tokenizer(
			tgt,
			max_length=self.max_target_length,
			padding='max_length',
			truncation=True,
			return_tensors='pt',
		)

		return {
			'task': task,  # must match model task_name
			'input_ids': enc['input_ids'].squeeze(0),
			'attention_mask': enc['attention_mask'].squeeze(0),
			'labels': dec['input_ids'].squeeze(0),  # model loss ignores pad_token_id
		}


class TaskBatchSampler(Sampler[List[int]]):
	"""Groups indices so each batch contains a single task.

	Yields lists of indices of size==batch_size; only full batches are yielded.
	"""

	def __init__(self, dataset: MultiTaskDataset, batch_size: int, shuffle: bool = True) -> None:
		self.dataset = dataset
		self.batch_size = int(batch_size)
		self.shuffle = shuffle

		# group indices by task
		self.by_task: Dict[str, List[int]] = {t: [] for t in TASKS}
		for i, ex in enumerate(dataset.examples):
			self.by_task[ex['task']].append(i)

		if self.shuffle:
			for t in self.by_task:
				random.shuffle(self.by_task[t])

		# precompute batches
		self._batches: List[List[int]] = []
		for t, lst in self.by_task.items():
			for i in range(0, len(lst), self.batch_size):
				chunk = lst[i:i + self.batch_size]
				if len(chunk) == self.batch_size:
					self._batches.append(chunk)

		if self.shuffle:
			random.shuffle(self._batches)

	def __iter__(self):
		for b in self._batches:
			yield b

	def __len__(self) -> int:
		return len(self._batches)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
	"""Stacks tensors and carries the task name for the batch.

	Assumes single-task batches when used with TaskBatchSampler.
	"""
	if 'input_ids' not in batch[0]:
		# non-tokenizer preview mode
		return {
			'task': batch[0]['task'],
			'input_text': [b['input_text'] for b in batch],
			'target_text': [b['target_text'] for b in batch],
		}

	task = batch[0]['task']
	input_ids = torch.stack([b['input_ids'] for b in batch])
	attention_mask = torch.stack([b['attention_mask'] for b in batch])
	labels = torch.stack([b['labels'] for b in batch])
	return {
		'task': task,  # pass to model as task_name
		'input_ids': input_ids,
		'attention_mask': attention_mask,
		'labels': labels,
	}


def build_dataloader(
	dataset: MultiTaskDataset,
	batch_size: int = 8,
	shuffle: bool = True,
	num_workers: int = 0,
	sampler: Optional[Sampler] = None,
) -> DataLoader:
	if sampler is None:
		sampler = TaskBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
		# When batch sampler is provided, DataLoader's batch_size/shuffle are ignored
		return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)
	else:
		return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


if __name__ == '__main__':
	# Lightweight preview CLI
	import argparse
	try:
		from transformers import PreTrainedTokenizerFast  # type: ignore
	except Exception:
		PreTrainedTokenizerFast = None

	script_dir = Path(__file__).resolve().parent
	default_data = str((script_dir.parent / 'data' / 'processed').resolve())

	p = argparse.ArgumentParser(description='Preview new_multitask dataset formatting/tokenization')
	p.add_argument('--data_dir', type=str, default=default_data)
	p.add_argument('--tokenizer_path', type=str, default=None)
	p.add_argument('--max_input_length', type=int, default=256)
	p.add_argument('--max_target_length', type=int, default=128)
	p.add_argument('--batch_size', type=int, default=4)
	p.add_argument('--samples_per_task', type=int, default=2)
	p.add_argument('--seed', type=int, default=42)
	args = p.parse_args()

	print('=' * 60)
	print('Preview new_multitask.dataset_loader')
	print(f"Resolved data_dir: {args.data_dir}")

	tok = None
	if args.tokenizer_path and PreTrainedTokenizerFast is not None:
		try:
			tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
			print(f"Loaded tokenizer from: {args.tokenizer_path}")
		except Exception as e:
			print(f"Warning: tokenizer load failed: {e}")
			tok = None

	random.seed(args.seed)
	ds = MultiTaskDataset(
		data_dir=args.data_dir,
		tokenizer=tok,
		max_input_length=args.max_input_length,
		max_target_length=args.max_target_length,
	)

	sampler = TaskBatchSampler(ds, batch_size=args.batch_size, shuffle=True)
	dl = build_dataloader(ds, sampler=sampler, num_workers=0)

	shown = {t: 0 for t in TASKS}
	want = args.samples_per_task
	pad_id = getattr(tok, 'pad_token_id', 0) if tok is not None else 0

	def dec_safe(toks: torch.Tensor) -> str:
		if tok is None:
			return ''
		try:
			return tok.decode(toks.tolist(), skip_special_tokens=True)
		except Exception:
			arr = [int(x) for x in toks.tolist() if int(x) != pad_id]
			try:
				return tok.decode(arr, skip_special_tokens=True)
			except Exception:
				return ''

	for batch in dl:
		t = batch['task']
		if shown.get(t, 0) >= want:
			continue
		if 'input_ids' in batch:
			x = batch['input_ids'][0]
			y = batch['labels'][0]
			attn = batch['attention_mask'][0]
			in_len = int(attn.sum().item())
			tg_len = int((y != pad_id).sum().item())
			xin = dec_safe(x)
			yout = dec_safe(y)
		else:
			xin = batch['input_text'][0]
			yout = batch['target_text'][0]
			in_len = len(xin)
			tg_len = len(yout)

		if shown.get(t, 0) == 0:
			print(f"\n[{t}]")
		print(f"- Input(len={in_len}): {xin[:160]}{'...' if len(xin)>160 else ''}")
		print(f"  Target(len={tg_len}): {yout[:160]}{'...' if len(yout)>160 else ''}")
		shown[t] = shown.get(t, 0) + 1
		if all(shown.get(tt, 0) >= want for tt in TASKS):
			break

	print('\n✓ Preview complete')

