"""
Compute evaluation metrics (F1, EM, ROUGE, BLEU) from prediction results.

Usage:
    # Compute metrics from saved predictions
    python -m new_multitask.compute_metrics --pred_file output/eval_results.jsonl
    
    # Compute metrics for specific task
    python -m new_multitask.compute_metrics --pred_file output/eval_style_transfer.jsonl --task style_transfer
"""

import argparse
import json
import re
from collections import Counter
from typing import List, Dict, Tuple


def normalize_answer(s: str) -> str:
    """Normalize answer for QA evaluation."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        import string
        exclude = set(string.punctuation)
        return ''.join(ch if ch not in exclude else ' ' for ch in text)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_korean(s: str) -> str:
    """Normalize Korean text (remove spaces and punctuation)."""
    import string
    # Remove punctuation
    exclude = set(string.punctuation + '、。，．！？；：''""（）【】《》')
    s = ''.join(ch if ch not in exclude else ' ' for ch in s)
    # Remove extra spaces
    return ' '.join(s.split())


def compute_f1(prediction: str, ground_truth: str, normalize_fn=normalize_answer) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_fn(prediction).split()
    gold_tokens = normalize_fn(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return int(pred_tokens == gold_tokens)
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def compute_exact_match(prediction: str, ground_truth: str, normalize_fn=normalize_answer) -> float:
    """Compute exact match score."""
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))


def compute_rouge_l(prediction: str, ground_truth: str, normalize_fn=normalize_korean) -> float:
    """Compute ROUGE-L score (longest common subsequence based)."""
    pred_tokens = normalize_fn(prediction).split()
    gold_tokens = normalize_fn(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    
    # LCS using dynamic programming
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == gold_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    if lcs_length == 0:
        return 0.0
    
    precision = lcs_length / len(pred_tokens)
    recall = lcs_length / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_bleu(prediction: str, ground_truth: str, normalize_fn=normalize_korean, n: int = 2) -> float:
    """Compute BLEU score (simplified version with n-gram precision)."""
    pred_tokens = normalize_fn(prediction).split()
    gold_tokens = normalize_fn(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    
    # Compute n-gram precisions
    precisions = []
    for i in range(1, n + 1):
        pred_ngrams = [tuple(pred_tokens[j:j+i]) for j in range(len(pred_tokens) - i + 1)]
        gold_ngrams = [tuple(gold_tokens[j:j+i]) for j in range(len(gold_tokens) - i + 1)]
        
        if len(pred_ngrams) == 0 or len(gold_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        pred_counter = Counter(pred_ngrams)
        gold_counter = Counter(gold_ngrams)
        
        matches = sum((pred_counter & gold_counter).values())
        total = len(pred_ngrams)
        
        precisions.append(matches / total if total > 0 else 0.0)
    
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        geo_mean = 1.0
        for p in precisions:
            geo_mean *= p
        geo_mean = geo_mean ** (1.0 / len(precisions))
    else:
        geo_mean = 0.0
    
    # Brevity penalty
    bp = 1.0
    if len(pred_tokens) < len(gold_tokens):
        bp = 0.0 if len(pred_tokens) == 0 else (len(pred_tokens) / len(gold_tokens)) ** 0.5
    
    return bp * geo_mean


def compute_char_accuracy(prediction: str, ground_truth: str) -> float:
    """Compute character-level accuracy (for style transfer)."""
    pred_chars = list(prediction.replace(' ', ''))
    gold_chars = list(ground_truth.replace(' ', ''))
    
    if len(pred_chars) == 0 and len(gold_chars) == 0:
        return 1.0
    if len(pred_chars) == 0 or len(gold_chars) == 0:
        return 0.0
    
    # Character-level F1
    pred_counter = Counter(pred_chars)
    gold_counter = Counter(gold_chars)
    
    common = pred_counter & gold_counter
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_chars)
    recall = num_same / len(gold_chars)
    
    return (2 * precision * recall) / (precision + recall)


def evaluate_predictions(predictions: List[Dict], task: str = None) -> Dict:
    """Evaluate predictions and return metrics."""
    
    # Group by task
    task_groups = {}
    for pred in predictions:
        t = pred.get('task', 'unknown')
        if task is None or t == task:
            task_groups.setdefault(t, []).append(pred)
    
    all_metrics = {}
    
    for task_name, preds in task_groups.items():
        metrics = {
            'count': len(preds),
            'avg_loss': sum(p.get('loss', 0.0) for p in preds) / len(preds) if preds else 0.0,
        }
        
        # Task-specific metrics
        if task_name == 'qa':
            # QA: F1, EM
            f1_scores = [compute_f1(p.get('prediction', p.get('pred', '')), p['target']) for p in preds]
            em_scores = [compute_exact_match(p.get('prediction', p.get('pred', '')), p['target']) for p in preds]
            
            metrics['f1'] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            metrics['em'] = sum(em_scores) / len(em_scores) if em_scores else 0.0
        
        elif task_name in ['dialogue_summarization', 'role_generation']:
            # Summarization: ROUGE-L, BLEU
            rouge_scores = [compute_rouge_l(p.get('prediction', p.get('pred', '')), p['target']) for p in preds]
            bleu_scores = [compute_bleu(p.get('prediction', p.get('pred', '')), p['target']) for p in preds]
            
            metrics['rouge_l'] = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
            metrics['bleu'] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        
        elif task_name == 'style_transfer':
            # Style transfer: Character accuracy, ROUGE-L
            char_acc_scores = [compute_char_accuracy(p.get('prediction', p.get('pred', '')), p['target']) for p in preds]
            rouge_scores = [compute_rouge_l(p.get('prediction', p.get('pred', '')), p['target'], normalize_fn=normalize_korean) for p in preds]
            
            metrics['char_accuracy'] = sum(char_acc_scores) / len(char_acc_scores) if char_acc_scores else 0.0
            metrics['rouge_l'] = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
        
        all_metrics[task_name] = metrics
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='Compute evaluation metrics')
    parser.add_argument('--pred_file', type=str, required=True, help='Path to predictions JSONL file')
    parser.add_argument('--task', type=str, default=None, help='Specific task to evaluate (optional)')
    
    args = parser.parse_args()
    
    # Load predictions
    print(f'Loading predictions from {args.pred_file}...')
    predictions = []
    with open(args.pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    
    print(f'Loaded {len(predictions)} predictions\n')
    
    # Compute metrics
    metrics = evaluate_predictions(predictions, args.task)
    
    # Print results
    print('=' * 70)
    print('EVALUATION METRICS')
    print('=' * 70)
    
    for task_name, task_metrics in sorted(metrics.items()):
        print(f'\n{task_name.upper()}:')
        print(f"  Samples: {task_metrics['count']}")
        print(f"  Avg Loss: {task_metrics['avg_loss']:.4f}")
        
        if 'f1' in task_metrics:
            print(f"  F1 Score: {task_metrics['f1']:.4f}")
        if 'em' in task_metrics:
            print(f"  Exact Match: {task_metrics['em']:.4f}")
        if 'rouge_l' in task_metrics:
            print(f"  ROUGE-L: {task_metrics['rouge_l']:.4f}")
        if 'bleu' in task_metrics:
            print(f"  BLEU: {task_metrics['bleu']:.4f}")
        if 'char_accuracy' in task_metrics:
            print(f"  Char Accuracy: {task_metrics['char_accuracy']:.4f}")
    
    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
