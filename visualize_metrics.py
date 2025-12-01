"""
Visualize evaluation metrics with charts

Usage:
    # Plot metrics from evaluation predictions
    python -m new_multitask.visualize_metrics --pred_file output/eval_old_model.jsonl

    # Plot training/validation loss curves
    python -m new_multitask.visualize_metrics --loss_log output/multitask_ckpt/training_log_YYYYMMDD-HHMMSS.json
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib
from new_multitask.compute_metrics import evaluate_predictions

# Use a font that supports Korean
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_metrics(metrics_dict: dict, save_path: str = None):
    """Create visualization of metrics"""
    
    tasks = list(metrics_dict.keys())
    n_tasks = len(tasks)
    
    # Create subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Loss comparison
    ax1 = plt.subplot(2, 3, 1)
    losses = [metrics_dict[t]['avg_loss'] for t in tasks]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax1.bar(tasks, losses, color=colors[:n_tasks])
    ax1.set_title('Average Loss by Task', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Task')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Sample counts
    ax2 = plt.subplot(2, 3, 2)
    counts = [metrics_dict[t]['count'] for t in tasks]
    ax2.pie(counts, labels=tasks, autopct='%1.1f%%', colors=colors[:n_tasks], startangle=90)
    ax2.set_title('Sample Distribution', fontsize=14, fontweight='bold')
    
    # 3. QA metrics
    if 'qa' in metrics_dict:
        ax3 = plt.subplot(2, 3, 3)
        qa_metrics = ['f1', 'em']
        qa_values = [metrics_dict['qa'].get(m, 0) for m in qa_metrics]
        ax3.bar(qa_metrics, qa_values, color=['#FF6B6B', '#FF8E8E'])
        ax3.set_title('QA Metrics', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_ylim([0, 1])
        
        for i, v in enumerate(qa_values):
            ax3.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Summarization metrics (dialogue_summarization)
    if 'dialogue_summarization' in metrics_dict:
        ax4 = plt.subplot(2, 3, 4)
        sum_metrics = ['rouge_l', 'bleu']
        sum_values = [metrics_dict['dialogue_summarization'].get(m, 0) for m in sum_metrics]
        ax4.bar(sum_metrics, sum_values, color=['#4ECDC4', '#5FD9CF'])
        ax4.set_title('Dialogue Summarization Metrics', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_ylim([0, 1])
        
        for i, v in enumerate(sum_values):
            ax4.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 5. Role generation metrics
    if 'role_generation' in metrics_dict:
        ax5 = plt.subplot(2, 3, 5)
        role_metrics = ['rouge_l', 'bleu']
        role_values = [metrics_dict['role_generation'].get(m, 0) for m in role_metrics]
        ax5.bar(role_metrics, role_values, color=['#45B7D1', '#60C2DC'])
        ax5.set_title('Role Generation Metrics', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Score')
        ax5.set_ylim([0, 1])
        
        for i, v in enumerate(role_values):
            ax5.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 6. Style transfer metrics
    if 'style_transfer' in metrics_dict:
        ax6 = plt.subplot(2, 3, 6)
        style_metrics = ['char_accuracy', 'rouge_l']
        style_values = [metrics_dict['style_transfer'].get(m, 0) for m in style_metrics]
        ax6.bar(style_metrics, style_values, color=['#FFA07A', '#FFB590'])
        ax6.set_title('Style Transfer Metrics', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Score')
        ax6.set_ylim([0, 1])
        
        for i, v in enumerate(style_values):
            ax6.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved visualization to {save_path}')
    
    plt.show()


def plot_loss_curves(log_path: str, save_path: str = None):
    """Plot training/validation loss curves from a training log."""

    with open(log_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)

    train_history = log_data.get('train_history', [])
    val_history = log_data.get('val_history', [])

    if not train_history and not val_history:
        print(f'No loss history found in {log_path}')
        return

    plt.figure(figsize=(12, 6))

    if train_history:
        train_steps = [entry['step'] for entry in train_history]
        train_losses = [entry['loss'] for entry in train_history]
        plt.plot(train_steps, train_losses, label='Train Loss', color='#FF6B6B', linewidth=1)

    if val_history:
        val_steps = [entry.get('step', i + 1) for i, entry in enumerate(val_history)]
        val_losses = [entry['loss'] for entry in val_history]
        plt.plot(val_steps, val_losses, label='Validation Loss', color='#4ECDC4', linewidth=2, marker='o')

    plt.title('Loss Curve', fontsize=16, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved loss curve to {save_path}')

    plt.show()


def plot_comparison(before_file: str, after_file: str, save_path: str = None):
    """Compare metrics before and after training"""
    
    # Load predictions
    with open(before_file, 'r', encoding='utf-8') as f:
        before_preds = [json.loads(line) for line in f if line.strip()]
    
    with open(after_file, 'r', encoding='utf-8') as f:
        after_preds = [json.loads(line) for line in f if line.strip()]
    
    # Compute metrics
    before_metrics = evaluate_predictions(before_preds)
    after_metrics = evaluate_predictions(after_preds)
    
    # Get common tasks
    tasks = sorted(set(before_metrics.keys()) & set(after_metrics.keys()))
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Before vs After Training Comparison', fontsize=16, fontweight='bold')
    
    # 1. Loss comparison
    ax = axes[0, 0]
    x = range(len(tasks))
    width = 0.35
    before_losses = [before_metrics[t]['avg_loss'] for t in tasks]
    after_losses = [after_metrics[t]['avg_loss'] for t in tasks]
    
    ax.bar([i - width/2 for i in x], before_losses, width, label='Before', color='#FF6B6B')
    ax.bar([i + width/2 for i in x], after_losses, width, label='After', color='#4ECDC4')
    ax.set_title('Average Loss Comparison', fontweight='bold')
    ax.set_ylabel('Loss')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Task-specific metrics
    if 'qa' in tasks:
        ax = axes[0, 1]
        metrics_names = ['F1', 'EM']
        before_vals = [before_metrics['qa'].get('f1', 0), before_metrics['qa'].get('em', 0)]
        after_vals = [after_metrics['qa'].get('f1', 0), after_metrics['qa'].get('em', 0)]
        
        x_pos = range(len(metrics_names))
        ax.bar([i - width/2 for i in x_pos], before_vals, width, label='Before', color='#FF6B6B')
        ax.bar([i + width/2 for i in x_pos], after_vals, width, label='After', color='#4ECDC4')
        ax.set_title('QA Metrics', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
    
    if 'style_transfer' in tasks:
        ax = axes[1, 0]
        metrics_names = ['Char Acc', 'ROUGE-L']
        before_vals = [
            before_metrics['style_transfer'].get('char_accuracy', 0),
            before_metrics['style_transfer'].get('rouge_l', 0)
        ]
        after_vals = [
            after_metrics['style_transfer'].get('char_accuracy', 0),
            after_metrics['style_transfer'].get('rouge_l', 0)
        ]
        
        x_pos = range(len(metrics_names))
        ax.bar([i - width/2 for i in x_pos], before_vals, width, label='Before', color='#FF6B6B')
        ax.bar([i + width/2 for i in x_pos], after_vals, width, label='After', color='#4ECDC4')
        ax.set_title('Style Transfer Metrics', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
    
    if 'dialogue_summarization' in tasks:
        ax = axes[1, 1]
        metrics_names = ['ROUGE-L', 'BLEU']
        before_vals = [
            before_metrics['dialogue_summarization'].get('rouge_l', 0),
            before_metrics['dialogue_summarization'].get('bleu', 0)
        ]
        after_vals = [
            after_metrics['dialogue_summarization'].get('rouge_l', 0),
            after_metrics['dialogue_summarization'].get('bleu', 0)
        ]
        
        x_pos = range(len(metrics_names))
        ax.bar([i - width/2 for i in x_pos], before_vals, width, label='Before', color='#FF6B6B')
        ax.bar([i + width/2 for i in x_pos], after_vals, width, label='After', color='#4ECDC4')
        ax.set_title('Dialogue Summarization Metrics', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved comparison to {save_path}')
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation metrics')
    parser.add_argument('--pred_file', type=str, help='Path to predictions JSONL file')
    parser.add_argument('--before_file', type=str, help='Before training predictions')
    parser.add_argument('--after_file', type=str, help='After training predictions')
    parser.add_argument('--save_path', type=str, help='Path to save figure')
    parser.add_argument('--loss_log', type=str, help='Path to training_log_*.json output')
    
    args = parser.parse_args()
    
    if args.loss_log:
        plot_loss_curves(args.loss_log, args.save_path)
    elif args.before_file and args.after_file:
        # Comparison mode
        plot_comparison(args.before_file, args.after_file, args.save_path)
    elif args.pred_file:
        # Single file mode
        with open(args.pred_file, 'r', encoding='utf-8') as f:
            predictions = [json.loads(line) for line in f if line.strip()]
        
        metrics = evaluate_predictions(predictions)
        plot_metrics(metrics, args.save_path)
    else:
        print('Error: Provide either --pred_file or both --before_file and --after_file')


if __name__ == '__main__':
    main()
