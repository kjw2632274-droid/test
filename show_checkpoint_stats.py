"""
Show training statistics from checkpoint.

Usage:
    python -m new_multitask.show_checkpoint_stats --ckpt_path output/multitask_ckpt/checkpoint_143247.pt
"""

import argparse
import torch


def format_number(num):
    """Format large numbers with K/M suffix."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return str(num)


def main():
    parser = argparse.ArgumentParser(description='Show checkpoint statistics')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint')
    args = parser.parse_args()
    
    print(f'Loading checkpoint: {args.ckpt_path}\n')
    state = torch.load(args.ckpt_path, map_location='cpu')
    
    print('=' * 70)
    print('CHECKPOINT STATISTICS')
    print('=' * 70)
    
    # Basic info
    if isinstance(state, dict):
        print(f"Checkpoint type: Dictionary")
        print(f"Keys: {list(state.keys())}\n")
        
        # Training step
        if 'step' in state:
            print(f"Training step: {format_number(state['step'])}")
        
        if 'epoch' in state:
            print(f"Epoch: {state['epoch']}")
        
        # Loss
        if 'loss' in state:
            print(f"Loss: {state['loss']:.4f}")
        
        if 'train_loss' in state:
            print(f"Train loss: {state['train_loss']:.4f}")
        
        if 'val_loss' in state:
            print(f"Val loss: {state['val_loss']:.4f}")
        
        # Learning rate
        if 'learning_rate' in state:
            print(f"Learning rate: {state['learning_rate']:.2e}")
        
        # Optimizer state
        if 'optimizer' in state:
            opt_state = state['optimizer']
            if isinstance(opt_state, dict):
                print(f"\nOptimizer state keys: {list(opt_state.keys())}")
                if 'param_groups' in opt_state and len(opt_state['param_groups']) > 0:
                    pg = opt_state['param_groups'][0]
                    print(f"Learning rate (from optimizer): {pg.get('lr', 'N/A')}")
                    print(f"Weight decay: {pg.get('weight_decay', 'N/A')}")
                    print(f"Betas: {pg.get('betas', 'N/A')}")
                    print(f"Eps: {pg.get('eps', 'N/A')}")
        
        # Scheduler state
        if 'scheduler' in state:
            sched_state = state['scheduler']
            if isinstance(sched_state, dict):
                print(f"\nScheduler state keys: {list(sched_state.keys())}")
                if 'last_epoch' in sched_state:
                    print(f"Last epoch: {sched_state['last_epoch']}")
                if '_step_count' in sched_state:
                    print(f"Step count: {format_number(sched_state['_step_count'])}")
                if 'base_lrs' in sched_state:
                    print(f"Base learning rates: {sched_state['base_lrs']}")
        
        # Model state
        if 'model' in state:
            model_state = state['model']
            print(f"\nModel state_dict:")
            print(f"  Total parameters: {len(model_state)}")
            
            # Count parameters by component
            encoder_params = sum(1 for k in model_state.keys() if 'encoder' in k)
            decoder_params = sum(1 for k in model_state.keys() if 'decoder' in k)
            lm_head_params = sum(1 for k in model_state.keys() if 'lm_head' in k)
            embedding_params = sum(1 for k in model_state.keys() if 'embedding' in k or 'shared' in k)
            
            print(f"  Encoder parameters: {encoder_params}")
            print(f"  Decoder parameters: {decoder_params}")
            print(f"  LM head parameters: {lm_head_params}")
            print(f"  Embedding parameters: {embedding_params}")
            
            # Total param count (actual values)
            total_params = sum(p.numel() for p in model_state.values())
            print(f"\n  Total parameter count: {format_number(total_params)} ({total_params:,})")
            
            # Show some parameter shapes
            print(f"\n  Sample parameter shapes:")
            for i, (k, v) in enumerate(list(model_state.items())[:10]):
                print(f"    {k}: {tuple(v.shape)}")
            if len(model_state) > 10:
                print(f"    ... and {len(model_state) - 10} more")
        
        # Custom metrics
        if 'metrics' in state:
            print(f"\nMetrics: {state['metrics']}")
        
        # Other info
        other_keys = [k for k in state.keys() if k not in ['model', 'optimizer', 'scheduler', 'step', 'epoch', 'loss', 'train_loss', 'val_loss', 'learning_rate', 'metrics']]
        if other_keys:
            print(f"\nOther keys in checkpoint:")
            for k in other_keys:
                val = state[k]
                if isinstance(val, (int, float, str)):
                    print(f"  {k}: {val}")
                elif isinstance(val, dict):
                    print(f"  {k}: dict with {len(val)} items")
                else:
                    print(f"  {k}: {type(val).__name__}")
    
    else:
        print("Checkpoint is a raw state_dict (not wrapped in dict)")
        print(f"Total parameters: {len(state)}")
        total_params = sum(p.numel() for p in state.values())
        print(f"Total parameter count: {format_number(total_params)} ({total_params:,})")
    
    print('=' * 70)


if __name__ == '__main__':
    main()
