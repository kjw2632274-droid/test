"""
Multi-head KoBART (shared encoder, 4 task-specific decoders)

Architecture:
- Shared: embedding + encoder (from pretrained KoBART)
- Task-specific: 4 decoders + 4 LM heads (one per task)

Tasks:
- qa
- role_generation
- style_transfer
- dialogue_summarization

Usage:
    from new_multitask.model import MultiHeadKoBART
    model = MultiHeadKoBART.from_pretrained('gogamza/kobart-base-v2')
    
    # Training
    out = model(input_ids=..., attention_mask=..., labels=..., task_name='qa')
    loss = out['loss']
    
    # Inference
    preds = model.generate(input_ids=..., attention_mask=..., task_name='qa')
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import (
    BartForConditionalGeneration,
    BartConfig,
)
from transformers.models.bart.modeling_bart import (
    BartEncoder,
    BartDecoder,
    BartPretrainedModel,
    shift_tokens_right,
)


TASKS = (
    'qa',
    'role_generation',
    'style_transfer',
    'dialogue_summarization',
)


class MultiHeadKoBART(BartPretrainedModel):
    """KoBART with shared encoder + 4 task-specific decoders and LM heads."""

    def __init__(self, base_model: BartForConditionalGeneration, task_names: Optional[list[str]] = None):
        super().__init__(base_model.config)

        if task_names is None:
            task_names = list(TASKS)

        self.task_names = task_names
        self.config: BartConfig = base_model.config

        # Shared components
        self.shared_embedding = base_model.model.shared
        self.encoder: BartEncoder = base_model.model.encoder

        # Task-specific decoders (deep copy from base)
        self.decoders = nn.ModuleDict()
        for t in self.task_names:
            dec = copy.deepcopy(base_model.model.decoder)
            self.decoders[t] = dec

        # Task-specific LM heads (tied to shared embedding)
        self.lm_heads = nn.ModuleDict()
        for t in self.task_names:
            lm = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
            lm.weight = self.shared_embedding.weight  # weight tying
            self.lm_heads[t] = lm

        self.post_init()

    def _select_decoder(self, task_name: str) -> BartDecoder:
        if task_name not in self.decoders:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(self.decoders.keys())}")
        return self.decoders[task_name]

    def _select_lm_head(self, task_name: str) -> nn.Linear:
        if task_name not in self.lm_heads:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(self.lm_heads.keys())}")
        return self.lm_heads[task_name]

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        task_name: str = 'qa',
        labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with task-specific decoder."""
        
        # Encode
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Prepare decoder inputs
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode with task-specific decoder
        decoder = self._select_decoder(task_name)
        dec_out = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=enc_out.last_hidden_state,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )

        # Task-specific LM head
        lm_head = self._select_lm_head(task_name)
        logits = lm_head(dec_out.last_hidden_state)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits, dec_out, enc_out)
            return (loss,) + output if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'encoder_last_hidden_state': enc_out.last_hidden_state,
            'decoder_hidden_states': dec_out.last_hidden_state,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_name: str = 'qa',
        max_length: int = 64,
        temperature: float = 1.0,
        repetition_penalty: float = 1.5,
        no_repeat_ngram_size: int = 2,
        **kwargs
    ) -> torch.LongTensor:
        """
        Greedy generation with anti-repetition techniques.
        
        Args:
            repetition_penalty: Penalty for repeated tokens (>1.0 discourages repetition)
            no_repeat_ngram_size: Block n-grams that already appeared (default 2)
        """
        device = input_ids.device
        bsz = input_ids.size(0)

        # Encode
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Select task-specific components
        decoder = self._select_decoder(task_name)
        lm_head = self._select_lm_head(task_name)

        # Start with BOS token
        dec_in = torch.full(
            (bsz, 1),
            self.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        # Calculate adaptive max_length based on input
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1)  # [bsz]
        else:
            input_lengths = torch.full((bsz,), input_ids.size(1), device=device)
        
        # For style_transfer task, output should be similar length to input
        # Very strict: output ~= input length (for 존댓말→반말 conversion)
        # Add only +2 tokens as buffer
        adaptive_max_lengths = (input_lengths * 0.95 + 2).long().clamp(min=5, max=max_length)

        # Track which sequences are done
        is_finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        
        # Greedy decoding loop
        for step in range(max_length - 1):
            # Decode
            dec_out = decoder(
                input_ids=dec_in,
                encoder_hidden_states=enc_out.last_hidden_state,
                encoder_attention_mask=attention_mask,
                return_dict=True,
            )
            logits = lm_head(dec_out.last_hidden_state[:, -1, :])  # [bsz, vocab]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(bsz):
                    if not is_finished[i]:
                        for token_id in set(dec_in[i].tolist()):
                            # Divide logits for repeated tokens (discourage them)
                            logits[i, token_id] = logits[i, token_id] / repetition_penalty
            
            # Aggressively boost EOS token probability as we approach adaptive max length
            current_len = dec_in.size(1)
            for i in range(bsz):
                if not is_finished[i]:
                    adaptive_max = adaptive_max_lengths[i].item()
                    if current_len >= adaptive_max * 0.6:  # After 60% of adaptive max
                        # Very aggressive EOS boosting
                        progress = (current_len - adaptive_max * 0.6) / (adaptive_max * 0.4)
                        boost_factor = 1.0 + progress * 10.0  # Up to 11x boost
                        logits[i, self.config.eos_token_id] = logits[i, self.config.eos_token_id] * boost_factor
                    
                    # Absolutely force EOS at adaptive max length
                    if current_len >= adaptive_max - 1:
                        logits[i, :] = -float('inf')
                        logits[i, self.config.eos_token_id] = 1e10

            # Apply n-gram blocking
            if no_repeat_ngram_size > 1 and dec_in.size(1) >= no_repeat_ngram_size:
                for i in range(bsz):
                    if not is_finished[i]:
                        seq = dec_in[i].tolist()
                        # Get current prefix (last n-1 tokens)
                        current_prefix = tuple(seq[-(no_repeat_ngram_size - 1):])
                        
                        # Find all tokens that would create repeated n-grams
                        banned_tokens = set()
                        for k in range(len(seq) - no_repeat_ngram_size + 1):
                            ngram_prefix = tuple(seq[k:k + no_repeat_ngram_size - 1])
                            if ngram_prefix == current_prefix:
                                banned_token = seq[k + no_repeat_ngram_size - 1]
                                banned_tokens.add(banned_token)
                        
                        # Block banned tokens
                        for token_id in banned_tokens:
                            logits[i, token_id] = -float('inf')

            # Greedy selection
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)  # [bsz, 1]
            
            # For finished sequences, keep padding with EOS
            for i in range(bsz):
                if is_finished[i]:
                    next_tok[i, 0] = self.config.eos_token_id
            
            dec_in = torch.cat([dec_in, next_tok], dim=-1)
            
            # Update finished status
            is_finished = is_finished | (next_tok.squeeze(-1) == self.config.eos_token_id)

            # Stop if all sequences are finished
            if is_finished.all():
                break

        return dec_in

    @classmethod
    def from_pretrained(
        cls,
        base_model_name_or_path: str,
        task_names: Optional[list[str]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        **kwargs,
    ) -> 'MultiHeadKoBART':
        """Load from pretrained KoBART checkpoint."""
        base = BartForConditionalGeneration.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
        model = cls(base_model=base, task_names=task_names)
        return model

    def print_model_info(self):
        """Print model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        print("\n" + "=" * 60)
        print("Multi-Head KoBART Model")
        print("=" * 60)
        print(f"Encoder layers: {self.config.encoder_layers}")
        print(f"Decoder layers: {self.config.decoder_layers}")
        print(f"d_model: {self.config.d_model}")
        print(f"Vocab size: {self.config.vocab_size}")
        print(f"Tasks: {list(self.decoders.keys())}")
        print(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    # Quick test
    import os
    base = os.environ.get('KOBART_MODEL', 'gogamza/kobart-base-v2')
    model = MultiHeadKoBART.from_pretrained(base)
    model.print_model_info()

    # Test forward
    bsz, seqlen = 2, 16
    x = torch.randint(5, 1000, (bsz, seqlen))
    attn = torch.ones_like(x)
    y = torch.randint(5, 1000, (bsz, 12))

    out = model(x, attention_mask=attn, labels=y, task_name='qa')
    print('loss:', float(out['loss']))

    # Test generate
    gen = model.generate(
        x, 
        attention_mask=attn, 
        task_name='qa',
        max_length=32,
        repetition_penalty=1.5,
        no_repeat_ngram_size=2
    )
    print('generated shape:', tuple(gen.shape))
