import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

from math import inf
from transformers import T5ForConditionalGeneration, TrainerCallback, Trainer, LogitsProcessor
from pathlib import Path


class T5WithModeLoss(T5ForConditionalGeneration):
    """
    T5 extended with a mode classifier (semantic vs code).
    """
    def __init__(self, config):
        super().__init__(config)

        def safe_init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                if torch.isnan(layer.weight).any() or torch.isnan(layer.bias).any():
                    print("!!! Reinitializing weights due to NaN...")
                    layer.weight.data = torch.randn_like(layer.weight) * 0.02
                    layer.bias.data.zero_()

        self.mode_classifier = nn.Linear(config.d_model, 1)
        nn.init.xavier_uniform_(self.mode_classifier.weight)
        nn.init.zeros_(self.mode_classifier.bias)
        self.mode_classifier.apply(safe_init_weights)


class CustomT5Trainer(Trainer):
    """
    Custom Trainer for T5 with mode loss and skipping loss for padding tokens and tokens predicted with a wrong context (mode).
    """
    def __init__(self, *args,
                 semantic_start_id,
                 semantic_stop_id,
                 semantic_token_ids,
                 code_token_ids,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_start_id = semantic_start_id
        self.semantic_stop_id = semantic_stop_id
        self.semantic_token_ids = set(semantic_token_ids)
        self.code_token_ids = set(code_token_ids)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def compute_mode_labels(self, labels):
        batch_size, seq_len = labels.shape
        mode_labels = torch.zeros_like(labels, dtype=torch.float)
        for b in range(batch_size):
            inside = False
            for t in range(seq_len):
                tok = labels[b, t]
                if tok == self.semantic_start_id:
                    inside = True
                    continue
                if tok == self.semantic_stop_id:
                    mode_labels[b, t] = 1.0
                    inside = False
                if inside:
                    mode_labels[b, t] = 1.0
        return mode_labels

    def compute_loss_mask(self, logits, mode_labels):
        batch_size, seq_len, vocab_size = logits.shape

        predicted_token_ids = torch.argmax(logits, dim=2)
        predicted_modes = torch.zeros_like(predicted_token_ids, dtype=torch.float)

        # It is not differentiable, so we good
        for b in range(batch_size):
            for t in range(seq_len):
                token_id = predicted_token_ids[b, t].item()
                if token_id in self.semantic_token_ids:
                    predicted_modes[b, t] = 1.0  # Semantic token
                elif token_id in self.code_token_ids:
                    predicted_modes[b, t] = 0.0  # Code token

        # Apply the mode mask to the mode_labels
        mode_mask = mode_labels * predicted_modes
        return mode_mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        decoder_input_ids = inputs.get("decoder_input_ids", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            labels=None,
            output_hidden_states=True,
            return_dict=True
        )

        logits = outputs.logits
        hidden = outputs.decoder_hidden_states[-1]
        vocab_size = logits.size(-1)

        mode_labels = self.compute_mode_labels(labels).to(logits.device)
        mode_mask = self.compute_loss_mask(logits, mode_labels)

        pad_token_id = model.config.pad_token_id

        # === MAIN LOSS ===
        # Flatten
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        mask_flat = mode_mask.view(-1).bool()

        # Padding mask
        non_pad_mask = labels_flat != pad_token_id

        # Final mask
        final_mask = mask_flat & non_pad_mask

        # Apply to logits and labels
        logits_masked = logits_flat[final_mask]
        labels_masked = labels_flat[final_mask]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        main_loss = loss_fct(logits_masked, labels_masked)

        # === MODE LOSS ==
        non_pad_mask_dims = labels != pad_token_id
        selected_hidden = hidden[non_pad_mask_dims]         # [N - padding, hidden_features]
        selected_labels = mode_labels[non_pad_mask_dims]    # [N - padding]

        mode_logits = model.mode_classifier(selected_hidden).squeeze(-1)  # [custom batch]
        mode_loss = self.bce_loss(mode_logits, selected_labels.float())

        total_loss = main_loss + 0.2 * mode_loss
        return (total_loss, outputs) if return_outputs else total_loss


class GradClippingCallBack(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        # print(f"\n[{state.global_step}] --- Gradients BEFORE clipping ---")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.data.norm(2).item()
        #         print(f"{name:60} | grad norm: {grad_norm:.4f}")
        # CLIPPING
        max_norm = args.max_grad_norm if hasattr(args, "max_grad_norm") else 1.0
        total_norm = nn_utils.clip_grad_norm_(model.parameters(), max_norm)
        # print(f"Total norm after clipping: {total_norm:.4f} (max allowed: {max_norm})")
        # print(f"\n[{state.global_step}] --- Gradients AFTER clipping ---")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.data.norm(2).item()
        #         print(f"{name:60} | grad norm: {grad_norm:.4f}")

class SemanticCodeLogitsMask(LogitsProcessor):
    def __init__(self, semantic_token_ids, code_token_ids, semantic_start_id, semantic_stop_id):
        self.semantic_token_ids = set(semantic_token_ids)
        self.code_token_ids = set(code_token_ids)
        self.semantic_start_id = semantic_start_id
        self.semantic_stop_id = semantic_stop_id

    def __call__(self, input_ids, scores):
        """
        Analizing this code it is worth remebering that "semantic start" token belongs to code tokens and "semantic stop" token belongs to semantic ones.
        """
        batch_size, cur_len = input_ids.shape
        for b in range(batch_size):
            latest_token_id = input_ids[b][-1].item()
            
            # Determine the context based on the latest token
            if latest_token_id in self.code_token_ids:
                if latest_token_id == self.semantic_start_id:
                    is_semantic = True
                else:
                    is_semantic = False
            elif latest_token_id == self.semantic_stop_id:
                is_semantic = False
                
            # Prepare the mask to exclude different different context tokens 
            if is_semantic:
                mask = torch.tensor([i in self.semantic_token_ids for i in range(scores.size(-1))], device=scores.device)
            else:
                mask = torch.tensor([i in self.code_token_ids for i in range(scores.size(-1))], device=scores.device)

            scores[b] = torch.where(mask, scores[b], torch.tensor(float("-inf"), device=scores.device))
            scores[b][0] = -inf

            # Condition the end of the sequence
            # if cur_len < 10: scores[b][1] = -inf
        return scores