import torch
import torch.nn as nn
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    LogitsProcessor,
    LogitsProcessorList
)

class T5WithModeLoss(T5ForConditionalGeneration):
    """
    Model wrapper with mode loss and loss mask.
    """

    def __init__(self, config, semantic_start_id, semantic_stop_id, semantic_token_ids, code_token_ids):
        super().__init__(config)
        self.mode_classifier = nn.Linear(config.d_model, 1)
        self.semantic_start_id = semantic_start_id
        self.semantic_stop_id = semantic_stop_id
        self.semantic_token_ids = set(semantic_token_ids)
        self.code_token_ids = set(code_token_ids)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def compute_mode_labels(self, labels):
        """
        Mask has 1.0 for semantic tokens and 0.0 for code tokens.
        """

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
                    inside = False
                    continue
                if inside:
                    mode_labels[b, t] = 1.0
        return mode_labels

    def compute_loss_mask(self, labels):
        """
        Prevents loss from being computed on tokens being generated within the wrong context.
        """

        batch_size, seq_len = labels.shape
        mask = torch.ones_like(labels, dtype=torch.float)
        for b in range(batch_size):
            inside = False
            for t in range(seq_len):
                tok = labels[b, t].item()
                if tok == self.semantic_start_id:
                    inside = True
                    continue
                if tok == self.semantic_stop_id:
                    inside = False
                    continue

                if inside and tok in self.code_token_ids:
                    mask[b, t] = 0.0
                elif not inside and tok in self.semantic_token_ids:
                    mask[b, t] = 0.0
        return mask

    def compute_loss(self, input_ids, attention_mask, labels, decoder_input_ids=None, **kwargs):
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=None,  # couse we use labels manually
            output_hidden_states=True,
            return_dict=True
        )
        logits = outputs.logits
        hidden = outputs.decoder_hidden_states[-1]
        vocab_size = logits.size(-1)

        # loss mask
        loss_mask = self.compute_loss_mask(labels)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        raw_loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
        masked_loss = raw_loss * loss_mask.view(-1)
        main_loss = masked_loss.mean()

        # mode loss
        mode_logits = self.mode_classifier(hidden).squeeze(-1)
        mode_labels = self.compute_mode_labels(labels)
        mode_loss = self.bce_loss(mode_logits, mode_labels)

        return main_loss + 0.2 * mode_loss
