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

import sys
from pathlib import Path

root = Path().resolve().parent
sys.path.insert(0, str(root))

from pretokenizers.firstpretokenizer import FirstPretokenizer


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

        # TODO mozliwe ze trzeba tez maskowac paddingi

        # loss mask
        loss_mask = self.compute_loss_mask(labels)
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=self.config.pad_token_id)
        raw_loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
        masked_loss = raw_loss * loss_mask.view(-1)
        main_loss = masked_loss.mean()

        # mode loss
        mode_logits = self.mode_classifier(hidden).squeeze(-1)
        mode_labels = self.compute_mode_labels(labels)
        mode_loss = self.bce_loss(mode_logits, mode_labels)

        return main_loss + 0.2 * mode_loss

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
            seq = input_ids[b].tolist()
            is_semantic = False
            for tok in seq:
                if tok == self.semantic_start_id:
                    is_semantic = True
                if tok == self.semantic_stop_id:
                    is_semantic = False

            if is_semantic:
                for tid in self.code_token_ids:
                    scores[b, tid] = -1e9
            else:
                for tid in self.semantic_token_ids:
                    scores[b, tid] = -1e9
        return scores


class LogitsMaskingCallback(TrainerCallback):
    """
    Callback to override the generate method of the model to apply logits masking during evaluation and prediction in training process.
    This is for benchmark purposes during training evaluation. Training itself doesnt use generate().
    This is needed for more accurate evaluation, couse model usage is based logits masking.
    """

    def __init__(self, semantic_ids, code_ids, start_id, stop_id):
        self.semantic_ids = semantic_ids
        self.code_ids = code_ids
        self.start_id = start_id
        self.stop_id = stop_id
        self.original_generate = None

    def _override_generate(self, model):
        processor = SemanticCodeLogitsMask(
            semantic_token_ids=self.semantic_ids,
            code_token_ids=self.code_ids,
            semantic_start_id=self.start_id,
            semantic_stop_id=self.stop_id
        )

        if self.original_generate is None:
            self.original_generate = model.generate

        def generate_with_masking(*args, **kwargs):
            kwargs["logits_processor"] = LogitsProcessorList([processor])
            return self.original_generate(*args, **kwargs)

        model.generate = generate_with_masking

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        self._override_generate(model)

    def on_predict(self, args, state, control, model=None, **kwargs):
        self._override_generate(model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self.original_generate:
            model.generate = self.original_generate


def main():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    pretokenizer = FirstPretokenizer(_use_dedent=True, _use_semantics=True)

    semantic_token_ids = [
        i for i in range(tokenizer.vocab_size)
        if i not in tokenizer.all_special_ids
    ]

    tags = [v for k, v in pretokenizer.tags.__dict__.items() if not k.startswith("_")]
    tokenizer.add_tokens(tags)

    code_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in tags]

    semantic_start_id = tokenizer.convert_tokens_to_ids(pretokenizer.tags.SEMANTIC_START)
    semantic_end_id = tokenizer.convert_tokens_to_ids(pretokenizer.tags.SEMANTIC_END)

    semantic_token_ids.append(semantic_end_id)
    code_token_ids.remove(semantic_end_id)
    code_token_ids.append(tokenizer.convert_tokens_to_ids("</s>"))

    model = T5WithModeLoss.from_pretrained(
        model_name,
        semantic_start_id=semantic_start_id,
        semantic_stop_id=semantic_end_id,
        semantic_token_ids=semantic_token_ids,
        code_token_ids=code_token_ids,
    )
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = ... 
    eval_dataset = ...

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="steps",
        eval_steps=200,
        logging_dir="./logs",
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            LogitsMaskingCallback(
                semantic_ids=semantic_token_ids,
                code_ids=code_token_ids,
                start_id=semantic_start_id,
                stop_id=semantic_end_id
            )
        ]
    )

    trainer.train()

if __name__ == "__main__":

    # Actually, I dont know if it works and it needs to be tested.

    main()

