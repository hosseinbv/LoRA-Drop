# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import TYPE_CHECKING, Optional

from transformers import DataCollatorForLanguageModeling

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .trainer import CustomTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


# ---- put near the top of workflow.py imports ----
from contextlib import suppress


def freeze_all_but_lora(model, allow_norm_and_lm_head: bool = False):
    # 1) freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2) unfreeze only LoRA-Drop adapters
    trainable_names = []
    for name, module in model.named_modules():
        if name.endswith("lora_adapter") or "lora_adapter" in name:
            for pn, p in module.named_parameters(recurse=True):
                p.requires_grad = True
            trainable_names.append(name)

    # (optional) if you also want to tune norms or lm_head, flip this flag
    if allow_norm_and_lm_head:
        for name, p in model.named_parameters():
            if any(k in name for k in ["norm.weight", "lm_head.weight", "embed_tokens.weight"]):
                p.requires_grad = True

    # Print a quick summary
    total, trainable = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"[LoRA-Drop] Trainable params: {trainable:,} / {total:,}")
    if trainable_names:
        print(f"[LoRA-Drop] Unfrozen adapters: {trainable_names}")
    else:
        print("[LoRA-Drop] WARNING: no lora_adapter modules were found. Check model wiring.")


def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="pt", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---- LoRA-Drop training knobs (safe: getattr defaults if fields don't exist) ----
    cfg = getattr(model, "config", None)
    if cfg is not None:
        # ❌ disable dual-pass; we now do single-pass token-accurate mixing
        setattr(cfg, "lora_drop_training_dual_path", False)

        # schedule and parity for LoRA-Drop
        setattr(cfg, "drop_cycle", getattr(finetuning_args, "lora_drop_cycle", 4))  # or hardcode 4
        setattr(cfg, "drop_parity", "even")  # skip even layers during drop-phase

        # LoRA adapter sizing
        setattr(cfg, "lora_rank", getattr(finetuning_args, "lora_drop_rank", 8))
        setattr(cfg, "lora_alpha", getattr(finetuning_args, "lora_drop_alpha", 16.0))

        # ✅ enable token-accurate mixing inside the layer
        setattr(cfg, "lora_drop_token_accurate_train", True)


    # >>> Freeze base model; train only LoRA adapters <<<
    freeze_all_but_lora(model, allow_norm_and_lm_head=False)

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
            else:
                keys += ["eval_loss"]

            plot_loss(training_args.output_dir, keys=keys)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")

        if isinstance(dataset_module.get("eval_dataset"), dict):
            for key in dataset_module["eval_dataset"].keys():
                try:
                    perplexity = math.exp(metrics[f"eval_{key}_loss"])
                except OverflowError:
                    perplexity = float("inf")

                metrics[f"eval_{key}_perplexity"] = perplexity
        else:
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")

            metrics["eval_perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
