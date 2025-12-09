# Copyright 2025 the LlamaFactory team.
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

from types import MethodType
from typing import TYPE_CHECKING, Optional

import torch
from transformers import Trainer
from typing_extensions import override

from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomTrainer(Trainer):
    r"""Inherit Trainer for custom optimizer."""

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    # @override
    # def compute_loss(self, model, inputs, *args, **kwargs):
    #     return super().compute_loss(model, inputs, *args, **kwargs)


    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        """
        Single-pass training for token-accurate LoRA-Drop.
        The decoder layer mixes per token (anchor vs drop), so we just compute LM loss.
        """
        # Important for memory & correctness in training
        kwargs.setdefault("use_cache", False)
        kwargs.setdefault("logits_to_keep", 0)

        outputs = model(**inputs, **kwargs)

        loss = getattr(outputs, "loss", None)
        if loss is not None:
            return loss

        # Fallback CE if model didn't compute loss internally
        logits = outputs.logits
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Labels are required to compute loss.")
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    # @override
    # def compute_loss(self, model, inputs, *args, **kwargs):
    #     """
    #     Dual-pass training:
    #       1) Teacher (full): no-grad, lora_drop_active=False -> teacher_logits (targets)
    #       2) Student (LoRA-Drop): with-grad, lora_drop_active=True -> lm loss
    #       3) Consistency: MSE(student_logits, teacher_logits.detach())
    #       loss = lm_loss + lambda * mse
    #     """
    #     labels = inputs.get("labels", None)
    #     teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}

    #     # ---- TEACHER (full) ----
    #     teacher_cache = {}  # << NEW: will collect per-layer full outputs
    #     with torch.no_grad():
    #         teacher_outputs = model(
    #             **teacher_inputs,
    #             use_cache=False,
    #             lora_drop_active=False,
    #             layer_output_cache=teacher_cache,  # save the layer outputs
    #             logits_to_keep=0
    #         )
    #         teacher_logits = teacher_outputs.logits.detach()

    #     # ---- STUDENT (LoRA-Drop) ----
    #     student_outputs = model(
    #         **inputs,
    #         use_cache=False,
    #         lora_drop_active=True,
    #         layer_output_cache=teacher_cache,  # reuse the teacher’s cache
    #         logits_to_keep=0
    #     )

    #     lm_loss = student_outputs.loss if hasattr(student_outputs, "loss") else None
    #     if lm_loss is None:
    #         # fall back to standard CE if model didn't compute loss internally
    #         logits = student_outputs.logits
    #         if labels is None:
    #             raise ValueError("Labels are required for LM loss when model does not return .loss")
    #         shift_logits = logits[:, :-1, :].contiguous()
    #         shift_labels = labels[:, 1:].contiguous()
    #         lm_loss = torch.nn.functional.cross_entropy(
    #             shift_logits.view(-1, shift_logits.size(-1)),
    #             shift_labels.view(-1),
    #             ignore_index=-100,
    #         )

    #     # ========= Consistency: logits-level MSE =========
    #     student_logits = student_outputs.logits
    #     # Align shapes if model used logits_to_keep slicing (we passed 0 above -> no slicing)
    #     if student_logits.shape != teacher_logits.shape:
    #         # last time-steps only if needed
    #         T = min(student_logits.size(1), teacher_logits.size(1))
    #         student_logits = student_logits[:, -T:, :]
    #         teacher_logits = teacher_logits[:, -T:, :]

    #     mse = torch.nn.functional.mse_loss(student_logits, teacher_logits)

    #     # Weight (from finetuning_args if present, else default)
    #     lam = getattr(self.finetuning_args, "lora_drop_consistency_weight", 0.05)

    #     loss = lm_loss + lam * mse
    #     return loss
