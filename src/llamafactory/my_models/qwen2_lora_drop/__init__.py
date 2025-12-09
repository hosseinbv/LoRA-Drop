# Copyright 2024 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING

from transformers.utils import _LazyModule
from transformers.utils.import_utils import define_import_structure

# import pdb; pdb.set_trace()

from .configuration_qwen2_lora_drop import *
from .modeling_qwen2_lora_drop import *
from .tokenization_qwen2_lora_drop import *
from .tokenization_qwen2_lora_drop_fast import *


from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# --- REQUIRED: register your custom architecture ---
AutoConfig.register("qwen2_lora_drop", Qwen2_Lora_DropConfig)
AutoModel.register(Qwen2_Lora_DropConfig, Qwen2_Lora_DropModel)
AutoModelForCausalLM.register(Qwen2_Lora_DropConfig, Qwen2_Lora_DropForCausalLM)

from transformers import AutoTokenizer
from .tokenization_qwen2_lora_drop import Qwen2_Lora_DropTokenizer          # slow (not *Fast)
from .tokenization_qwen2_lora_drop_fast import Qwen2_Lora_DropTokenizerFast  # fast

from .configuration_qwen2_lora_drop import Qwen2_Lora_DropConfig

AutoTokenizer.register(
    Qwen2_Lora_DropConfig,
    slow_tokenizer_class=Qwen2_Lora_DropTokenizer,
    fast_tokenizer_class=Qwen2_Lora_DropTokenizerFast,
)

__all__ = [
    "Qwen2_Lora_DropConfig",
    "Qwen2_Lora_DropModel",
    "Qwen2_Lora_DropForCausalLM",
]

