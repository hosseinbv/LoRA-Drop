# Based on QWEN2 code
from typing import Callable, Optional, Union

import torch
from torch import nn
import math
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from .configuration_qwen2_lora_drop import Qwen2_Lora_DropConfig

#-------for caching of layers
# --- Anchor cache: per-layer final outputs at the last anchor token ---
class AnchorOutputCache:
    def __init__(self, num_layers: int, device=None, dtype=None):
        self.num_layers = num_layers
        self.device, self.dtype = device, dtype
        self.layers = [None] * num_layers  # each: [B, 1, D]
        self.anchor_step_id = -1

    def clear(self, new_step_id: int | None = None):
        for i in range(self.num_layers):
            self.layers[i] = None
        if new_step_id is not None:
            self.anchor_step_id = int(new_step_id)

    def write_layer(self, layer_idx: int, final_out: torch.Tensor):
        # keep only last token to minimize memory: [B,T,D] -> [B,1,D]
        self.layers[layer_idx] = final_out[:, -1:, :].detach()

    def read_layer(self, layer_idx: int) -> torch.Tensor | None:
        return self.layers[layer_idx]

    # beam ops
    def reorder_cache(self, beam_idx: torch.LongTensor):
        for i, t in enumerate(self.layers):
            if t is not None:
                self.layers[i] = t.index_select(0, beam_idx)

    def reset(self): self.clear(-1)
    def crop(self, max_length: int): pass
    def batch_repeat_interleave(self, repeats: int):
        for i, t in enumerate(self.layers):
            if t is not None:
                self.layers[i] = t.repeat_interleave(repeats, dim=0)
    def batch_select_indices(self, indices: torch.Tensor):
        for i, t in enumerate(self.layers):
            if t is not None:
                self.layers[i] = t.index_select(0, indices)


class DynamicCacheWithAnchors(DynamicCache):
    def __init__(self, *args, config=None, **kwargs):
        try:
            super().__init__(config=config, *args, **kwargs)
        except TypeError:
            super().__init__(*args, **kwargs)
        nl = getattr(config, "num_hidden_layers", len(self.layers) or 1) if config is not None else (len(self.layers) or 1)
        self.anchor_cache = AnchorOutputCache(num_layers=nl)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        super().reorder_cache(beam_idx); self.anchor_cache.reorder_cache(beam_idx)
    def reset(self):
        super().reset(); self.anchor_cache.reset()
    def crop(self, max_length: int):
        super().crop(max_length); self.anchor_cache.crop(max_length)
    def batch_repeat_interleave(self, repeats: int):
        super().batch_repeat_interleave(repeats); self.anchor_cache.batch_repeat_interleave(repeats)
    def batch_select_indices(self, indices: torch.Tensor):
        super().batch_select_indices(indices); self.anchor_cache.batch_select_indices(indices)


#------------
class LoRADropAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.A = nn.Linear(hidden_size, rank, bias=False)
        self.B = nn.Linear(rank, hidden_size, bias=False)
        self.scaling = alpha / max(1, rank)

        # init like LoRA
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x)) * self.scaling


class Qwen2_Lora_DropMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen2_Lora_DropAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2_Lora_DropConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        just_save_kv: bool = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        if just_save_kv:
            return None, None
        
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@use_kernel_forward_from_hub("RMSNorm")
class Qwen2_Lora_DropRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2_Lora_DropRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2_Lora_DropDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2_Lora_DropConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2_Lora_DropAttention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen2_Lora_DropMLP(config)
        self.input_layernorm = Qwen2_Lora_DropRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2_Lora_DropRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_lora_layernorm = Qwen2_Lora_DropRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
        self.layer_idx = layer_idx
        # self.is_drop_layer = (layer_idx % 2 == 1)  # p = 1/2 → skip odd layers during drop-phase
        self.lora_adapter = LoRADropAdapter(config.hidden_size, getattr(config, "lora_rank", 8), getattr(config, "lora_alpha", 16.0))


    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        lora_drop_active: bool = False,           # used for inference
        layer_output_cache: Optional[AnchorOutputCache] = None,# used for inference cache across tokens
        training_dual_path: bool = False,         # legacy; not used in token-accurate path
        gen_token_idx: int = -1,              # legacy; not used in token-accurate path
        **kwargs,
    ) -> torch.Tensor:
        cfg = self.self_attn.config if hasattr(self.self_attn, "config") else None
        drop_cycle = getattr(cfg, "drop_cycle", 4)
        # drop_parity = getattr(cfg, "drop_parity", "even")  # "even" or "odd"
        # skip_this_layer_in_drop = self.layer_idx>0 and (self.layer_idx % 2 == 0) if drop_parity == "even" else (self.layer_idx % 2 == 1)
        skip_this_layer_in_drop = self.layer_idx>0 and (self.layer_idx % 4 != 0)
        # skip_this_layer_in_drop = True
        # ----- Pre-norm input (the layer input) -----
        residual_pre_norm = hidden_states
        x = self.input_layernorm(hidden_states)   # normalized input for attn/MLP

        # =========================
        # Inference: keep fast early-return skip
        # =========================
        # import pdb; pdb.set_trace()
        just_save_kv = False
        if (not self.training) and gen_token_idx%8!=0 and lora_drop_active \
            and skip_this_layer_in_drop and layer_output_cache is not None:
            # cached_full = layer_output_cache.get(self.layer_idx, None)
            cached_full = layer_output_cache.anchor_cache.read_layer(self.layer_idx)
            if cached_full is not None:
                # align seq length if needed (e.g., generation with T=1)
                if cached_full.size(1) != x.size(1):
                    T = x.size(1)
                    cached_full = cached_full[:, -T:, :]
                lora_delta = self.lora_adapter(residual_pre_norm)
                # just_save_kv=True
                # attn_out, _ = self.self_attn(
                #     hidden_states=x,
                #     attention_mask=attention_mask,
                    
                #     past_key_values=past_key_values,
                    
                #     cache_position=cache_position,
                #     position_embeddings=position_embeddings,
                #     just_save_kv=just_save_kv,
                #     **kwargs,
                # )
                return cached_full + lora_delta
            # if cache miss, fall through to full compute

        # =========================
        # Full path (always computed in training; used directly in inference if not skipping)
        # =========================
        attn_out, _ = self.self_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        h = residual_pre_norm + attn_out

        residual = h
        h_norm = self.post_attention_layernorm(h)
        mlp_out = self.mlp(h_norm)
        full_out = residual + mlp_out          # shape [B, T, D]

        # =========================
        # Training: token-accurate LoRA-Drop mixing
        # =========================
        if self.training and getattr(cfg, "lora_drop_token_accurate_train", False):
            # Build per-token mask from position_ids relative to first token in sequence
            # anchor: rel_pos % drop_cycle == 0   |   drop: rel_pos % drop_cycle != 0
            # position_ids: [B, T]
            if position_ids is None:
                # fallback to a simple [0..T-1] if position_ids not provided
                rel_pos = torch.arange(full_out.size(1), device=full_out.device).unsqueeze(0).expand(full_out.size(0), -1)
            else:
                # make it relative so anchors fall within this window
                rel_pos = position_ids - position_ids[:, :1]

            drop_mask = (rel_pos % drop_cycle != 0)    # [B, T] → True means "use LoRA-Drop"
            # For each token, find its anchor index within this window
            anchor_idx = (rel_pos // drop_cycle) * drop_cycle          # [B, T]
            T, D = full_out.size(1), full_out.size(2)
            # clamp in case of edge effects
            anchor_idx = anchor_idx.clamp_(min=0, max=T - 1)

            # Gather full_out at each token's anchor position
            anchor_full = full_out.gather(
                dim=1,
                index=anchor_idx.unsqueeze(-1).expand(-1, -1, D)      # [B, T, D]
            )

            # LoRA delta from PRE-norm input, per token
            lora_delta = self.lora_adapter(residual_pre_norm)          # [B, T, D]

            if skip_this_layer_in_drop:
                # mix: on drop tokens use LoRA-Drop; on anchors use full_out
                mixed = torch.where(drop_mask.unsqueeze(-1), anchor_full + lora_delta, full_out)
                out = mixed
            else:
                out = full_out
            # Log
            # num_drop = drop_mask.sum().item()
            # (Optional) save anchor reference only for eval/inference
            if layer_output_cache is not None and (not self.training):
                # layer_output_cache[self.layer_idx] = out.detach()
                layer_output_cache.anchor_cache.write_layer(self.layer_idx, out.detach())
            return out

        # =========================
        # Default: store cache (for inference) and return full_out
        # =========================
        if gen_token_idx %4 == 0 and \
            layer_output_cache is not None and (not self.training):
            # layer_output_cache[self.layer_idx] = full_out.detach()
            layer_output_cache.anchor_cache.write_layer(self.layer_idx, full_out.detach())

        return full_out


    # def forward(
    #     self,
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     use_cache: Optional[bool] = False,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    #     *,
    #     lora_drop_active: bool = False,           # NEW: are we in the 3-token drop-phase?
    #     layer_output_cache: Optional[dict] = None,# NEW: dict[layer_idx] -> last full output at anchor
    #     training_dual_path: bool = False,         # NEW: if True (only used in training), compute both full & lora paths
    #     **kwargs,
    # ) -> torch.Tensor:
    #     residual = hidden_states
    #     hidden_states = self.input_layernorm(hidden_states)
    #     # ---- LoRA-Drop short-circuit for odd layers during drop-phase ----
    #     if lora_drop_active and self.is_drop_layer:
    #         # Use the cached full output from the most recent anchor token
    #         if layer_output_cache is not None and self.layer_idx in layer_output_cache:
    #             cached_full = layer_output_cache[self.layer_idx]
    #             # LoRA adapter consumes "current" layer input (pre-attn norm input)
    #             # Note: we used residual before the input_layernorm to match your description.
    #             hidden_states = self.lora_adapter(hidden_states)  # whole-layer surrogate
    #             hidden_states = self.post_lora_layernorm(cached_full + hidden_states) + residual
    #             return hidden_states
    #         # If for some reason no cache yet (e.g., first few tokens), fall back to full compute.

    #     # Self Attention
    #     hidden_states, _ = self.self_attn(
    #         hidden_states=hidden_states,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         use_cache=use_cache,
    #         cache_position=cache_position,
    #         position_embeddings=position_embeddings,
    #         **kwargs,
    #     )
    #     hidden_states = residual + hidden_states

    #     # Fully Connected
    #     residual = hidden_states
    #     hidden_states = self.post_attention_layernorm(hidden_states)
    #     hidden_states = self.mlp(hidden_states)
    #     hidden_states = residual + hidden_states
    #     # Save this layer's full output as the new anchor reference
    #     if layer_output_cache is not None:
    #         layer_output_cache[self.layer_idx] = hidden_states.detach()  # detach so the cache is not a grad source
    #     return hidden_states


@auto_docstring
class Qwen2_Lora_DropPreTrainedModel(PreTrainedModel):
    config: Qwen2_Lora_DropConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_Lora_DropDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen2_Lora_DropDecoderLayer,
        "attentions": Qwen2_Lora_DropAttention,
    }


class Qwen2_Lora_DropRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen2_Lora_DropConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@auto_docstring
class Qwen2_Lora_DropModel(Qwen2_Lora_DropPreTrainedModel):
    def __init__(self, config: Qwen2_Lora_DropConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_Lora_DropDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2_Lora_DropRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_Lora_DropRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.drop_cycle = getattr(config, "drop_cycle", 4)     # token t anchor + next 3 drop
        self.drop_ratio = getattr(config, "drop_ratio", 0.5)   # p = 1/2 (matches is_drop_layer rule above)

        # self.layer_output_cache = {}
        self.layer_output_cache = DynamicCacheWithAnchors(config=config)
        # if not self.training:
        self.gen_token_counter = -1
        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
    
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        # Per-step storage for LoRA-Drop to keep "last full outputs" at the anchor
        # if past_key_values is not None and hasattr(past_key_values, "_extra_state"):
        #     layer_output_cache = past_key_values._extra_state.setdefault("layer_outputs", {})
        # else:
        #     layer_output_cache = {}
        #     if past_key_values is not None:
        #         # Create a small side-channel on DynamicCache to persist across calls
        #         setattr(past_key_values, "_extra_state", {"layer_outputs": layer_output_cache})

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # Anchor token if (step % drop_cycle == 0) else drop-phase
        # step_id = int(cache_position[-1].item()) if cache_position.ndim > 0 else int(cache_position)
        # lora_drop_active = (step_id % self.drop_cycle) != 0
        # --- LoRA-Drop: extract overrides from **kwargs to avoid double-passing ---
        
        lora_drop_override = kwargs.pop("lora_drop_active", None)
        training_dual_path_override = kwargs.pop("training_dual_path", None)
        # external_layer_output_cache = kwargs.pop("layer_output_cache", None)  # << NEW
        # Anchor/drop schedule (only used if caller didn't override)
        step_id = int(cache_position[-1].item()) if cache_position.ndim > 0 else int(cache_position)
        default_lora_drop_active = (step_id % self.drop_cycle) != 0

        # Final flags (caller override wins)
        lora_drop_active = bool(default_lora_drop_active) if lora_drop_override is None else bool(lora_drop_override)
        training_dual_path = (
            self.training and getattr(self.config, "lora_drop_training_dual_path", False)
            if training_dual_path_override is None else bool(training_dual_path_override)
        )
        
        if not self.training:
            self.gen_token_counter += 1
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                lora_drop_active=lora_drop_active,
                layer_output_cache=self.layer_output_cache,
                training_dual_path=training_dual_path,
                gen_token_idx=self.gen_token_counter,
                **kwargs,
            )
        # if not self.training:
        #     if  self.gen_token_counter %4 ==0:
        #         self.layer_output_cache = layer_output_cache
            # setattr(past_key_values, "_extra_state", {"layer_outputs": layer_output_cache})

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class Qwen2_Lora_DropForCausalLM(Qwen2_Lora_DropPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2_Lora_DropModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2_Lora_DropForCausalLM

        >>> model = Qwen2_Lora_DropForCausalLM.from_pretrained("meta-qwen2_lora_drop/Qwen2_Lora_Drop-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2_lora_drop/Qwen2_Lora_Drop-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        if not self.training:
            use_cache = True
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen2_Lora_DropForSequenceClassification(GenericForSequenceClassification, Qwen2_Lora_DropPreTrainedModel):
    pass


class Qwen2_Lora_DropForTokenClassification(GenericForTokenClassification, Qwen2_Lora_DropPreTrainedModel):
    pass


class Qwen2_Lora_DropForQuestionAnswering(GenericForQuestionAnswering, Qwen2_Lora_DropPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


__all__ = [
    "Qwen2_Lora_DropPreTrainedModel",
    "Qwen2_Lora_DropModel",
    "Qwen2_Lora_DropForCausalLM",
    "Qwen2_Lora_DropRMSNorm",
    "Qwen2_Lora_DropForSequenceClassification",
    "Qwen2_Lora_DropForTokenClassification",
    "Qwen2_Lora_DropForQuestionAnswering",
]
