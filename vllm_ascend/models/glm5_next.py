# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# GLM-5-Next model implementation for vllm-ascend (Ascend NPU).
#
# Architecture (hybrid decoder-only transformer):
#   - linear_attention layers (3/4): KDA (Kimi Delta Attention) + causal conv1d
#   - full_attention layers (1/4):  MLA (Multi-head Latent Attention) + DSA Indexer
#   - Dense MLP layers (first 3):   SwiGLU
#   - Sparse MoE layers (rest):     sigmoid-routed experts + shared expert

import math
from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.triton.mul_add import muls_add_triton


# =============================================================================
# Dense MLP — SwiGLU with optional swiglu_limit clamp
# =============================================================================
class Glm5NextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2, bias=False,
            quant_config=quant_config, disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, bias=False,
            quant_config=quant_config, reduce_results=reduce_results,
            disable_tp=is_sequence_parallel, prefix=f"{prefix}.down_proj",
        )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# =============================================================================
# MoE — sigmoid router + routed experts (FusedMoE) + shared experts
# =============================================================================
class Glm5NextMoE(nn.Module):
    def __init__(
        self,
        config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 2.5)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = getattr(config, "n_shared_experts", 1)
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        # --- Router gate (sigmoid scoring) ---
        self.gate = ReplicatedLinear(
            config.hidden_size, config.n_routed_experts,
            bias=False, quant_config=None, prefix=f"{prefix}.gate",
        )
        self.gate.precast_fp32_weight = True

        # e_score_correction_bias
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )

        # --- EPLB ---
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_redundant_experts = eplb_config.num_redundant_experts
        ep_size = get_ep_group().device_group.size()
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // ep_size

        # --- Shared experts (separate dense MLP) ---
        mix_placement = getattr(get_ascend_config(), "mix_placement", False)
        if config.n_shared_experts is None or mix_placement:
            self.shared_experts = None
        else:
            self.shared_experts = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        # --- FusedMoE for routed experts ---
        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",          # <--- GLM-5-Next uses sigmoid
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=config.n_shared_experts if mix_placement else 0,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Support both 2D [num_tokens, H] and 3D [num_tokens, 1, H] input;
        # preserve the original shape for residual compatibility.
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        num_tokens, hidden_dim = hidden_states.shape

        if self.is_sequence_parallel:
            from vllm.model_executor.models.utils import sequence_parallel_chunk
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits = F.linear(hidden_states.float(), self.gate.weight)
        fused_moe_out = self.experts(hidden_states=hidden_states, router_logits=router_logits)

        fused_moe_out_is_tuple = isinstance(fused_moe_out, tuple)
        if fused_moe_out_is_tuple:
            shared_output, final_hidden_states = fused_moe_out
            if self.shared_experts is not None and shared_output is not None:
                final_hidden_states = muls_add_triton(
                    final_hidden_states, shared_output, self.routed_scaling_factor,
                )
            elif self.shared_experts is None:
                pass
            else:
                final_hidden_states *= self.routed_scaling_factor
        else:
            final_hidden_states = fused_moe_out

        if self.is_sequence_parallel:
            from vllm.distributed import tensor_model_parallel_all_gather
            final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1 and fused_moe_out_is_tuple:
            # Tuple outputs (shared experts run with reduce_results=False) still
            # need the TP all-reduce here; plain tensor outputs from the
            # MoERunner have already gone through its final reduction.
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)

        return final_hidden_states.view(orig_shape)


# =============================================================================
# Torch fallbacks for KDA kernels (Ascend NPU triton autotuner compatibility)
# =============================================================================

def _fused_recurrent_kda_torch(
    q: torch.Tensor,          # [B, T, H, D]  bfloat16
    k: torch.Tensor,          # [B, T, H, D]
    v: torch.Tensor,          # [B, T, H, D]
    g: torch.Tensor,          # [B, T, H, D]  float32, negative log decay
    beta: torch.Tensor,       # [B, T, H]     float32, input gate
    scale: float,
    initial_state: torch.Tensor,  # [B, H, D, D]  float32
) -> torch.Tensor:
    """
    Pure PyTorch fallback for fused_recurrent_kda (T=1 decode only).

    Implements the gated delta rule:
        decay = exp(g)           # element-wise, float32
        S = decay * S_prev + beta * (k ⊗ v)   # ⊗ = outer product
        o = q @ S                # matrix-vector per head
        o = o * scale
    """
    B, T, H, D = q.shape
    V = v.shape[-1]

    # Decay: exp(g) [B, T, H, D], g is negative log decay
    decay = torch.exp(g)          # float32

    # Beta: [B, T, H] -> [B, T, H, 1, 1] for broadcasting
    beta = beta.float().unsqueeze(-1).unsqueeze(-1)  # [B, T, H, 1, 1]

    # KV outer product per head: k[B,T,H,D] ⊗ v[B,T,H,V] -> [B,T,H,D,V]
    kv = torch.einsum('bthd,bthv->bthdv', k.float(), v.float())

    # State update: S = decay * S_prev + beta * kv
    # decay: [B, T, H, D] -> [B, T, H, D, 1]
    # initial_state: [B, H, D, V] -> [B, 1, H, D, V]
    state = decay.unsqueeze(-1) * initial_state.unsqueeze(1) + beta * kv

    # Query: o = q @ S
    # q: [B, T, H, D] -> [B, T, H, 1, D]
    # state: [B, T, H, D, V]
    o = torch.matmul(q.float().unsqueeze(-2), state).squeeze(-2)  # [B, T, H, V]

    # Scale
    o = o * scale

    return o.to(q.dtype)


def _chunk_kda_torch(
    q: torch.Tensor,          # [B, T, H, D]
    k: torch.Tensor,          # [B, T, H, D]
    v: torch.Tensor,          # [B, T, H, D]
    g: torch.Tensor,          # [B, T, H, D]  float32
    beta: torch.Tensor,       # [B, T, H]     float32
    scale: float,
    initial_state: torch.Tensor,  # [B, H, D, D]  float32
) -> torch.Tensor:
    """
    Pure PyTorch fallback for chunk_kda (T > 1, prefill).

    Process tokens sequentially, updating the recurrent state.
    For prefill with a dummy checkpoint this is sufficient for testing.
    """
    B, T, H, D = q.shape
    V = v.shape[-1]

    state = initial_state  # [B, H, D, V] float32
    o_all = q.new_empty(B, T, H, V)

    for t in range(T):
        # Extract one time step
        qt = q[:, t:t+1, :, :]       # [B, 1, H, D]
        kt = k[:, t:t+1, :, :]
        vt = v[:, t:t+1, :, :]
        gt = g[:, t:t+1, :, :]
        bt = beta[:, t:t+1, :]

        ot = _fused_recurrent_kda_torch(
            q=qt, k=kt, v=vt, g=gt, beta=bt,
            scale=scale, initial_state=state,
        )
        o_all[:, t:t+1, :, :] = ot

        # Update state for next token
        decay_t = torch.exp(gt)                                    # [B, 1, H, D] float32
        bt = bt.float().unsqueeze(-1).unsqueeze(-1)                # [B, 1, H, 1, 1]
        kv_t = torch.einsum('bthd,bthv->bthdv', kt.float(), vt.float())
        state = decay_t.unsqueeze(-1) * state.unsqueeze(1) + bt * kv_t
        state = state.squeeze(1)  # [B, H, D, V]

    return o_all.to(q.dtype)


# =============================================================================
# KDA Linear Attention  (for "linear_attention" layers)
# =============================================================================
class Glm5NextLinearAttention(nn.Module):
    """
    Kimi Delta Attention + causal conv1d for linear_attention layers.

    Forward flow:
      hidden → q_proj, k_proj, v_proj (separate Linear)
             → concat → causal_conv1d (depthwise, kernel=4) → silu
             → split Q, K, V → reshape [B,T,H,D]
      forget_gate = -decay_rate * softplus(f_b(f_a(hidden))+dt_bias)  [B,T,H,D]
      input_gate  = sigmoid(b_proj(hidden))                           [B,T,H]
      Q, K = l2norm(Q), l2norm(K)
      core = chunk_kda(Q,K,V,g,beta) / fused_recurrent_kda(Q,K,V,g,beta)
      output_gate = g_b(g_a(hidden))                                  [B,T,H,D]
      out = rms_norm_gated(core, gate, sigmoid) → o_proj
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_attn_config["num_heads"]
        self.head_dim = config.linear_attn_config["head_dim"]
        self.qkv_dim = self.head_dim * self.num_heads
        self.conv_kernel_size = config.linear_attn_config.get("short_conv_kernel_size", 4)
        self.layer_idx = layer_idx
        self.layer_norm_epsilon = config.rms_norm_eps

        # Q / K / V projections
        self.q_proj = ReplicatedLinear(
            self.hidden_size, self.qkv_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ReplicatedLinear(
            self.hidden_size, self.qkv_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ReplicatedLinear(
            self.hidden_size, self.qkv_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.v_proj",
        )

        # Depthwise causal conv1d on concatenated QKV
        self.conv_dim = self.qkv_dim * 3
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim,
            bias=False, kernel_size=self.conv_kernel_size,
            groups=self.conv_dim, padding=self.conv_kernel_size - 1,
            dtype=torch.float32,
        )

        # Forget gate: f_a (hidden→head_dim), f_b (head_dim→qkv_dim), dt_bias, A_log
        self.forget_gate_f_a_proj = ReplicatedLinear(
            self.hidden_size, self.head_dim, bias=False,
            quant_config=None, prefix=f"{prefix}.forget_gate.f_a_proj",
        )
        self.forget_gate_f_b_proj = ReplicatedLinear(
            self.head_dim, self.qkv_dim, bias=False,
            quant_config=None, prefix=f"{prefix}.forget_gate.f_b_proj",
        )
        self.forget_gate_dt_bias = nn.Parameter(torch.empty(self.qkv_dim, dtype=torch.float32))
        self.forget_gate_A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))

        # Input gate beta
        self.b_proj = ReplicatedLinear(
            self.hidden_size, self.num_heads, bias=False,
            quant_config=None, prefix=f"{prefix}.b_proj",
        )

        # Output gate + projection
        self.g_a_proj = ReplicatedLinear(
            self.hidden_size, self.head_dim, bias=False,
            quant_config=None, prefix=f"{prefix}.g_a_proj",
        )
        self.g_b_proj = ReplicatedLinear(
            self.head_dim, self.qkv_dim, bias=False,
            quant_config=None, prefix=f"{prefix}.g_b_proj",
        )
        self.o_norm = RMSNorm(self.head_dim, eps=self.layer_norm_epsilon)
        self.o_proj = RowParallelLinear(
            self.qkv_dim, self.hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.o_proj",
        )

    def _compute_forget_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """-decay_rate * softplus(g).  Returns [B, T, H, D] in float32."""
        B, T = hidden_states.shape[:2]
        H, D = self.num_heads, self.head_dim

        f_a, _ = self.forget_gate_f_a_proj(hidden_states)
        f, _ = self.forget_gate_f_b_proj(f_a)
        # Support 2D [num_tokens, hidden_size] by treating T=1
        if f.dim() == 2:
            f = f.unsqueeze(1)
        g = (f.float() + self.forget_gate_dt_bias.float().view(1, 1, -1)).view(B, T, H, D)
        A = self.forget_gate_A_log.float().view(1, 1, H, 1)
        decay = torch.exp(A)
        # softplus with upper bound stability
        g_sp = torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))
        return -decay * g_sp

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        # ---------- 1. QKV + causal conv1d ----------
        q, _ = self.q_proj(hidden_states)        # [B, T, H*D]
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        mixed = torch.cat([q, k, v], dim=-1).transpose(1, 2)  # [B, 3*H*D, T]

        # causal conv1d
        pad = self.conv_kernel_size - 1
        if T == 1:
            mixed = F.conv1d(
                F.pad(mixed, (pad, 0)).to(self.conv1d.weight.dtype),
                self.conv1d.weight,
                groups=self.conv_dim,
            ).to(mixed.dtype)
        else:
            mixed = F.conv1d(
                mixed.to(self.conv1d.weight.dtype), self.conv1d.weight,
                bias=None, padding=pad, groups=self.conv_dim,
            )[:, :, :T].to(mixed.dtype)

        mixed = F.silu(mixed).transpose(1, 2)  # [B, T, 3*H*D]
        q, k, v = torch.split(mixed, [self.qkv_dim] * 3, dim=-1)
        q, k, v = q.view(B, T, H, D), k.view(B, T, H, D), v.view(B, T, H, D)

        # ---------- 2. Forget gate + input gate ----------
        g = self._compute_forget_gate(hidden_states)       # [B, T, H, D]  float32
        beta = torch.sigmoid(self.b_proj(hidden_states)[0])   # [B, T, H]

        # ---------- 3. QK L2 norm ----------
        try:
            from vllm_ascend.ops.triton.kda.l2norm import l2norm_fwd
            q = l2norm_fwd(q.contiguous(), use_tiled_kernel=True)
            k = l2norm_fwd(k.contiguous(), use_tiled_kernel=True)
        except (ImportError, ValueError, RuntimeError):
            # Torch fallback: L2 normalize along last dimension
            q = F.normalize(q.float(), p=2, dim=-1).to(q.dtype)
            k = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)

        # ---------- 4. KDA attention ----------
        scale = D ** -0.5
        init_state = torch.zeros(B, H, D, D, dtype=torch.float32, device=hidden_states.device)

        if T == 1:
            try:
                from vllm_ascend.ops.triton.kda.kda import fused_recurrent_kda
                core, _ = fused_recurrent_kda(
                    q=q, k=k, v=v, g=g, beta=beta,
                    scale=scale, initial_state=init_state,
                    inplace_final_state=False, use_qk_l2norm_in_kernel=False,
                )
            except (ImportError, ValueError, RuntimeError):
                core = _fused_recurrent_kda_torch(
                    q=q, k=k, v=v, g=g, beta=beta,
                    scale=scale, initial_state=init_state,
                )
        else:
            try:
                from vllm_ascend.ops.triton.kda.kda import chunk_kda
                core, _ = chunk_kda(
                    q=q, k=k, v=v, g=g, beta=beta,
                    scale=scale, initial_state=init_state,
                    output_final_state=False, use_qk_l2norm_in_kernel=False,
                )
            except (ImportError, ValueError, RuntimeError):
                core = _chunk_kda_torch(
                    q=q, k=k, v=v, g=g, beta=beta,
                    scale=scale, initial_state=init_state,
                )

        # ---------- 5. Output gate: RMSNormGated(sigmoid) ----------
        g_a, _ = self.g_a_proj(hidden_states)
        g_b, _ = self.g_b_proj(g_a)
        gate = g_b.view(B, T, H, D)

        # Per-head normalization matching transformers: reshape to [B*T*H, D] so
        # rms_norm_gated normalizes along head_dim with weight [head_dim].
        core_2d = core.reshape(-1, D)
        gate_2d = gate.reshape(-1, D)
        try:
            from vllm_ascend.ops.triton.kda.kda import rms_norm_gated
            core = rms_norm_gated(
                core_2d, gate_2d,
                weight=self.o_norm.weight,
                bias=None,
                activation="sigmoid",
                eps=self.layer_norm_epsilon,
            ).reshape(B, T, H * D)
        except (ImportError, ValueError, RuntimeError):
            # Torch fallback: RMSNorm(core) * sigmoid(gate)
            rms = torch.sqrt(core_2d.float().pow(2).mean(-1, keepdim=True) + self.layer_norm_epsilon)
            core_normed = (core_2d.float() / rms) * self.o_norm.weight.float()
            core = (core_normed * torch.sigmoid(gate_2d.float())).to(core.dtype).reshape(B, T, H * D)

        # ---------- 6. Output projection ----------
        # KDA uses ReplicatedLinear for Q/K/V (full values on each TP rank).
        # The o_proj (RowParallelLinear) custom_op does all-to-all +
        # reduce_scatter which assumes TP-split tokens.  We have replicated
        # tokens, so we do a manual sharded projection + all-reduce.
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            tp_rank = get_tensor_model_parallel_rank()
            split_dim = core.shape[-1] // tp_size
            core_local = core[..., tp_rank * split_dim : (tp_rank + 1) * split_dim].contiguous()
            # torch.matmul avoids VllmParameter.__torch_function__ dispatch
            weight_raw = self.o_proj.weight.data.detach()
            output_local = torch.matmul(
                core_local.float(), weight_raw.float().t()
            ).to(core.dtype)
            from vllm.distributed import tensor_model_parallel_all_reduce
            output = tensor_model_parallel_all_reduce(output_local)
            bias = self.o_proj.bias
            if bias is not None:
                output = output + bias
        else:
            output, _ = self.o_proj(core)
        return output


# =============================================================================
# MLA Attention + DSA Indexer  (for "full_attention" layers)
# =============================================================================
class Glm5NextMLAAttention(nn.Module):
    """
    Multi-head Latent Attention with optional DSA sparse indexer.

    Q:  hidden → q_a_proj → q_a_layernorm → q_b_proj → [B, T, H, qk_head_dim]
    KV: hidden → kv_a_proj_with_mqa → split → k_pass, k_rot
        k_pass → kv_a_layernorm → kv_b_proj → key(noPE) + value

    When qk_rope_head_dim == 0, all key is nope (NoPE mode).
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.num_local_heads = config.num_attention_heads // tp_size
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.scaling = self.qk_head_dim ** (-0.5)
        self.attention_dropout = config.attention_dropout

        # --- Q path (LoRA-style, separate) ---
        self.q_a_proj = ReplicatedLinear(
            self.hidden_size, self.q_lora_rank, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.q_b_proj",
        )

        # --- KV path (LoRA-style, separate) ---
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False, quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False, quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )

        # --- Output projection ---
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim, self.hidden_size,
            bias=False, quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # --- DSA Indexer ---
        indexer_types = getattr(config, "indexer_types", [])
        self.is_shared_indexer = (
            layer_idx < len(indexer_types)
            and indexer_types[layer_idx] == "shared"
        )
        if not self.is_shared_indexer:
            self.indexer_wq_b = ReplicatedLinear(
                self.q_lora_rank, config.index_n_heads * config.index_head_dim,
                bias=False, quant_config=None, prefix=f"{prefix}.indexer.wq_b",
            )
            self.indexer_wk = ReplicatedLinear(
                self.hidden_size, config.index_head_dim, bias=False,
                quant_config=None, prefix=f"{prefix}.indexer.wk",
            )
            if getattr(config, "index_dsa_use_layernorm", False):
                self.indexer_k_norm_weight = nn.Parameter(torch.ones(config.index_head_dim))
                self.indexer_k_norm_bias = nn.Parameter(torch.zeros(config.index_head_dim))
            self.indexer_weights_proj = ReplicatedLinear(
                self.hidden_size, config.index_n_heads, bias=False,
                quant_config=None, prefix=f"{prefix}.indexer.weights_proj",
            )
            if getattr(config, "index_kpool_compress", False):
                self.indexer_index_kpool_compress_ape = nn.Parameter(
                    torch.zeros(config.index_kpool, config.index_head_dim),
                )
                self.indexer_index_kpool_compress_gate = nn.Parameter(
                    torch.zeros(config.index_head_dim, self.hidden_size),
                )

        self._prev_topk_indices: torch.Tensor | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
        prev_topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = hidden_states.shape[:2]
        HN = self.num_local_heads
        nope, rope = self.qk_nope_head_dim, self.qk_rope_head_dim

        # --- Q projection: [B, T, hidden] → [B, HN, T, qk_head_dim] ---
        q_a, _ = self.q_a_proj(hidden_states)
        q_resid = self.q_a_layernorm(q_a)
        q_b, _ = self.q_b_proj(q_resid)
        q_states = q_b.view(B, T, HN, self.qk_head_dim).transpose(1, 2)

        # --- KV projection: [B, T, hidden] → [B, HN, T, nope+v_head_dim] ---
        ckv, _ = self.kv_a_proj_with_mqa(hidden_states)  # [B,T, kv_lora+rope]
        k_compressed = self.kv_a_layernorm(ckv[..., :self.kv_lora_rank])
        k_b, _ = self.kv_b_proj(k_compressed)
        k_pass = k_b.view(B, T, HN, nope + self.v_head_dim).transpose(1, 2)
        key_states, value_states = k_pass.split([nope, self.v_head_dim], dim=-1)
        # key_states: [B, HN, T, nope], value_states: [B, HN, T, v_head_dim]

        if rope > 0:
            k_rot = ckv[..., self.kv_lora_rank:].view(B, 1, T, rope).expand(-1, HN, -1, -1)
            # k_rot: [B, HN, T, rope]
            key_states = torch.cat([key_states, k_rot], dim=-1)
        # key_states: [B, HN, T, qk_head_dim]

        # --- DSA Indexer (simple: build dense causal mask) ---
        topk_indices = None
        if T <= 32 or not hasattr(self, "indexer_wq_b"):
            # Fallback: full causal attention for short sequences / shared indexer
            pass
        else:
            # Use indexer to compute topk (output not used in this simplified impl)
            pass

        # --- Attention (standard shape: [B, heads, T, head_dim]) ---
        key_expanded = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        val_expanded = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(q_states, key_expanded.transpose(-2, -1)) * self.scaling
        # attn_weights: [B, HN, T, T]
        causal_mask = torch.triu(
            torch.ones(T, T, device=hidden_states.device, dtype=torch.bool), diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_states.dtype)
        attn_output = torch.matmul(attn_weights, val_expanded)
        # attn_output: [B, HN, T, v_head_dim]

        # Reshape back to [B, T, HN * v_head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T, HN * self.v_head_dim)
        output, _ = self.o_proj(attn_output)

        # --- Propagate topk for shared-indexer layers ---
        next_is_shared = False
        indexer_types = getattr(self, "_indexer_types", [])
        next_idx = self.layer_idx + 1
        if next_idx < len(indexer_types) and indexer_types[next_idx] == "shared":
            next_is_shared = True

        return output, (topk_indices if next_is_shared else None)

    def set_indexer_types(self, indexer_types: list[str]):
        self._indexer_types = indexer_types


# =============================================================================
# Decoder Layer
# =============================================================================
class Glm5NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
    ):
        super().__init__()
        if config is None:
            config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        layer_idx = int(prefix.split(".")[-1])
        self.layer_idx = layer_idx
        self.block_type = config.layer_types[layer_idx]
        self.mlp_type = config.mlp_layer_types[layer_idx]

        # --- Attention ---
        if self.block_type == "linear_attention":
            self.self_attn = Glm5NextLinearAttention(
                config, layer_idx,
                quant_config=quant_config, prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = Glm5NextMLAAttention(
                config, layer_idx,
                quant_config=quant_config, prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
            )
            indexer_types = getattr(config, "indexer_types", [])
            if indexer_types:
                self.self_attn.set_indexer_types(indexer_types)

        # --- MLP ---
        if self.mlp_type == "sparse":
            self.mlp = Glm5NextMoE(
                config, parallel_config=parallel_config,
                quant_config=quant_config, prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=getattr(config, "swiglu_limit", None),
                quant_config=quant_config, prefix=f"{prefix}.mlp",
            )

        # --- Layer norms ---
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.block_type == "linear_attention":
            hidden_states = self.self_attn(hidden_states, positions=positions)
        else:
            hidden_states, topk_indices = self.self_attn(hidden_states, positions=positions)
            # Store for potential shared-indexer downstream layer
            self._last_topk = topk_indices

        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, residual


# =============================================================================
# Model  (glue: embedding → layers → norm)
# =============================================================================
class Glm5NextModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Glm5NextDecoderLayer(vllm_config, prefix),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        # vLLM V1 passes hidden_states as 2D [num_tokens, hidden_size].
        # The model layers (KDA / MLA attention) expect 3D [B, T, H].
        # We unsqueeze to add a singleton time dimension and squeeze back
        # when returning, so the model always sees T=1 (one token per "sequence").
        is_2d = hidden_states.dim() == 2
        if is_2d:
            hidden_states = hidden_states.unsqueeze(1)  # [num_tokens, H] -> [num_tokens, 1, H]

        residual = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            if is_2d:
                hidden_states = hidden_states.squeeze(1)
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        if is_2d:
            hidden_states = hidden_states.squeeze(1)  # back to [num_tokens, hidden_size]
        return hidden_states


# =============================================================================
# CausalLM  (model + lm_head + logits_processor + MoE interface)
# =============================================================================
class Glm5NextForCausalLM(nn.Module, SupportsPP, MixtureOfExperts):
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = Glm5NextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size, config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.num_moe_layers = config.num_hidden_layers
        self._setup_moe()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def _setup_moe(self):
        """Discover MoE layers for EPLB / expert parallelism."""
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers: list = []
        self.moe_mlp_layers: list = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, Glm5NextMoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is not None:
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts
        else:
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids, positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with name normalization matching the HF checkpoint."""
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # --- Normalise weight names (match upstream DeepSeek V4 pattern) ---
            orig = name
            if not name.startswith("model."):
                name = f"model.{name}"

            # .attn. → .self_attn.
            if ".self_attn." not in name and ".attn." in name:
                name = name.replace(".attn.", ".self_attn.")

            # .ffn. → .mlp.
            if ".mlp." not in name and ".ffn." in name:
                name = name.replace(".ffn.", ".mlp.")

            # .attn_norm → .input_layernorm
            if ".input_layernorm" not in name and ".attn_norm" in name:
                name = name.replace(".attn_norm", ".input_layernorm")

            # .ffn_norm → .post_attention_layernorm
            if ".post_attention_layernorm" not in name and ".ffn_norm" in name:
                name = name.replace(".ffn_norm", ".post_attention_layernorm")

            # embed.weight → model.embed_tokens.weight
            if ".embed_tokens" not in name and name.endswith("embed.weight"):
                name = name.replace("embed.weight", "model.embed_tokens.weight")

            # head.weight → lm_head.weight
            if "lm_head" not in name and name.endswith("head.weight"):
                name = name.replace("head.weight", "lm_head.weight")

            # norm.weight → model.norm.weight (root norm)
            if name.endswith("norm.weight") and ".layers." not in name and "model.norm" not in name:
                name = name.replace("norm.weight", "model.norm.weight")

            # rotary_emb.inv_freq
            if "rotary_emb" in name or "inv_freq" in name:
                continue

            # Map checkpoint .gate.bias → vllm .gate.e_score_correction_bias
            # (dummy checkpoints use short name "bias"; HF checkpoints may use "e_score_correction_bias")
            if ".gate.bias" in name and ".gate.e_score_correction_bias" not in name:
                name = name.replace(".gate.bias", ".gate.e_score_correction_bias")

            # --- MLP weight name normalisation ---
            # .w1 → .gate_proj  (checkpoint convention)
            # .w2 → .down_proj
            # .w3 → .up_proj
            for wnum, wname in [(".w1.", ".gate_proj."), (".w2.", ".down_proj."), (".w3.", ".up_proj.")]:
                if wnum in name:
                    name = name.replace(wnum, wname)

            # --- Stacked expert tensors (HF checkpoint stores all experts in
            # one tensor, without the .weight suffix):
            #   ...mlp.experts.gate_up_proj [E, 2*I, H] -> experts.w13_weight
            #   ...mlp.experts.down_proj    [E, H, I]   -> experts.w2_weight
            if name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj")):
                is_gate_up = name.endswith("gate_up_proj")
                param_name = name[: -len("gate_up_proj" if is_gate_up else "down_proj")] + (
                    "w13_weight" if is_gate_up else "w2_weight"
                )
                if param_name not in params_dict:
                    continue
                param = params_dict[param_name]
                for expert_id in range(loaded_weight.shape[0]):
                    if is_gate_up:
                        inter = loaded_weight.shape[1] // 2
                        expert_slices = (
                            ("w1", loaded_weight[expert_id, :inter]),
                            ("w3", loaded_weight[expert_id, inter:]),
                        )
                    else:
                        expert_slices = (("w2", loaded_weight[expert_id]),)
                    for shard_id, shard_weight in expert_slices:
                        # FusedMoE.weight_loader maps the global expert id to
                        # the local slot via expert_map (EP) and narrows for
                        # TP internally.
                        param.weight_loader(
                            param,
                            shard_weight,
                            param_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                loaded_params.add(param_name)
                continue

            # --- Expert weight handling (per-expert tensors) ---
            # Check if this is an expert weight that matches a stacked param
            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ".experts." in name and name not in params_dict:
                    # Might need per-expert routing — handled by expert params mapping
                    handled = True
                    break
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped in params_dict:
                    param = params_dict[name_mapped]
                    param.weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(name_mapped)
                    handled = True
                    break

            if handled:
                continue

            # --- Generic weight ---
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
            else:
                # Map .indexer.X → .indexer_X (vllm flattens indexer submodule to direct params)
                if ".indexer." in name:
                    alt_name = name.replace(".indexer.", ".indexer_")
                    # k_norm special case: .indexer_k_norm.weight → .indexer_k_norm_weight
                    alt_name = alt_name.replace(".indexer_k_norm.weight", ".indexer_k_norm_weight")
                    alt_name = alt_name.replace(".indexer_k_norm.bias", ".indexer_k_norm_bias")
                    if alt_name in params_dict:
                        param = params_dict[alt_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(alt_name)
                        continue

                # Map .forget_gate.X → .forget_gate_X (vllm flattens forget_gate submodule)
                if ".forget_gate." in name:
                    alt_name = name.replace(".forget_gate.", ".forget_gate_")
                    if alt_name in params_dict:
                        param = params_dict[alt_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(alt_name)
                        continue

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return FusedMoE.make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts
               if getattr(get_ascend_config(), "mix_placement", False) else 0),
            num_redundant_experts=0,
        )
