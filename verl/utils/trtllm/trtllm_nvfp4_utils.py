# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import logging
import os

import torch

from verl.utils.fp8_utils import FP8QuantizerHelper

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Check if Triton is available
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    logger.debug("Triton not available, FP4 Triton kernels will not be used")

# Environment variable to control Triton FP4 usage (set to "1" to disable)
_DISABLE_TRITON_FP4 = os.environ.get("VERL_DISABLE_TRITON_FP4", "0").lower() in ("1", "true", "yes")

# FP4 (E2M1) lookup tables (used by PyTorch fallback)
# Adapted from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant
_e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _nvfp4_get_weights_scaling_factor(
    input: torch.Tensor,
    block_size: int,
    weights_scaling_factor_2: torch.Tensor | None = None,
    keep_high_precision: bool = False,
):
    """Returns (fp8_per_block_scale, float32_per_tensor_scale)."""
    if weights_scaling_factor_2 is None:
        weights_scaling_factor_2 = input.abs().amax().float() / (6.0 * 448.0)

    [n, k] = input.shape[-2:]
    assert block_size != 0, "Block size is zero."
    assert k % block_size == 0, "Weight shape is not divisible by block size."

    input = input.reshape((*tuple(input.shape[:-2]), n, k // block_size, block_size))
    per_block_amax = input.abs().amax(dim=-1).float()
    per_block_scale = per_block_amax / 6.0
    q_per_block_scale = per_block_scale / weights_scaling_factor_2
    q_per_block_scale[per_block_scale == 0] = 1.0
    if not keep_high_precision:
        q_per_block_scale = q_per_block_scale.to(torch.float8_e4m3fn)
    return q_per_block_scale, weights_scaling_factor_2


# ---------------------------------------------------------------------------
# Triton kernel: fused scale + FP4 E2M1 cast + pack
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:

    @triton.jit
    def _scale_and_cast_fp4_pack_kernel(
        X,
        SF,
        SF2,
        OUT,
        stride_xm,
        stride_xk,
        stride_sfm,
        stride_sfn,
        stride_om,
        stride_ok,
        M,
        K,
        QUANT_BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K2: tl.constexpr,
    ):
        """Fused kernel: scale by per-block/per-tensor factors, cast to FP4 E2M1
        with round-to-nearest-even, and pack adjacent pairs into uint8.

        Each program processes a [BLOCK_M, BLOCK_K2*2] tile of input and
        produces a [BLOCK_M, BLOCK_K2] tile of packed uint8 output.
        """
        pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
        pid_k = tl.cast(tl.program_id(axis=1), tl.int64)

        off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_k2 = pid_k * BLOCK_K2 + tl.arange(0, BLOCK_K2)

        # Input column indices for even/odd elements of each packed pair
        off_k_even = off_k2 * 2
        off_k_odd = off_k2 * 2 + 1

        K2: tl.constexpr = K // 2
        mask_m = off_m < M
        mask_k = off_k2 < K2
        mask = mask_m[:, None] & mask_k[None, :]

        # Load even and odd input elements
        x_even = tl.load(
            X + off_m[:, None] * stride_xm + off_k_even[None, :] * stride_xk,
            mask=mask, other=0.0,
        ).to(tl.float32)
        x_odd = tl.load(
            X + off_m[:, None] * stride_xm + off_k_odd[None, :] * stride_xk,
            mask=mask, other=0.0,
        ).to(tl.float32)

        # Load per-block scale (adjacent pairs always share the same block)
        sf_col = off_k_even // QUANT_BLOCK_SIZE
        sf = tl.load(
            SF + off_m[:, None] * stride_sfm + sf_col[None, :] * stride_sfn,
            mask=mask, other=1.0,
        ).to(tl.float32)

        # Load per-tensor scale and compute inverse
        sf2 = tl.load(SF2).to(tl.float32)
        inv_scale = 1.0 / (sf * sf2)

        x_even = x_even * inv_scale
        x_odd = x_odd * inv_scale

        # --- FP4 E2M1 round-to-nearest-even via comparison chain ---
        # Bounds: [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
        # mask=0 (even ordinal target) → strict ">",
        # mask=1 (odd ordinal target)  → ">=" to round ties to even.

        # Even elements
        abs_even = tl.abs(x_even)
        sign_even = tl.where(x_even < 0, 8, 0)
        ord_even = (abs_even > 0.25).to(tl.int32)
        ord_even += (abs_even >= 0.75).to(tl.int32)
        ord_even += (abs_even > 1.25).to(tl.int32)
        ord_even += (abs_even >= 1.75).to(tl.int32)
        ord_even += (abs_even > 2.5).to(tl.int32)
        ord_even += (abs_even >= 3.5).to(tl.int32)
        ord_even += (abs_even > 5.0).to(tl.int32)
        fp4_even = sign_even + ord_even

        # Odd elements
        abs_odd = tl.abs(x_odd)
        sign_odd = tl.where(x_odd < 0, 8, 0)
        ord_odd = (abs_odd > 0.25).to(tl.int32)
        ord_odd += (abs_odd >= 0.75).to(tl.int32)
        ord_odd += (abs_odd > 1.25).to(tl.int32)
        ord_odd += (abs_odd >= 1.75).to(tl.int32)
        ord_odd += (abs_odd > 2.5).to(tl.int32)
        ord_odd += (abs_odd >= 3.5).to(tl.int32)
        ord_odd += (abs_odd > 5.0).to(tl.int32)
        fp4_odd = sign_odd + ord_odd

        # Pack: even in low nibble, odd in high nibble
        packed = ((fp4_odd << 4) | fp4_even).to(tl.uint8)

        tl.store(
            OUT + off_m[:, None] * stride_om + off_k2[None, :] * stride_ok,
            packed, mask=mask,
        )

    def _scale_and_cast_fp4_pack_triton(
        x: torch.Tensor,
        weights_scaling_factor: torch.Tensor,
        weights_scaling_factor_2: torch.Tensor,
        quant_block_size: int,
    ) -> torch.Tensor:
        """Triton-accelerated fused scale + FP4 E2M1 cast + pack.

        No intermediate full-size tensors are allocated — the kernel reads from
        the original weight, computes scaling / FP4 rounding / packing in
        registers, and writes the packed uint8 output directly.

        Args:
            x: Input weight tensor (arbitrary leading batch dims, last 2 dims are [M, K]).
            weights_scaling_factor: Per-block FP8 E4M3 scale, shape [*batch, M, K // quant_block_size].
            weights_scaling_factor_2: Per-tensor float32 scale (scalar tensor).
            quant_block_size: Quantization block size (typically 16).

        Returns:
            Packed uint8 tensor of shape [*batch, M, K // 2].
        """
        orig_shape = x.shape
        K = orig_shape[-1]
        assert K % 2 == 0, f"K ({K}) must be even for FP4 packing"

        # Flatten batch dims into M for the 2-D kernel
        M = x[..., 0, 0].numel() if x.dim() > 2 else x.shape[0]
        x_2d = x.reshape(M, K).contiguous()

        sf_2d = weights_scaling_factor.reshape(M, K // quant_block_size).to(torch.float32).contiguous()
        sf2_1d = weights_scaling_factor_2.to(torch.float32).contiguous().reshape(1)

        out = torch.empty(M, K // 2, device=x.device, dtype=torch.uint8)

        BLOCK_M = 64
        BLOCK_K2 = 64  # each program packs 128 input elements into 64 bytes

        grid = (ceil_div(M, BLOCK_M), ceil_div(K // 2, BLOCK_K2))

        _scale_and_cast_fp4_pack_kernel[grid](
            x_2d, sf_2d, sf2_1d, out,
            *x_2d.stride(),
            *sf_2d.stride(),
            *out.stride(),
            M, K,
            QUANT_BLOCK_SIZE=quant_block_size,
            BLOCK_M=BLOCK_M,
            BLOCK_K2=BLOCK_K2,
            num_warps=4,
            num_stages=2,
        )

        del sf_2d, sf2_1d
        # Restore batch dims
        out_shape = (*orig_shape[:-1], K // 2)
        return out.reshape(out_shape)


# ---------------------------------------------------------------------------
# PyTorch fallback for FP4 E2M1 cast (kept for non-Triton environments)
# ---------------------------------------------------------------------------
def _cast_fp4(weight: torch.Tensor) -> torch.Tensor:
    """Round-nearest to FP4 E2M1 and return as uint8 nibble values [0, 15]."""
    device = weight.device
    mask = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8, device=device)
    mask = mask.expand([*list(weight.shape), 7])

    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs()
    ord_ = torch.searchsorted(_e2m1_bounds.to(device), weight_abs, out_int32=True).to(torch.uint8)
    round_ = torch.any((weight_abs.unsqueeze(-1) == _e2m1_bounds.to(device)) * mask, dim=-1)
    return (sign_bit * 0b1000 + ord_ + round_).to(torch.uint8)


def _trtllm_quantize_nvfp4(
    input: torch.Tensor,
    block_size: int,
    weights_scaling_factor_2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight to packed FP4 E2M1.

    Returns:
        packed_weight: uint8 tensor of shape [M, K//2] (two FP4 nibbles per byte)
        fp8_per_block_scale: FP8 E4M3 per-block scale
    """
    weights_scaling_factor, weights_scaling_factor_2 = _nvfp4_get_weights_scaling_factor(
        input, block_size, weights_scaling_factor_2
    )

    if _TRITON_AVAILABLE and not _DISABLE_TRITON_FP4:
        # Triton path: fused scale + cast + pack in a single kernel.
        # No intermediate full-size tensors are allocated.
        packed = _scale_and_cast_fp4_pack_triton(
            input, weights_scaling_factor, weights_scaling_factor_2, block_size
        )
    else:
        # PyTorch fallback with in-place scaling to reduce peak memory.
        # Convert to float32 first (new tensor, not a view of input),
        # then use in-place div_ to avoid allocating a separate `scaled` tensor.
        orig_shape = input.shape
        x = input.float().reshape((*tuple(orig_shape[:-1]), -1, block_size))
        del input
        x.div_((weights_scaling_factor.to(torch.float32) * weights_scaling_factor_2).unsqueeze(-1))
        x = x.view((*tuple(x.shape[:-2]), -1))
        q = _cast_fp4(x)
        del x
        packed = (q[..., 1::2] << 4) | q[..., 0::2]
        del q

    return packed, weights_scaling_factor


# Fusion groups: projections that will be merged by TRT-LLM and must share scale_2.
#   - qkv: q_proj + k_proj + v_proj  -> fused as qkv_proj
#   - gate_up: gate_proj + up_proj    -> fused as gate_up_proj
_FUSION_GROUPS: tuple[tuple[str, ...], ...] = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
)

# Quick lookup: proj_type -> its fusion group tuple
_PROJ_TO_GROUP: dict[str, tuple[str, ...]] = {
    proj: group for group in _FUSION_GROUPS for proj in group
}


class TRTLLMNVFP4QuantizerHelper(FP8QuantizerHelper):

    def __init__(self, quant_config):
        super().__init__(quant_config)

    def _get_proj_info(self, param_name: str):
        """Return (proj_type, fusion_group) if param belongs to a fusion group, else (None, None)."""
        for proj, group in _PROJ_TO_GROUP.items():
            if proj in param_name:
                return proj, group
        return None, None

    def _get_layer_prefix(self, param_name: str, proj_type: str) -> str:
        """Strip the proj suffix to get the shared layer prefix."""
        idx = param_name.index(proj_type)
        return param_name[:idx].rstrip(".")

    def _quantize_nvfp4(self, key, v, group_size, dtype, weights_scaling_factor_2=None):
        """Quantize a single weight tensor to NVFP4 and yield (key, tensor) pairs.

        Yields:
          key              -> packed uint8 weight, shape [M, K//2]  (float4_e2m1x2 format)
          key + "_scale"   -> per-block FP8 E4M3 scale
          key + "_scale_2" -> global float32 per-tensor scale
        """
        v_hp = v.to(dtype)

        # If no global scale supplied, derive it from this tensor's own amax so
        # that _trtllm_quantize_nvfp4 receives a consistent scale_2.
        if weights_scaling_factor_2 is None:
            weights_scaling_factor_2 = v_hp.abs().amax().float() / (6.0 * 448.0)

        # _trtllm_quantize_nvfp4 returns (packed_uint8 [M, K//2], fp8_per_block_scale).
        packed_weight, q_per_block_scale = _trtllm_quantize_nvfp4(
            v_hp,
            block_size=group_size,
            weights_scaling_factor_2=weights_scaling_factor_2,
        )
        del v_hp
        yield (key, packed_weight)
        yield (key + "_scale", q_per_block_scale)
        yield (key + "_scale_2", weights_scaling_factor_2)
        del packed_weight, q_per_block_scale

    def _flush_fusion_group(self, group: tuple[str, ...], projs: dict, group_size: int, dtype):
        """Quantize a buffered fusion group with a unified scale_2 and yield results.

        A single scale_2 is computed from the joint amax of all tensors in the
        group so that the global scale is consistent after TRT-LLM fuses them.

        projs: {proj_type: (key, tensor)}
        """
        all_amaxes = [v.to(torch.float32).abs().amax() for _, v in projs.values()]
        unified_scale_2 = torch.stack(all_amaxes).amax() / (6.0 * 448.0)
        del all_amaxes

        for proj_type in group:
            if proj_type not in projs:
                continue
            key, v = projs[proj_type]
            try:
                yield from self._quantize_nvfp4(key, v, group_size, dtype,
                                                weights_scaling_factor_2=unified_scale_2)
            except Exception as e:
                logger.error(f"Failed to quantize {key}: {e}")
                yield (key, v)

        del unified_scale_2

    def quantize_weights(self, weights, dtype=torch.bfloat16):
        """NVFP4 quantization based on parameter name using a memory-efficient generator.

        Projections that TRT-LLM fuses together share a single scale_2 computed
        from their joint amax:
          - q_proj / k_proj / v_proj  -> fused as qkv_proj
          - gate_proj / up_proj       -> fused as gate_up_proj

        Args:
            weights: Generator or iterable of (name, tensor) pairs
            dtype: Data type for intermediate computation

        Yields:
            Tuples of (name, tensor) for each weight and its scales
        """
        if isinstance(self.quant_config, dict):
            group_size = self.quant_config.get("group_size", 16)
        else:
            group_size = getattr(self.quant_config, "group_size", 16)

        # Buffer keyed by (layer_prefix, group): {proj_type: (key, tensor)}
        fusion_buffer: dict[tuple, dict] = {}

        for k, v in weights:
            if not self.should_quantize_param(k):
                yield (k, v)
                continue

            proj_type, group = self._get_proj_info(k)

            if proj_type is not None:
                layer_prefix = self._get_layer_prefix(k, proj_type)
                buf_key = (layer_prefix, group)
                fusion_buffer.setdefault(buf_key, {})[proj_type] = (k, v)

                if all(p in fusion_buffer[buf_key] for p in group):
                    if torch.distributed.get_rank() == 0:
                        logger.debug(
                            f"Quantizing fused group {group} to NVFP4 (unified scale_2): {layer_prefix}"
                        )
                    yield from self._flush_fusion_group(
                        group, fusion_buffer.pop(buf_key), group_size, dtype
                    )
            else:
                # Non-fused weight: individual quantization
                try:
                    if torch.distributed.get_rank() == 0:
                        logger.debug(f"Quantizing to NVFP4 blockwise: {k}")
                    yield from self._quantize_nvfp4(k, v, group_size, dtype)
                except Exception as e:
                    logger.error(f"Failed to quantize {k}: {e}")
                    yield (k, v)

        # All fusion groups must be complete after iterating all weights
        for (layer_prefix, group), projs in fusion_buffer.items():
            assert False, (
                f"Incomplete fusion group {group} at {layer_prefix}: "
                f"found {list(projs.keys())}, expected {list(group)}."
            )
