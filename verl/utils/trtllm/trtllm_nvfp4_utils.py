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

# FP4 (E2M1) lookup tables
# Adapted from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant
_e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])


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
    # Reshape, scale, cast, pack
    x = input.view((*tuple(input.shape[:-1]), -1, block_size))
    scaled = x / (weights_scaling_factor.to(torch.float32) * weights_scaling_factor_2).unsqueeze(-1)
    scaled = scaled.view((*tuple(scaled.shape[:-2]), -1))
    q = _cast_fp4(scaled)
    packed = (q[..., 1::2] << 4) | q[..., 0::2]
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
        yield (key, packed_weight)
        yield (key + "_scale", q_per_block_scale)
        yield (key + "_scale_2", weights_scaling_factor_2)
        del v_hp, packed_weight, q_per_block_scale

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
