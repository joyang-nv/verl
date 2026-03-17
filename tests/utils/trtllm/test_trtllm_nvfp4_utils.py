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

import asyncio
import json
import math
import os
import time
from unittest.mock import patch

import pytest
import ray
import torch
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.trtllm.trtllm_nvfp4_utils import TRTLLMNVFP4QuantizerHelper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# NVFP4 quantisation config passed to TRT-LLM as model_kwargs.
# This tells TRT-LLM to instantiate NVFP4 Linear modules so they accept
# {weight, weight_scale, weight_scale_2} on reload.
_NVFP4_QUANT_CONFIG = {
    "quant_method": "nvfp4",
    "activation_scheme": "dynamic",
    "group_size": 16,
}

_GROUP_SIZE = 16  # must match group_size above

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qwen3_model_path():
    path = os.environ.get("QWEN3_MODEL_PATH")
    if not path:
        pytest.skip("Set QWEN3_MODEL_PATH to a local Qwen3 checkpoint to run this test.")
    if not os.path.isdir(path):
        pytest.skip(f"QWEN3_MODEL_PATH={path!r} is not a directory.")
    return path


@pytest.fixture(scope="module")
def ray_context():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_hf_weights(model_path: str) -> dict[str, torch.Tensor]:
    """Return the HF state-dict on CPU (bf16)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    del model
    torch.cuda.empty_cache()
    return weights


def _quantize_weights(raw_weights: dict) -> dict[str, torch.Tensor]:
    """Run NVFP4 quantisation; mock dist.get_rank for non-distributed context."""
    helper = TRTLLMNVFP4QuantizerHelper({"group_size": _GROUP_SIZE})
    with patch("torch.distributed.get_rank", return_value=0):
        result = dict(helper.quantize_weights(raw_weights.items()))
    return result


def _pack_ipc_handles(
    weights: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, list]:
    """Move tensors to *device*, pack them as IPC handles, and return
    ``{device_uuid: [(name, handle), ...]}``.

    The IPC-handle format is what TRT-LLM's WorkerExtension.update_weights()
    expects when called via collective_rpc.
    """
    from verl.workers.rollout.trtllm_rollout.trtllm_rollout import get_device_uuid

    gpu_weights = {k: v.to(device).contiguous() for k, v in weights.items()}
    handles = [(name, reduce_tensor(tensor)) for name, tensor in gpu_weights.items()]
    del gpu_weights
    torch.cuda.synchronize(device)

    uuid = get_device_uuid(device.index)
    return {uuid: handles}


async def _build_async_llm(model_path: str):
    """Construct an AsyncLLM with dummy weights and NVFP4 quant config.

    AsyncLLM is an awaitable: its internal executor, event-loop bridge and
    Ray workers are only fully initialised after the await completes.
    """
    from tensorrt_llm import AsyncLLM
    from tensorrt_llm.llmapi import KvCacheConfig

    return await AsyncLLM(
        model=model_path,
        backend="pytorch",
        dtype="bfloat16",
        load_format="dummy",
        model_kwargs={"quantization_config": _NVFP4_QUANT_CONFIG},
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_batch_size=4,
        max_seq_len=256,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.5),
        orchestrator_type="ray",
        # Use TRT-LLM's built-in WorkerExtension which implements update_weights.
        ray_worker_extension_cls="tensorrt_llm.llmapi.rlhf_utils.WorkerExtension",
        allreduce_strategy="NCCL",
        force_dynamic_quantization=True,
    )


async def _reload_weights(llm, ipc_handles: dict) -> None:
    """Execute the standard two-step weight-reload flow.

    Step 1: pass ipc_handles  → WorkerExtension calls pre_reload_weights()
                                 then model_loader.reload(model, weights).
    Step 2: pass None         → WorkerExtension calls process_weights_after_loading()
                                 and finalises (resets prefix cache, etc.).
    """
    await llm.collective_rpc("update_weights", args=(ipc_handles,))
    await llm.collective_rpc("update_weights", args=(None,))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestNVFP4TRTLLMIntegration:
    """Smoke tests that exercise the full NVFP4 quantise → TRT-LLM reload pipeline."""

    def test_inference_produces_valid_output(self, qwen3_model_path, ray_context):
        """Full pipeline: quantise weights, load into TRT-LLM, reload, generate."""
        from tensorrt_llm.llmapi import SamplingParams
        from transformers import AutoTokenizer

        # 1. Quantise weights
        raw_weights = _load_hf_weights(qwen3_model_path)
        quantized_weights = _quantize_weights(raw_weights)

        # 2. Tokenise prompt
        tokenizer = AutoTokenizer.from_pretrained(qwen3_model_path, trust_remote_code=True)
        prompt = "The president of the United States is"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()

        # 3. Pack weights as IPC handles for the GPU worker
        device = torch.device("cuda:0")
        ipc_handles = _pack_ipc_handles(quantized_weights, device)

        async def run():
            llm = await _build_async_llm(qwen3_model_path)
            await _reload_weights(llm, ipc_handles)
            sampling_params = SamplingParams(
                max_tokens=16,
                temperature=0.0,
                logprobs=1,
            )
            return await llm.generate_async(
                inputs=input_ids,
                sampling_params=sampling_params,
            )

        output = asyncio.run(run())

        assert output is not None, "generate_async returned None"
        assert output.outputs, "No output sequences returned"
        token_ids = output.outputs[0].token_ids
        assert len(token_ids) > 0, "Generated sequence is empty"
        print(output.outputs[0].text)

        # Validate logprobs if returned
        if output.outputs[0].logprobs:
            for step_logprobs in output.outputs[0].logprobs:
                for lp in step_logprobs.values():
                    score = lp.logprob if hasattr(lp, "logprob") else float(lp)
                    assert not math.isnan(score), f"NaN logprob at token step"
                    assert not math.isinf(score), f"Inf logprob at token step"
