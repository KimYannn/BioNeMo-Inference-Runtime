# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Opt-in GB10 pair-bias attention with an SDPA fallback."""

from functools import lru_cache
from importlib.util import find_spec
from threading import Lock

import torch

from bionemo_ir.dsl_kernels.cute_cache import CuteKernelCache

from ..interface import AttentionMetadata
from .sdpa import SDPAPairwiseAttention

_HEADS = 16
_HEAD_DIM = 48
_SCALE = 1 / (48**0.5)
_compiled: dict[tuple[int, int, int, int], object] = {}
_compile_lock = Lock()


@lru_cache(maxsize=1)
def _compiler_available() -> bool:
    """The compiler is optional in released BioIR wheels."""
    return find_spec("cutlass") is not None and find_spec("tvm_ffi") is not None


def _supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, biases: list[torch.Tensor] | None) -> bool:
    """Match the measured BF16 packed-QGKV, sample-shared pair-bias layout."""
    if torch.compiler.is_compiling() or q.device.type != "cuda" or q.dtype != torch.bfloat16 or q.ndim != 4:
        return False
    if torch.cuda.get_device_capability(q.device) != (12, 1) or biases is None or len(biases) != 2:
        return False
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v, *biases)):
        return False
    batch, samples, seq, channels = q.shape
    if channels != _HEADS * _HEAD_DIM or not (1 <= batch <= 2 and 1 <= samples <= 5 and 31 <= seq <= 1762):
        return False
    strides = (samples * seq * channels * 4, seq * channels * 4, channels * 4, 1)
    if any(
        x.device != q.device or x.dtype != q.dtype or x.shape != q.shape or x.stride() != strides for x in (q, k, v)
    ):
        return False
    mask, pair = biases
    if (
        mask.device != q.device
        or mask.dtype != torch.float32
        or mask.shape != (batch, 1, 1, 1, seq)
        or not mask.is_contiguous()
        or pair.device != q.device
        or pair.dtype != q.dtype
        or pair.shape != (batch, 1, _HEADS, seq, seq)
        or not pair.is_contiguous()
    ):
        return False
    return (
        all(x.data_ptr() % 16 == 0 for x in (q, k, v, mask, pair))
        and not CuteKernelCache.force_cubin()
        and _compiler_available()
    )


def _run(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, pair: torch.Tensor
) -> torch.Tensor | None:
    """Compile once per device and exact shape; a cold CUDA graph uses SDPA."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from bionemo_ir.dsl_kernels.gb10_pair_bias_cute import FlashAttentionForwardAmpere

    batch, samples, seq, _ = q.shape
    key = (q.device.index, batch, samples, seq)
    q4, k4, v4 = (x.view(batch * samples, seq, _HEADS, _HEAD_DIM) for x in (q, k, v))
    pair4 = pair.view(batch, _HEADS, seq, seq)
    mask2 = mask.view(batch, seq)
    output = torch.empty((batch, samples, seq, _HEADS, _HEAD_DIM), dtype=q.dtype, device=q.device)
    out4 = output.view(batch * samples, seq, _HEADS, _HEAD_DIM)
    with torch.cuda.device(q.device):
        stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        executable = _compiled.get(key)
        if executable is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            with _compile_lock:
                executable = _compiled.get(key)
                if executable is None:
                    tensors = [
                        from_dlpack(x, assumed_align=16, enable_tvm_ffi=True) for x in (q4, k4, v4, pair4, mask2, out4)
                    ]
                    use_bc32 = seq <= 256 and samples > 1
                    query_tile = 64 if samples > 1 or seq <= 512 else 128
                    kernel = FlashAttentionForwardAmpere(
                        _HEAD_DIM,
                        samples,
                        pad_to_64=use_bc32,
                        m_block_size=query_tile,
                        n_block_size=32 if use_bc32 else 64,
                        num_threads=128,
                        is_causal=False,
                    )
                    executable = cute.compile(kernel, *tensors, _SCALE, stream, options="--enable-tvm-ffi")
                    _compiled[key] = executable
        executable(q4, k4, v4, pair4, mask2, out4, _SCALE, stream)
    return output


class GB10PairwiseAttention(SDPAPairwiseAttention):
    """Run the GB10 kernel for its supported inference layout, otherwise SDPA.

    This source-only backend is opt-in. It requires the CuTeDSL compiler and
    TVM FFI from the development dependency group, and compiles each new
    shape on first use. The regular SDPA backend remains the default.
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: list[torch.Tensor] | None = None,
        metadata: AttentionMetadata | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.num_heads == _HEADS and self.head_dim == _HEAD_DIM and _supported(q, k, v, biases):
            result = _run(q, k, v, biases[0], biases[1])
            if result is not None:
                return result
        return super().forward(q, k, v, biases=biases, metadata=metadata, **kwargs)
