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
"""Numerical and fallback checks for the opt-in GB10 pair-bias backend."""

import pytest
import torch

from bionemo_ir._torch.attention_backend.pairwise_attention import gb10
from bionemo_ir._torch.attention_backend.pairwise_attention.gb10 import (
    GB10PairwiseAttention,
    _compiler_available,
    _supported,
)
from bionemo_ir._torch.attention_backend.pairwise_attention.sdpa import SDPAPairwiseAttention

_GB10_AVAILABLE = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (12, 1)
    and _compiler_available()
    and not gb10.CuteKernelCache.force_cubin()
)
requires_gb10 = pytest.mark.skipif(not _GB10_AVAILABLE, reason="requires SM121 and the CuTeDSL compiler")


def _inputs(batch: int, samples: int, seq: int):
    packed = torch.randn(batch, samples, seq, 4 * 16 * 48, device="cuda", dtype=torch.bfloat16)
    q, _, k, v = packed.split(16 * 48, dim=-1)
    pair = torch.randn(batch, 1, 16, seq, seq, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(batch, 1, 1, 1, seq, device="cuda", dtype=torch.float32)
    return q, k, v, mask, pair


def test_cpu_fallback_preserves_gradients():
    q, k, v = (torch.randn(1, 2, 7, 16 * 48, requires_grad=True) for _ in range(3))
    biases = [torch.zeros(1, 1, 1, 1, 7), torch.randn(1, 1, 16, 7, 7)]
    backend = GB10PairwiseAttention(0, 16, 48)
    reference = SDPAPairwiseAttention(0, 16, 48)
    actual = backend.forward(q, k, v, biases)
    expected = reference.forward(q, k, v, biases)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.sum().backward()
    assert all(x.grad is not None for x in (q, k, v))


def test_compiled_cpu_fallback_matches_sdpa():
    q, k, v = (torch.randn(1, 1, 7, 16 * 48) for _ in range(3))
    biases = [torch.zeros(1, 1, 1, 1, 7), torch.randn(1, 1, 16, 7, 7)]
    backend = GB10PairwiseAttention(0, 16, 48)
    reference = SDPAPairwiseAttention(0, 16, 48)
    compiled = torch.compile(backend.forward, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(q, k, v, biases), reference.forward(q, k, v, biases), atol=0, rtol=0)


@requires_gb10
@pytest.mark.parametrize(
    ("batch", "samples", "seq"), [(1, 5, 31), (2, 2, 129), (1, 1, 203), (1, 5, 257), (1, 1, 575), (1, 5, 575)]
)
@pytest.mark.parametrize("mask_kind", ["none", "holes", "empty_finite", "empty_infinite", "trailing_infinite"])
def test_matches_sdpa_for_mask_patterns(batch: int, samples: int, seq: int, mask_kind: str):
    torch.manual_seed(42)
    q, k, v, mask, pair = _inputs(batch, samples, seq)
    if mask_kind == "holes":
        mask[..., 2::5] = -1e9
    elif mask_kind == "empty_finite":
        mask.fill_(-1e9)
    elif mask_kind == "empty_infinite":
        mask.fill_(-torch.inf)
    elif mask_kind == "trailing_infinite":
        mask[..., seq // 2 :] = -torch.inf

    backend = GB10PairwiseAttention(0, 16, 48)
    reference = SDPAPairwiseAttention(0, 16, 48)
    with torch.inference_mode():
        assert _supported(q, k, v, [mask, pair])
        actual = backend.forward(q, k, v, [mask, pair])
        expected = reference.forward(q, k, v, [mask, pair])
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@requires_gb10
def test_graph_replay_uses_new_inputs():
    q, k, v, mask, pair = _inputs(1, 5, 203)
    backend = GB10PairwiseAttention(0, 16, 48)
    reference = SDPAPairwiseAttention(0, 16, 48)
    with torch.inference_mode():
        backend.forward(q, k, v, [mask, pair])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = backend.forward(q, k, v, [mask, pair])
        q.neg_()
        graph.replay()
        expected = reference.forward(q, k, v, [mask, pair])
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@requires_gb10
def test_cold_graph_capture_uses_sdpa():
    q, k, v, mask, pair = _inputs(1, 1, 143)
    backend = GB10PairwiseAttention(0, 16, 48)
    reference = SDPAPairwiseAttention(0, 16, 48)
    key = (q.device.index, 1, 1, 143)
    gb10._compiled.pop(key, None)
    with torch.inference_mode():
        reference.forward(q, k, v, [mask, pair])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = backend.forward(q, k, v, [mask, pair])
        q.neg_()
        graph.replay()
        expected = reference.forward(q, k, v, [mask, pair])
    assert key not in gb10._compiled
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@requires_gb10
def test_cuda_gradients_use_sdpa():
    q, k, v, mask, pair = _inputs(1, 1, 31)
    q.requires_grad_()
    backend = GB10PairwiseAttention(0, 16, 48)
    assert not _supported(q, k, v, [mask, pair])
    backend.forward(q, k, v, [mask, pair]).sum().backward()
    assert q.grad is not None


@requires_gb10
def test_fallback_when_compiler_is_missing_or_cubin_is_forced(monkeypatch: pytest.MonkeyPatch):
    q, k, v, mask, pair = _inputs(1, 1, 31)
    backend = GB10PairwiseAttention(0, 16, 48)
    reference = SDPAPairwiseAttention(0, 16, 48)
    with torch.inference_mode():
        expected = reference.forward(q, k, v, [mask, pair])
        monkeypatch.setattr(gb10, "_compiler_available", lambda: False)
        assert not _supported(q, k, v, [mask, pair])
        torch.testing.assert_close(backend.forward(q, k, v, [mask, pair]), expected, atol=0, rtol=0)
        monkeypatch.setattr(gb10, "_compiler_available", lambda: True)
        monkeypatch.setenv("CUTEDSL_FORCE_CUBIN", "1")
        assert not _supported(q, k, v, [mask, pair])
        torch.testing.assert_close(backend.forward(q, k, v, [mask, pair]), expected, atol=0, rtol=0)


@requires_gb10
def test_unsupported_qkv_layout_uses_sdpa():
    q, k, v, mask, pair = _inputs(1, 1, 31)
    q = q.contiguous()
    backend = GB10PairwiseAttention(0, 16, 48)
    reference = SDPAPairwiseAttention(0, 16, 48)
    with torch.inference_mode():
        assert not _supported(q, k, v, [mask, pair])
        torch.testing.assert_close(
            backend.forward(q, k, v, [mask, pair]), reference.forward(q, k, v, [mask, pair]), atol=0, rtol=0
        )


@requires_gb10
def test_attention_pair_bias_layer_dispatches_to_kernel(monkeypatch: pytest.MonkeyPatch):
    from bionemo_ir._torch.layers.attention import AttentionPairBias

    torch.manual_seed(42)
    kwargs = {
        "layer_idx": 0,
        "c_s": 768,
        "c_z": 32,
        "num_heads": 16,
        "dtype": torch.bfloat16,
        "bias_proj": True,
        "initial_norm": False,
    }
    layer = AttentionPairBias(**kwargs, attn_backend="CuTeDSLGB10").cuda()
    reference = AttentionPairBias(**kwargs, attn_backend="SDPA").cuda()
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(mean=0, std=0.02)
        reference.load_state_dict(layer.state_dict())
    seq = 129
    single = torch.randn(1, 5, seq, 768, device="cuda", dtype=torch.bfloat16)
    pair = torch.randn(1, seq, seq, 32, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, seq, device="cuda")
    mask[..., -13:] = 0
    calls = []
    original = gb10._run

    def record(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(gb10, "_run", record)
    with torch.inference_mode():
        actual = layer(single, pair, mask)
        expected = reference(single, pair, mask)
    assert calls, "the real layer must reach the GB10 kernel"
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@requires_gb10
def test_openfold3_token_transformer_uses_kernel(monkeypatch: pytest.MonkeyPatch):
    from bionemo_ir._torch.layers.transformers.diffusion_transformer import OpenFold3DiffusionTransformer
    from bionemo_ir.configs import DiffusionTransformerConfig

    torch.manual_seed(43)
    config = DiffusionTransformerConfig(
        num_blocks=2,
        num_heads=16,
        dim=768,
        dim_single_cond=384,
        dim_pairwise=32,
        bias_proj=True,
        precompute_bias=True,
        dtype="bfloat16",
        pairwise_attention_backend="CuTeDSLGB10",
        version="v1",
    )
    model = OpenFold3DiffusionTransformer(config).cuda().eval()
    reference_config = config.model_copy(update={"pairwise_attention_backend": "SDPA"})
    reference = OpenFold3DiffusionTransformer(reference_config).cuda().eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim > 1:
                parameter.normal_(std=0.02)
            elif name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.zero_()
        reference.load_state_dict(model.state_dict())

    seq = 129
    a = torch.randn(1, 5, seq, 768, device="cuda", dtype=torch.bfloat16)
    s = torch.randn(1, 1, seq, 384, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(1, seq, seq, 32, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, seq, device="cuda")
    mask[..., -13:] = 0
    calls = []
    original = gb10._run

    def record(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(gb10, "_run", record)
    with torch.inference_mode():
        actual = model(a=a, s=s, z=z, mask=mask)
        expected = reference(a=a, s=s, z=z, mask=mask)
    assert len(calls) == config.num_blocks
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
