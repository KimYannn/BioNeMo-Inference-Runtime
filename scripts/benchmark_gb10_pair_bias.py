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

"""Benchmark the opt-in GB10 pair-bias backend against SDPA."""

import argparse
import importlib.metadata
import json
import statistics
from functools import partial
from pathlib import Path

import torch

from bionemo_ir._torch.attention_backend.pairwise_attention.gb10 import GB10PairwiseAttention, _supported
from bionemo_ir._torch.attention_backend.pairwise_attention.sdpa import SDPAPairwiseAttention
from bionemo_ir._torch.layers.attention import AttentionPairBias


def capture(fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def elapsed(graph, iterations=20):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def measure(reference, candidate):
    for _ in range(3):
        reference.replay()
        candidate.replay()
    torch.cuda.synchronize()
    ref_times, new_times = [], []
    for trial in range(3):
        if trial % 2:
            new_times.append(elapsed(candidate))
            ref_times.append(elapsed(reference))
        else:
            ref_times.append(elapsed(reference))
            new_times.append(elapsed(candidate))
    return {
        "sdpa_ms": statistics.median(ref_times),
        "gb10_ms": statistics.median(new_times),
        "speedup": statistics.median(ref_times) / statistics.median(new_times),
        "sdpa_trials_ms": ref_times,
        "gb10_trials_ms": new_times,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure GB10 pair-bias attention and its enclosing layer.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("The benchmark requires a CUDA SM121 device")
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "operators": [],
        "layers": [],
    }
    target = args.output

    with torch.inference_mode():
        for n in (203, 575, 1286, 1762):
            for samples in (1, 5):
                torch.manual_seed(712)
                packed = torch.randn(1, samples, n, 3072, device="cuda", dtype=torch.bfloat16)
                q, _, k, v = packed.split(768, -1)
                pair = torch.randn(1, 1, 16, n, n, device="cuda", dtype=torch.bfloat16)
                mask = torch.zeros(1, 1, 1, 1, n, device="cuda", dtype=torch.float32)
                mask[..., n * 7 // 8 :] = -1e9
                optimized, baseline = GB10PairwiseAttention(0, 16, 48), SDPAPairwiseAttention(0, 16, 48)
                assert _supported(q, k, v, [mask, pair])
                optimized.forward(q, k, v, [mask, pair])
                baseline.forward(q, k, v, [mask, pair])
                g_opt, out_opt = capture(partial(optimized.forward, q, k, v, [mask, pair]))
                g_ref, out_ref = capture(partial(baseline.forward, q, k, v, [mask, pair]))
                g_opt.replay()
                g_ref.replay()
                torch.testing.assert_close(out_opt, out_ref, atol=0.02, rtol=0.02)
                row = {"n": n, "samples": samples, **measure(g_ref, g_opt)}
                result["operators"].append(row)
                target.write_text(json.dumps(result, indent=2))
                print(json.dumps(row), flush=True)

        for n in (203, 575):
            torch.manual_seed(714)
            kwargs = {
                "layer_idx": 0,
                "c_s": 768,
                "c_z": 128,
                "num_heads": 16,
                "dtype": torch.bfloat16,
                "bias_proj": True,
                "initial_norm": False,
            }
            optimized = AttentionPairBias(**kwargs, attn_backend="CuTeDSLGB10").cuda()
            baseline = AttentionPairBias(**kwargs, attn_backend="SDPA").cuda()
            for parameter in optimized.parameters():
                parameter.normal_(0, 0.02)
            baseline.load_state_dict(optimized.state_dict())
            single = torch.randn(1, 5, n, 768, device="cuda", dtype=torch.bfloat16)
            pair = torch.randn(1, n, n, 128, device="cuda", dtype=torch.bfloat16)
            mask = torch.ones(1, n, device="cuda")
            mask[..., n * 7 // 8 :] = 0
            optimized(single, pair, mask)
            baseline(single, pair, mask)
            g_opt, out_opt = capture(partial(optimized, single, pair, mask))
            g_ref, out_ref = capture(partial(baseline, single, pair, mask))
            g_opt.replay()
            g_ref.replay()
            torch.testing.assert_close(out_opt, out_ref, atol=0.02, rtol=0.02)
            row = {"n": n, "samples": 5, **measure(g_ref, g_opt)}
            result["layers"].append(row)
            target.write_text(json.dumps(result, indent=2))
            print(json.dumps(row), flush=True)

    result["status"] = "passed"
    target.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
