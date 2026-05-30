"""format_sft_row support for kernel datasets (GPUMODE/KernelBook family)."""
from __future__ import annotations

import json

from densify.reconstruction_data import format_sft_row


def test_kernelbook_python_to_triton_pair():
    row = {
        "python_code": "import torch\nclass M(torch.nn.Module):\n    def forward(self, x): return x.sum()",
        "triton_code": "import triton\n@triton.jit\ndef k(): pass",
        "entry_point": "M",
    }
    out = format_sft_row(row)
    assert "<user>" in out and "<assistant>" in out
    assert "Triton kernel" in out
    assert "@triton.jit" in out


def test_pytorch_code_alias_and_final_triton():
    row = {"pytorch_code": "x=1", "final_triton_code": "y=2"}
    out = format_sft_row(row)
    assert "x=1" in out and "y=2" in out


def test_messages_as_json_string():
    msgs = json.dumps([
        {"role": "user", "content": "convert this kernel"},
        {"role": "assistant", "content": "here is the triton code"},
    ])
    out = format_sft_row({"messages": msgs})
    assert "<user>" in out and "convert this kernel" in out
    assert "<assistant>" in out and "here is the triton code" in out


def test_full_messages_json_string_key():
    msgs = json.dumps([{"role": "system", "content": "you are a kernel expert"}])
    out = format_sft_row({"full_messages": msgs})
    assert "you are a kernel expert" in out


def test_sakana_pytorch_to_cuda_pair():
    row = {"PyTorch_Code_Module": "import torch\nclass M(torch.nn.Module): ...",
           "CUDA_Code": "#include <cuda_runtime.h>\n__global__ void k(){}"}
    out = format_sft_row(row)
    assert "CUDA kernel" in out and "__global__" in out
