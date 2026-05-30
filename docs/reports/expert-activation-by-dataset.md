# Laguna-XS.2 — Expert Activation, Dataset by Dataset

**Summary.** Each densification dataset (+ SWE-bench) was fed through the frozen
`poolside/Laguna-XS.2` teacher (top-8 of 256 experts, 39 sparse layers); we hooked every
router and counted which experts fire. **One finding dominates: the lower-level the input,
the narrower the routing.** CUDA/Triton collapse onto ~100–108 effective experts/layer;
natural-language instructions stay as broad as web text (~158–183). Each dataset below shows
its **256-expert activation grid (16×16)** — brighter = more activated.

<img src="figures/sq_summary.png" width="430">

| Dataset | Eff. experts | Gini | Routing |
|---|--:|--:|---|
| Magicoder | 183 | 0.43 | most diffuse |
| SWE-bench | 163 | 0.49 | web-like |
| *C4 (baseline)* | *158* | *0.53* | *baseline* |
| CodeFeedback | 151 | 0.52 | mild |
| OpenCodeInstruct | 145 | 0.54 | mild |
| KernelBook | 108 | 0.65 | concentrated |
| cuda_kernels | 100 | 0.68 | most concentrated |

---

## 1. `allenai/c4`
> General web text. The broad baseline — routing spreads across ~158 experts.

<img src="figures/sq_c4.png" width="380">

**web text (baseline)** · `400 questions / 161,932 tokens` · **eff 158/256** · Gini 0.528 · *(baseline — see C4 gist)*

---

## 2. `ise-uiuc/Magicoder-Evol-Instruct-110K`
> Natural-language coding instructions. The **most diffuse** of all — even wider than web text.

<img src="figures/sq_magicoder.png" width="380">

**NL instruction** · `300 questions / 47,874 tokens` · **eff 183/256** · Gini 0.425 · top: E63, E160, E70, E2, E200, E164, E79, E107

Sample fed through Laguna:

````text
Please amend the subsequent Python script so that it includes a 'while' loop rather than the existing 'for' loop, which iterates through the items of an integer list.

The script currently has a bug where it attempts to print an object that is outside the bounds of the list. Fix this error and modify the script to use 'while' instead of 'for' loop. Ensure your script correctly handles empty lists. 

```python
  # Establish an integer list
  arr = [1, 2, 3, 4]

  # Determine the length of the lis
````

---

## 3. `princeton-nlp/SWE-bench_Lite`
> Real GitHub issue text. Prose-heavy, so it routes almost exactly like web text.

<img src="figures/sq_swebench_lite.png" width="380">

**NL problem statement** · `300 questions / 115,888 tokens` · **eff 163/256** · Gini 0.494 · top: E76, E205, E228, E18, E117, E180, E107, E52

Sample fed through Laguna:

````text
Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels
Consider the following model:

```python
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

cm = m.Linear1D(10) & m.Linear1D(5)
```

It's separability matrix as you might expect is a diagonal:

```python
>>> separability_matrix(cm)
array([[ True, False],
       [False,  True]])
```

If I make the model more complex:
```python
>>>
````

---

## 4. `m-a-p/CodeFeedback-Filtered-Instruction`
> Code Q&A queries. Slightly tighter than pure NL as concrete code creeps in.

<img src="figures/sq_codefeedback.png" width="380">

**NL query** · `300 questions / 50,164 tokens` · **eff 151/256** · Gini 0.519 · top: E70, E63, E160, E2, E200, E235, E159, E205

Sample fed through Laguna:

````text
Create a nested loop to print every combination of numbers between 0-9, excluding any combination that contains the number 5. Additionally, exclude any combination that contains a repeating digit. Implement the solution without using any built-in functions or libraries to check for repeating digits.
````

---

## 5. `nvidia/OpenCodeInstruct`
> Python coding tasks. Tighter still — Python tokens start concentrating the router.

<img src="figures/sq_opencodeinstruct.png" width="380">

**NL → Python** · `300 questions / 66,590 tokens` · **eff 145/256** · Gini 0.540 · top: E63, E70, E160, E47, E159, E200, E2, E117

Sample fed through Laguna:

````text
You are given a list of `n` tasks, each represented as a tuple `(start, end)`, indicating the start and end times of the task. The tasks are sorted by their start times. Your goal is to determine the maximum number of non-overlapping tasks that can be selected. Two tasks are considered non-overlapping if the start time of one task is greater than or equal to the end time of the other.

**Input:**
- An integer `n` representing the number of tasks.
- A list of `n` tuples, where each tuple `(start,
````

---

## 6. `GPUMODE/KernelBook`
> PyTorch-module source (the Triton anchor). Routing **collapses** to ~108 experts.

<img src="figures/sq_kernelbook.png" width="380">

**Triton / PyTorch source** · `300 questions / 108,815 tokens` · **eff 108/256** · Gini 0.646 · top: E66, E173, E21, E93, E176, E189, E247, E198

Sample fed through Laguna:

````python
import torch
import torch.nn as nn


class SumAggregator(nn.Module):

    def __init__(self):
        super(SumAggregator, self).__init__()

    def forward(self, neighbor):
        return torch.sum(neighbor, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
````

---

## 7. `andrew-wang/cuda_kernels`
> Raw CUDA kernel source. The **most concentrated** input — only ~100 effective experts.

<img src="figures/sq_cuda_kernels.png" width="380">

**CUDA C++ source** · `137 questions / 75,709 tokens` · **eff 100/256** · Gini 0.683 · top: E66, E198, E173, E21, E178, E163, E167, E166

Sample fed through Laguna:

````python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise multiplication and bias addition
elementwise_multiply_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_multiply_add_kernel(const float* x, const float* weight, const float* bias, float* out, int n_channels) {
    int c = blockIdx.x;
    int n = blockIdx.y;
    if (c < n_channels && n < warpSize) {
        int id
````

---

## 8. Cross-dataset expert overlap

> Three near-disjoint neighborhoods: web text, code-instruct (OpenCode↔CodeFeedback 56%),
> and kernels (KernelBook↔CUDA 49%). Cross-cluster overlap is only 3–8%.

<img src="figures/sq_overlap.png" width="430">

**Implication for the mix:** densifying on one domain overfits the dense surrogate to that
domain's experts. Kernel-specialist experts (E66, E198, E21, E173 …) are near-silent on
general text — the kernel anchor must be explicit in the `--datasets` interleave or they're
never reconstructed.

---

*Method: `analyze_datasets_expert.py` — one forward pass per question, top-8 membership
accumulated per layer. Per-dataset stats in `dataset_diag/`. Each dataset's question/prompt
field is fed (not prompt+answer); code-only datasets use their source as input.*
