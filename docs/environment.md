# Verified environments

These runtimes are used for the real responses bundled under `examples/`.
The frame-based verification uses one GPU for Transformers and two for vLLM.
They describe observed working configurations, not minimum hardware requirements.

| Component | Native Transformers | vLLM server |
| --- | --- | --- |
| Operating system | Linux x86_64 | Linux x86_64 |
| Python | 3.12.3 | 3.11.14 |
| PyTorch | 2.11.0+cu129 | 2.10.0+cu129 |
| CUDA build reported by `torch.version.cuda` | 12.9 | 12.9 |
| Transformers | 5.12.1 | 4.57.6 |
| vLLM | Not used | 0.18.0 |
| Precision | BF16 | BF16 |
| Execution | SDPA | Eager |
| GPU | One NVIDIA H100 80GB HBM3 | Two NVIDIA H100 80GB HBM3 (TP=2) |

Video decoding and the vLLM client ran with Python 3.12.3, PyAV 16.1.0, and
Pillow 12.3.0. The frame path sends the original JPEG bytes; the server decodes these images.
The optional video path sends decoded frames as PNG images.
The observed host driver reported `535.129.03`.

## Installation

Use Python 3.12 for `scripts/install.sh`; choose `transformers`, `vllm`, or
`client` and activate the resulting virtual environment. The installer supports
Python 3.11–3.13. Python 3.11 is also
supported; vLLM 0.18.0 requires Python below 3.14. Keep Transformers and vLLM in
separate environments. Install the PyTorch CUDA build that matches your
deployment before installing `requirements.txt`. Dependency groups are declared
in `pyproject.toml`;
`requirements.txt` installs the Transformers group and `requirements-vllm.txt`
installs the vLLM group. The core installation (`pip install -e .`) is sufficient
for the HTTP client and dataset downloads. PyTorch and Transformers versions are pinned for each
backend; video decoder and image library requirements accept compatible ranges.

The CUDA version above identifies the PyTorch build. It does not identify an
installed `nvcc` toolkit or establish a universal minimum driver. Kernel features,
GPU architecture, and CUDA compatibility libraries affect driver requirements;
consult [NVIDIA's compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
when selecting your own driver and runtime.

To inspect an environment:

```bash
python --version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m pip show transformers av Pillow
nvidia-smi
```

## Memory and validation scope

BF16 weights occupy approximately 17.53 GB. Loading also uses host RAM; inference
uses additional GPU memory for visual inputs, activations, and the key-value
cache. We have not established a minimum GPU memory requirement.

The real examples use 20 frames, a 65,536-pixel upper budget per image, and an
8,192-token output limit. The frame protocol uses a 65,536-token context,
tensor parallel size 2, up to four concurrent requests, eager execution, and
`--gpu-memory-utilization 0.85`. The earlier MP4 smoke responses used one GPU,
a 16,384-token context, one concurrent request, and memory utilization 0.6.
Neither configuration establishes a measured minimum.

CPU tests cover manifest validation, batching, failure recording, resume checks,
video processing, exact frame hashes and prompt messages, and client transport
of unchanged JPEG bytes. A fresh single-sample JSONL batch ran on vLLM; the native
script ran the same frames through `--frames-json`. The dataset preparation
script exports manifests accepted by the inference loader; the 166 samples
with distributed media were checked through this interface. A complete GPU
run of all 192 released benchmark samples is not included here. Test doubles used in those tests
do not represent model predictions. The JSON files under `examples/` contain
actual GPU responses, with their input mode identified in the example README.
