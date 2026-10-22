<div align="center">

<img src="docs/assets/verus-banner.svg" alt="Verus-Insight — Towards Explainable Artifact Detection in AI-Generated Videos" width="100%">

**Inspection Planning → Evidence Verification → Evidence Aggregation**

[Homepage](https://verus-team.github.io/Verus-Insight-Preview/) ·
[Model](https://huggingface.co/Verus-Team/Verus-Insight-8B) ·
[Verus-Bench](https://huggingface.co/datasets/Verus-Team/Verus-Bench) ·
[Examples](examples/README.md) · [Documentation](#documentation)

**8B vision-language model** &nbsp; · &nbsp; **Transformers / vLLM** &nbsp; · &nbsp; **Apache-2.0 code**

</div>

Verus-Insight inspects human-perceivable artifacts in AI-generated videos.
Built on **Qwen3-VL-8B-Instruct**, it generates video-specific inspection criteria,
verifies them against the visual input, and aggregates the observations into a
structured real/fake judgment. Human-anchored full-process supervision follows
**Context → Rubric → Audit → Aggregation**.

This is the **inference release**: environment setup, benchmark input preparation,
model download, and single-input or batch inference. It contains no training
launchers, training recipes, optimizers, or training framework dependencies.

## Method overview

<img src="docs/assets/verus_data_training.png" alt="The paper's human-anchored data construction and progressive learning pipeline" width="100%">

The figure describes the method in the paper. The scripts in this repository
run the released model; the figure is not a set of executable training instructions.

## Quick start

```bash
git clone https://github.com/zhangzhuoran997997/Verus-Insight.git
cd Verus-Insight
PYTHON=python3.12 bash scripts/install.sh transformers
source .venv-transformers/bin/activate

# Download the model once; reuse the local directory for inference.
hf download Verus-Team/Verus-Insight-8B --local-dir models/Verus-Insight-8B

# Run the bundled original benchmark frames.
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
  --model models/Verus-Insight-8B \
  --frames-json examples/frames.json --output outputs/example.json
```

Use a Linux CUDA environment. The verified Transformers example uses one H100
80GB GPU; this is an observed configuration, not a minimum memory requirement.
If Hugging Face access is required, run `hf auth login` before downloading.
The bundled [saved responses](examples/README.md) can be inspected without a GPU.

## Release contents

The repository has four public scripts:

| Script | Purpose |
| --- | --- |
| `scripts/install.sh` | Install a Transformers, vLLM, or HTTP-client environment |
| `scripts/prepare_data.py` | Download/check Verus-Bench and export model inputs |
| `scripts/batch_infer.py` | Run a manifest, save per-sample predictions, and resume |
| `scripts/infer.py` | Convert a video, image, or frame sequence into model input and infer |

## 1. Install the environment

Use Linux and Python 3.12 (Python 3.11–3.13 is supported by the installer):

```bash
git clone https://github.com/zhangzhuoran997997/Verus-Insight.git
cd Verus-Insight
PYTHON=python3.12 bash scripts/install.sh transformers
source .venv-transformers/bin/activate
```

The installer creates a virtual environment, installs the package and backend,
and runs `pip check`. To use vLLM, install a separate environment:

```bash
PYTHON=python3.12 bash scripts/install.sh vllm
source .venv-vllm/bin/activate
```

Use `client` instead of `vllm` if you only need to call an existing server.
An optional second argument selects the environment directory. Dependency pins
are in `pyproject.toml`; [environment details](docs/environment.md) record the
verified Python, CUDA, package versions, and hardware. The examples used one
H100 80GB GPU for Transformers and two for the reference vLLM configuration.

## 2. Download the model

```bash
hf download Verus-Team/Verus-Insight-8B --local-dir models/Verus-Insight-8B
```

To freeze a particular release, add `--revision <model-commit>` to `hf download`.
Transformers can also load the Hub ID directly, or accept `--revision` with that
ID. For a local snapshot, pass `--model models/Verus-Insight-8B`. Use the same
snapshot throughout an inference run and its retries. The vLLM client instead
uses the model name exposed by the running server.

## 3. Prepare the data

Download the released frames and metadata and prepare the available subset:

```bash
python scripts/prepare_data.py --download \
  --dataset-root data/Verus-Bench \
  --available-only --output metadata/inputs.available.jsonl
```

The release includes media for **166/192 samples**. The remaining 26 samples
have GT and acquisition references; their media are omitted under the dataset's
license policy. `--available-only` explicitly selects complete local samples and
reports partial coverage. Without it, export requires all 192 samples.

For a dataset you already downloaded, omit `--download`. Use `--check` to inspect
availability without changes; missing samples produce exit code 2. If you have
obtained the missing original frames or upstream videos, pass `--frames-root` or
`--source-media-root`. Video reconstruction also requires ffmpeg. The dataset's
own preparation helper handles acquisition metadata and extraction recipes.

Generated paths are relative to the output manifest. For a full run, export
`metadata/inputs.local.jsonl` without `--available-only`. See
[benchmark usage](docs/benchmark.md) and the dataset's `SETUP_ALL_DATA.md`.
Hub downloads require uploaded files and access; use `hf auth login` for private
repositories. `--revision` can pin a dataset commit with `--download`.

## 4. Run batch inference

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_infer.py \
  --backend transformers \
  --input-jsonl data/Verus-Bench/metadata/inputs.available.jsonl \
  --output outputs/predictions.jsonl
```

The default model is `Verus-Team/Verus-Insight-8B`. Set
`--model /path/to/Verus-Insight-8B` for local weights. To test the batch path
without the full dataset, use `--input-jsonl examples/inputs.jsonl`.

Add `--resume` to continue with the same settings and output. Successful samples
are skipped; failed and unprocessed samples are retried. Each attempt is saved
immediately. **Use the last row for each sample as its current result**, and keep
the adjacent `.run.json` file. GT is never passed to the model. This script
produces predictions; it does not compute benchmark scores.

### Use vLLM

In the vLLM environment, start the server with the reference settings:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Verus-Team/Verus-Insight-8B \
  --served-model-name Verus-Insight-8B \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 2 --max-model-len 65536 \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image":128}' \
  --mm-processor-kwargs '{"min_pixels":3136,"max_pixels":65536}' \
  --generation-config vllm --enforce-eager --gpu-memory-utilization 0.85
```

Replace the repository ID after `vllm serve` with a local model path if needed.
In another terminal:

```bash
python scripts/batch_infer.py --backend vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --input-jsonl data/Verus-Bench/metadata/inputs.available.jsonl \
  --output outputs/predictions_vllm.jsonl
```

`--model` on the vLLM client selects the served name. Set `VLLM_API_KEY` for an
authenticated server. A single GPU can use `--tensor-parallel-size 1`, subject to
available memory; this changes the reference serving configuration.

## 5. Infer on your own input

The single-input script converts local media into ordered images and timestamp
text, applies the independent [prompt](src/verus_insight/prompts/inspect.json),
and calls the selected backend:

```bash
# Video: sample the selected interval at 4 FPS.
python scripts/infer.py --video examples/pole_cam.mp4 \
  --start 8.47 --end 13.47 --output outputs/video.json

# Original benchmark frames and supplied timestamps.
python scripts/infer.py --frames-json examples/frames.json \
  --output outputs/frames.json

# A single image at time zero.
python scripts/infer.py --image examples/frames/pole_cam/frame_0000.jpg \
  --output outputs/image.json

# An image directory in natural filename order, with uniform timestamps.
python scripts/infer.py --frames-dir examples/frames/pole_cam --fps 4 \
  --output outputs/directory.json
```

For nonuniform frame times, add `--timestamps-json /path/to/timestamps.json`,
a JSON list of seconds matching that directory's image order. For explicit order
and timestamps, use `--frames-json`. Append `--backend vllm` to call a running
server; Transformers is the default.

Supported image extensions for a directory are JPEG, PNG, WebP, and BMP. Video
formats depend on PyAV/FFmpeg. Inputs are local files; URLs and audio are not
accepted. The script does not persist intermediate conversions. Use the original
frame manifest for benchmark reproduction: re-extracting MP4 can change pixels.

The output preserves `raw_text`, `parsed_output`, input hashes and timestamps,
generation settings, actual token counts, and `finish_reason`. Invalid JSON or
truncated output is saved and reported as a failure. Predictions can be incorrect.
See [real example outputs](examples/README.md) and the
[input/output reference](docs/input_output.md).

Defaults: at most 64 images, 3,136–65,536 pixels per image, greedy decoding,
8,192 output tokens, seed `20260913`, thinking disabled, context budget 65,536.
Images are never silently truncated. `--help` lists backend and input options.

## Reported results

The latest manuscript reports the following percentages for the final model.
These are paper results, not measurements produced by the installation checks
or bundled single-video examples.

| Benchmark | Balanced accuracy ↑ | Plan F1 ↑ | Finding F1 ↑ |
| --- | ---: | ---: | ---: |
| Verus-Bench | 89.06 | 24.70 | 15.97 |
| RewardData | 94.79 | 31.33 | 35.42 |

Verus-Bench contains **192 videos** (96 real / 96 generated), **320 annotated
artifact findings**, and **17 artifact categories**. Distributed media cover
166 samples; the full benchmark and available-media subset are distinct.

Plan F1 and Finding F1 use independent one-to-one semantic matching. They measure
inspection coverage and reference artifact recovery; they do not measure box or
timestamp accuracy. A reference nonmatch does not by itself establish that a
visual observation is unsupported. This release generates predictions and does
not run the paper's scoring pipeline.

## Documentation

| Guide | What it covers |
| --- | --- |
| [Environment](docs/environment.md) | Tested hardware, dependencies, precision, and memory scope |
| [Benchmark](docs/benchmark.md) | Full vs. available data, frame protocol, and reference serving settings |
| [Inputs and outputs](docs/input_output.md) | Local media, timestamp and coordinate conventions, failure records, and resume |
| [Real examples](examples/README.md) | Original frames and saved Transformers / vLLM responses |

## Development and license

The four scripts reuse `src/verus_insight/`; the prompt remains in its own JSON
file. Run CPU checks in the Transformers environment:

```bash
python -m unittest discover -s tests -v
```

Code: [Apache-2.0](LICENSE). Model and Verus-owned dataset material use the
non-commercial terms in their respective repositories. Third-party media retain
their original terms; the bundled example is [CC BY 2.0](examples/ATTRIBUTION.md).
