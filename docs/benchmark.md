# Running Verus-Bench inputs

Dataset: [Verus-Team/Verus-Bench](https://huggingface.co/datasets/Verus-Team/Verus-Bench).
Follow its release instructions to obtain the input manifest and permitted media.
Access requires uploaded files and authorization while the repository is private.

## Input protocol

Use the supplied frame files and timestamps to reproduce the benchmark inputs:

```json
{"sample_id":"clip_001","images":["frames/clip_001/0000.jpg","frames/clip_001/0001.jpg"],"timestamps":[0.0,0.25]}
```

Prepare a runnable manifest from the downloaded dataset:

```bash
python scripts/prepare_data.py --dataset-root /path/to/Verus-Bench \
  --output metadata/inputs.local.jsonl
```

The release distributes frames for 166 of 192 samples. The other 26 require
local acquisition under the dataset's media policy. To explicitly run only the
available subset, add `--available-only` and use an output such as
`metadata/inputs.available.jsonl`. Report its evaluated sample IDs and coverage.
Without that option, incomplete data prevents export. The distributed
`metadata/inputs.jsonl` is a release index with missing-media placeholders;
pass the generated manifest to inference instead.

Paths in generated manifests are relative to the manifest. Keep frame bytes, order, and timestamps
unchanged. Re-extracting an MP4 is not guaranteed to recover the same pixels.
Only `sample_id`, `images`, and `timestamps` are accepted for frame rows.
GT belongs in a separate file and is never passed to the model.

## Run inference

The reference backend is vLLM. Start the server as documented in the README,
then run:

```bash
python scripts/batch_infer.py --backend vllm \
  --input-jsonl /path/to/Verus-Bench/metadata/inputs.local.jsonl \
  --output outputs/verus_bench.jsonl
```

For native Transformers, select `--backend transformers` with the same manifest. Add `--resume` to continue interrupted runs. Preserve code, model
revision, prompt, media, settings, and the output's `.run.json` sidecar.
Use the last output row for each `sample_id` after retries.

## Reference inference settings

| Setting | Value |
| --- | --- |
| Input | Ordered frame images and supplied timestamps, at most 64 |
| Prompt | `src/verus_insight/prompts/inspect.json` |
| Timestamp text | `<{timestamp:.2f} seconds>` |
| Pixel limits | min 3,136 / max 65,536 |
| Thinking | Disabled |
| Decoding | temperature 0, top-p 1, seed 20260913, max output 8,192 |
| Other sampling defaults | top-k 20, repetition penalty 1 (greedy decoding) |
| Model context | 65,536 |
| Backend | vLLM 0.18.0, Transformers 4.57.6, PyTorch 2.10.0+cu129 |
| Precision and parallelism | BF16, tensor parallel size 2 |
| Server scheduling | Eager, max sequences 4, max batched tokens 8,192 |
| Server capacity | 128 images per request; client limits single-video input to 64 |
| GPU memory budget | 0.85 |

The client explicitly sets sampling parameters, so they do not depend on a
server's inherited generation configuration. The reference evaluation used
H800 GPUs. The public smoke runs use H100 GPUs. Changing hardware, parallelism,
request batching, library versions, or attention kernels can change generated
text even with greedy decoding. Native Transformers is an alternative backend;
matching its inputs does not establish token-for-token equivalence to vLLM.

The bundled frame fixture checks exact message content and image SHA-256 hashes.
The released benchmark has 192 samples and 5,932 frame descriptors. A complete
GPU inference and scoring run for that dataset is not included in this repository.
See [examples](../examples/README.md) for GPU validation, and
[environment details](environment.md) for the tested runtimes.

The loader validates every manifest row and media path before inference. Missing
media fail explicitly; no samples are silently omitted. This repository produces
predictions. Dataset-wide inference and benchmark scoring remain separate steps.
