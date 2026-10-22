# Real inference examples

## Original benchmark frames

[`frames.json`](frames.json) lists the 20 JPEG images in `frames/pole_cam/`
and timestamps 0.00–4.75 seconds. They are the original input frames for
Verus-Bench sample `real_nextqa_2983727948`, copied without re-encoding.
[`inputs.jsonl`](inputs.jsonl) contains the same sample for batch inference.
See [attribution](ATTRIBUTION.md) for the CC BY 2.0 source and extraction details.

[`frame_request.json`](frame_request.json) stores the reference message content
with image placeholders and the original image SHA-256 hashes. CPU tests compare
the current input builder against this fixture. The source video is supplied for
viewing and optional MP4 inference; sampling it again need not recover the exact
benchmark pixels.

### vLLM frame result

[`vllm_frames_output.jsonl`](vllm_frames_output.jsonl) is a fresh response from
the public batch script and bundled frame manifest:

- Two NVIDIA H100 80GB GPUs; BF16; tensor parallel size 2.
- vLLM 0.18.0, Transformers 4.57.6, PyTorch 2.10.0+cu129.
- Context 65,536, eager execution, max sequences 4, max batched tokens 8,192.
- Input tokens: 1,331; generated tokens: 1,065.
- Finish: `stop`; valid JSON; predicted verdict: `real`.
- HTTP request time: 64.928 seconds, excluding startup.

```bash
# Start the vLLM server using the command in the main README.
python scripts/batch_infer.py --backend vllm \
  --input-jsonl examples/inputs.jsonl \
  --output outputs/frames_vllm.jsonl
```

This verifies the frame and batch inference path. It does not establish benchmark
accuracy or token-for-token equality with an earlier response. Device and runtime
numerics can change generated text even when the frames and prompt are identical.

### Transformers frame result

[`transformers_frames_output.json`](transformers_frames_output.json) is a fresh
response from the native script using the same JPEG frames and timestamps:

- One NVIDIA H100 80GB GPU; BF16; SDPA.
- Transformers 5.12.1, PyTorch 2.11.0+cu129.
- Input tokens: 1,331; generated tokens: 1,072.
- Finish: `stop`; valid JSON; predicted verdict: `real`.
- Load, preprocessing, and generation time: 74.136 seconds.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py --backend transformers \
  --frames-json examples/frames.json \
  --output outputs/frames_transformers.json
```

Both new responses record matching image and prompt hashes. They were generated
from the local release weights; Hub commands require those weights to be uploaded
and accessible. Use `--model /path/to/Verus-Insight-8B` before upload.

## Earlier MP4 smoke responses

The input is the 8.47–13.47 second interval of `pole_cam.mp4`, a source video used
by Verus-Bench sample `real_nextqa_2983727948`. The script samples 20 frames at
4 FPS and timestamps them relative to the interval start. See
[ATTRIBUTION.md](ATTRIBUTION.md) for the source and video license.

### Transformers MP4 result

[transformers_output.json](transformers_output.json) contains the unedited
decoded response and the same response parsed as JSON.

- Hardware: one NVIDIA H100 80GB.
- Runtime: PyTorch 2.11.0+cu129, Transformers 5.12.1, BF16, SDPA.
- Input tokens: 1,331.
- Generated tokens: 1,086.
- Finish reason: `stop`; JSON parsing succeeded.
- Load, preprocessing, and generation time: approximately 104 seconds.

The model describes a helmeted skier on a snowy mountain ridge and checks the
helmet and goggles, the hand holding the camera, and the mountain contours.
Its final prediction is `stage_targets.audit.verdict = "real"`.

Run the optional MP4 input path from the repository root:

```bash
python scripts/infer.py --backend transformers \
  --model Verus-Team/Verus-Insight-8B \
  --video examples/pole_cam.mp4 \
  --start 8.47 --end 13.47 \
  --output outputs/transformers.json
```

Input video and prompt SHA-256 hashes are recorded in the output. This example
verifies that the released weights and inference path produce a real response;
it is not a benchmark accuracy result. Numerical differences between devices or
backends may change the generated text even with greedy decoding.

### vLLM MP4 result

[vllm_output.json](vllm_output.json) contains a separate, real model response
from the vLLM server on the same 20 frames and prompt.

- Hardware: one NVIDIA H100 80GB.
- Server: vLLM 0.18.0, PyTorch 2.10.0+cu129, Transformers 4.57.6, BF16.
- Context limit: 16,384 tokens; eager execution; one concurrent request.
- Input tokens: 1,331.
- Generated tokens: 1,121.
- Finish reason: `stop`; JSON parsing succeeded.
- Request time: approximately 18.8 seconds, excluding server startup.
- Final prediction: `stage_targets.audit.verdict = "real"`.

For the earlier one-GPU server configuration (override the README server flags
with `--tensor-parallel-size 1 --max-model-len 16384`,
`--gpu-memory-utilization 0.6 --max-num-seqs 1`), run:

```bash
python scripts/infer.py --backend vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --video examples/pole_cam.mp4 \
  --start 8.47 --end 13.47 \
  --output outputs/vllm.json
```

The two responses are retained independently; their wording differs. The timing
measurements cover different scopes and should not be compared as a speed test.

These responses were generated from the local release weights. Hub commands
require those weights to be uploaded and accessible; use `--model` with a local
model directory to reproduce before upload.
