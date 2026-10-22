# Inputs and outputs

Use `scripts/infer.py` for one input and `scripts/batch_infer.py` for JSONL
manifests. Both accept `--backend transformers` (default) or `--backend vllm`.

## Image and image directory

`--image /path/to/image.jpg` supplies one image at time zero. `--frames-dir`
accepts JPEG, PNG, WebP, and BMP files directly inside a directory, sorted in
natural filename order (`frame2.jpg` before `frame10.jpg`). Subdirectories and
other files are ignored. Without explicit times, frame `i` gets timestamp
`i / fps`; default `--fps` is 4. These are assumed uniform times, not measured
video timestamps. `--timestamps-json` supplies a JSON list of nonnegative finite
seconds in the same image order. Use `--frames-json` for explicit file ordering.
Empty directories, time-count mismatches, and decreasing timestamps fail.
A single image cannot provide evidence of temporal consistency.

## Frame input (benchmark and default example)

`--frames-json examples/frames.json` reads one JSON object:

```json
{"sample_id":"clip_001","images":["frames/0000.jpg","frames/0001.jpg"],"timestamps":[0.0,0.25]}
```

These three fields are required. `sample_id` is a nonempty string; `images` and
`timestamps` are nonempty lists of equal length. Paths are local files, relative
to the manifest directory or absolute. Timestamps are finite, nonnegative,
chronologically ordered seconds. The loader preserves their values and the
image order; it does not sort, rebase, resample, or re-encode the supplied files.
The model receives timestamp text formatted as `<0.00 seconds>` followed by
its image. The first timestamp shares a text part with the prompt prefix.

Transformers decodes the images to RGB and applies the model processor. The
vLLM client sends the original image bytes as data URLs. Both use pixel limits
3,136–65,536. Inputs above `--max-frames` (default 64) fail without truncation.
Do not set `--start`, `--end`, or a different `--fps` for frame inputs.

## Optional video input

`--video` accepts a local file decoded by PyAV/FFmpeg. The bundled source video is decoded by the
example; other containers and codecs depend on the installed decoder. Its
`pole_cam.mp4` filename is retained from upstream, but its container is FLV.
The first video stream is used, and audio is ignored. Video URLs are not accepted.

| Option | Meaning | Default |
| --- | --- | --- |
| `--start` | Interval start in seconds from the first decoded video frame | `0` |
| `--end` | Exclusive interval end in the same timeline | End of file |
| `--fps` | Requested frame sampling rate | `4` |
| `--max-frames` | Maximum sampled images; exceeding it raises an error | `64` |
| `--max-new-tokens` | Maximum generated tokens | `8192` |
| `--prompt` | JSON file with `system`, `prefix`, and `suffix` strings | `src/verus_insight/prompts/inspect.json` |

Frames are selected at sampling grid points starting at `--start`. The first
decoded frame at or after each point is used. Several points may select the same
frame if sampling is faster than the source. Inputs include text timestamps
relative to the interval start, formatted as `<0.00 seconds>`, followed by each
image. Both backends use the same sampler and prompt.

`--end` beyond the available video reads to the available end. An interval with
no decoded frames fails. A long video is not automatically shortened to fit the
frame limit. Image resizing preserves aspect ratio within the processor's pixel
budget of 3,136–65,536, subject to its patch-size rounding.

## Batch manifest

Use `--input-jsonl` in place of `--frames-json` or `--video`. Each nonempty line
is one frame object as above, or an optional video object:

```json
{"sample_id":"clip_001","video":"videos/clip_001.mp4","start":0.0,"end":5.0}
```

- `sample_id`: required, unique, nonempty string.
- `video`: required local file path, resolved relative to the manifest directory.
  Absolute paths are also accepted.
- `start`: optional nonnegative finite number, default `0`.
- `end`: optional finite number greater than `start`, default end of file.

A video row accepts only these four fields; a frame row accepts only
`sample_id`, `images`, and `timestamps`. The two forms cannot be mixed within a row. Keep GT, labels, and reference answers in a
separate file. Sample IDs and file names are used for bookkeeping and are not
included in the model's messages. All rows and media paths are checked before
inference begins. Set per-video intervals in the manifest; do not combine batch
mode with CLI `--start` or `--end`.

Processing is sequential. Transformers loads the model once and reuses it for
every row. vLLM receives one request at a time. This batch interface is intended
for reproducible dataset runs, not simultaneous tensor batching.

## Output record

Single-input mode writes one JSON object. Batch mode writes one JSON object per
line. A successful record contains:

| Field | Meaning |
| --- | --- |
| `model`, `backend` | Public model name and selected inference backend |
| `input` | Frame basenames and SHA-256 hashes, or video basename/hash and interval; timestamps, frame count, prompt SHA-256 |
| `parameters` | Decoding and image pixel limits |
| `raw_text` | Original decoded response, with special tokens removed; no JSON repair or rewriting |
| `parsed_output` | JSON-decoded response; an object on success |
| `parse_error` | `null` on successful JSON-object parsing, otherwise an error description |
| `finish_reason` | `stop` on normal completion; `length` means the output budget was reached |
| `usage` | Actual prompt and completion token counts, or `null` if unavailable |
| `runtime` | Native Transformers versions, precision, and attention implementation, when available |
| `elapsed_seconds` | Time measured by the backend script |

Native timing includes input preparation and, for the first row, model loading.
vLLM timing covers the HTTP request and excludes server startup and local input
preparation. These measurements are not a backend speed comparison.

Batch rows add `sample_id`, `run_signature`, `status` (`ok` or `error`), and
`error`. Decode, transport, or model exceptions produce an error row with
`raw_text: null`. Invalid or truncated model responses retain their raw text.
Later samples continue after a per-sample exception. A batch exits with code 1
if any sample's latest attempt failed; progress is printed to stderr and the
final counts to stdout. Interruptions preserve rows already flushed to disk.

## Model response

The released model typically returns a `stage_targets` object:

- `video_describe`: scene summary and interpretation of the video.
- `domain_selection`: the visual aspects chosen for inspection.
- `rubric`: inspection criteria. The model may use `items`, or separate
  `suspicion_items` and `authenticity_items` arrays.
- `audit`: findings, evidence assessment, and a final `verdict`, usually `real`
  or `fake`. A `confidence` field may also be present.

For example, the bundled response contains
`parsed_output.stage_targets.audit.verdict = "real"`.

When localization is supplied, `timestamp` intervals, `keyframe_time`, and
`frame_boxes[].time` use the supplied frame timestamp timeline. For `--video`,
they are relative to the selected interval; add the input start time to map
them back to the original video. Boxes use
`[x_min, y_min, x_max, y_max]` on a 0–1000 coordinate scale; for image width `W`
and height `H`, convert using `x * W / 1000` and `y * H / 1000`.
`bbox: null` and empty `frame_boxes` mean that no box was supplied.

These fields are generated predictions. The scripts validate JSON syntax and
the top-level object type; they do not enforce the full response schema or
validate verdicts, coordinate bounds, timestamps, or factual correctness. Check
those fields before consuming them. Normal findings are not positive artifacts.

## Resume and retry

Add `--resume` with the same manifest, prompt, backend, model reference, settings,
and output path. Successful rows are skipped; failed and unprocessed samples are
retried. Historical rows stay in the file, so a retried sample can appear more
than once. **Use the last row for each `sample_id` as its current result.**

The adjacent `predictions.jsonl.run.json` stores a run signature. It checks the
manifest, image/video contents, prompt, inference code, model reference, settings, and
available local model configuration/checksum files. A mismatch fails before new
predictions are appended. Keep the sidecar with the output. Model weights behind
a remote repository ID, local path, or vLLM alias must remain unchanged; resume
does not rehash every weight shard or inspect the server's weights.

Without `--resume`, an existing output is refused. Only one process may write to
an output at a time. If an interrupted write leaves an invalid JSONL line, resume
reports that line and leaves the original file untouched; preserve the evidence
and use a new output or repair a copy before resuming.
