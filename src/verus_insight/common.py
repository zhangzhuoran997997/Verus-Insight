"""Shared video sampling, prompt formatting, and output handling."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

DEFAULT_PROMPT = Path(__file__).parent / "prompts" / "inspect.json"
MIN_PIXELS = 3136
MAX_PIXELS = 65536
MODEL_CONTEXT = 65536
SEED = 20260913


def dispatch(mode: str) -> None:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    selected, _ = selector.parse_known_args()
    if selected.backend == "vllm":
        from .infer_vllm import main
    else:
        from .infer_transformers import main
    main(mode=mode)


def arguments(description: str, mode: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.set_defaults(video=None, frames_json=None, input_jsonl=None, image=None,
                        frames_dir=None, timestamps_json=None, resume=False)
    source = parser.add_mutually_exclusive_group(required=True)
    if mode != "batch":
        source.add_argument("--video", type=Path)
        source.add_argument("--image", type=Path, help="One still image, timestamped at zero")
        source.add_argument("--frames-dir", type=Path, help="Image directory in natural filename order")
        source.add_argument("--frames-json", type=Path, help="One sample with images and timestamps; paths relative to this file")
        parser.add_argument("--timestamps-json", type=Path, help="JSON list of seconds for --frames-dir; otherwise use --fps")
    if mode != "single":
        source.add_argument("--input-jsonl", type=Path, help="Batch manifest; media paths are relative to this file")
        parser.add_argument("--resume", action="store_true", help="Skip successful batch rows and retry failed rows")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def validate_frames(row: dict, directory: Path) -> dict:
    """Resolve supplied images without sampling, sorting, or changing timestamps."""
    if not isinstance(row, dict) or set(row) != {"sample_id", "images", "timestamps"}:
        raise ValueError("Frame input requires exactly sample_id, images, timestamps")
    if not isinstance(row["sample_id"], str) or not row["sample_id"].strip():
        raise ValueError("sample_id must be a nonempty string")
    images, times = row["images"], row["timestamps"]
    if (not isinstance(images, list) or not images or not isinstance(times, list)
            or len(images) != len(times)):
        raise ValueError("images and timestamps must be nonempty lists of equal length")
    if any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times):
        raise ValueError("timestamps must be nonnegative finite numbers")
    if any(a > b for a, b in zip(times, times[1:])):
        raise ValueError("timestamps must be in chronological order")
    paths = []
    for name in images:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each image must be a local file path")
        path = (directory / name).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image missing: {path}")
        paths.append(path)
    return dict(sample_id=row["sample_id"], images=paths, timestamps=times)


def sample_video(path: Path, start: float, end: float | None, fps: float,
                 max_frames: int) -> tuple[list, list[float]]:
    """Use the first decoded frame at/after each sampling time; never silently truncate."""
    import av

    if not math.isfinite(start) or start < 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("start must be nonnegative and fps must be positive and finite")
    if end is not None and (not math.isfinite(end) or end <= start):
        raise ValueError("end must be finite and greater than start")
    if max_frames < 1:
        raise ValueError("max-frames must be positive")
    frames, timestamps = [], []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Input has no video stream")
        stream = container.streams.video[0]
        origin = None
        for frame in container.decode(stream):
            if frame.time is None:
                raise ValueError("Video has no frame timestamps")
            if origin is None:
                origin = float(frame.time)
            time = float(frame.time) - origin
            if end is not None and time >= end - 1e-7:
                break
            image = None
            while time + 1e-7 >= start + len(frames) / fps:
                if len(frames) == max_frames:
                    raise ValueError("Frame limit exceeded; select a shorter interval with --start/--end")
                if image is None:
                    image = frame.to_image().convert("RGB")
                timestamps.append(round(len(frames) / fps, 6))
                frames.append(image)
    if not frames:
        raise ValueError("No frames in the requested interval")
    return frames, timestamps


def prepare(args) -> tuple[list, list[dict], dict]:
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    prompt_bytes = args.prompt.read_bytes()
    prompt = json.loads(prompt_bytes)
    if set(prompt) != {"system", "prefix", "suffix"} or not all(
        isinstance(value, str) and value.strip() for value in prompt.values()
    ):
        raise ValueError("Prompt must contain nonempty system, prefix, and suffix strings")
    if getattr(args, "timestamps_json", None) and not getattr(args, "frames_dir", None):
        raise ValueError("--timestamps-json requires --frames-dir")
    if getattr(args, "image", None):
        row = validate_frames(dict(sample_id=args.image.stem,
                                   images=[str(args.image.resolve())], timestamps=[0.0]), Path.cwd())
    elif getattr(args, "frames_dir", None):
        if not math.isfinite(args.fps) or args.fps <= 0:
            raise ValueError("fps must be positive and finite")
        paths = sorted((p for p in args.frames_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}),
                       key=lambda p: ([(1, int(v)) if v.isdigit() else (0, v.lower())
                                       for v in re.split(r"(\d+)", p.name)], p.name))
        times = (json.loads(args.timestamps_json.read_text()) if args.timestamps_json
                 else [round(i / args.fps, 6) for i in range(len(paths))])
        row = validate_frames(dict(sample_id=args.frames_dir.resolve().name,
                                   images=[str(p.resolve()) for p in paths], timestamps=times), Path.cwd())
    elif getattr(args, "frames_json", None):
        row = validate_frames(json.loads(args.frames_json.read_text()), args.frames_json.parent)
    elif getattr(args, "images", None):
        row = dict(sample_id=args.sample_id, images=args.images, timestamps=args.timestamps)
    else:
        row = None
    if row is not None:
        from PIL import Image
        if args.start != 0 or args.end is not None or (args.fps != 4 and not getattr(args, "frames_dir", None)):
            raise ValueError("Frame inputs use supplied timestamps; do not set start, end, or fps")
        if not 0 < len(row["images"]) <= args.max_frames:
            raise ValueError("Frame limit exceeded; supplied frames are never truncated")
        frames = []
        args.images = row["images"]
        for path in row["images"]:
            with Image.open(path) as image:
                frames.append(image.convert("RGB"))
        timestamps = row["timestamps"]
        metadata = {"source": "frames", "sample_id": row["sample_id"],
                    "images": [p.name for p in row["images"]],
                    "image_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in row["images"]]}
    else:
        frames, timestamps = sample_video(args.video, args.start, args.end, args.fps, args.max_frames)
        with args.video.open("rb") as handle:
            video_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        metadata = {"source": "video", "video": args.video.name, "sha256": video_hash,
                    "start_seconds": args.start, "end_seconds": args.end, "sampling_fps": args.fps}
    content = []
    for index, timestamp in enumerate(timestamps):
        content.extend([
            {"type": "text", "text": (prompt["prefix"] if index == 0 else "") + f"<{timestamp:.2f} seconds>"},
            {"type": "image"},
        ])
    content.append({"type": "text", "text": prompt["suffix"]})
    messages = [{"role": "system", "content": prompt["system"]},
                {"role": "user", "content": content}]
    metadata = {
        **metadata, "num_frames": len(frames),
        "timestamps_seconds": timestamps,
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
    }
    return frames, messages, metadata


def make_result(raw_text: str, finish_reason: str, **metadata) -> dict:
    error, parsed = None, None
    def reject_constant(value: str):
        raise ValueError(f"Non-finite number is not valid JSON: {value}")
    try:
        parsed = json.loads(raw_text, parse_constant=reject_constant)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object")
    except ValueError as exc:
        error = str(exc)
    return dict(metadata, raw_text=raw_text, parsed_output=parsed,
                parse_error=error, finish_reason=finish_reason)


def write_result(path: Path, result: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(result["raw_text"], flush=True)
    if result["parse_error"] or result["finish_reason"] != "stop":
        raise RuntimeError(f"Incomplete or invalid response; raw output saved to {path}")
    return result


def save_result(path: Path, raw_text: str, finish_reason: str, **metadata) -> dict:
    return write_result(path, make_result(raw_text, finish_reason, **metadata))
