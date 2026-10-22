"""Sequential JSONL inference with durable per-sample results and resume checks."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Callable

from .common import validate_frames, write_result


def read_samples(path: Path) -> list[dict]:
    samples, seen = [], set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and "images" in row:
            item = validate_frames(row, path.parent)
            if item["sample_id"] in seen:
                raise ValueError(f"Line {number}: duplicate sample_id")
            seen.add(item["sample_id"])
            samples.append(item)
            continue
        if (not isinstance(row, dict) or not {"sample_id", "video"} <= row.keys()
                or row.keys() - {"sample_id", "video", "start", "end"}):
            raise ValueError(f"Line {number}: only sample_id, video, start, end are accepted")
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen:
            raise ValueError(f"Line {number}: sample_id must be nonempty and unique")
        if not isinstance(row["video"], str) or not row["video"].strip():
            raise ValueError(f"Line {number}: video must be a local file path")
        start, end = row.get("start", 0.0), row.get("end")
        if (type(start) not in (int, float) or not math.isfinite(start) or start < 0
                or (end is not None and (type(end) not in (int, float)
                    or not math.isfinite(end) or end <= start))):
            raise ValueError(f"Line {number}: invalid start/end interval")
        video = (path.parent / row["video"]).resolve()
        if not video.is_file():
            raise FileNotFoundError(f"Line {number}: video missing: {video}")
        samples.append(dict(sample_id=sample_id, video=video, start=start, end=end))
        seen.add(sample_id)
    if not samples:
        raise ValueError("Input manifest is empty")
    return samples


def run_signature(args: argparse.Namespace, backend: str, samples: list[dict]) -> str:
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in {"output", "resume", "input_jsonl", "video"}}
    root = Path(__file__).parent
    digest = hashlib.sha256(json.dumps({"backend": backend, "settings": settings}, sort_keys=True).encode())
    for path in [args.input_jsonl, args.prompt, root / "__init__.py", root / "common.py", root / "batch.py",
                 root / f"infer_{backend}.py"]:
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    media = {p for row in samples for p in (row["images"] if "images" in row else [row["video"]])}
    for path in sorted(media):
        with path.open("rb") as handle:
            digest.update(hashlib.file_digest(handle, "sha256").digest())
    model = Path(args.model)
    if backend == "transformers" and model.is_dir():
        for name in ["config.json", "model.safetensors.index.json", "SHA256SUMS"]:
            if (model / name).is_file():
                digest.update(hashlib.sha256((model / name).read_bytes()).digest())
    return digest.hexdigest()


def run(args: argparse.Namespace, predict: Callable[[argparse.Namespace], dict], backend: str) -> None:
    if not args.input_jsonl:
        if args.resume:
            raise ValueError("--resume requires --input-jsonl")
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        write_result(args.output, predict(args))
        return
    if args.start != 0 or args.end is not None:
        raise ValueError("Set start/end in manifest rows when using --input-jsonl")
    if not math.isfinite(args.fps) or args.fps <= 0 or args.max_frames < 1 or args.max_new_tokens < 1:
        raise ValueError("FPS and token/frame limits must be positive and finite")
    samples = read_samples(args.input_jsonl)
    signature = run_signature(args, backend, samples)
    sidecar = args.output.with_name(args.output.name + ".run.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    latest = {}
    with args.output.open("r+" if args.resume else "x+", encoding="utf-8") as handle:
        # A single writer owns the output even if two resume commands are started.
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if json.loads(sidecar.read_text()).get("signature") != signature:
                raise ValueError("Manifest, prompt, code, model reference, or settings changed; use a new output")
            ids = {row["sample_id"] for row in samples}
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"Output line {number} is incomplete/invalid; original file preserved") from exc
                if record.get("sample_id") not in ids or record.get("run_signature") != signature:
                    raise ValueError(f"Output line {number} belongs to a different run")
                latest[record["sample_id"]] = record
        else:
            with sidecar.open("x", encoding="utf-8") as metadata:
                json.dump({"signature": signature, "backend": backend, "samples": len(samples)}, metadata)
                metadata.write("\n")
        handle.seek(0)
        contents = handle.read()
        handle.seek(0, os.SEEK_END)
        if contents and not contents.endswith("\n"):
            handle.write("\n")
        skipped, attempted = 0, 0
        for position, row in enumerate(samples, 1):
            sample_id = row["sample_id"]
            previous = latest.get(sample_id, {})
            if (previous.get("status") == "ok" and previous.get("finish_reason") == "stop"
                    and previous.get("parse_error") is None and isinstance(previous.get("parsed_output"), dict)):
                skipped += 1
                continue
            item = argparse.Namespace(**(vars(args) | row))
            try:
                record = predict(item)
                error = record["parse_error"]
                if record["finish_reason"] != "stop":
                    error = error or f"Generation finished with {record['finish_reason']}"
                record.update(status="error" if error else "ok", error=error)
            except Exception as exc:
                record = dict(model="Verus-Insight", backend=backend, status="error",
                              error=f"{type(exc).__name__}: {exc}", raw_text=None, parsed_output=None,
                              parse_error=None, finish_reason=None,
                              input={"sample_id": sample_id})
            record.update(sample_id=sample_id, run_signature=signature)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            latest[sample_id] = record
            attempted += 1
            print(f"[{position}/{len(samples)}] {sample_id}: {record['status']}", file=sys.stderr, flush=True)
    failed = sum(record["status"] != "ok" for record in latest.values())
    print(json.dumps(dict(samples=len(samples), attempted=attempted, skipped=skipped,
                          succeeded=len(samples) - failed, failed=failed)), flush=True)
    if failed:
        raise SystemExit(1)
