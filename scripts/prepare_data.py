"""Download/check Verus-Bench and export a manifest for batch_infer.py."""

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Verus-Bench"))
    parser.add_argument("--download", action="store_true", help="Download the published dataset from Hugging Face")
    parser.add_argument("--revision", help="Optional dataset commit or tag for --download")
    parser.add_argument("--available-only", action="store_true", help="Explicitly prepare only complete samples")
    parser.add_argument("--frames-root", type=Path, help="Locally obtained missing benchmark frames")
    parser.add_argument("--source-media-root", type=Path, help="Locally obtained upstream videos for missing samples")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--output", default="metadata/inputs.local.jsonl", help="Output path relative to dataset root")
    parser.add_argument("--check", action="store_true", help="Only report available and missing samples")
    args = parser.parse_args()
    if args.check and (args.download or args.frames_root or args.source_media_root or args.available_only):
        parser.error("--check cannot download, import, or select data")
    if args.revision and not args.download:
        parser.error("--revision requires --download")
    root = args.dataset_root.resolve()
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="Verus-Team/Verus-Bench", repo_type="dataset",
                          revision=args.revision, local_dir=root,
                          allow_patterns=["metadata/*", "frames/*", "scripts/*", "docs/*", "*.md",
                                          "LICENSE", "THIRD_PARTY_DATA.csv", "ground_truth.jsonl",
                                          "sample_ids.txt", "RELEASE_MANIFEST.json"])
    helper = root / "scripts" / "prepare_data.py"
    if not helper.is_file():
        parser.error("Dataset is missing scripts/prepare_data.py; set --dataset-root or use --download")
    command = [sys.executable, str(helper), "--dataset-root", str(root)]
    if args.check:
        command.append("--check")
    else:
        command.extend(["--write-inputs", args.output, "--ffmpeg", args.ffmpeg])
        if args.available_only:
            command.append("--available-only")
        for flag, value in [("--frames-root", args.frames_root), ("--source-media-root", args.source_media_root)]:
            if value:
                command.extend([flag, str(value.resolve())])
    # The dataset owns media availability and extraction recipes; reuse its helper.
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
