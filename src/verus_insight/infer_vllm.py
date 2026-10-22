"""Send the same timestamped frames to a vLLM chat-completions server."""

import base64
import io
import json
import os
import time
from urllib.request import Request, urlopen
from pathlib import Path

from . import DEFAULT_SERVED_MODEL
from .batch import run
from .common import MAX_PIXELS, MIN_PIXELS, MODEL_CONTEXT, SEED, arguments, prepare, make_result


def main(mode: str | None = None) -> None:
    parser = arguments(__doc__, mode)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=DEFAULT_SERVED_MODEL, help="vLLM served model name")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    run(args, predict, "vllm")


def predict(args) -> dict:
    frames, messages, metadata = prepare(args)
    images = iter(frames)
    paths = getattr(args, "images", None)
    paths = iter(paths) if paths else None
    for part in messages[-1]["content"]:
        if part["type"] == "image":
            if paths is not None:
                from PIL import Image
                path = Path(next(paths))
                with Image.open(path) as image:
                    mime = Image.MIME[image.format]
                data = path.read_bytes()
            else:
                buffer = io.BytesIO()
                next(images).save(buffer, format="PNG")
                data, mime = buffer.getvalue(), "image/png"
            url = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
            part.clear()
            part.update(type="image_url", image_url={"url": url})
    payload = {"model": args.model, "messages": messages, "temperature": 0.0,
               "top_p": 1.0, "max_tokens": args.max_new_tokens,
               "seed": SEED, "top_k": 20, "repetition_penalty": 1.0,
               "chat_template_kwargs": {"enable_thinking": False},
               "mm_processor_kwargs": {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}}
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    request = Request(args.base_url.rstrip("/") + "/chat/completions",
                      data=json.dumps(payload).encode(), headers=headers)
    started = time.monotonic()
    with urlopen(request, timeout=args.timeout) as response:
        result = json.load(response)
    choice = result["choices"][0]
    raw = choice["message"].get("content") or ""
    return make_result(
        raw, choice["finish_reason"], model="Verus-Insight", backend="vllm",
        input=metadata, parameters={"temperature": 0.0, "max_new_tokens": args.max_new_tokens,
                                    "top_p": 1.0, "top_k": 20, "repetition_penalty": 1.0,
                                    "seed": SEED, "enable_thinking": False, "model_context": MODEL_CONTEXT,
                                    "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        usage=result.get("usage"), elapsed_seconds=round(time.monotonic() - started, 3),
    )


if __name__ == "__main__":
    main()
