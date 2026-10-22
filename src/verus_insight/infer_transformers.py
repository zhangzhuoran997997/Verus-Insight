"""Run Verus-Insight directly with Hugging Face Transformers."""

import time

from . import DEFAULT_MODEL
from .batch import run
from .common import MAX_PIXELS, MIN_PIXELS, MODEL_CONTEXT, SEED, arguments, prepare, make_result


def main(mode: str | None = None) -> None:
    parser = arguments(__doc__, mode)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model directory or Hugging Face repository ID")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face model commit or tag")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()
    model = processor = None

    def predict(args) -> dict:
        nonlocal model, processor
        started = time.monotonic()
        frames, messages, metadata = prepare(args)
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use a GPU environment or explicitly select --device cpu")
        if model is None:
            processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.model, revision=args.revision, dtype=getattr(torch, args.dtype), attn_implementation="sdpa",
            ).to(args.device).eval()
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        inputs = processor(
            text=[text], images=frames, return_tensors="pt",
            images_kwargs={"size": {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}},
        )
        inputs.pop("token_type_ids", None)
        prompt_tokens = inputs["input_ids"].shape[-1]
        if prompt_tokens + args.max_new_tokens > min(MODEL_CONTEXT, model.config.text_config.max_position_embeddings):
            raise ValueError("Input and requested output exceed the model context window")
        inputs = inputs.to(args.device)
        torch.manual_seed(SEED)
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                       repetition_penalty=1.0)
        new_tokens = generated[0, prompt_tokens:]
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        finish = "stop" if len(new_tokens) and new_tokens[-1].item() in eos else "length"
        raw = processor.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return make_result(
            raw, finish, model="Verus-Insight", backend="transformers",
            input=metadata, parameters={"do_sample": False, "max_new_tokens": args.max_new_tokens,
                                        "seed": SEED, "enable_thinking": False, "model_context": MODEL_CONTEXT,
                                        "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": len(new_tokens)},
            runtime={"torch": torch.__version__, "transformers": transformers.__version__,
                     "dtype": args.dtype, "attention": "sdpa"},
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

    run(args, predict, "transformers")


if __name__ == "__main__":
    main()
