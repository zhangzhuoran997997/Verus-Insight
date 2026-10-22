"""Run a JSONL input manifest through Transformers or a vLLM server."""

from verus_insight.common import dispatch

if __name__ == "__main__":
    dispatch("batch")
