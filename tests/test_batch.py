"""Batch integrity checks; model responses here are explicit test doubles."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verus_insight.batch import read_samples, run
from verus_insight.common import arguments, dispatch, make_result
from verus_insight import infer_transformers


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "inputs.jsonl"
        self.output = self.root / "predictions.jsonl"
        self.video = ROOT / "examples/pole_cam.mp4"
        self.rows = [{"sample_id": name, "video": str(self.video), "start": 8.47, "end": 8.97}
                     for name in ["first", "second"]]
        self.write_rows(self.rows)
        parser = arguments("batch test")
        parser.add_argument("--model", default="Verus-Insight")
        self.args = parser.parse_args(["--input-jsonl", str(self.manifest), "--output", str(self.output)])

    def write_rows(self, rows):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def execute(self, predict):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            run(self.args, predict, "vllm")

    def records(self):
        return [json.loads(line) for line in self.output.read_text().splitlines()]

    def test_failures_resume_and_no_overwrite(self):
        def first_attempt(args):
            if args.sample_id == "second":
                return make_result('{"cut":', "length")
            return make_result('{}', "stop")
        with self.assertRaises(SystemExit):
            self.execute(first_attempt)
        records = self.records()
        self.assertEqual([r["status"] for r in records], ["ok", "error"])
        self.assertEqual(records[1]["raw_text"], '{"cut":')
        with self.assertRaises(FileExistsError):
            self.execute(first_attempt)
        self.args.resume = True
        retried = []
        def retry(args):
            retried.append(args.sample_id)
            return make_result('{}', "stop")
        self.execute(retry)
        self.assertEqual(retried, ["second"])
        self.assertEqual(len(self.records()), 3)
        self.execute(lambda args: self.fail("Completed sample was rerun"))
        self.assertEqual(len(self.records()), 3)

    def test_exception_does_not_drop_later_samples(self):
        def predict(args):
            if args.sample_id == "first":
                raise RuntimeError("test decoder failure")
            return make_result('{}', "stop")
        with self.assertRaises(SystemExit):
            self.execute(predict)
        self.assertEqual([r["status"] for r in self.records()], ["error", "ok"])

    def test_resume_rejects_changed_settings_and_truncated_output(self):
        self.execute(lambda args: make_result('{}', "stop"))
        self.args.resume = True
        original = self.output.read_bytes()
        self.args.fps = 2
        with self.assertRaises(ValueError):
            self.execute(lambda args: self.fail("Should fail before inference"))
        self.assertEqual(original, self.output.read_bytes())
        self.args.fps = 4
        with self.output.open('a') as handle:
            handle.write('{"unfinished":')
        damaged = self.output.read_bytes()
        with self.assertRaises(ValueError):
            self.execute(lambda args: self.fail("Should fail before inference"))
        self.assertEqual(damaged, self.output.read_bytes())

    def test_manifest_validation_and_relative_paths(self):
        (self.root / 'clip.mp4').write_bytes(b'path validation only')
        self.write_rows([{"sample_id": "valid", "video": "clip.mp4"}])
        self.assertEqual(read_samples(self.manifest)[0]['video'], self.root / 'clip.mp4')
        for rows in [[self.rows[0], self.rows[0]], [self.rows[0] | {"label": "real"}],
                     [self.rows[0] | {"start": True}], [self.rows[0] | {"video": "missing.mp4"}]]:
            self.write_rows(rows)
            with self.assertRaises((ValueError, FileNotFoundError)):
                read_samples(self.manifest)

    def test_transformers_loads_once_and_resume_loads_nothing(self):
        import torch

        class Inputs(dict):
            def to(self, device):
                return self

        processor = MagicMock()
        processor.return_value = Inputs(input_ids=torch.tensor([[1, 2]]))
        processor.decode.return_value = '{}'
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        model.config.text_config.max_position_embeddings = 16384
        model.generation_config.eos_token_id = 99
        model.generate.return_value = torch.tensor([[1, 2, 5, 99]])
        loader = MagicMock(return_value=model)
        processor_loader = MagicMock(return_value=processor)
        transformers = SimpleNamespace(__version__="test-double",
            AutoProcessor=SimpleNamespace(from_pretrained=processor_loader),
            Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=loader))
        argv = ['batch_infer.py', '--backend', 'transformers', '--model', 'Verus-Insight', '--device', 'cpu',
                '--input-jsonl', str(self.manifest), '--output', str(self.output)]
        with patch.dict(sys.modules, {'transformers': transformers}), patch.object(sys, 'argv', argv):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                dispatch("batch")
                argv.append('--resume')
                dispatch("batch")
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(processor_loader.call_count, 1)
        self.assertEqual(model.generate.call_count, 2)
        self.assertEqual([r['status'] for r in self.records()], ['ok', 'ok'])

    def test_frame_batch_resume_detects_changed_image(self):
        image = self.root / 'frame.jpg'
        image.write_bytes((ROOT / 'examples/frames/pole_cam/frame_0000.jpg').read_bytes())
        self.write_rows([{'sample_id': 'frames', 'images': ['frame.jpg'], 'timestamps': [1.25]}])
        def predict(args):
            self.assertEqual(args.images, [image])
            self.assertEqual(args.timestamps, [1.25])
            return make_result('{}', 'stop')
        self.execute(predict)
        self.args.resume = True
        self.execute(lambda args: self.fail('Already completed'))
        image.write_bytes(image.read_bytes() + b'changed')
        with self.assertRaises(ValueError):
            self.execute(predict)


if __name__ == '__main__':
    unittest.main()
