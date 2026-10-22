"""Small CPU checks for input handling and preservation of unsuccessful output."""

import contextlib
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from verus_insight.common import arguments, dispatch, prepare, sample_video, save_result, make_result, validate_frames
from verus_insight import infer_vllm


class InferenceTests(unittest.TestCase):
    def test_nonfinite_model_numbers_are_invalid_json(self):
        for value in ['NaN', 'Infinity', '-Infinity']:
            with self.subTest(value=value):
                result = make_result('{"value":' + value + '}', 'stop')
                self.assertIsNotNone(result['parse_error'])

    def test_video_and_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments("test").parse_args([
                "--video", str(ROOT / "examples/pole_cam.mp4"),
                "--start", "8.47", "--end", "13.47",
                "--output", str(Path(directory) / "output.json"),
            ])
            frames, messages, metadata = prepare(args)
            self.assertEqual(len(frames), 20)
            self.assertEqual(metadata["timestamps_seconds"], [i / 4 for i in range(20)])
            content = messages[1]["content"]
            self.assertEqual(sum(x["type"] == "image" for x in content), 20)
            self.assertTrue(content[0]["text"].endswith("\n<0.00 seconds>"))
            self.assertEqual(content[-3]["text"], "<4.75 seconds>")
            self.assertNotIn("label", metadata)

    def test_invalid_intervals_and_frame_limit(self):
        video = ROOT / "examples/pole_cam.mp4"
        for start, end, fps, limit in [(0, 1, 0, 64), (0, 0, 4, 64),
                                       (0, 1, float("nan"), 64), (0, 1, 4, 1)]:
            with self.subTest(start=start, end=end, fps=fps, limit=limit):
                with self.assertRaises(ValueError):
                    sample_video(video, start, end, fps, limit)

    def test_raw_failure_preserved(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "output.json"
            with self.assertRaises(RuntimeError):
                save_result(output, '{"unfinished":', "length")
            result = json.loads(output.read_text())
            self.assertEqual(result["raw_text"], '{"unfinished":')
            self.assertEqual(result["finish_reason"], "length")
            self.assertTrue(result["parse_error"])
            with self.assertRaises(FileExistsError):
                save_result(output, '{}', "stop")

    def test_sampling_above_source_fps(self):
        frames, timestamps = sample_video(ROOT / "examples/pole_cam.mp4", 0, 0.2, 60, 64)
        self.assertGreater(len(frames), 6)
        self.assertEqual(timestamps, [round(i / 60, 6) for i in range(len(frames))])
        self.assertTrue(any(a is b for a, b in zip(frames, frames[1:])))

    def test_vllm_request_transport(self):
        # This checks the client wire format; it does not simulate model quality.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.json"
            argv = ["infer.py", "--backend", "vllm", "--video", str(ROOT / "examples/pole_cam.mp4"),
                    "--start", "8.47", "--end", "8.97", "--output", str(output)]
            def response(request, timeout):
                payload = json.loads(request.data)
                images = [p for p in payload["messages"][-1]["content"] if p["type"] == "image_url"]
                self.assertEqual(len(images), 2)
                self.assertTrue(images[0]["image_url"]["url"].startswith("data:image/png;base64,"))
                self.assertEqual(payload["temperature"], 0)
                return io.BytesIO(json.dumps({"choices": [{"message": {"content": "{}"},
                                                          "finish_reason": "stop"}]}).encode())
            with patch.object(sys, "argv", argv), patch.object(infer_vllm, "urlopen", response):
                with contextlib.redirect_stdout(io.StringIO()):
                    dispatch("single")
            self.assertEqual(json.loads(output.read_text())["backend"], "vllm")

    def test_reference_frames_and_request(self):
        args = arguments("test").parse_args([
            "--frames-json", str(ROOT / "examples/frames.json"), "--output", "unused.json"])
        frames, messages, metadata = prepare(args)
        reference = json.loads((ROOT / "examples/frame_request.json").read_text())
        self.assertEqual(messages, reference["messages"])
        self.assertEqual(metadata["image_sha256"], reference["image_sha256"])
        self.assertEqual(len(frames), 20)
        args.model, args.base_url, args.timeout = "Verus-Insight-8B", "http://localhost/v1", 60
        def response(request, timeout):
            payload = json.loads(request.data)
            for key, value in {"seed": 20260913, "top_p": 1.0, "top_k": 20,
                               "temperature": 0.0, "max_tokens": 8192,
                               "chat_template_kwargs": {"enable_thinking": False},
                               "mm_processor_kwargs": {"min_pixels": 3136, "max_pixels": 65536}}.items():
                self.assertEqual(payload[key], value)
            parts = payload['messages'][1]['content']
            paths = json.loads(args.frames_json.read_text())['images']
            images = [p for p in parts if p['type'] == 'image_url']
            for part, path in zip(images, paths):
                uri = part['image_url']['url']
                self.assertTrue(uri.startswith('data:image/jpeg;base64,'))
                self.assertEqual(base64.b64decode(uri.split(',', 1)[1]),
                                 (args.frames_json.parent / path).read_bytes())
            return io.BytesIO(b'{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}')
        with patch.object(infer_vllm, 'urlopen', response):
            self.assertIsNone(infer_vllm.predict(args)['parse_error'])

    def test_frame_validation_and_supplied_timestamps(self):
        row = json.loads((ROOT / 'examples/frames.json').read_text())
        directory = ROOT / 'examples'
        invalid = [row | {'timestamps': []}, row | {'timestamps': [float('nan')] * 20},
                   row | {'timestamps': list(reversed(row['timestamps']))},
                   row | {'target': {}}, row | {'images': ['missing.jpg'] * 20}]
        for item in invalid:
            with self.assertRaises((ValueError, FileNotFoundError)):
                validate_frames(item, directory)
        args = arguments('test').parse_args(['--frames-json', str(directory / 'frames.json'),
                                            '--output', 'unused.json', '--max-frames', '1'])
        with self.assertRaises(ValueError):
            prepare(args)
        args.frames_json = None
        args.images = [directory / row['images'][0], directory / row['images'][1]]
        args.timestamps, args.sample_id, args.max_frames = [1.37, 2.89], 'offset', 64
        _, messages, metadata = prepare(args)
        self.assertEqual(metadata['timestamps_seconds'], [1.37, 2.89])
        self.assertTrue(messages[1]['content'][0]['text'].endswith('<1.37 seconds>'))

    def test_image_and_directory_conversion(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = ROOT / 'examples/frames/pole_cam/frame_0000.jpg'
            for name in ['frame10.jpg', 'frame2.jpg', 'frame1.jpg']:
                shutil.copyfile(source, folder / name)
            parser = arguments('single', 'single')
            args = parser.parse_args(['--frames-dir', str(folder), '--fps', '2', '--output', 'unused.json'])
            _, messages, metadata = prepare(args)
            self.assertEqual(metadata['images'], ['frame1.jpg', 'frame2.jpg', 'frame10.jpg'])
            self.assertEqual(metadata['timestamps_seconds'], [0, 0.5, 1])
            times = folder / 'timestamps.json'
            times.write_text('[0.12, 1.37, 4.89]')
            args.timestamps_json = times
            _, _, metadata = prepare(args)
            self.assertEqual(metadata['timestamps_seconds'], [0.12, 1.37, 4.89])
            args = parser.parse_args(['--image', str(source), '--output', 'unused.json'])
            _, messages, metadata = prepare(args)
            self.assertEqual(metadata['num_frames'], 1)
            self.assertEqual(metadata['timestamps_seconds'], [0])
            self.assertTrue(messages[1]['content'][0]['text'].endswith('<0.00 seconds>'))
            args.timestamps_json = times
            with self.assertRaises(ValueError):
                prepare(args)


if __name__ == "__main__":
    unittest.main()
