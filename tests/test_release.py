"""Check the four public entry points without installing dependencies or using GPUs."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_four_entry_points_and_backend_help(self):
        self.assertEqual({p.name for p in (ROOT / 'scripts').iterdir() if p.is_file()},
                         {'install.sh', 'prepare_data.py', 'batch_infer.py', 'infer.py'})
        with tempfile.TemporaryDirectory() as directory:
            for name in ['infer.py', 'batch_infer.py']:
                for backend in ['transformers', 'vllm']:
                    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / name),
                                             '--backend', backend, '--help'], cwd=directory,
                                            capture_output=True, text=True, check=True)
                    self.assertIn('--backend', result.stdout)
                    self.assertIn('--base-url' if backend == 'vllm' else '--device', result.stdout)
                    self.assertIn('--input-jsonl' if name == 'batch_infer.py' else '--frames-dir', result.stdout)
                    self.assertNotIn('--video' if name == 'batch_infer.py' else '--input-jsonl', result.stdout)
            subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_data.py'), '--help'],
                           cwd=directory, capture_output=True, check=True)
            subprocess.run(['bash', str(ROOT / 'scripts/install.sh'), '--help'],
                           cwd=directory, capture_output=True, check=True)

    def test_installer_selects_environment_and_dependency_group(self):
        # Record commands instead of installing packages or contacting the network.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            executable = folder / 'fake-python'
            executable.write_text('#!' + sys.executable + '\n' + '''import json,os,shutil,sys
from pathlib import Path
with open(os.environ['INSTALL_TEST_LOG'], 'a') as handle:
    handle.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1:3] == ['-m', 'venv']:
    target = Path(sys.argv[3]) / 'bin/python'
    target.parent.mkdir(parents=True)
    shutil.copyfile(__file__, target)
    target.chmod(0o755)
''')
            executable.chmod(0o755)
            env = os.environ.copy()
            env.update(PYTHON=str(executable), INSTALL_TEST_LOG=str(folder / 'commands.jsonl'))
            command = ['bash', str(ROOT / 'scripts/install.sh'), 'vllm', str(folder / 'env with spaces')]
            subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            calls = [json.loads(line) for line in (folder / 'commands.jsonl').read_text().splitlines()]
            self.assertIn(['-m', 'pip', 'install', '-e', str(ROOT) + '[vllm]'], calls)
            self.assertIn(['-m', 'pip', 'check'], calls)
            result = subprocess.run(command[:2] + ['unknown'], env=env, capture_output=True)
            self.assertEqual(result.returncode, 2)

    def test_data_entry_forwards_paths_and_failure_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset with spaces'
            (root / 'scripts').mkdir(parents=True)
            (root / 'scripts/prepare_data.py').write_text(
                "import json,sys\nprint(json.dumps(sys.argv[1:]))\nraise SystemExit(2)\n")
            command = [sys.executable, str(ROOT / 'scripts/prepare_data.py'),
                       '--dataset-root', str(root), '--available-only', '--output', 'metadata/subset.jsonl']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            forwarded = json.loads(result.stdout)
            self.assertEqual(forwarded[forwarded.index('--dataset-root') + 1], str(root))
            self.assertIn('--available-only', forwarded)
            self.assertEqual(forwarded[forwarded.index('--write-inputs') + 1], 'metadata/subset.jsonl')


if __name__ == '__main__':
    unittest.main()
