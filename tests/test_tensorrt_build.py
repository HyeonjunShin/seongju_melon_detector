import json
import tempfile
import unittest
from pathlib import Path

from melon_pose_nmsfree.tensorrt_deploy import build_engine


class TensorRTBuildTest(unittest.TestCase):
    def test_refuses_to_overwrite_existing_engine_or_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = Path(folder) / 'local.engine'
            for existing in (engine, engine.with_suffix('.json')):
                with self.subTest(existing=existing.name):
                    existing.write_bytes(b'keep')
                    with self.assertRaises(FileExistsError):
                        build_engine(Path(folder) / 'missing.onnx', engine)
                    self.assertEqual(existing.read_bytes(), b'keep')
                    existing.unlink()

    def test_rejects_onnx_checksum_mismatch_before_building(self):
        with tempfile.TemporaryDirectory() as folder:
            onnx = Path(folder) / 'model.onnx'
            onnx.write_bytes(b'changed model')
            onnx.with_suffix('.json').write_text(json.dumps({'onnx_sha256': 'wrong'}))
            engine = Path(folder) / 'local.engine'
            with self.assertRaisesRegex(ValueError, 'checksum'):
                build_engine(onnx, engine)
            self.assertFalse(engine.exists())
            self.assertFalse(engine.with_suffix('.json').exists())
