"""Unit tests for the standalone handoff QNP01 binary verifier."""
import importlib.util
from pathlib import Path
import struct
import unittest
import tempfile

SRC = Path(__file__).resolve().parents[1]/"tools/qntc_handoff_check.py"
spec = importlib.util.spec_from_file_location("qntc_handoff_check", SRC)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class TestStage(unittest.TestCase):
    def test_valid(self):
        raw = module.HEADER.pack(b"QNP01",114,0,1556,1)
        raw += module.RECORD.pack(93,0.125,bytes(32))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"stage.qblk"
            p.write_bytes(raw)
            self.assertEqual(module.read_stage(p),(114,0,1556,1))

    def test_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"stage.qblk"
            p.write_bytes(b"junk")
            with self.assertRaises(ValueError):
                module.read_stage(p)

if __name__ == "__main__":
    unittest.main()
