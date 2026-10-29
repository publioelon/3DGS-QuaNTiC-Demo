"""Source-level safety checks that run on CPU-only GitHub Actions."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/gpu_validation.py"
spec = importlib.util.spec_from_file_location("q3dg_gpu_validation", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class TestGpuValidation(unittest.TestCase):
    def test_source_compiles(self):
        compile(SCRIPT.read_text(), str(SCRIPT), "exec")

    def test_real_format_parser_on_synthetic_stage(self):
        # The test uses only synthetic QNP01 records: no CUDA is required.
        original = module.STAGES
        with tempfile.TemporaryDirectory() as folder:
            module.STAGES = Path(folder)
            try:
                for priority, size in enumerate((389, 389, 389, 389)):
                    ids = list(range(priority * 389, (priority + 1) * 389))
                    data = module.HEADER.pack(b"QNP01", 114, priority, 1556, size)
                    data += b"".join(module.RECORD.pack(i, 0.01, bytes(32)) for i in ids)
                    (module.STAGES / f"NTC_000114_P{priority}.qblk").write_bytes(data)
                result = module.stage_check(114)
                self.assertIn("1556 unique blocks", result)
            finally:
                module.STAGES = original

if __name__ == "__main__":
    unittest.main()
