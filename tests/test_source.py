"""Check that the shipped Python source is syntactically valid."""
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
class SourceTests(unittest.TestCase):
    def test_all_py_compiles(self):
        candidates=sorted((ROOT/'research').rglob('*.py'))+sorted((ROOT/'tools').rglob('*.py'))
        self.assertGreater(len(candidates),20)
        for path in candidates:
            with self.subTest(path=str(path.relative_to(ROOT))):
                compile(path.read_text(encoding='utf-8'),str(path),'exec')
if __name__=='__main__':unittest.main()
