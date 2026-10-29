import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('q3dgstream_cli',ROOT/'q3dgstream.py')
app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)
class CLIStageTests(unittest.TestCase):
    def test_valid_demo_stage(self):
        payload=app.HEADER.pack(b'QNP01',114,0,1556,1)+app.RECORD.pack(93,.25,bytes([0x89]*32))
        frame,stage,total,records=app.parse_qblk(payload)
        self.assertEqual((frame,stage,total,len(records)),(114,0,1556,1))
    def test_duplicate_record(self):
        rec=app.RECORD.pack(93,.25,bytes(32))
        with self.assertRaises(ValueError):app.parse_qblk(app.HEADER.pack(b'QNP01',114,0,1556,2)+rec+rec)
    def test_invalid_magic(self):
        with self.assertRaises(ValueError):app.parse_qblk(app.HEADER.pack(b'HELLO',114,0,1556,0))
