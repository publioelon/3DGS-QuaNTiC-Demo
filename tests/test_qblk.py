"""CPU-only tests for the experimental QNP01 sparse INT4 block format."""
import importlib.util
from pathlib import Path
import struct
import sys
import unittest

try: import numpy as np
except ImportError: np=None

ROOT=Path(__file__).resolve().parents[1]
WIRE=ROOT/'research/priority/dynamic_299/qntc_progressive_wire.py'

@unittest.skipIf(np is None,'numpy not installed')
class StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('q3dgs_qblk',WIRE)
        cls.mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.mod)

    def test_record_roundtrip_and_dequantization(self):
        packed=bytes([0x89]*32)   # low nibble 9 -> +1; high nibble 8 -> 0
        blob=self.mod.encode_stage(114,0,[(93,0.25,packed)])
        self.assertEqual(len(blob),14+40)
        frame,stage,records=self.mod.decode_stage(blob)
        self.assertEqual((frame,stage),(114,0))
        full=np.zeros((self.mod.TOTAL,self.mod.BLOCK),dtype=np.float32)
        self.mod.decode_records_into(full,records)
        self.assertTrue(np.all(full[93,0::2]==0.25))
        self.assertTrue(np.all(full[93,1::2]==0.0))
        self.assertEqual(float(full[92].sum()),0.0)

    def test_bad_header_and_duplicates(self):
        rec=(0,0.125,bytes(32))
        data=self.mod.encode_stage(2,1,[rec])
        with self.assertRaises(ValueError):self.mod.decode_stage(data+b'garbage')
        with self.assertRaises(ValueError):self.mod.decode_stage(b'BAD!!'+data[5:])
        dupe=self.mod.encode_stage(2,1,[rec,rec])
        with self.assertRaises(ValueError):self.mod.decode_stage(dupe)

    def test_stage_integrity(self):
        self.assertEqual(self.mod.HEADER.size,14)
        self.assertEqual(self.mod.RECORD.size,40)
        self.assertEqual(self.mod.TOTAL,31120)

if __name__=='__main__':unittest.main()
