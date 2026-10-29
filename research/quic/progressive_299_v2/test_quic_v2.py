#!/usr/bin/env python3
"""CPU-only record decode parity check, no actual GPU/QUIC required."""
import sys
import types
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
# LiveQUICState class import needs live_quic, but decode_stages itself does not.
stub=types.ModuleType('live_quic')
stub.LiveQUICState=type('LiveQUICState',(),{})
sys.modules['live_quic']=stub
from qntc_progressive_live_v2 import decode_stages
from qntc_progressive_wire import encode_stage, decode_stage, decode_records_into, TOTAL, BLOCK
import tempfile

rng=np.random.default_rng(42)
ids=rng.choice(TOTAL,1556,replace=False)
scales=rng.uniform(.00001,.05,1556).astype(np.float32)
packed=rng.integers(0,256,(1556,32),dtype=np.uint8)
records=[(int(ids[k]),float(scales[k]),packed[k].tobytes()) for k in range(1556)]
with tempfile.TemporaryDirectory() as d:
    paths=[]
    for stage in range(4):
        part=records[stage*389:(stage+1)*389]
        p=Path(d)/f'NTC_000114_P{stage}.qblk'
        p.write_bytes(encode_stage(114,stage,part))
        paths.append((stage,p))
        ref=np.zeros((TOTAL,BLOCK),np.float32)
        for _,path in paths:
            _,_,r=decode_stage(path.read_bytes())
            decode_records_into(ref,r)
        frame,dense,num=decode_stages(paths)
        assert frame==114 and num==(stage+1)*389
        np.testing.assert_array_equal(dense,ref)
    print('[PASS] four-stage INT4 decoding equals original reference (1,991,680 params)')
    try:
        decode_stages(paths+[paths[0]])
    except ValueError:
        print('[PASS] duplicate block/stage rejected')
    else:
        raise AssertionError('duplicate stage accepted')
