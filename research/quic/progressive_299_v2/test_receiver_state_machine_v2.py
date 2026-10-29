#!/usr/bin/env python3
"""Minimal no-GPU receiver state-machine smoke test."""
import sys, types, tempfile, time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))

class Dummy:
    def __init__(self, cache_root, total_frames_hint=300, autoplay=True, **kw):
        self.cache_root=Path(cache_root); self.total_frames_hint=total_frames_hint
        self.init_ply_ok=True; self.config_ok=True; self.max_play=-1
        self.autoplay=autoplay; self.status='idle'; self.err=''
    def process_new_files(self, renderer, camera):
        self._renderer_ref=renderer
        self.max_play=-1
    def ui_snapshot(self): return {'ready_ntc':0,'ntc_files':0,'max_play':-1}

stub=types.ModuleType('live_quic');stub.LiveQUICState=Dummy;sys.modules['live_quic']=stub
from qntc_progressive_live_v2 import LiveQUICState
from qntc_progressive_wire import encode_stage

class Renderer: current_timestep=0

def tick_until(pred,live, renderer, limit=3):
    t0=time.time()
    while time.time()-t0<limit:
        live.process_new_files(renderer,None)
        if pred(): return
        time.sleep(.020)
    raise AssertionError('timeout '+repr((live.part_versions,live.max_play,live.err)))

with tempfile.TemporaryDirectory() as d:
    root=Path(d)/'NTCs/progressive';root.mkdir(parents=True)
    live=LiveQUICState(cache_root=d)
    renderer=Renderer()
    data=bytes([0x88])*32
    for stage in range(4):
        rec=[(stage*389+k,.002,data) for k in range(389)]
        p=root/f'NTC_000000_P{stage}.qblk'; p.write_bytes(encode_stage(0,stage,rec))
        tick_until(lambda:live.part_versions.get(0,0)==stage+1,live,renderer)
        assert live.max_play==1
    # Second frame may start only when previous one is FULLY prepared.
    p=root/'NTC_000001_P0.qblk';p.write_bytes(encode_stage(1,0,[(7,.01,data)]))
    renderer.current_timestep=1
    tick_until(lambda: live.part_versions.get(1,0)==1,live,renderer)
    assert live.max_play==2
    assert live._prepared[0][1].shape==(31120,64)
    print('[PASS] progressive updates 1/4 ... 4/4; next frame unlocked; bounded decoded CPU cache')
