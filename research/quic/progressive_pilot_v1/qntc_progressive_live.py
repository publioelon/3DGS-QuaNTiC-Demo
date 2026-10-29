"""Live QUIC receiver for progressive sparse INT4 block records.

Processes each *actually received* QUIC priority part in main/UI thread and
swaps parameters of the existing one-network live NTC loader. No color overlay.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import torch
from live_quic import LiveQUICState as FileLiveQUICState
from qntc_progressive_wire import TOTAL, BLOCK, decode_stage, decode_records_into

STAGE_RE = re.compile(r'^NTC_(\d{6})_P([0-3])\.qblk$')


class LiveQUICState(FileLiveQUICState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.part_versions = {}
        self._parts_seen = set()
        self._frame_data = {}       # up to last 4 frames of zero-initialized tensors
        self._frame_stages = {}
        self._frame_ids = {}
        self._frame_counts = {}
        self.hold_last_enabled = False  # never apply one frame's motion as another frame's motion
        self._progressive_path = self.cache_root/'NTCs/progressive'
        self._bounds_path = self._progressive_path/'bounds.json'

    def _ensure_bounds(self, renderer):
        if self._xyz_min is not None and self._xyz_max is not None:
            return
        if not self._bounds_path.is_file():
            raise RuntimeError('progressive bounds.json must arrive before config/initialization')
        d=json.loads(self._bounds_path.read_text())
        lo=torch.tensor(d['xyz_bound_min'],device='cuda',dtype=torch.float32)
        hi=torch.tensor(d['xyz_bound_max'],device='cuda',dtype=torch.float32)
        if lo.shape != (3,) or hi.shape != (3,) or not torch.all(hi>lo):
            raise ValueError('bad NTC xyz bounds')
        self._xyz_min,self._xyz_max=lo,hi
        print('[REAL QUIC] using exact source xyz bounds',flush=True)

    def process_new_files(self, renderer, camera):
        # Original file receiver still handles PLY/config and status panel.
        super().process_new_files(renderer,camera)
        renderer.qntc_live_priority_state=self
        if not self.init_ply_ok or not self.config_ok or not self._progressive_path.is_dir():
            return
        # Files are committed atomically by the existing QUIC transport.
        # Look only for the next three file indices to avoid scanning all old updates.
        candidates = sorted(self._progressive_path.glob('NTC_*_P*.qblk'))
        for path in candidates:
            if path.name in self._parts_seen:
                continue
            m=STAGE_RE.match(path.name)
            if not m:
                continue
            try:
                frame,stage,records=decode_stage(path.read_bytes())
                if int(m.group(1)) != frame or int(m.group(2)) != stage:
                    raise ValueError('QUIC filename disagrees with payload header')
                if frame not in self._frame_data:
                    self._frame_data[frame]=np.zeros((TOTAL,BLOCK),dtype=np.float32)
                    self._frame_stages[frame]=set()
                    self._frame_ids[frame]=set()
                    self._frame_counts[frame]=0
                if stage in self._frame_stages[frame]:
                    raise ValueError('duplicate stage')
                ids=[rec[0] for rec in records]
                if self._frame_ids[frame].intersection(ids):
                    raise ValueError('duplicate original block ID in frame')
                decode_records_into(self._frame_data[frame],records)
                self._frame_stages[frame].add(stage)
                self._frame_ids[frame].update(ids)
                self._frame_counts[frame]+=len(records)
                self.part_versions[frame]=self.part_versions.get(frame,0)+1
                self._parts_seen.add(path.name)
                self.max_play=max(self.max_play,frame+1)
                self.status=f'QUIC partial NTC {frame:03d} stage {stage}/3'
                print(f'[REAL QUIC APPLY] frame={frame:03d} part=P{stage} received_blocks={self._frame_counts[frame]}/1556 version={self.part_versions[frame]}',flush=True)
                if len(self._frame_stages[frame])==4:
                    if self._frame_counts[frame]!=1556:
                        raise RuntimeError('full NTC missing blocks')
                    print(f'[REAL QUIC COMPLETE] frame={frame:03d}: all 1556 records received',flush=True)
                # Safe retention: restrict to 6 CPU network states (48 MiB).
                for old in sorted(self._frame_data)[:-6]:
                    if len(self._frame_stages.get(old,())) == 4:
                        self._frame_data.pop(old,None)
                        self._frame_ids.pop(old,None)
                        self._frame_stages.pop(old,None)
                        self._frame_counts.pop(old,None)
            except Exception as exc:
                self.err=f'progressive QUIC frame ingest {path.name}: {exc!r}'
                print('[REAL QUIC ERROR]',self.err,flush=True)
                self._parts_seen.add(path.name)
        # super() recomputes availability from complete .pth files (there are none).
        # Preserve progression between QUIC part arrivals instead of falling back
        # to the total-frames hint, which would skip the entire animation.
        if self.part_versions:
            self.max_play = max(self.part_versions) + 1

    def get_ntc_for_index(self, idx):
        frame=int(idx)
        if frame not in self._frame_data:
            return None
        if not (self.init_ply_ok and self.config_ok):
            return None
        with self._ntc_lock:
            try:
                self._ensure_ntc_single(self._renderer_ref)
                version=self.part_versions[frame]
                signature=(frame,version)
                if getattr(self,'_progressive_loaded',None)!=signature:
                    params=torch.from_numpy(self._frame_data[frame].reshape(-1).copy())
                    self._ntc_single.model.params.data.copy_(
                        params.to(device=self._ntc_single.model.params.device,
                                  dtype=self._ntc_single.model.params.dtype))
                    self._progressive_loaded=signature
                self.active_ntc_idx=frame
                self.active_ntc_age=0
                return self._ntc_single
            except Exception as exc:
                self.err=f'progressive NTC CUDA load {frame}: {exc!r}'
                print('[REAL QUIC ERROR]',self.err,flush=True)
                return None

    def ui_snapshot(self):
        snap=super().ui_snapshot()
        snap['ready_ntc']=len(self.part_versions)
        snap['ntc_files']=sum(len(stages)==4 for stages in self._frame_stages.values())
        snap['max_play']=self.max_play
        return snap
