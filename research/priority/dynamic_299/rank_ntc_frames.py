#!/usr/bin/env python3
"""Rank retained sparse INT4 HashGrid records separately for every selected NTC.

Uses saved 12-camera SalFoM 3D fused anchors (interpolated tile scores) and
fresh gradient times weight sensitivity on each decoded sparse NTC. The probe
coordinates are sampled from fixed initial persistent Gaussian positions;
these are approximate local dependencies, NOT dynamic Gaussian trajectories.

Nothing changes the working viewer, original FP32 checkpoints, compressed
checkpoint bytes, or the active QUIC sender/receiver. Emits small per-frame
order .npz files; optionally emits four exact-record .qblk QUIC stage files.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
from dynamic_priority_core import interpolate_saliency, priority_tiers, rank_retained_blocks, partition_in_four
from qntc_progressive_wire import extract_records_from_torch, encode_stage, decode_stage, HASH, TOTAL

BASE = Path('/data/3dgs_storage/QNTC-Stream-PQAT-Chunks/qoe_eval/lowbit299_v11_20261011T014600_039285Z/cases/int4_keep05/pqat')
DEFAULT_TIMELINE = Path.home()/'snap/firefox/common/NTC_dynamic_saliency_timeline.json'
DEFAULT_OUT = Path.home()/'snap/firefox/common/QNTC_dynamic_299_priorities_v1'
BS=64


def balanced_tiles(xyz, n):
    # Must match prepare_dynamic_saliency.py and NTC114 audit ordering exactly.
    tiles=[np.arange(len(xyz),dtype=np.int64)]
    while len(tiles)<n:
        _,i=max((len(ids),i) for i,ids in enumerate(tiles) if len(ids)>1)
        ids=tiles.pop(i)
        axis=int(np.argmax(np.ptp(xyz[ids],axis=0)))
        ordered=ids[np.argsort(xyz[ids,axis],kind='stable')]
        half=len(ordered)//2
        tiles.extend([ordered[:half],ordered[half:]])
    tiles.sort(key=lambda ids:tuple(np.mean(xyz[ids],axis=0).tolist()))
    return tiles


def decode_full_records(records):
    """Rebuild exact sparse decoder weights from wire source; no FP32 source substitution."""
    full=np.zeros((TOTAL,BS),dtype=np.float32)
    for block_id,scale,packed in records:
        data=np.frombuffer(packed,dtype=np.uint8)
        if len(data)!=32 or block_id<0 or block_id>=TOTAL or not np.isfinite(scale) or scale<0:
            raise ValueError('Invalid input record')
        values=np.empty(BS,dtype=np.float32)
        values[0::2]=(data & 15).astype(np.float32)-8
        values[1::2]=(data >> 4).astype(np.float32)-8
        full[int(block_id)]=values*np.float32(scale)
    return full.reshape(-1)


def load_timeline(path, scene, num_points, n_tiles):
    timeline=json.loads(path.read_text())
    if timeline.get('tiles')!=n_tiles or timeline.get('points')!=num_points:
        raise ValueError('Timeline was built for a different Gaussian count or tile count')
    source=Path(timeline['scene_ply']).resolve()
    if source!=(scene/'init_3dgs.ply').resolve():
        raise ValueError('Saliency timeline scene does not match selected PQAT scene; rebuild timeline')
    if len(timeline['frames'])<2:
        raise ValueError('At least two measured saliency anchors required')
    print('[SALIENCY] saved anchors:',len(timeline['frames']),'first/last:',timeline['frames'][0],timeline['frames'][-1],flush=True)
    return timeline


def save_frame_npz(path, ids, scores, tiers, sal, wire_path, elapsed, frame):
    # Atomic create avoids treating interrupted runs as complete; tiny files.
    temp=path.with_suffix('.npz.partial')
    with open(temp,'wb') as f:
        np.savez_compressed(f, ordered_block_ids=ids.astype(np.int32),
           priority_scores=scores.astype(np.float32), tile_tiers=tiers.astype(np.uint8),
           tile_saliency=sal.astype(np.float32), source_wire=str(wire_path),
           ntc_index=np.int32(frame), probe_elapsed_seconds=np.float32(elapsed))
    os.replace(temp,path)


def emit_stages(folder, frame, ordered_ids, records):
    by_id={int(r[0]):r for r in records}
    hash_ids=ordered_ids[ordered_ids < HASH]
    tail_ids=ordered_ids[ordered_ids >= HASH]
    stages=partition_in_four(hash_ids,tail_ids)
    flatten=np.concatenate(stages)
    assert len(flatten)==len(records) and set(map(int,flatten))==set(by_id)
    folder.mkdir(parents=True,exist_ok=True)
    for st,ids in enumerate(stages):
        data=encode_stage(frame,st,[by_id[int(x)] for x in ids])
        decoded_frame,decoded_st,decoded_records=decode_stage(data)
        assert decoded_frame==frame and decoded_st==st and len(decoded_records)==len(ids)
        dst=folder/f'NTC_{frame:06d}_P{st}.qblk'
        temp=dst.with_suffix('.qblk.partial')
        temp.write_bytes(data)
        os.replace(temp,dst)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--scene',type=Path,default=BASE/'scenes/candidate')
    a.add_argument('--wire',type=Path,default=BASE/'wire')
    a.add_argument('--timeline',type=Path,default=DEFAULT_TIMELINE)
    a.add_argument('--viewer',type=Path,default=Path.home()/'3DGS-QuaNTiC-Demo')
    a.add_argument('--out',type=Path,default=DEFAULT_OUT)
    a.add_argument('--start',type=int,default=114)
    a.add_argument('--end',type=int,default=114)
    a.add_argument('--tiles',type=int,default=32)
    a.add_argument('--samples-per-tile',type=int,default=128)
    a.add_argument('--probes',type=int,default=2)
    a.add_argument('--seed',type=int,default=114)
    a.add_argument('--emit-stages',action='store_true',help='Also create exact INT4 QUIC stage payloads without sending')
    a.add_argument('--overwrite',action='store_true',help='Recalculate existing completed schedules')
    o=a.parse_args()
    if not (0<=o.start<=o.end<=298 and 8<=o.tiles<=256 and o.samples_per_tile>=16 and 1<=o.probes<=16):
        a.error('Invalid frames/tiles/samples/probes')
    scene=o.scene.expanduser().resolve(); wire=o.wire.expanduser().resolve()
    out=o.out.expanduser(); timeline_path=o.timeline.expanduser()
    for f in [scene/'init_3dgs.ply',scene/'NTCs/config.json',timeline_path]:
        if not f.is_file(): raise FileNotFoundError(f)
    for i in range(o.start,o.end+1):
        if not (wire/f'NTC_{i:06d}.pth').is_file():
            raise FileNotFoundError(wire/f'NTC_{i:06d}.pth')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('tiny-cuda-nn gradient experiment requires CUDA')
    import tinycudann as tcnn
    sys.path.insert(0,str(o.viewer.expanduser().resolve()))
    import util_gau
    xyz=np.asarray(util_gau.load_ply(str(scene/'init_3dgs.ply')).xyz,dtype=np.float32)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
        raise ValueError('Invalid scene XYZ')
    print('[SCENE] persistent Gaussians:',len(xyz),flush=True)
    tiles=balanced_tiles(xyz,o.tiles)
    timeline=load_timeline(timeline_path,scene,len(xyz),o.tiles)
    centroids=np.asarray(timeline['centroids'],dtype=np.float64)
    if centroids.shape!=(o.tiles,3): raise ValueError('Invalid timeline centroids')
    expected=np.asarray([xyz[ids].mean(axis=0) for ids in tiles],dtype=np.float64)
    if not np.allclose(centroids,expected,atol=1e-4,rtol=1e-5):
        raise ValueError('Timeline tiles do not match NTC ranking spatial partitions')
    cfg=json.loads((scene/'NTCs/config.json').read_text())
    dev=torch.device('cuda')
    model=tcnn.NetworkWithInputEncoding(n_input_dims=3,n_output_dims=8,
        encoding_config=cfg['encoding'],network_config=cfg['network']).to(dev)
    model.train()
    if model.params.numel()!=TOTAL*BS:
        raise ValueError('Unexpected NTC architecture')
    records_dir=out/'orders'; stage_dir=out/'stages'
    records_dir.mkdir(parents=True,exist_ok=True)
    rng=torch.Generator(device='cuda')
    summary=[]
    for fnum in range(o.start,o.end+1):
        dst=records_dir/f'NTC_{fnum:06d}.npz'
        stage_paths=[stage_dir/f'NTC_{fnum:06d}_P{s}.qblk' for s in range(4)]
        if dst.is_file() and not o.overwrite and (not o.emit_stages or all(p.is_file() for p in stage_paths)):
            print(f'[SKIP] NTC {fnum:03d} already scheduled; --overwrite to recompute',flush=True)
            continue
        started=time.monotonic()
        rng.manual_seed(o.seed + fnum * 100003)
        file=wire/f'NTC_{fnum:06d}.pth'
        w=torch.load(file,map_location='cpu',weights_only=True)
        records=extract_records_from_torch(w)
        ids=np.asarray([r[0] for r in records],dtype=np.int32)
        h_ids=ids[ids < HASH]
        tail=ids[ids >= HASH]
        if len(h_ids)+len(tail)!=1556 or len(np.unique(ids))!=len(ids):
            raise ValueError(f'NTC {fnum}: unexpected sparse wire IDs')
        full_params=decode_full_records(records)
        with torch.no_grad():
            model.params.copy_(torch.from_numpy(full_params).to(device=dev,dtype=model.params.dtype))
        lo=torch.as_tensor(w['xyz_bound_min'],device=dev,dtype=torch.float32).reshape(1,3)
        hi=torch.as_tensor(w['xyz_bound_max'],device=dev,dtype=torch.float32).reshape(1,3)
        if (hi<=lo).any(): raise ValueError('Invalid NTC bounds')
        span=hi-lo
        sal=interpolate_saliency(timeline,fnum)
        tiers=priority_tiers(sal,centroids)
        # Unique NTC and unique retained IDs: fresh sensitivity, not NTC114 reuse.
        sens=np.zeros((o.tiles,len(h_ids)),dtype=np.float64)
        selected_gpu=torch.as_tensor(h_ids,dtype=torch.long,device=dev)
        for tile_idx,base_ids in enumerate(tiles):
            sample=np.linspace(0,len(base_ids)-1,min(len(base_ids),o.samples_per_tile),dtype=np.int64)
            pos=torch.as_tensor(xyz[base_ids[sample]],device=dev,dtype=torch.float32)
            x=(pos-lo)/span
            x=x[((x>=0)&(x<=1)).all(dim=1)].contiguous()
            if len(x)==0: continue
            for _ in range(o.probes):
                model.zero_grad(set_to_none=True)
                y=model(x)[:,:7].float()
                signs=torch.randint(0,2,y.shape,device=dev,dtype=torch.int8,generator=rng).float().mul_(2).sub_(1)
                loss=(y*signs).sum()/math.sqrt(float(y.numel()))
                loss.backward()
                if model.params.grad is None: raise RuntimeError('No tiny-cuda-nn parameter gradients')
                grad=model.params.grad.detach().float().reshape(-1,BS)
                weights=model.params.detach().float().reshape(-1,BS)
                v=((grad[selected_gpu]*weights[selected_gpu]).square().sum(dim=1))
                sens[tile_idx] += v.cpu().numpy()/o.probes
            del x,y,signs,loss,grad,weights,v,pos
        model.zero_grad(set_to_none=True)
        ordered_hash,rank_scores=rank_retained_blocks(h_ids,sens,tiers,sal)
        if not np.isfinite(rank_scores).all() or float(rank_scores.sum()) <= 0:
            raise RuntimeError(f'NTC {fnum} produced zero/nonfinite sensitivity; do not invent rankings')
        # The 9 MLP decoder blocks precede all spatial HashGrid records.
        final_ids=np.concatenate([tail,ordered_hash]).astype(np.int32)
        decoder_score = float(np.max(rank_scores)) + max(1.0, abs(float(np.max(rank_scores))))
        final_scores=np.concatenate([np.full(len(tail),decoder_score),rank_scores])
        if set(map(int,final_ids))!=set(map(int,ids)) or len(final_ids)!=len(ids):
            raise RuntimeError('Record integrity mismatch')
        elapsed=time.monotonic()-started
        save_frame_npz(dst,final_ids,final_scores,tiers,sal,file,elapsed,fnum)
        if o.emit_stages: emit_stages(stage_dir,fnum,final_ids,records)
        msg=f'[DONE] NTC {fnum:03d}: {len(tail)} MLP + {len(h_ids)} HashGrid blocks; '
        msg+=f'P0 tiles={np.flatnonzero(tiers==0).tolist()} P1={np.flatnonzero(tiers==1).tolist()}; {elapsed:.1f}s'
        print(msg,flush=True)
        summary.append({'ntc':fnum,'elapsed_seconds':round(elapsed,3),
                        'p0':np.flatnonzero(tiers==0).tolist(),
                        'p1':np.flatnonzero(tiers==1).tolist(),
                        'top_hash_blocks':ordered_hash[:12].tolist(),
                        'wire_bytes':file.stat().st_size})
        del w,full_params,records,sens,ids,h_ids,tail,final_ids,final_scores
    meta={
       'format':'qntc_per_ntc_saliency_sensitivity_rank_v1','scheduled_frames':summary,
       'source_scene':str(scene),'wire':str(wire),'timeline':str(timeline_path),
       'source_saliency':'saved SalFoM temporal anchors, interpolated tile importance',
       'sensitivity':'gradient times weight squared on each exact decoded sparse INT4 NTC',
       'coordinates':'sampled persistent Gaussians at initial xyz (NOT time-updated positions)',
       'wire_contents_unchanged':True,
       'interpretation':'priority schedule, not evidence of rendered image quality or original FP32 fidelity'
    }
    out.joinpath(f'run_{o.start:03d}_{o.end:03d}.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('[OUTPUT]',records_dir,flush=True)
    if o.emit_stages: print('[STAGES]',stage_dir,flush=True)
    print('[SAFE] Nothing in the viewer or source checkpoint directories was modified.',flush=True)

if __name__=='__main__': main()
