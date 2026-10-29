#!/usr/bin/env python3
"""299-frame progressive NTC block evaluation. Run in qntcstream CUDA environment.
Read-only inputs; compares complete INT4 NTC with partial saliency/random blocks.
"""
import argparse,csv,json,struct,sys,time
from pathlib import Path
import numpy as np
import torch

H=struct.Struct('!5sIBHH'); R=struct.Struct('!If32s')
BASE=Path.home()/'snap/firefox/common'
CASE=Path('/data/3dgs_storage/QNTC-Stream-PQAT-Chunks/qoe_eval/lowbit299_v11_20261011T014600_039285Z/cases/int4_keep05/pqat')
PRIO=BASE/'QNTC_dynamic_299_priorities_v1'
STATES=BASE/'QNTC_299_student_states'
SCENE=Path.home()/'qntc_scenes/flame_steak_official'
VIEWER=Path.home()/'3DGS-QuaNTiC-Demo'


def records(frame,stage_dir):
    parts=[]; allids=[]; scales=[]; packed=[]; sizes=[]
    for p in range(4):
        path=stage_dir/f'NTC_{frame:06d}_P{p}.qblk'
        raw=path.read_bytes(); magic,f,priority,total,count=H.unpack_from(raw)
        if (magic,f,priority)!=(b'QNP01',frame,p) or len(raw)!=H.size+count*R.size:
            raise ValueError(f'Bad stage file: {path}')
        ids=[]
        for i in range(count):
            bid,scale,blob=R.unpack_from(raw,H.size+i*R.size)
            ids.append(bid); allids.append(bid); scales.append(scale);packed.append(blob)
        parts.append(ids); sizes.append(len(raw))
    if len(set(allids))!=len(allids) or len(allids)!=total:raise ValueError(f'Block count/duplicate frame {frame}')
    return parts,np.asarray(allids,dtype=np.int64),np.asarray(scales,dtype=np.float32),packed,sizes


def check_wire(frame,parts,ids,scales,packed,wire):
    d=torch.load(wire,map_location='cpu',weights_only=True)
    wids=d['model.params_sparse_int4_block_indices'].numpy().astype(np.int64)
    ws=d['model.params_sparse_int4_scales'].numpy().astype(np.float32)
    wp=d['model.params_sparse_qint4_packed'].numpy().reshape(-1,32)
    if len(ids)!=len(wids) or len(set(map(int,wids)))!=len(wids):raise ValueError(f'IDs count frame {frame}')
    lookup={int(b):i for i,b in enumerate(wids)}
    for b,s,blob in zip(ids,scales,packed):
        j=lookup.get(int(b))
        if j is None or struct.pack('!f',float(s))!=struct.pack('!f',float(ws[j])) or blob!=wp[j].tobytes():
            raise ValueError(f'Wire/stage mismatch frame {frame}, block {b}')
    if int(d['model.params_sparse_int4_block_size'])!=64:raise ValueError('Block size not 64')
    return d


def decode(ids,scales,packed,nblocks):
    result=np.zeros((nblocks,64),np.float32)
    for b,s,blob in zip(ids,scales,packed):
        a=np.frombuffer(blob,dtype=np.uint8)
        q=np.empty(64,np.float32);q[::2]=(a&15).astype(np.float32)-8;q[1::2]=(a>>4).astype(np.float32)-8
        result[b]=q*s
    return result


def balanced_tiles(xyz,n=32):
    tiles=[np.arange(len(xyz),dtype=np.int64)]
    while len(tiles)<n:
        _,i=max((len(ids),i) for i,ids in enumerate(tiles) if len(ids)>1)
        ids=tiles.pop(i);axis=int(np.argmax(np.ptp(xyz[ids],axis=0)))
        ordered=ids[np.argsort(xyz[ids,axis],kind='stable')]; mid=len(ordered)//2
        tiles.extend([ordered[:mid],ordered[mid:]])
    tiles.sort(key=lambda ids:tuple(np.mean(xyz[ids],axis=0).tolist()))
    return tiles


def infer(model,x,batch):
    ys=[]
    with torch.no_grad():
        for i in range(0,len(x),batch):
            z=x[i:i+batch].contiguous(); n=len(z)
            if n%128:z=torch.cat((z,z[-1:].expand(128-n%128,-1)),dim=0)
            ys.append(model(z)[:n,:7].float().cpu().numpy())
    return np.concatenate(ys) if ys else np.empty((0,7),np.float32)


def metrics(y,ref):
    dt=y[:,:3].astype(np.float64)-ref[:,:3].astype(np.float64)
    t=np.sqrt(np.mean(np.sum(dt*dt,axis=1))) if len(dt) else np.nan
    tr=ref[:,:3].astype(np.float64)
    trms=np.sqrt(np.mean(np.sum(tr*tr,axis=1))) if len(tr) else np.nan
    # Interpret raw 4-vector as quaternion update; compare sign-invariant orientation.
    a=y[:,3:7].astype(np.float64);b=ref[:,3:7].astype(np.float64)
    an=np.linalg.norm(a,axis=1);bn=np.linalg.norm(b,axis=1)
    valid=(an>1e-12)&(bn>1e-12)
    if valid.any():
        dots=np.abs(np.sum(a[valid]*b[valid],axis=1)/(an[valid]*bn[valid]))
        angle=2*np.degrees(np.arccos(np.clip(dots,0,1)))
        ang_rms=float(np.sqrt(np.mean(angle**2)));ang_mean=float(np.mean(angle))
    else:ang_rms=ang_mean=np.nan
    return {'translation_rms':float(t),'translation_reference_rms':float(trms),
            'translation_relative_rms':float(t/max(trms,1e-12)),
            'rotation_angle_rms_deg':ang_rms,'rotation_angle_mean_deg':ang_mean,
            'rotation_valid_count':int(valid.sum())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--verify-only',action='store_true')
    ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=298)
    ap.add_argument('--seeds',type=int,default=5);ap.add_argument('--batch',type=int,default=8192)
    ap.add_argument('--out',type=Path,default=BASE/'QNTC_299_saliency_vs_random_v1')
    a=ap.parse_args()
    if not 0<=a.start<=a.end<=298 or a.seeds<1 or a.batch%128:ap.error('Invalid range/seeds/batch')
    a.out.mkdir(parents=True,exist_ok=True)
    verified=[]; records_cache={}
    print('[VERIFY] Checking ALL 299 NTC wire checkpoints against priority stages',flush=True)
    for frame in range(299):
        p,ids,scales,packed,sizes=records(frame,PRIO/'stages')
        wire=CASE/'wire'/f'NTC_{frame:06d}.pth'
        d=check_wire(frame,p,ids,scales,packed,wire)
        verified.append({'frame':frame,'blocks':len(ids),'stage_counts':[len(v) for v in p],
                         'stage_bytes':sizes,'wire_bytes':wire.stat().st_size})
        if frame%25==0 or frame==298:print(f'[VERIFY] {frame+1}/299 passed',flush=True)
    (a.out/'wire_stage_verification.json').write_text(json.dumps(verified,indent=2))
    print('[PASS] 299/299 compressed wire models match priority stages exactly',flush=True)
    if a.verify_only:return
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for inference')
    import tinycudann as tcnn
    sys.path.insert(0,str(VIEWER));import util_gau
    init=np.asarray(util_gau.load_ply(str(SCENE/'init_3dgs.ply')).xyz,dtype=np.float32)
    tiles=balanced_tiles(init,32)
    cfg=json.loads((SCENE/'NTCs/config.json').read_text())
    model=tcnn.NetworkWithInputEncoding(n_input_dims=3,n_output_dims=8,
            encoding_config=cfg['encoding'],network_config=cfg['network']).cuda().eval()
    nblocks=model.params.numel()//64
    if model.params.numel()%64:raise ValueError('Invalid param shape')
    rows=[];summary=[]; csvpath=a.out/f'metrics_{a.start:03d}_{a.end:03d}.csv'
    colors=['red','orange','yellow','blue']
    for frame in range(a.start,a.end+1):
        tic=time.time()
        p,ids,scales,packed,sizes=records(frame,PRIO/'stages')
        order=np.load(PRIO/'orders'/f'NTC_{frame:06d}.npz',allow_pickle=False)
        tiers=np.asarray(order['tile_tiers'],dtype=np.int64)
        scheduled=np.asarray(order['ordered_block_ids'],dtype=np.int64)
        if len(tiers)!=32 or not np.array_equal(scheduled,ids):raise ValueError(f'Priority order mismatch {frame}')
        labels=np.full(len(init),-1,dtype=np.int8)
        for i,g in enumerate(tiles):labels[g]=int(tiers[i])
        if (labels<0).any() or (labels>3).any():raise ValueError('Invalid color tier')
        state=torch.load(STATES/f'state_{frame:06d}.pt',map_location='cpu',weights_only=True)
        if int(state['timestep'])!=frame:raise ValueError(f'State timestep mismatch {frame}')
        xyz=state['xyz'].numpy().astype(np.float32)
        if xyz.shape!=init.shape:raise ValueError('Gaussian count mismatch')
        wire=torch.load(CASE/'wire'/f'NTC_{frame:06d}.pth',map_location='cpu',weights_only=True)
        lo=wire['xyz_bound_min'].numpy().reshape(1,3);hi=wire['xyz_bound_max'].numpy().reshape(1,3)
        norm=(xyz-lo)/(hi-lo)
        # Match existing NTC114 experiment: only in-bounds Gaussians get active NTC output.
        active=np.logical_and(norm>=0,norm<=1).all(axis=1)
        full=decode(ids,scales,packed,nblocks)
        with torch.no_grad():model.params.copy_(torch.from_numpy(full.reshape(-1)).to(device='cuda',dtype=model.params.dtype))
        gid=np.flatnonzero(active);x=torch.from_numpy(np.ascontiguousarray(norm[gid])).cuda()
        reference=infer(model,x,a.batch)
        # all Gaussian population is represented; out-of-bounds Gaussians have no active transform.
        gids_by_color=[np.flatnonzero(labels[gid]==t) for t in range(4)]
        stage_cuts=np.cumsum([len(z) for z in p])
        for method in ['saliency']+[f'random_{seed}' for seed in range(a.seeds)]:
            ordering=ids if method=='saliency' else np.random.default_rng(20261028+frame*1009+int(method.split('_')[1])).permutation(ids)
            partial=np.zeros_like(full)
            previous=0
            for stage,cut in enumerate(stage_cuts):
                chosen=ordering[previous:cut]
                partial[chosen]=full[chosen]
                previous=cut
                with torch.no_grad():model.params.copy_(torch.from_numpy(partial.reshape(-1)).to(device='cuda',dtype=model.params.dtype))
                y=infer(model,x,a.batch)
                for tier,index in enumerate(gids_by_color):
                    if not len(index):continue
                    row={'frame':frame,'ordering':method,'stage':stage,'color':colors[tier],
                         'evaluated_gaussians':len(index),'total_gaussians':len(init),
                         'active_gaussians':len(gid),'blocks_received':int(cut),
                         'block_budget_fraction':float(cut/len(ids)),
                         'stage_payload_bytes':int(sum(sizes[:stage+1]))}
                    row.update(metrics(y[index],reference[index]));rows.append(row)
            print(f'[FRAME {frame:03d}] {method} done',flush=True)
        print(f'[FRAME {frame:03d}] finished in {time.time()-tic:.1f}s',flush=True)
        # Persist each frame atomically; allow progress inspection if run is interrupted.
        with csvpath.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('[DONE]',csvpath,flush=True)

if __name__=='__main__':main()
