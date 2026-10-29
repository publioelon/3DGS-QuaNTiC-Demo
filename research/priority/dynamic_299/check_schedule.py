#!/usr/bin/env python3
"""Verify produced ordering/stages include source sparse records without mutations."""
import argparse
from pathlib import Path
import sys
import numpy as np

from qntc_progressive_wire import extract_records_from_torch,decode_stage


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path.home()/'snap/firefox/common/QNTC_dynamic_299_priorities_v1')
    p.add_argument('--wire',type=Path,default=Path('/data/3dgs_storage/QNTC-Stream-PQAT-Chunks/qoe_eval/lowbit299_v11_20261011T014600_039285Z/cases/int4_keep05/pqat/wire'))
    p.add_argument('--start',type=int,default=114)
    p.add_argument('--end',type=int,default=114)
    p.add_argument('--stages',action='store_true',help='Check four QUIC stage files per NTC')
    a=p.parse_args()
    import torch
    for frame in range(a.start,a.end+1):
        source=a.wire/f'NTC_{frame:06d}.pth'
        order_path=a.out/'orders'/f'NTC_{frame:06d}.npz'
        with np.load(order_path,allow_pickle=False) as f:
            ids=f['ordered_block_ids'].astype(np.int64)
            assert int(f['ntc_index'])==frame
        source_records=extract_records_from_torch(torch.load(source,map_location='cpu',weights_only=True))
        source_by_id={r[0]:r for r in source_records}
        assert len(ids)==len(source_records)==1556
        assert len(set(map(int,ids)))==1556
        assert set(map(int,ids))==set(source_by_id)
        assert np.all(ids[:9]>=30976) and np.all(ids[9:]<30976), 'MLP blocks must arrive first'
        if a.stages:
            decoded=[]
            for stage in range(4):
                filename=a.out/'stages'/f'NTC_{frame:06d}_P{stage}.qblk'
                fr,st,records=decode_stage(filename.read_bytes())
                assert fr==frame and st==stage
                decoded.extend(records)
            assert [r[0] for r in decoded]==ids.tolist(), 'Stage order differs from schedule'
            assert all(record==source_by_id[record[0]] for record in decoded), 'Stage record bytes differ from source'
        print(f'[PASS] NTC {frame:03d}: {len(ids)} records, exact source IDs'+(' and bytes' if a.stages else ''),flush=True)
    print('[SUCCESS] All checked schedules intact; no source files changed')

if __name__=='__main__':main()
