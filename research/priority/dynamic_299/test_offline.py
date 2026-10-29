#!/usr/bin/env python3
"""CPU-only synthetic tests; does not touch original PQAT/FP32 data."""
import numpy as np
from dynamic_priority_core import interpolate_saliency,priority_tiers,rank_retained_blocks,partition_in_four
from qntc_progressive_wire import encode_stage,decode_stage


def main():
    n=32
    anchors={'tiles':32,'frames':[15,23,31], 'scores':[
       [1.0 if j==0 else 0.1 for j in range(n)],
       [1.0 if j==8 else 0.1 for j in range(n)],
       [1.0 if j==16 else 0.1 for j in range(n)],
    ]}
    points=np.stack([np.arange(n,dtype=float),np.zeros(n),np.zeros(n)],axis=1)
    xa=interpolate_saliency(anchors,15)
    xb=interpolate_saliency(anchors,23)
    assert xa.argmax()==0 and xb.argmax()==8
    ta=priority_tiers(xa,points)
    tb=priority_tiers(xb,points)
    assert ta[0]==0 and tb[8]==0
    ids=np.array([0,1,2,3,30976,30977],dtype=np.int32)
    sens=np.zeros((n,4))
    sens[0,0]=2; sens[8,1]=2; sens[16,2]=2; sens[24,3]=2
    order_a,values_a=rank_retained_blocks(ids[:4],sens,ta,xa)
    order_b,values_b=rank_retained_blocks(ids[:4],sens,tb,xb)
    assert order_a[0]==0 and order_b[0]==1
    assert set(order_a)==set(ids[:4]) and set(order_b)==set(ids[:4])
    stages=partition_in_four(order_a,ids[-2:])
    assert len(stages)==4 and np.array_equal(np.concatenate(stages),np.r_[ids[-2:],order_a])
    by_id={i:(int(i),np.float32(.5),bytes([i%16]*32)) for i in ids}
    rebuilt=[]
    for j,stage in enumerate(stages):
        packet=encode_stage(114,j,[by_id[int(i)] for i in stage],expected_total=1556)
        frame,idx,recs=decode_stage(packet)
        assert frame==114 and idx==j
        rebuilt.extend(recs)
    assert set(r[0] for r in rebuilt)==set(int(i) for i in ids)
    assert all(r[2]==by_id[r[0]][2] for r in rebuilt)
    print('[PASS] dynamic saliency changes tile and block ranking')
    print('[PASS] MLP decoder records first; four stages cover all records exactly once')
    print('[PASS] binary stage encoding/decoding keeps original block bytes')
    print('[NOTE] Real CUDA gradient tests require the user\'s qntcstream environment')

if __name__=='__main__':main()
