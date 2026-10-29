#!/usr/bin/env python3
"""Read-only analysis of NTC 114 HashGrid->spatial-color vs QUIC-stage overlap.

Input: CSV produced by inspect_qntc_priorities.py --frame 114 --with-tiles.
Outputs: two new CSV reports in QNTC_priority_audit; does not change any NTC or QUIC data.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ('P0', 'P1', 'P2', 'P3')
COLORS = ('red', 'orange', 'yellow', 'blue')


def main():
    home = Path.home()
    base = home / 'snap/firefox/common/QNTC_priority_audit'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=base/'ntc_000114_stage_block_ids.csv')
    parser.add_argument('--out', type=Path, default=base)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error('Missing input CSV. Run inspect_qntc_priorities.py --frame 114 --with-tiles first.')
    with args.input.open(newline='') as fp:
        rd = csv.DictReader(fp)
        needed = {'ntc', 'block_id', 'network', 'transmission_priority'} | {f'gaussians_in_{p}' for p in LABELS}
        missing = needed - set(rd.fieldnames or ())
        if missing:
            parser.error('Input CSV is missing tile columns ' + str(sorted(missing)) + '. Re-run --with-tiles.')
        data = list(rd)

    seen = set()
    group = defaultdict(list)
    hash_rows = []
    mlp_by_stage = Counter()
    for row in data:
        if int(row['ntc']) != 114:
            raise ValueError('This mapping is validated for NTC 114 only')
        bid = int(row['block_id'])
        st = row['transmission_priority']
        if st not in LABELS or bid in seen:
            raise ValueError(f'Invalid stage or duplicate block ID: {bid} {st}')
        seen.add(bid)
        if row['network'] == 'MLP':
            mlp_by_stage[st] += 1
            continue
        if row['network'] != 'HashGrid':
            raise ValueError('Unrecognized network type: ' + row['network'])
        counts = [int(row[f'gaussians_in_{p}']) for p in LABELS]
        hash_rows.append((bid, st, counts))
        for i, c in enumerate(counts):
            if c > 0:
                group[(i, st)].append(bid)
    if len(seen) != len(data):
        raise ValueError('Unexpected duplicate block IDs')
    if len(data) != 1556 or len(hash_rows) != 1547 or sum(mlp_by_stage.values()) != 9:
        raise ValueError(f'Unexpected NTC114 counts: total={len(data)}, HashGrid={len(hash_rows)}, MLP={sum(mlp_by_stage.values())}')

    args.out.mkdir(parents=True, exist_ok=True)
    crossfile = args.out/'ntc_000114_spatial_vs_transmission_crosswalk.csv'
    with crossfile.open('w', newline='') as fp:
        wr = csv.writer(fp)
        wr.writerow(['block_id', 'transmission_priority', 'spatial_P0_red_gaussians',
                     'spatial_P1_orange_gaussians', 'spatial_P2_yellow_gaussians',
                     'spatial_P3_blue_gaussians', 'spatial_memberships', 'exclusive_spatial_priority'])
        for bid, stage, c in sorted(hash_rows):
            touched = [LABELS[i] for i, n in enumerate(c) if n > 0]
            wr.writerow([bid, stage, *c, ' '.join(touched), touched[0] if len(touched) == 1 else ''])

    stagefile = args.out/'ntc_000114_spatial_color_vs_quic_stage_ids.csv'
    with stagefile.open('w', newline='') as fp:
        wr = csv.writer(fp)
        wr.writerow(['spatial_priority', 'color', 'quic_transmission_priority', 'unique_hashgrid_blocks', 'original_block_ids'])
        for i, (color, label) in enumerate(zip(COLORS, LABELS)):
            for st in LABELS:
                ids = sorted(group[(i, st)])
                wr.writerow([label, color, st, len(ids), ' '.join(map(str, ids))])

    print('\n=== NTC 114: UNIQUE HashGrid BLOCKS CONNECTING EACH COLORED AREA ===')
    print('Rows: spatial colors. Columns: QUIC transmission stages (P0 sent first).')
    print('A HashGrid block may count in multiple spatial-color rows. MLP is omitted.')
    print(f"{'Spatial':<21}" + ''.join(f'{p:>9}' for p in LABELS) + f"{'TOTAL':>9}")
    totals = []
    for i, color in enumerate(COLORS):
        ns = [len(group[(i, st)]) for st in LABELS]
        total = sum(ns)
        totals.append(total)
        print(f'{LABELS[i]} {color:<18}' + ''.join(f'{n:9,d}' for n in ns) + f'{total:9,d}')
    assert totals == [1056, 1026, 1097, 1466], f'Unexpected tile coverage {totals}'

    print('\n=== SHARE OF EACH COLOR\'S STRUCTURAL BLOCKS AVAILABLE AFTER EACH STAGE ===')
    for i, color in enumerate(COLORS):
        cumulative = 0
        pieces = []
        for st in LABELS:
            cumulative += len(group[(i, st)])
            pieces.append(f'{st}: {cumulative:,}/{totals[i]:,} ({100*cumulative/totals[i]:.1f}%)')
        print(f'{LABELS[i]} {color}: ' + ' | '.join(pieces))

    exclusive = Counter()
    shared = Counter()
    all_colors = Counter()
    for bid, stage, counts in hash_rows:
        memberships = [LABELS[i] for i, count in enumerate(counts) if count > 0]
        shared[len(memberships)] += 1
        if len(memberships) == 1:
            exclusive[(memberships[0], stage)] += 1
        all_colors[stage] += 1
    print('\n=== SPATIAL MEMBERSHIP OVERLAP ===')
    for n in range(5):
        if shared[n]:
            print(f'HashGrid blocks touching {n} distinct color groups: {shared[n]}')
    print('Blocks touching exactly one spatial color, by QUIC stage:')
    for color in LABELS:
        print(f'  {color}: ' + ', '.join(f'{st}={exclusive[(color,st)]}' for st in LABELS))
    print('QUIC stage HashGrid block counts:', dict(sorted(all_colors.items())))
    print('QUIC stage MLP blocks:', {p: mlp_by_stage[p] for p in LABELS})
    print('\n[SAVED]', crossfile)
    print('[SAVED]', stagefile)
    print('[SAFE] Read-only with respect to NTC checkpoints, original .qblk, scene and receiver')

if __name__ == '__main__':
    main()
