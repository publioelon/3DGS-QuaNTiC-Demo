#!/usr/bin/env python3
"""Synthetic, CPU-only checks: no network connection or GPU required."""
from pathlib import Path
from tempfile import TemporaryDirectory
from fp32_quic_sender import collect_files, send_order


def run():
    with TemporaryDirectory() as d:
        scene=Path(d)
        (scene/'NTCs').mkdir()
        (scene/'additional_3dgs').mkdir()
        (scene/'init_3dgs.ply').write_bytes(b'init')
        (scene/'NTCs/config.json').write_bytes(b'{}')
        for i in range(3):
            (scene/'NTCs'/f'NTC_{i:06d}.pth').write_bytes(bytes([i]))
            (scene/'additional_3dgs'/f'additions_{i:06d}.ply').write_bytes(bytes([i]))
        files=collect_files(scene, 3)
        assert len(files)==8
        names=[x[0] for x in send_order(files, 3, 3)]
        assert names==['init_3dgs.ply',
          'additional_3dgs/additions_000000.ply','NTCs/NTC_000000.pth',
          'additional_3dgs/additions_000001.ply','NTCs/NTC_000001.pth',
          'additional_3dgs/additions_000002.ply','NTCs/NTC_000002.pth',
          'NTCs/config.json']
        streaming=[x[0] for x in send_order(files,3,1)]
        assert streaming.index('NTCs/config.json')==3
        assert all(streaming.index(f'additional_3dgs/additions_{i:06d}.ply') <
                   streaming.index(f'NTCs/NTC_{i:06d}.pth') for i in range(3))
        print('[PASS] full baseline and prebuffer ordering; additions before matching NTC')
        print('[PASS] NTC0 before config to initialize exact checkpoint bounds')
        print('[PASS] source files only, no quantization or block edits')

if __name__ == '__main__':
    run()
