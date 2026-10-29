#!/usr/bin/env python3
"""Fail-closed installer of bounded CPU-decoding progressive QUIC v2.

Dry run by default. Preserves the previously patched main.py and renderer_cuda.py.
Only replaces the two experimental QUIC sidecar modules after compatibility checks.
"""
import argparse
import ast
import os
from pathlib import Path
import shutil
import tempfile

PAIR = {
    'qntc_progressive_live.py': 'qntc_progressive_live_v2.py',
    'qntc_progressive_send.py': 'qntc_progressive_send_v2.py',
}
BACKUP_SUFFIX = '.before_bounded_progressive_v2'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--viewer', type=Path, default=Path.home()/'3DGS-QuaNTiC-Demo')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--restore', action='store_true')
    args = ap.parse_args()
    root = args.viewer.expanduser().resolve()
    here = Path(__file__).resolve().parent
    for path, marker in [('main.py','QNTC_PROGRESSIVE_QUIC_REAL_V1'),
                         ('renderer_cuda.py','QNTC_PROGRESSIVE_REDRAW_REAL_V1')]:
        dst = root/path
        if not dst.is_file() or marker not in dst.read_text():
            raise SystemExit(f'[STOP] {dst}: expected original pilot hook is missing; nothing changed')
    for target_name, src_name in PAIR.items():
        target = root/target_name
        source = here/src_name
        backup = root/(target_name+BACKUP_SUFFIX)
        if args.restore:
            if not backup.is_file():
                raise SystemExit(f'[STOP] Missing original backup {backup}')
        else:
            if not target.is_file() or not source.is_file():
                raise SystemExit(f'[STOP] Missing expected module {target} or {source}')
            if target_name == 'qntc_progressive_live.py':
                current = target.read_text()
                if ('class LiveQUICState' not in current or
                    'def get_ntc_for_index' not in current or
                    'def process_new_files' not in current):
                    raise SystemExit(f'[STOP] Unexpected receiver module {target}')
            ast.parse(source.read_text(),filename=str(source))
            if backup.exists() and target.read_bytes() != source.read_bytes():
                raise SystemExit(f'[STOP] Existing v2 backup and modified module {target}; review manually')
    if args.restore:
        if not args.apply:
            print('[RESTORE DRY RUN OK] Sidecar backups available. Pass --apply --restore.')
            return
        for target_name in PAIR:
            target = root/target_name
            backup = root/(target_name+BACKUP_SUFFIX)
            shutil.copy2(backup,target)
            print('[RESTORED]',target)
        print('[RESTORED] No main.py/renderer_cuda.py/NTC data touched')
        return
    print('[DRY RUN OK] Compatible existing QUIC hooks; new CPU-worker receiver and full-sequence sender')
    print('[CHECK] No changes to main.py, renderer_cuda.py, colors, or NTC compression')
    for target_name,src_name in PAIR.items():
        print('[CHANGE]',root/target_name,'<=',here/src_name)
    if not args.apply:
        print('[NEXT] Run with --apply after closing the viewer and sender')
        return
    for target_name,src_name in PAIR.items():
        target = root/target_name
        backup = root/(target_name+BACKUP_SUFFIX)
        if target.read_bytes() == (here/src_name).read_bytes():
            print('[ALREADY INSTALLED]',target)
            continue
        if not backup.exists():
            shutil.copy2(target,backup)
        # Atomic replacement, never truncate installed source in place.
        with tempfile.NamedTemporaryFile('wb',dir=root,prefix=target_name+'.',suffix='.part',delete=False) as tmp:
            tmp.write((here/src_name).read_bytes())
            pending=Path(tmp.name)
        try:
            os.chmod(pending,target.stat().st_mode)
            os.replace(pending,target)
        finally:
            pending.unlink(missing_ok=True)
        print('[INSTALLED]',target,'[BACKUP]',backup)
    print('[DONE] Restart receiver with fresh --quic_clear_cache; send all 299 updates.')


if __name__=='__main__':
    main()
