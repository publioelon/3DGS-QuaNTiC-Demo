#!/usr/bin/env python3
"""Backed-up, fail-closed installer for the existing viewer; dry run by default."""
import argparse
import ast
import shutil
from pathlib import Path

MARK_MAIN='QNTC_PROGRESSIVE_QUIC_REAL_V1'
MARK_RENDER='QNTC_PROGRESSIVE_REDRAW_REAL_V1'


def transform_main(src):
    old='        from live_quic import LiveQUICState'
    replacement=(
        f'        # {MARK_MAIN}\n'
        '        if os.environ.get("QNTC_PROGRESSIVE_QUIC") == "1":\n'
        '            from qntc_progressive_live import LiveQUICState\n'
        '        else:\n'
        '            from live_quic import LiveQUICState')
    if MARK_MAIN in src:
        return src
    if src.count(old)!=1:
        raise RuntimeError('main.py QUIC import hook differs; refusing to patch')
    return src.replace(old,replacement,1)


def transform_renderer(src):
    if MARK_RENDER in src:
        return src
    # Main loop calls draw(timestep) each UI tick; update one already played
    # timestep by restoring the PRE-update state on receipt of another part.
    anchor='        with torch.no_grad():\n            while timestep - self.current_timestep > 0:'
    if src.count(anchor)!=1:
        raise RuntimeError('renderer_cuda.py draw loop hook differs; refusing to patch')
    replacement='''        # QNTC_PROGRESSIVE_REDRAW_REAL_V1
        _pq_state = getattr(self, 'qntc_live_priority_state', None)
        if _pq_state is not None and timestep > 0:
            _frame_id = timestep - 1
            _ver = _pq_state.part_versions.get(_frame_id, 0)
            _last = getattr(self, '_qntc_progressive_last', None)
            if (_last is not None and _last[0] == timestep
                    and _last[1] != _ver and _ver > 0
                    and getattr(self, '_qntc_progressive_before', None) is not None):
                with torch.no_grad():
                    self.gaussians.xyz.copy_(self._qntc_progressive_before[0])
                    self.gaussians.rot.copy_(self._qntc_progressive_before[1])
                self.current_timestep = timestep - 1
            if self.current_timestep == timestep - 1:
                self._qntc_progressive_before = (
                    self.gaussians.xyz.detach().clone(),
                    self.gaussians.rot.detach().clone(),
                )
                self._qntc_progressive_last = (timestep, _ver)

        with torch.no_grad():
            while timestep - self.current_timestep > 0:'''
    return src.replace(anchor,replacement,1)


def run(args):
    root=args.viewer.expanduser().resolve()
    main=root/'main.py'
    renderer=root/'renderer_cuda.py'
    for p in [main,renderer]:
        if not p.is_file():
            raise FileNotFoundError(p)
    if args.restore:
        for p in [main,renderer]:
            backup=p.with_name(p.name+'.before_progressive_quic')
            if not backup.is_file():
                raise FileNotFoundError('Missing backup: '+str(backup))
        for p in [main,renderer]:
            backup=p.with_name(p.name+'.before_progressive_quic')
            shutil.copy2(backup,p)
            print('[RESTORED]',p)
        print('[RESTORED] viewer code before progressive patch; sidecar files are inert')
        return
    original={main:main.read_text(),renderer:renderer.read_text()}
    patched={main:transform_main(original[main]),renderer:transform_renderer(original[renderer])}
    for p,text in patched.items():
        ast.parse(text,filename=str(p))
    here=Path(__file__).resolve().parent
    need=['qntc_progressive_wire.py','qntc_progressive_send.py','qntc_progressive_live.py']
    for name in need:
        ast.parse((here/name).read_text(),filename=name)
    print('[DRY RUN OK] main.py QUIC receive hook and renderer reapply hook compatible')
    print('[DRY RUN] colored overlay remains untouched; start with QNTC_PRIORITY_VIS=0')
    if args.apply:
        for p,contents in patched.items():
            backup=p.with_name(p.name+'.before_progressive_quic')
            if not backup.exists():
                shutil.copy2(p,backup)
            p.write_text(contents)
            print('[PATCHED]',p,'backup:',backup)
        for name in need:
            src=here/name
            shutil.copy2(src,root/name)
            print('[COPIED]',root/name)
        print('[DONE] Existing NTCs and original on-disk PLY were not modified')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer',type=Path,default=Path.home()/'3DGS-QuaNTiC-Demo')
    p.add_argument('--apply',action='store_true')
    p.add_argument('--restore',action='store_true')
    args=p.parse_args()
    if args.apply and args.restore:
        p.error('choose one of --apply or --restore')
    run(args)

if __name__=='__main__':
    main()
