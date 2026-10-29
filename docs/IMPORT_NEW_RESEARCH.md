# Bringing the newer research code into the public repository

The research handoff archive was assembled from the verified experiment packages and is available separately from the owner:
`3DGS_QuaNTiC_Undergraduate_Handoff_2026-10-29.zip`.

It includes `research/priority/dynamic_299`, `research/quic/dynamic_priority_adapter`, `research/quic/progressive_299_v2`, the NTC114 HashGrid mapper, and evaluation code.

## One-command import on Ubuntu

Download `3DGS_QuaNTiC_Undergraduate_Handoff_2026-10-29.zip` to your usual Firefox download folder, then:

```bash
cd ~/snap/firefox/common
unzip -q 3DGS_QuaNTiC_Undergraduate_Handoff_2026-10-29.zip
cd 3DGS-QuaNTiC-Undergraduate-Handoff
bash tools/import_to_github.sh --push
```

This uses a **separate Git worktree**, runs the archived CPU tests and pushes source changes to the handoff branch. Your original GPU viewer tree is not overwritten. This cannot synchronize locally patched `main.py` / `renderer_cuda.py` by itself; those require review.

## Review before pushing

```bash
cd ~/3DGS-QuaNTiC-Demo
git status --short
git fetch origin
git switch undergraduate-handoff-2026-10-29

# In another folder, unzip the handoff archive.
# Copy the source directories (not weights or scenes) into this checkout.
cp -a /path/to/3DGS-QuaNTiC-Undergraduate-Handoff/research ./
cp -a /path/to/3DGS-QuaNTiC-Undergraduate-Handoff/tools/handoff.py ./tools/
cp -a /path/to/3DGS-QuaNTiC-Undergraduate-Handoff/tests/test_qblk.py ./tests/
cp -a /path/to/3DGS-QuaNTiC-Undergraduate-Handoff/tests/test_source.py ./tests/

python tools/handoff.py test
git diff --check
git status --short
# Once reviewed:
git add research tools tests
git commit -m "Add recovered PQAT, saliency and QUIC research experiments"
git push -u origin undergraduate-handoff-2026-10-29
```

**Local QUIC viewer is a separate step.** The GitHub viewer is older than the tested patched local copy. The local handoff tool can copy a *review-only snapshot* of relevant viewer source files outside the Git checkout; inspect source, secrets, licensing and behavior before intentionally committing changes. No automated force-push or overwrite of a working renderer is performed.

## Keep out of Git

Original scene `.ply` files, NTC `.pth`, states `.pt`, generated `.npz` and `.qblk`, private TLS keys, certificates, experiment caches and full video/GIF media.

Before sharing new source publicly, consider whether unpublished research methods should instead be reviewed in a private repository.
