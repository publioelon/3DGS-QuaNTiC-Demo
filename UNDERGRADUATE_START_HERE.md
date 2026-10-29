# 3DGS-QuaNTiC · Undergraduate handoff

> **Start with the working viewer.** This repository originally contained a TCP receiver and dynamic NTC renderer. Later QUIC, PQAT and saliency-prioritization research lives in separate experimental packages, whose **exact local GPU-tested viewer hooks are not yet present on `main`**.

## The pipeline

```text
Initial 3D Gaussians ───────────────────────────────────────────────┐
                                                                   ▼
Frame t NTC → sparse INT4 PQAT → saliency priority → P0 P1 P2 P3 .qblk
                                                                   │
                                      QUIC (ordered send)          ▼
                               decode → receiver-side NTC → render
```

An NTC transforms the **previous Gaussian state**, so skipping an NTC step is different from omitting some of its quantized weight blocks. Every partial reconstruction for one frame must use **the same pre-step state** to avoid double-applying motion.

## Quick start

For a fresh Ubuntu install, follow the existing [README](README.md) for the CUDA/PyTorch/tiny-cuda-nn setup, then:

```bash
git clone https://github.com/publioelon/3DGS-QuaNTiC-Demo.git
cd 3DGS-QuaNTiC-Demo
git fetch origin undergraduate-handoff-2026-10-29
git switch --track origin/undergraduate-handoff-2026-10-29
source ~/venvs/qntcstream/bin/activate
python tools/qntc_handoff_check.py doctor
python -m unittest discover -s tests -p 'test_qntc_handoff_check.py' -v
```

The **full recent research source bundle** is separately available as `3DGS_QuaNTiC_Undergraduate_Handoff_2026-10-29.zip` from the research owner. The archive includes source code and offline tests for priority generation, dynamic QUIC adapter, HashGrid mapping, and 299-frame evaluation. See [import instructions](docs/IMPORT_NEW_RESEARCH.md).

## Roadmap for the undergraduate

| First reproduce | Then integrate | Finally evaluate |
|---|---|---|
| Existing viewer loads initial PLY and NTCs | QUIC receiver accepts frame-indexed INT4 records | Stage arrival-to-render delay |
| Independent `.qblk` decode tests | Progressive NTC partial inference from same frame state | Translation and rotation error |
| 30-update local QUIC pilot | Initial scene, additions, P0–P3 and state-age reporting | Bandwidth/RTT/QoE tradeoff |

## Read next

- [What each piece does](docs/priority_streaming.md)
- [Binary `.qblk` specification](docs/qblk_wire_format.md)
- [Exact experimental evidence and limitations](docs/experimental_status_2026-10-29.md)
- [Import newer research source safely](docs/IMPORT_NEW_RESEARCH.md)

**Do not commit weights, scene datasets, Gaussian state dumps or QUIC TLS private keys.** Research code is public on this repository; review unpublished material before pushing it.
