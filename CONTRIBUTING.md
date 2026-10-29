# Contributing to Q3DGStream

## Development environment

Python 3.10+ is sufficient to run the synthetic `.qblk` demo and source checks. GPU experiments require Ubuntu, a compatible CUDA toolchain, PyTorch and tiny-cuda-nn as described in [Quickstart](docs/QUICKSTART.md).

Before opening a pull request:

```bash
python3 q3dgstream.py test
python3 q3dgstream.py demo
python3 -m compileall -q research tools q3dgstream.py
```

## Project organization

- Keep stable viewer improvements at the repository root and standalone experiments under `research/`.
- Avoid adding new experiment scripts with hardcoded user-specific filesystem paths when configurable command-line arguments will work.
- Include a minimal reproduction command, tested Python version, necessary scene/model inputs and expected output with every benchmark change.
- Maintain explicit distinction among **PTQ, PQAT, quantization bit depth, pruning ratio**, and **stage reception ratio**.
- Preserve model serialization compatibility when modifying the QNP01 `.qblk` parser.
- Do not commit private TLS credentials, trained weight checkpoints, scene PLY files or large generated data.

## Reporting results

Always identify the image rendering protocol, dataset sequence and frame count. Describe local QUIC pilot results separately from network-adaptation benchmarks. For progressive reconstruction, state whether the reference is the full **compressed** NTC or original uncompressed teacher.
