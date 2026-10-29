REAL QUIC PROGRESSIVE INT4 NTC DEMO (PILOT, NOT AN OFFLINE SIMULATION)
==================================================================
Goal: watch uncolored, normally shaded Flame Steak Gaussians update as
real application-prioritized QUIC block-record streams ARRIVE. This modifies
motion (NTC params), NOT the SH colors, geometry attributes, or compression.

CAVEATS / ACCURACY
- This IS a real local QUIC connection. Existing project uses aioquic QUIC
  streams, and the four stage files are sent sequentially as distinct streams.
- Priorities are application-level ordering, NOT transport-internal
  set_priority() or network QoS markings.
- Pilot 'saliency' order uses the previously measured (binary) HashGrid
  dependency map for NTC114 and dynamic fused 3D SalFoM tile priorities.
  It is NOT the exact weighted Jacobian/Taylor ranking from the successful
  one-NTC experiment; the 81%/91% error reductions must not be claimed for
  this new live run. Blocks absent from NTC114's map rank last.
- P0..P3 stage names are delivery quartiles, not independently decodable
  Gaussian tiles or verified multi-camera agreement.
- Receiver applies actual incoming 40-byte original INT4 records, zeros the
  not-yet-received HashGrid blocks, and preserves the original shared MLP first.
  It re-runs that frame's motion prediction on each stage arrival.
- This pilot deliberately SLOWS sending with stage-delay so a human can see
  changes; it is not a performance or latency benchmark.
- Stable viewer mode should be no-additions, NTC_ROTATION_MODE=none; preserve
  original colors by setting QNTC_PRIORITY_VIS=0.
- Read-only scene/wire inputs. Viewer main.py and renderer_cuda.py are backed
  up before modification. No existing scientific result files overwritten.
- We have checked Python syntax, pilot QUIC record round-trip, and source
  compatibility with the viewer archive; real CUDA+QUIC playback requires the
  user's actual Ubuntu environment and has NOT been run in this container.

UNPACK (download ZIP to ~/snap/firefox/common)
  cd ~/snap/firefox/common
  unzip -o QNTC_Real_QUIC_Progressive_Pilot.zip
  source ~/venvs/qntcstream/bin/activate

BUILD DYNAMIC PRIORITY TIMELINE IF NOT ALREADY DONE
  python QNTC_Real_QUIC_Progressive_Pilot/prepare_dynamic_saliency.py

INSTALL (viewer must be closed)
  cd ~/snap/firefox/common/QNTC_Real_QUIC_Progressive_Pilot
  python install_progressive_quic.py --viewer ~/3DGS-QuaNTiC-Demo
  # Must show [DRY RUN OK]. If not, STOP and provide diagnostic.
  python install_progressive_quic.py --viewer ~/3DGS-QuaNTiC-Demo --apply

TEST WITH TWO TERMINALS (sender and viewer on same Ubuntu laptop):

TERMINAL A: local TLS certificate if you do not already have one
  openssl req -x509 -newkey rsa:2048 -nodes -days 1 \
    -subj '/CN=localhost' \
    -keyout /tmp/qntc_progressive.key \
    -out /tmp/qntc_progressive.crt

TERMINAL A: start QUIC viewer (normal Gaussian colors; NOT offline autoload)
  cd ~/3DGS-QuaNTiC-Demo
  source ~/venvs/qntcstream/bin/activate
  QNTC_PRIORITY_VIS=0 QNTC_PROGRESSIVE_QUIC=1 NTC_DEBUG=0 \
    python main.py --quic_listen 5002 --quic_bind 127.0.0.1 \
      --quic_cache /tmp/qntc_progressive_cache --quic_clear_cache \
      --quic_cert /tmp/qntc_progressive.crt \
      --quic_key /tmp/qntc_progressive.key \
      --frames 300 --autoplay --video_fps 30

TERMINAL B: send first 20 NTC updates as actual QUIC priority parts
  cd ~/3DGS-QuaNTiC-Demo
  source ~/venvs/qntcstream/bin/activate
  python qntc_progressive_send.py --insecure --host 127.0.0.1 \
    --port 5002 --start 0 --end 19 --order saliency \
    --stage-delay 0.65

Expected receiver terminal:
  [REAL QUIC APPLY] frame=000 part=P0 received_blocks=396/1556 ...
  [REAL QUIC APPLY] frame=000 part=P1 received_blocks=.../1556 ...
  [REAL QUIC APPLY] frame=000 part=P2 ...
  [REAL QUIC COMPLETE] frame=000: all 1556 records received

Each stage arrival causes the last rendered timestep to be recalculated with
its now-larger NTC parameter set, preserving original colors and lighting.
Slow frame progression is intentional and caused by staged network delivery.
To send the full movie, use --end 298 after restarting viewer/cache.

COMPARISON
  To compare with ordinary parameter ordering, restart both and change
  sender to --order original. The same 40-byte records and same QUIC streams
  are sent; only their order changes.

RESTORE YOUR EXISTING WORKING VIEWER
  Close viewer, then:
  cd ~/snap/firefox/common/QNTC_Real_QUIC_Progressive_Pilot
  python install_progressive_quic.py --viewer ~/3DGS-QuaNTiC-Demo --restore

RESTORE does not remove your earlier priority-overlay patches, it restores the
viewer state immediately before THIS progressive QUIC patch.
