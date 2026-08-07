# ps3-cluster end-to-end test — layer client CLI + primary→standby failover

Commit: `cfcc56e` on branch `kimi-k3-async-coordination` (`/home/ubuntu/repos/ram-coffers`).
CPU-only Python + C, loopback, no UI. No source code was edited.

## Summary

Built / re-used host binaries, generated a three-tier config with the new `--head-standby 1 --region-standby 1` flags, launched a real **14-process farm** (8 `run_expert.py` workers + 3 `run_subcluster.py` heads + 3 `run_region.py` coordinators), drove the new `tools/run_layer.py` CLI for exact and fast modes, killed primary processes, and verified primary→standby failover, `--ping`, and clean shutdown. The suite passed with 225 tests and zero `ResourceWarning`s. No code bugs or doc gaps were found on this commit.

## Environment and build

```bash
cd /home/ubuntu/repos/ram-coffers/ps3-cluster
make -j$(nproc) host
```

Result:
```
make: Nothing to be done for 'host'.
```

The C sources were unchanged by `cfcc56e`; the previously-built `build/expert_node_host` and `build/expert_node_rsxemu` binaries were still valid. numpy 2.x was already installed.

## Generate config and artifacts

```bash
mkdir -p /tmp/farm-cfcc56e/logs
python3 tools/gen_cluster_config.py --layer 0 --experts 8 --size 4 \
  --expert-host 127.0.0.1 --expert-port-base 9100 \
  --head-host 127.0.0.1 --head-port-base 9200 --head-standby 1 \
  --regions 2 --region-host 127.0.0.1 --region-port-base 9300 --region-standby 1 \
  -o /tmp/farm-cfcc56e/cluster.json
```

Output:
```
wrote /tmp/farm-cfcc56e/cluster.json: 2 regions, 2 subclusters, 8 consoles
```

The JSON contains:
- `sc-0000` on `127.0.0.1:9200`, standby `127.0.0.1:9202`, experts 0-3 on ports 9100-9103
- `sc-0001` on `127.0.0.1:9201`, standby `127.0.0.1:9203`, experts 4-7 on ports 9104-9107
- `rg-0000` on `127.0.0.1:9300`, standby `127.0.0.1:9302`, fronts `sc-0000`
- `rg-0001` on `127.0.0.1:9301`, standby `127.0.0.1:9303`, fronts `sc-0001`

Pack 8 distinct `.exp` files and an activation vector:

```bash
for e in $(seq 0 7); do
  python3 tools/pack_expert.py /tmp/farm-cfcc56e/L000-E000${e}.exp \
    --hidden 32 --inter 32 --layer 0 --expert $e --seed $e
done
python3 -c "
import numpy as np
rng = np.random.default_rng(12345)
x = rng.standard_normal(32).astype(np.float32)
np.save('/tmp/farm-cfcc56e/x.npy', x)
print('x shape', x.shape, 'dtype', x.dtype)
"
```

Output:
```
wrote /tmp/farm-cfcc56e/L000-E0000.exp hidden=32 inter=32
... (8 lines)
x shape (32,) dtype float32
```

## Launch the farm

A launch script spawned 14 processes:

```bash
# 8 expert workers
python3 tools/run_expert.py /tmp/farm-cfcc56e/L000-E0000.exp --host 127.0.0.1 --port 9100
...
python3 tools/run_expert.py /tmp/farm-cfcc56e/L000-E0007.exp --host 127.0.0.1 --port 9107

# 3 subcluster heads
python3 tools/run_subcluster.py --config /tmp/farm-cfcc56e/cluster.json --subcluster sc-0000
python3 tools/run_subcluster.py --config /tmp/farm-cfcc56e/cluster.json --subcluster sc-0000 --standby 0
python3 tools/run_subcluster.py --config /tmp/farm-cfcc56e/cluster.json --subcluster sc-0001

# 3 regional coordinators
python3 tools/run_region.py --config /tmp/farm-cfcc56e/cluster.json --region rg-0000
python3 tools/run_region.py --config /tmp/farm-cfcc56e/cluster.json --region rg-0000 --standby 0
python3 tools/run_region.py --config /tmp/farm-cfcc56e/cluster.json --region rg-0001
```

First line of each log:

```
layer 0 expert 0 (hidden=32 inter=32) listening on 127.0.0.1:9100
...
sc-0000 listening on 127.0.0.1:9200 for 4 consoles
sc-0000 listening on 127.0.0.1:9202 for 4 consoles
sc-0001 listening on 127.0.0.1:9201 for 4 consoles
rg-0000 listening on 127.0.0.1:9300 for 1 subclusters
rg-0000 listening on 127.0.0.1:9302 for 1 subclusters
rg-0001 listening on 127.0.0.1:9301 for 1 subclusters
```

A port-poll script confirmed all expected TCP listeners were up before any client calls were made:

```
all ports up: [9100, 9101, 9102, 9103, 9104, 9105, 9106, 9107, 9200, 9201, 9202, 9300, 9301, 9302]
```

## `run_layer.py --ping` (healthy farm)

```bash
python3 tools/run_layer.py --config /tmp/farm-cfcc56e/cluster.json --layer 0 --token 1 \
  --experts 0 1 --gates 0.5 0.5 --activation /tmp/farm-cfcc56e/x.npy \
  --output /tmp/farm-cfcc56e/ping_out.json --ping
```

Output:
```
rg-0000	up
rg-0001	up
```

## Exact-mode output vs flat dispatcher

Three interleaved tokens were run with `run_layer.py`, and each output was compared bit-for-bit to `DistributedExpertDispatcher` over the same 8 worker processes via `PersistentSocketTransport(config.expert_endpoints())`.

| token | experts | gates | `np.array_equal` to flat | non-degenerate |
|-------|---------|-------|--------------------------|----------------|
| 1 | `[2,7,0,5,3,6,1,4]` | `[0.6,0.2,0.5,0.1,0.7,0.4,0.3,0.8]` | **True** | yes |
| 2 | `[7,0,4,3,1,6,5,2]` | `[0.15,0.35,0.45,0.55,0.25,0.65,0.05,0.75]` | **True** | yes |
| 3 | `[3,5,1,7,2,4,0,6]` | `[-0.5,0.4,-0.3,0.2,0.6,-0.1,0.7,0.9]` | **True** | yes |

Example command for token 1:

```bash
python3 tools/run_layer.py --config /tmp/farm-cfcc56e/cluster.json --layer 0 --token 1 \
  --experts 2 7 0 5 3 6 1 4 \
  --gates 0.6 0.2 0.5 0.1 0.7 0.4 0.3 0.8 \
  --activation /tmp/farm-cfcc56e/x.npy --output /tmp/farm-cfcc56e/tiered_t1.npy
```

Output:
```
wrote /tmp/farm-cfcc56e/tiered_t1.npy: shape (32,) dtype float32
```

The same token via the public Python API (`SubclusterTransport` + `HierarchicalExpertDispatcher`) produced:

```
API exact vs flat array_equal=True requests_sent={'rg-0000': 1, 'rg-0001': 1}
```

This confirms the CLI and the Python API both use one request per region and return the exact same bytes as the flat dispatcher.

## Fast-mode inequality

Token 1 was rerun with `--fast`:

```bash
python3 tools/run_layer.py --config /tmp/farm-cfcc56e/cluster.json --layer 0 --token 1 \
  --experts 2 7 0 5 3 6 1 4 \
  --gates 0.6 0.2 0.5 0.1 0.7 0.4 0.3 0.8 \
  --activation /tmp/farm-cfcc56e/x.npy --output /tmp/farm-cfcc56e/tiered_t1_fast.npy --fast
```

Output:
```
wrote /tmp/farm-cfcc56e/tiered_t1_fast.npy: shape (32,) dtype float32
```

Comparison to the exact-mode file:

```
np.array_equal(fast, exact) = False
np.allclose(fast, exact, rtol=1e-5) = True
```

This is the expected result: fast mode re-associates float32 additions and produces a close but not bit-identical vector.

## Primary→standby failover

### 1. Kill primary subcluster head `sc-0000` (`127.0.0.1:9200`)

```bash
ss -ltnp | grep ':9200'
# LISTEN ... pid=52312 ...
kill -9 52312
ss -ltn | grep -E ':9200|:9202'
# LISTEN 0 5 127.0.0.1:9202 ...
```

Re-run token 2:

```bash
python3 tools/run_layer.py --config /tmp/farm-cfcc56e/cluster.json --layer 0 --token 2 \
  --experts 7 0 4 3 1 6 5 2 \
  --gates 0.15 0.35 0.45 0.55 0.25 0.65 0.05 0.75 \
  --activation /tmp/farm-cfcc56e/x.npy --output /tmp/farm-cfcc56e/head_fail_t2.npy
```

Output:
```
wrote /tmp/farm-cfcc56e/head_fail_t2.npy: shape (32,) dtype float32
np.array_equal(head_fail_t2, flat_t2) = True
```

The layer transparently reached `sc-0000` on its standby port `9202` under `rg-0000`.

### 2. Kill primary regional coordinator `rg-0000` (`127.0.0.1:9300`)

```bash
ss -ltnp | grep ':9300'
# LISTEN ... pid=52315 ...
kill -9 52315
ss -ltn | grep -E ':9300|:9302'
# LISTEN 0 5 127.0.0.1:9302 ...
```

Re-run token 3:

```bash
python3 tools/run_layer.py --config /tmp/farm-cfcc56e/cluster.json --layer 0 --token 3 \
  --experts 3 5 1 7 2 4 0 6 \
  --gates -0.5 0.4 -0.3 0.2 0.6 -0.1 0.7 0.9 \
  --activation /tmp/farm-cfcc56e/x.npy --output /tmp/farm-cfcc56e/region_fail_t3.npy
```

Output:
```
wrote /tmp/farm-cfcc56e/region_fail_t3.npy: shape (32,) dtype float32
np.array_equal(region_fail_t3, flat_t3) = True
```

The layer transparently reached `rg-0000` on its standby port `9302`, which then reached the `sc-0000` standby on `9202`.

### 3. `--ping` after failover

```bash
python3 tools/run_layer.py --config /tmp/farm-cfcc56e/cluster.json --layer 0 --token 1 \
  --experts 0 1 --gates 0.5 0.5 --activation /tmp/farm-cfcc56e/x.npy \
  --output /tmp/farm-cfcc56e/ping2.json --ping
```

Output:
```
rg-0000	up
rg-0001	up
```

`rg-0000` is still reported `up` because its standby is responding. `run_region.py --check-members` also kept `sc-0000` as `true` because a head standby was alive:

```bash
python3 tools/run_region.py --config /tmp/farm-cfcc56e/cluster.json --region rg-0000 --check-members
```

Output:
```
{
  "sc-0000": true
}
```

### 4. Kill both `rg-0000` addresses

```bash
ss -ltnp | grep ':9302'
# LISTEN ... pid=52316 ...
kill -9 52316
python3 tools/run_layer.py --config /tmp/farm-cfcc56e/cluster.json --layer 0 --token 1 \
  --experts 0 1 --gates 0.5 0.5 --activation /tmp/farm-cfcc56e/x.npy \
  --output /tmp/farm-cfcc56e/ping3.json --ping
```

Output:
```
rg-0000	down
rg-0001	up
```

`--ping` correctly reported `rg-0000` as `down` once both primary and standby endpoints were unreachable.

## Full test suite

```bash
PYTHONWARNINGS=error::ResourceWarning ./run_tests.sh
```

Final output:
```
Ran 225 tests in 262.458s

OK
```

Exit code `0`, and the test harness reported zero `ResourceWarning` occurrences.

## Cleanup

All remaining farm PIDs were terminated with `SIGTERM`:

```bash
for pid in $(cat /tmp/farm-cfcc56e/pids.txt); do kill -TERM "$pid" 2>/dev/null || true; done
sleep 1
pgrep -af 'python3.*run_(expert|subcluster|region)' || echo 'none'
# none
ss -ltn | grep -E '910[0-7]|92[0-3][0-9]|93[0-3][0-9]' || echo 'none'
# none
```

The only remaining listener on the box was an unrelated port (`29229`). Every worker log ended with `layer 0 expert E stopped`; every surviving coordinator log ended with `<id> stopped after N batches`. The two primaries killed with `SIGKILL` did not write a shutdown line, which is expected for an abrupt failure test.

## Code bugs and doc gaps

- **No code bugs found.** Exact-mode hierarchical output matches the flat dispatcher, fast mode is close but not bit-identical, failover works through both head and regional standbys, `--ping` reports up/down correctly, and the suite passes without warnings.
- **No doc gaps found on this commit.** `README.md` and `docs/PS3_CLUSTER_PORT.md` contain `run_layer.py` examples, and the new `--head-standby` / `--region-standby` generator flags are correctly wired to produce standby address blocks (verified by the deployment). `run_subcluster.py --standby 0` and `run_region.py --standby 0` used those addresses exactly as documented.

## Artifacts

- This report: `/home/ubuntu/repos/ram-coffers/ps3-cluster/test-report.md`
- Farm directory: `/tmp/farm-cfcc56e/`
  - config: `cluster.json`
  - activation: `x.npy`
  - expert files: `L000-E0000.exp` … `L000-E0007.exp`
  - outputs: `tiered_t1.npy`, `tiered_t2.npy`, `tiered_t3.npy`, `tiered_t1_fast.npy`, `head_fail_t2.npy`, `region_fail_t3.npy`
  - flat references: `flat_t1.npy`, `flat_t2.npy`, `flat_t3.npy`
  - verification result: `verify_summary.json`
  - launch log: `logs/*.log`
  - launch script: `launch.sh`
  - verification script: `verify.py`
- Updated skill: `/home/ubuntu/repos/ram-coffers/.agents/skills/testing-ps3-cluster/SKILL.md`
