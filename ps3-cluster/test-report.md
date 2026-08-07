# PS3 cluster test report

## Commit

`eff57a3` on branch `kimi-k3-async-coordination`.

## Scope

- Persistent pooled P3XC transport (`transport.py`).
- Concurrent flat and hierarchical dispatch (`dispatch.py`, `hierarchy.py`).
- Deployable subcluster and regional coordinator services (`coordinator.py`, `regional.py`).
- Fixed-big-endian batch protocol with per-expert and per-region routing (`batch.py`).
- Request IDs, bounded dedup cache, safe/ambiguous retry, coordinator failover (`dedup.py`, `errors.py`).
- Three-tier configuration and CLI deployment (`deployment.py`, `tools/`).

## Test command

```bash
cd ps3-cluster
make -j$(nproc) host
PYTHONWARNINGS=error::ResourceWarning ./run_tests.sh
```

## Results

```text
Ran 213 tests in 248.485s
OK
```

Zero `ResourceWarning`s.

## Independent multi-process 3-tier failover deployment

Run by the testing agent on the same branch (commit `dd745ad`, and this report
revalidates `eff57a3`):

- 8 `tools/run_expert.py` console workers.
- 4 `tools/run_subcluster.py` subcluster heads (one with a standby).
- 2 `tools/run_region.py` regional coordinators (one with a standby).
- Separate layer-side client.

Verified:

- Three tokens with interleaved expert positions produced outputs byte-for-byte
equal to `DistributedExpertDispatcher` (`np.array_equal`).
- Persistent regional/subcluster connections were reused across tokens.
- Exactly one batch per immediate downstream group per token.
- Regional primary failure and subcluster primary failure both recovered through
standby endpoints.
- Both region endpoints unavailable raised an attributed `NodeConnectError` and
was not reduced.
- Fast mode produced a partial sum, differed from exact mode, and was correctly
rejected by `--refuse-fast`.
- `--list` and `--check-members` reported the tree.
- `run_region.py` correctly refused a two-tier (no regions) config.
- SIGTERM left no orphan processes, listening ports, or leaked connections.

## Known limitations

- No physical PlayStation 3 has executed this code; all validation is CPU-only
  on loopback sockets.
- A fourth configurable coordinator tier is not exposed by `ClusterConfig`, though
  the coordinator abstraction is recursive.
- Cross-process deduplication is unavailable without shared state; retries that
  reach a different coordinator process are at-least-once execution, while the
  caller reduces exactly once.
- Health steering is advisory; a fresh transport still pays one probe to
  discover a dead primary.
- Batching is per immediate downstream group per token, not across tokens.
- Requests on one console connection remain sequential.
- For non-PS3 model validation, the tiny checkpoint
  `inference-optimization/Kimi-K3-0.40B` on Hugging Face is a suitable
  single-node/few-node target.
