"""The console-side worker: holds shards, runs experts, answers G9XC.

One of these runs on each console. It is deliberately simple — a threaded TCP
server over :mod:`gen9_cluster.protocol` — because the interesting engineering
is in *what it holds* and *where it holds it*, not in its concurrency model.

Two things here are worth reading closely:

:class:`ShardStore` is the coffer manager. It keeps a shard either in RAM or
memory-mapped from NVMe, and the distinction is visible to the caller, because a
routing hit on a memory-mapped expert costs a 2.4-5.5 GB/s read and the
coordinator deserves to know that is why the layer was slow. Mapping rather than
reading is what makes the NVMe tier usable at all: the page cache keeps the
experts that actually get routed to, which converges on the popular ones without
anybody having to profile them. That is the same effect KTransformers gets by
pinning hot experts to the GPU, arrived at by letting the kernel do it.

:class:`ExpertRunner` is the compute seam. The reference implementation is
numpy, which is honest and slow; a console plugs in the AVX2, Vulkan or HIP
kernel from ``kernels/`` by passing a different runner. Nothing else in the
stack knows which backend is in use. A runner implements
:meth:`ExpertRunner.rows` — one weighted output per expert, no cross-expert
addition — and the base class derives the collapsed sum from it, so no backend
gets to choose its own reduction order.

The node does not sum a batch unless it is asked to. Its reply is one row per
expert, tagged, and the coordinator adds them in top-k order; see
:mod:`gen9_cluster.protocol`. ``Flags.FAST`` opts into the collapsed sum.
"""

from __future__ import annotations

import socket
import socketserver
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from . import fp8, quants
from .dedup import (DedupCache, DedupCapacityError, MismatchedBatchError,
                    batch_fingerprint)
from .errors import CapacityError, ShardMissing
from .protocol import (HEADER_SIZE, DType, ExpertBatchPayload,
                       ExpertRowsPayload, Flags, Frame, HelloPayload, MsgType,
                       ShardHeader, accumulate, decode_vector, encode_error,
                       encode_vector)

#: (layer, expert) -> the three matrices of a DeepSeek expert.
ExpertKey = Tuple[int, int]


@dataclass
class ExpertWeights:
    """One routed expert: gate, up, down.

    Stored as whatever the loader produced. FP8 shards arrive as ``uint8``
    plus a scale array and are dequantised by the kernel, and block-packed
    formats (GGUF quants, mxfp4, fp4) arrive as runs of scale-interleaved
    blocks; the reference runner up-converts either kind once per call rather
    than pretending numpy speaks them.
    """

    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray
    scales: Optional[np.ndarray] = None
    #: The storage encoding: "fp32" for an ordinary buffer, "fp16"/"bf16"
    #: for a dense shard kept at its wire width, "fp8-e4m3-b128" for FP8
    #: codes whose scales live in ``scales``, "fp8-e4m3-tile" for the
    #: tile-scaled FP8 layouts with fp32 scales, "fp8-e4m3-tile-ue8m0" for
    #: the same with one-byte exponent scales, or a
    #: :mod:`gen9_cluster.quants` block format whose scales are packed into
    #: the blocks themselves.
    format: str = "fp32"
    #: (rows, cols) one scale covers, for the tile-scaled FP8 encodings.
    #: None is the flat per-128 layout of ``fp8-e4m3-b128``.
    scale_tile: Optional[Tuple[int, int]] = None
    #: Per-matrix fp32 tensor scale, for NVFP4 — three entries, gate, up,
    #: down, applied on top of the in-block scales when the expert is
    #: dequantised. None for every other format.
    tensor_scale: Optional[np.ndarray] = None
    #: "fast" | "slow" | "ssd" — which coffer this came out of.
    tier: str = "fast"

    @property
    def nbytes(self) -> int:
        total = self.gate.nbytes + self.up.nbytes + self.down.nbytes
        if self.scales is not None:
            total += self.scales.nbytes
        if self.tensor_scale is not None:
            total += self.tensor_scale.nbytes
        return total

    @property
    def quantised(self) -> bool:
        return self.gate.dtype == np.uint8

    @property
    def packed_fp8(self) -> bool:
        """Flat per-128-block FP8 the compiled kernel can read directly."""
        return (self.quantised and self.scales is not None
                and self.scale_tile is None)

    def dequantised(self) -> "ExpertWeights":
        """An fp32 copy, for runners that cannot read the storage encoding.

        This throws away the whole point of the packed formats — the expert is
        several times its stored size while it is being used — so it is done
        per call and discarded, never cached. A console that does this for
        every token is one whose compiled kernel failed to build, and its
        throughput will say so loudly.
        """
        if not self.quantised:
            if self.format in ("fp16", "bf16"):
                return ExpertWeights(
                    gate=quants.decode_dense(self.gate, self.format),
                    up=quants.decode_dense(self.up, self.format),
                    down=quants.decode_dense(self.down, self.format),
                    tier=self.tier)
            return self
        if self.scales is None and self.format not in (
                "fp32", "fp8-e4m3-b128", "fp8-e4m3-tile",
                "fp8-e4m3-tile-ue8m0"):
            # A block-packed codec: scales live inside the blocks, and NVFP4
            # adds a per-matrix tensor scale on top.
            decoded = ExpertWeights(
                gate=quants.dequantize_rows(self.gate, self.format),
                up=quants.dequantize_rows(self.up, self.format),
                down=quants.dequantize_rows(self.down, self.format),
                tier=self.tier)
            if self.tensor_scale is not None:
                decoded.gate *= self.tensor_scale[0]
                decoded.up *= self.tensor_scale[1]
                decoded.down *= self.tensor_scale[2]
            return decoded
        if self.format not in ("fp32", "fp8-e4m3-b128", "fp8-e4m3-tile",
                               "fp8-e4m3-tile-ue8m0"):
            raise ValueError(f"a {self.format} expert packs its scales into "
                             "the blocks; a separate scale array means the "
                             "shard is malformed")
        # ``scales`` present, or an explicitly FP8 shard: a codes-then-scales
        # layout, flat or tiled. (Experts built ad-hoc with uint8 codes and a
        # scale array — the format field left at its default — are the flat
        # kind too.)
        if self.scales is None:
            raise ValueError("a quantised expert carries neither a format "
                             "nor block scales; the shard is incomplete")
        # UE8M0 scales stay one byte each in the coffer — the whole point of
        # the encoding is that footprint — and decode here, per use.
        scales = (fp8.decode_ue8m0(self.scales)
                  if self.format == "fp8-e4m3-tile-ue8m0" else self.scales)
        flat = np.asarray(scales, dtype=np.float32).reshape(-1)
        if self.scale_tile is not None:
            counts = [fp8.tile_scales_needed(array.shape, self.scale_tile)
                      for array in (self.gate, self.up, self.down)]
            if flat.size != sum(counts):
                raise ValueError(f"expert needs {sum(counts)} tile scales, "
                                 f"carries {flat.size}")
            first, second = counts[0], counts[0] + counts[1]
            return ExpertWeights(
                gate=fp8.dequantize_tiled(self.gate, flat[:first],
                                          self.scale_tile),
                up=fp8.dequantize_tiled(self.up, flat[first:second],
                                        self.scale_tile),
                down=fp8.dequantize_tiled(self.down, flat[second:],
                                          self.scale_tile),
                tier=self.tier)
        sizes = [array.size for array in (self.gate, self.up, self.down)]
        counts = [fp8.n_blocks(size) for size in sizes]
        if flat.size != sum(counts):
            raise ValueError(f"expert needs {sum(counts)} block scales, "
                             f"carries {flat.size}")
        first, second = counts[0], counts[0] + counts[1]
        return ExpertWeights(
            gate=fp8.dequantize(self.gate, flat[:first]),
            up=fp8.dequantize(self.up, flat[first:second]),
            down=fp8.dequantize(self.down, flat[second:]),
            tier=self.tier)


class ExpertRunner:
    """Runs a batch of experts on one activation. Replaceable per backend.

    Backends override :meth:`rows`, never :meth:`__call__`. Returning the
    experts separately and letting the caller decide whether and in what order
    to add them is what keeps the answer independent of where the experts
    happen to live: a backend that summed internally would bake this node's
    particular shard assignment into the arithmetic.
    """

    name = "numpy-reference"

    def rows(self, activation: np.ndarray,
             experts: Sequence[ExpertWeights],
             gates: Sequence[float]) -> np.ndarray:
        """``gate_i * expert_i(activation)`` for each expert, as a 2-D array."""
        out = np.empty((len(experts), activation.size), dtype=np.float32)
        for index, (weights, gate) in enumerate(zip(experts, gates)):
            if weights.quantised or weights.format != "fp32":
                weights = weights.dequantised()
            hidden = activation @ weights.gate.T
            hidden = hidden * (1.0 / (1.0 + np.exp(-hidden)))   # SiLU
            hidden = hidden * (activation @ weights.up.T)
            out[index] = np.float32(gate) * (hidden @ weights.down.T)
        return out

    def __call__(self, activation: np.ndarray,
                 experts: Sequence[ExpertWeights],
                 gates: Sequence[float]) -> np.ndarray:
        """Gate-weighted sum of SwiGLU experts, folded left to right.

        Only used for a ``FAST`` batch, and defined here rather than in each
        backend so that every collapsed sum in the fleet folds in the same
        order whatever kernel produced the rows.
        """
        if not experts:
            return np.zeros_like(activation, dtype=np.float32)
        return accumulate(list(self.rows(activation, experts, gates)))


class ShardStore:
    """The node's residency: which experts it holds, and in which coffer."""

    def __init__(self, *, capacity_bytes: int = 0,
                 storage_dir: Optional[Path] = None):
        self.capacity_bytes = capacity_bytes
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self._experts: Dict[ExpertKey, ExpertWeights] = {}
        self._lock = threading.RLock()
        self.resident_bytes = 0

    def put(self, layer: int, expert: int, weights: ExpertWeights) -> None:
        with self._lock:
            if (self.capacity_bytes and weights.tier != "ssd"
                    and self.resident_bytes + weights.nbytes
                    > self.capacity_bytes):
                raise CapacityError(
                    f"shard would take the node to "
                    f"{(self.resident_bytes + weights.nbytes) / 2**30:.2f} GiB "
                    f"against a {self.capacity_bytes / 2**30:.2f} GiB budget",
                    layer=layer, expert=expert)
            self._experts[(layer, expert)] = weights
            if weights.tier != "ssd":
                self.resident_bytes += weights.nbytes

    def get(self, layer: int, expert: int) -> ExpertWeights:
        try:
            return self._experts[(layer, expert)]
        except KeyError:
            raise ShardMissing("shard not resident on this node", layer=layer,
                               expert=expert) from None

    def holds(self, layer: int, expert: int) -> bool:
        return (layer, expert) in self._experts

    def layers(self) -> Iterable[int]:
        return sorted({layer for layer, _ in self._experts})

    def __len__(self) -> int:
        return len(self._experts)

    def map_from_storage(self, layer: int, expert: int, path: Path, *,
                         hidden: int, intermediate: int,
                         dtype: str = "float32") -> ExpertWeights:
        """Memory-map an expert from NVMe instead of loading it.

        The OS page cache then decides which experts stay hot, which is both
        free and better than a hand-written policy: a routing distribution is
        exactly the access pattern an LRU is good at.
        """
        itemsize = np.dtype(dtype).itemsize
        counts = intermediate * hidden
        gate = np.memmap(path, dtype=dtype, mode="r", offset=0,
                         shape=(intermediate, hidden))
        up = np.memmap(path, dtype=dtype, mode="r", offset=counts * itemsize,
                       shape=(intermediate, hidden))
        down = np.memmap(path, dtype=dtype, mode="r",
                         offset=2 * counts * itemsize,
                         shape=(hidden, intermediate))
        weights = ExpertWeights(gate=gate, up=up, down=down, tier="ssd")
        self.put(layer, expert, weights)
        return weights


class NodeServer:
    """Serves G9XC on a console.

    ``block_fn`` is the hook for the hot block (attention + router + shared
    expert) on the shelf's host console; nodes that only hold experts leave it
    unset and reject BLOCK_FWD, which is a plan/deployment mismatch worth
    hearing about loudly.
    """

    def __init__(self, store: ShardStore, *, unit_id: str, host: str = "0.0.0.0",
                 port: int = 9713, runner: Optional[ExpertRunner] = None,
                 block_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
                 sku: str = "unknown", backend: str = "cpu-avx2",
                 runtime: str = "host-sim", weight_bytes: int = 0,
                 fast_bytes: int = 0, gemv_gflops: float = 0.0,
                 dedup: Optional[DedupCache] = None):
        self.store = store
        self.unit_id = unit_id
        self.host = host
        self.port = port
        self.runner = runner or ExpertRunner()
        self.block_fn = block_fn
        self.hello = HelloPayload(unit_id=unit_id, sku=sku, backend=backend,
                                  runtime=runtime, weight_bytes=weight_bytes,
                                  fast_bytes=fast_bytes,
                                  gemv_gflops=gemv_gflops)
        self.dedup = dedup if dedup is not None else DedupCache()
        self.tokens_served = 0
        self.experts_run = 0
        self.storage_reads = 0
        self._server: Optional[socketserver.ThreadingTCPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- serving ----------------------------------------------------------
    def serve_forever(self) -> None:
        server = self._make_server()
        server.serve_forever(poll_interval=0.2)

    def start(self) -> int:
        """Start in a background thread; returns the bound port."""
        server = self._make_server()
        self._thread = threading.Thread(target=server.serve_forever,
                                        kwargs={"poll_interval": 0.05},
                                        name=f"g9-node-{self.unit_id}",
                                        daemon=True)
        self._thread.start()
        return server.server_address[1]

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> "NodeServer":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def _make_server(self) -> socketserver.ThreadingTCPServer:
        node = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY,
                                        1)
                while True:
                    head = _read_exactly(self.request, HEADER_SIZE)
                    if head is None:
                        return
                    try:
                        frame, length = Frame.decode_header(head)
                    except ValueError:
                        return          # framing is gone; the peer must reopen
                    payload = b""
                    if length:
                        body = _read_exactly(self.request, length)
                        if body is None:
                            return
                        payload = body
                    frame.payload = payload
                    reply = node.handle_frame(frame)
                    if reply is None:
                        return
                    self.request.sendall(reply.encode())

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        server = Server((self.host, self.port), Handler)
        self._server = server
        self.port = server.server_address[1]
        return server

    # -- message handling -------------------------------------------------
    def handle_frame(self, frame: Frame) -> Optional[Frame]:
        """Dispatch one frame. Returns the reply, or None to close."""
        handler = {
            MsgType.HELLO: self._on_hello,
            MsgType.EXPERT_BATCH: self._on_expert_batch,
            MsgType.BLOCK_FWD: self._on_block,
            MsgType.LOAD_SHARD: self._on_load,
            MsgType.PING: self._on_ping,
            MsgType.STATUS: self._on_status,
        }.get(frame.msg_type)
        if frame.msg_type is MsgType.SHUTDOWN:
            return None
        if handler is None:
            return self._error(frame, f"unsupported message "
                                      f"{frame.msg_type.name}")
        try:
            return handler(frame)
        except (ShardMissing, CapacityError) as exc:
            return self._error(frame, str(exc))
        except Exception as exc:                    # noqa: BLE001 - reported
            return self._error(frame, f"{type(exc).__name__}: {exc}")

    def _on_hello(self, frame: Frame) -> Frame:
        return Frame(MsgType.HELLO_ACK, frame.request_id,
                     self.hello.encode())

    def _on_expert_batch(self, frame: Frame) -> Frame:
        """Run a batch once, and answer with rows unless FAST was asked for.

        The dedup cache wraps the *whole* reply including its flags, so a
        replay is byte-identical to what the first attempt sent — apart from
        ``REPLAYED``, which is set on the way out so the coordinator's stats
        can tell a cheap retry from an expensive one.
        """
        payload = ExpertBatchPayload.decode(frame.payload)
        fast = bool(frame.flags & Flags.FAST)
        fingerprint = batch_fingerprint(frame.layer, payload.expert_ids,
                                        payload.gates, payload.activation,
                                        fast)

        def compute() -> bytes:
            return self._run_batch(frame, payload, fast).encode()

        try:
            raw, replayed = self.dedup.run(payload.batch_id, fingerprint,
                                           compute)
        except MismatchedBatchError as exc:
            return self._error(frame, str(exc))
        except DedupCapacityError as exc:
            reply = self._error(frame, str(exc))
            reply.flags |= Flags.BACKPRESSURE
            return reply
        reply, _ = Frame.decode_header(raw[:HEADER_SIZE])
        reply.payload = raw[HEADER_SIZE:]
        reply.request_id = frame.request_id
        if replayed:
            reply.flags |= Flags.REPLAYED
        return reply

    def _run_batch(self, frame: Frame, payload: ExpertBatchPayload,
                   fast: bool) -> Frame:
        weights = [self.store.get(frame.layer, eid)
                   for eid in payload.expert_ids]
        rows = self.runner.rows(payload.activation, weights, payload.gates)
        self.experts_run += len(weights)
        self.tokens_served += 1
        flags = Flags.NONE
        if any(w.tier == "ssd" for w in weights):
            self.storage_reads += 1
            flags |= Flags.FROM_STORAGE
        if fast:
            flags |= Flags.PARTIAL
            body = encode_vector(accumulate(list(rows)))
        else:
            flags |= Flags.PER_EXPERT
            body = ExpertRowsPayload(payload.expert_ids, rows).encode()
        return Frame(MsgType.EXPERT_RESULT, frame.request_id, body,
                     layer=frame.layer, token=frame.token, flags=flags)

    def _on_block(self, frame: Frame) -> Frame:
        if self.block_fn is None:
            return self._error(frame, "this node hosts no hot blocks; the plan "
                                      "and the deployment disagree")
        out = self.block_fn(frame.layer, decode_vector(frame.payload))
        return Frame(MsgType.BLOCK_RESULT, frame.request_id, encode_vector(out),
                     layer=frame.layer, token=frame.token)

    def _on_load(self, frame: Frame) -> Frame:
        header, offset = ShardHeader.decode(frame.payload)
        body = frame.payload[offset:]
        try:
            if header.dtype is DType.FP8_E4M3_B128:
                self._load_fp8(header, body)
            elif header.dtype.fp8_tile is not None:
                self._load_fp8_tiled(header, body)
            elif header.dtype.packed:
                self._load_packed(header, body)
            else:
                self._load_dense(header, body)
        except ValueError as exc:
            return self._error(frame, str(exc))
        return Frame(MsgType.LOAD_ACK, frame.request_id, layer=header.layer,
                     expert=header.first_expert, dtype=header.dtype)

    def _load_dense(self, header: ShardHeader, body: bytes) -> None:
        """A dense shard: fp32, fp16 or bf16, kept at its wire width.

        Dense formats carry no scales, but fp16 and bf16 still stay stored as
        they arrived — the plan budgeted two bytes per parameter and decoding
        to fp32 here would double it. ``dequantised`` up-converts per use,
        exactly like the block-packed formats.
        """
        per_expert = 3 * header.hidden_size * header.intermediate_size
        array = np.frombuffer(body, dtype=header.dtype.numpy)
        expected = header.n_experts * per_expert
        if array.size != expected:
            raise ValueError(f"shard body has {array.size} {header.dtype.name} "
                             f"elements, expected {expected}")
        dtype_name = {DType.FP32: "fp32", DType.FP16: "fp16",
                      DType.BF16: "bf16"}[header.dtype]
        for index in range(header.n_experts):
            chunk = array[index * per_expert:(index + 1) * per_expert]
            gate, up, down = np.split(chunk, 3)
            self.store.put(
                header.layer, header.first_expert + index,
                ExpertWeights(
                    gate=gate.reshape(header.intermediate_size,
                                      header.hidden_size),
                    up=up.reshape(header.intermediate_size,
                                  header.hidden_size),
                    down=down.reshape(header.hidden_size,
                                      header.intermediate_size),
                    format=dtype_name,
                    tier=header.tier))

    def _load_fp8(self, header: ShardHeader, body: bytes) -> None:
        """An FP8 shard: all the codes, then all the block scales.

        Codes and scales are separated rather than interleaved so the codes
        land contiguous and page-aligned in the coffer; a GPU backend uploads
        them as one buffer and a memory-mapped shard reads them without a
        gather. The scales are three orders of magnitude smaller and can
        afford to be a second, small transfer.
        """
        matrix = header.hidden_size * header.intermediate_size
        per_expert = 3 * matrix
        n_codes = header.n_experts * per_expert
        blocks_per_expert = 3 * fp8.n_blocks(matrix)
        n_scales = header.n_experts * blocks_per_expert
        expected = n_codes + 4 * n_scales
        if len(body) != expected:
            raise ValueError(f"fp8 shard body is {len(body)} bytes, expected "
                             f"{expected} ({n_codes} codes + {n_scales} "
                             f"block scales)")
        codes = np.frombuffer(body, dtype=np.uint8, count=n_codes)
        scales = np.frombuffer(body, dtype="<f4", count=n_scales,
                               offset=n_codes)
        for index in range(header.n_experts):
            chunk = codes[index * per_expert:(index + 1) * per_expert]
            gate, up, down = np.split(chunk, 3)
            self.store.put(
                header.layer, header.first_expert + index,
                ExpertWeights(
                    gate=gate.reshape(header.intermediate_size,
                                      header.hidden_size),
                    up=up.reshape(header.intermediate_size,
                                  header.hidden_size),
                    down=down.reshape(header.hidden_size,
                                      header.intermediate_size),
                    scales=scales[index * blocks_per_expert:
                                  (index + 1) * blocks_per_expert],
                    format="fp8-e4m3-b128",
                    tier=header.tier))

    def _load_packed(self, header: ShardHeader, body: bytes) -> None:
        """A block-packed shard: GGUF blocks, mxfp4 or fp4, kept packed.

        The layout mirrors the dense and FP8 paths — per expert, gate then
        up then down, each matrix packed row by row — but the bytes stay in
        their blocks: the whole point of these formats is the footprint, and
        decoding at load would give it back. Each matrix's row width must
        divide the block size; a shape that cannot is a plan/checkpoint
        mismatch worth refusing loudly.
        """
        codec = header.dtype.codec
        #: NVFP4 packs the same blocks but ships one fp32 tensor scale as a
        #: trailer after each matrix's blocks.
        matrix_trailer = 4 if header.dtype is DType.NVFP4 else 0
        hidden_row = quants.block_bytes_per_row(header.hidden_size, codec)
        inter_row = quants.block_bytes_per_row(header.intermediate_size, codec)
        gate_bytes = header.intermediate_size * hidden_row
        down_bytes = header.hidden_size * inter_row
        per_expert = 2 * (gate_bytes + matrix_trailer) + down_bytes + matrix_trailer
        expected = header.n_experts * per_expert
        if len(body) != expected:
            raise ValueError(f"{codec} shard body is {len(body)} bytes, "
                             f"expected {expected}")
        for index in range(header.n_experts):
            base = index * per_expert
            gate = np.frombuffer(body, dtype=np.uint8, count=gate_bytes,
                                 offset=base)
            up_at = base + gate_bytes + matrix_trailer
            up = np.frombuffer(body, dtype=np.uint8, count=gate_bytes,
                               offset=up_at)
            down_at = up_at + gate_bytes + matrix_trailer
            down = np.frombuffer(body, dtype=np.uint8, count=down_bytes,
                                 offset=down_at)
            tensor_scale = None
            if matrix_trailer:
                tensor_scale = np.array([
                    np.frombuffer(body, dtype="<f4", count=1,
                                  offset=base + gate_bytes)[0],
                    np.frombuffer(body, dtype="<f4", count=1,
                                  offset=up_at + gate_bytes)[0],
                    np.frombuffer(body, dtype="<f4", count=1,
                                  offset=down_at + down_bytes)[0]],
                    dtype=np.float32)
            self.store.put(
                header.layer, header.first_expert + index,
                ExpertWeights(
                    gate=gate.reshape(header.intermediate_size, hidden_row),
                    up=up.reshape(header.intermediate_size, hidden_row),
                    down=down.reshape(header.hidden_size, inter_row),
                    format=codec,
                    tensor_scale=tensor_scale,
                    tier=header.tier))

    def _load_fp8_tiled(self, header: ShardHeader, body: bytes) -> None:
        """An FP8 shard whose scales tile the 2-D matrices, not the flat run.

        Same codes-then-scales separation as the flat layout; the scale dtype
        and tile shape come from the wire code. Scales stay at wire width —
        a one-byte UE8M0 exponent is one byte in the coffer too, because the
        whole point of the encoding is that footprint; ``dequantised`` does
        the lossless fp32 expansion per use instead.
        """
        tile = header.dtype.fp8_tile
        assert tile is not None
        scale_np = np.dtype(header.dtype.fp8_scale_dtype)
        inter, hidden = header.intermediate_size, header.hidden_size
        matrix = hidden * inter
        per_expert = 3 * matrix
        n_codes = header.n_experts * per_expert
        gate_up_tiles = fp8.tile_scales_needed((inter, hidden), tile)
        down_tiles = fp8.tile_scales_needed((hidden, inter), tile)
        scales_per_expert = 2 * gate_up_tiles + down_tiles
        n_scales = header.n_experts * scales_per_expert
        expected = n_codes + scale_np.itemsize * n_scales
        if len(body) != expected:
            raise ValueError(f"{header.dtype.name} shard body is {len(body)} "
                             f"bytes, expected {expected} ({n_codes} codes + "
                             f"{n_scales} tile scales)")
        codes = np.frombuffer(body, dtype=np.uint8, count=n_codes)
        raw_scales = np.frombuffer(body, dtype=scale_np, count=n_scales,
                                   offset=n_codes)
        scales = np.asarray(raw_scales)
        ue8m0 = scale_np.itemsize == 1
        for index in range(header.n_experts):
            chunk = codes[index * per_expert:(index + 1) * per_expert]
            gate, up, down = np.split(chunk, 3)
            self.store.put(
                header.layer, header.first_expert + index,
                ExpertWeights(
                    gate=gate.reshape(header.intermediate_size,
                                      header.hidden_size),
                    up=up.reshape(header.intermediate_size,
                                  header.hidden_size),
                    down=down.reshape(header.hidden_size,
                                      header.intermediate_size),
                    scales=scales[index * scales_per_expert:
                                  (index + 1) * scales_per_expert],
                    format=("fp8-e4m3-tile-ue8m0" if ue8m0
                            else "fp8-e4m3-tile"),
                    scale_tile=tile,
                    tier=header.tier))

    def _on_ping(self, frame: Frame) -> Frame:
        return Frame(MsgType.PONG, frame.request_id, frame.payload)

    def _on_status(self, frame: Frame) -> Frame:
        text = (f"unit={self.unit_id} shards={len(self.store)} "
                f"resident={self.store.resident_bytes} "
                f"tokens={self.tokens_served} experts={self.experts_run} "
                f"storage_reads={self.storage_reads} backend={self.runner.name}")
        return Frame(MsgType.STATUS_REPLY, frame.request_id,
                     text.encode("utf-8"))

    def _error(self, frame: Frame, message: str) -> Frame:
        return Frame(MsgType.ERROR, frame.request_id, encode_error(message),
                     layer=frame.layer, expert=frame.expert)


def _read_exactly(sock: socket.socket, count: int) -> Optional[bytes]:
    chunks = []
    remaining = count
    while remaining:
        try:
            chunk = sock.recv(remaining)
        except OSError:
            return None
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
