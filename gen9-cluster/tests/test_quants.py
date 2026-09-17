"""The packed-format decoders, exercised against hand-packed fixtures.

Each test *packs* its own block: the helpers below stuff fields into the
published ggml layouts (``ggml-common.h``) rather than quantising anything,
so encode and decode are independent and a disagreement means a real layout
bug, not a matched pair of mistakes. Expected values are computed from the
format definitions, not round-tripped through the decoder.
"""

import unittest

import numpy as np

from gen9_cluster import fp8, quants
from gen9_cluster.errors import Gen9Error
from gen9_cluster.model import QuantSpec
from gen9_cluster.node import NodeServer, ShardStore
from gen9_cluster.protocol import (DType, ExpertBatchPayload,
                                   ExpertRowsPayload, Frame, MsgType,
                                   ShardHeader)
from gen9_cluster.transport import NodeConnection

HIDDEN = 256
INTERMEDIATE = 256

_E2M1_X2 = np.array((0, 1, 2, 3, 4, 6, 8, 12,
                     0, -1, -2, -3, -4, -6, -8, -12), dtype=np.float32)


def _f16(values):
    return np.asarray(values, dtype=np.float16).reshape(-1).view(np.uint8)


def _pack_q2_qs(q):
    """256 two-bit values into the 64 qs bytes shared by q2_K and q3_K."""
    f = np.arange(256)
    byte = (f // 128) * 32 + f % 32
    shift = 2 * ((f % 128) // 32)
    packed = np.zeros(64, np.uint8)
    for s in (0, 2, 4, 6):
        mask = shift == s
        packed[byte[mask]] |= (q[mask] << s).astype(np.uint8)
    return packed


def _pack_k4_scales(sc, m):
    """The q4_K/q5_K 12-byte packing: eight 6-bit scales and mins."""
    b = np.zeros(12, np.uint8)
    for i in range(4):
        b[i] = sc[i] & 0x3F
        b[4 + i] = m[i] & 0x3F
    for i in range(4, 8):
        b[8 + i - 4] |= (sc[i] & 0xF) | ((m[i] & 0xF) << 4)
        b[i - 4] |= (sc[i] >> 4) << 6
        b[4 + i - 4] |= (m[i] >> 4) << 6
    return b


def _pack_nibbles(q):
    """4-bit values into pairs of (low, high) nibbles per byte."""
    packed = np.zeros(len(q) // 2, np.uint8)
    packed |= (q[:len(q) // 2] & 0xF).astype(np.uint8)
    packed |= ((q[len(q) // 2:] & 0xF) << 4).astype(np.uint8)
    return packed


class TestBlockDecoders(unittest.TestCase):
    def test_q8_0(self):
        rng = np.random.default_rng(0)
        q = rng.integers(-128, 128, size=(2, 32), dtype=np.int8)
        d = np.array([1.0, 0.5], dtype=np.float16)
        blocks = np.concatenate(
            [_f16(d).reshape(2, 2), q.view(np.uint8)], axis=1)
        out = quants.dequantize_blocks(blocks, "q8_0")
        np.testing.assert_allclose(out, q * d.astype(np.float32)[:, None])

    def test_q2_k(self):
        sc = np.arange(1, 17, dtype=np.uint8) & 0xF
        m = np.array([3, 1, 2, 7] * 4, dtype=np.uint8)
        q = np.tile(np.array([0, 1, 2, 3], np.uint8), 64)
        d, dmin = np.float16(1.5), np.float16(0.25)

        block = np.zeros(84, np.uint8)
        block[:16] = sc | (m << 4)
        block[16:80] = _pack_q2_qs(q)
        block[80:82] = _f16(d)
        block[82:84] = _f16(dmin)

        f = np.arange(256)
        expected = (float(d) * sc[f // 16] * q
                    - float(dmin) * m[f // 16]).astype(np.float32)
        out = quants.dequantize_blocks(block.reshape(1, -1), "q2_k")
        np.testing.assert_allclose(out[0], expected)

    def test_q3_k(self):
        sc = np.arange(16) - 8                       # unpacked, -32 offset later
        q = np.tile(np.array([0, 1, 2, 3], np.uint8), 64)
        hm = (np.arange(256) // 16) % 2              # hmask bit per element
        d = np.float16(0.75)

        u = (sc + 32).astype(np.uint8)
        scales = np.zeros(12, np.uint8)
        for i in range(8):
            scales[i] = (u[i] & 0xF) | ((u[i + 8] & 0xF) << 4)
        for i in range(16):
            scales[8 + (i % 4)] |= (u[i] >> 4) << (2 * (i // 4))

        hmask = np.zeros(32, np.uint8)
        f = np.arange(256)
        for k in range(8):
            mask = (f // 32) == k
            hmask[f[mask] % 32] |= (hm[mask] << k).astype(np.uint8)

        block = np.concatenate(
            [hmask, _pack_q2_qs(q), scales, _f16(d)])

        qv = q.astype(np.int8) - np.where(hm == 1, 0, 4)
        expected = (float(d) * sc[f // 16] * qv).astype(np.float32)
        out = quants.dequantize_blocks(block.reshape(1, -1), "q3_k")
        np.testing.assert_allclose(out[0], expected)

    def test_q4_k(self):
        sc = np.array([1, 9, 3, 40, 5, 63, 2, 17], np.uint8)
        m = np.array([7, 0, 11, 3, 60, 1, 22, 4], np.uint8)
        q = np.tile(np.arange(16, dtype=np.uint8), 16)
        d, dmin = np.float16(2.0), np.float16(0.125)

        packed = np.zeros(128, np.uint8)
        f = np.arange(256)
        byte = (f // 64) * 32 + f % 32
        nib = (f % 64) // 32
        for h in (0, 1):
            mask = nib == h
            packed[byte[mask]] |= (q[mask] << (4 * h)).astype(np.uint8)

        block = np.concatenate(
            [_f16(d), _f16(dmin), _pack_k4_scales(sc, m), packed])
        expected = (float(d) * sc[f // 32] * q
                    - float(dmin) * m[f // 32]).astype(np.float32)
        out = quants.dequantize_blocks(block.reshape(1, -1), "q4_k")
        np.testing.assert_allclose(out[0], expected)

    def test_q5_k(self):
        sc = np.array([1, 9, 3, 40, 5, 63, 2, 17], np.uint8)
        m = np.array([7, 0, 11, 3, 60, 1, 22, 4], np.uint8)
        q = np.tile(np.arange(32, dtype=np.uint8), 8)   # 5-bit codes 0..31
        d, dmin = np.float16(2.0), np.float16(0.125)

        f = np.arange(256)
        ql = np.zeros(128, np.uint8)
        byte = (f // 64) * 32 + f % 32
        nib = (f % 64) // 32
        for h in (0, 1):
            mask = nib == h
            ql[byte[mask]] |= ((q[mask] & 0xF) << (4 * h)).astype(np.uint8)
        qh = np.zeros(32, np.uint8)
        for k in range(8):
            mask = (f // 32) == k
            qh[f[mask] % 32] |= ((q[mask] >> 4) << k).astype(np.uint8)

        block = np.concatenate(
            [_f16(d), _f16(dmin), _pack_k4_scales(sc, m), qh, ql])
        expected = (float(d) * sc[f // 32] * q
                    - float(dmin) * m[f // 32]).astype(np.float32)
        out = quants.dequantize_blocks(block.reshape(1, -1), "q5_k")
        np.testing.assert_allclose(out[0], expected)

    def test_q6_k(self):
        sc = np.arange(-8, 8, dtype=np.int8)
        q = np.tile(np.arange(64, dtype=np.int8) - 32, 4)  # codes -32..31
        d = np.float16(0.5)

        u = (q.astype(np.int16) + 32).astype(np.uint8)
        f = np.arange(256)
        ql = np.zeros(128, np.uint8)
        for h in (0, 1):
            mask = (f // 64) % 2 == h
            ql[(f[mask] // 128) * 64 + f[mask] % 64] |= (
                (u[mask] & 0xF) << (4 * h)).astype(np.uint8)
        qh = np.zeros(64, np.uint8)
        for k in range(4):
            mask = ((f % 128) // 32) == k
            qh[(f[mask] // 128) * 32 + f[mask] % 32] |= (
                (u[mask] >> 4) << (2 * k)).astype(np.uint8)

        block = np.concatenate([ql, qh, sc.view(np.uint8), _f16(d)])
        expected = (float(d) * sc[f // 16] * q).astype(np.float32)
        out = quants.dequantize_blocks(block.reshape(1, -1), "q6_k")
        np.testing.assert_allclose(out[0], expected)

    def test_mxfp4(self):
        e = np.uint8(128)                            # 2**1 at true scale
        v = np.arange(32, dtype=np.uint8) % 16
        block = np.concatenate([[e], _pack_nibbles(v)])
        expected = 2.0 ** (128 - 127) * (_E2M1_X2[v] / 2)
        out = quants.dequantize_blocks(block.reshape(1, -1), "mxfp4")
        np.testing.assert_allclose(out[0], expected)

    def test_fp4_and_its_aliases(self):
        d = np.array([0x38, 0x40, 0x48, 0x30], np.uint8)  # ue4m3 sub-scales
        v = np.arange(64, dtype=np.uint8) % 16
        qs = np.zeros(32, np.uint8)
        for j in range(4):
            for i in range(8):
                qs[j * 8 + i] = v[j * 16 + i] | (v[j * 16 + 8 + i] << 4)
        block = np.concatenate([d, qs])

        exp = (d.astype(np.int32) >> 3) & 0xF
        man = (d & 0x7).astype(np.float32)
        ue4m3 = np.where(exp == 0, man * 2.0 ** -9,
                         (1.0 + man / 8.0) * 2.0 ** (exp - 7))
        expected = np.repeat(ue4m3, 16) * _E2M1_X2[v] / 2

        for name in ("fp4", "fp4-e4m3-16", "nvfp4"):
            out = quants.dequantize_blocks(block.reshape(1, -1), name)
            np.testing.assert_allclose(out[0], expected, err_msg=name)

    def test_rows_decode_multiple_blocks_per_row(self):
        rng = np.random.default_rng(3)
        q = rng.integers(-128, 128, size=(4, 64), dtype=np.int8)
        d = np.full((4, 2), 0.25, dtype=np.float16)
        d_bytes = _f16(d).reshape(4, 4)
        rows = np.concatenate(
            [d_bytes[:, :2], q[:, :32].view(np.uint8),
             d_bytes[:, 2:], q[:, 32:].view(np.uint8)], axis=1)
        out = quants.dequantize_rows(rows, "q8_0")
        expected = (q.reshape(4, 2, 32)
                    * d.astype(np.float32).reshape(4, 2, 1)).reshape(4, 64)
        np.testing.assert_allclose(out, expected)

    def test_row_width_must_divide_the_block(self):
        with self.assertRaises(ValueError):
            quants.block_bytes_per_row(100, "q2_k")
        with self.assertRaises(ValueError):
            quants.dequantize_rows(np.zeros((2, 33), np.uint8), "q8_0")

    def test_dense_decode(self):
        x = np.linspace(-2, 2, 8, dtype=np.float32)
        np.testing.assert_allclose(
            quants.decode_dense(x.astype(np.float16), "fp16"), x, atol=1e-3)
        bf16 = (x.view(np.uint32) >> 16).astype(np.uint16)
        np.testing.assert_allclose(
            quants.decode_dense(bf16, "bf16"), x, atol=0.02)


class TestWireFormats(unittest.TestCase):
    def test_every_spec_dtype_has_a_wire_dtype(self):
        for name, wire in (("fp32", DType.FP32), ("fp16", DType.FP16),
                           ("bf16", DType.BF16), ("fp8", DType.FP8_E4M3_B128),
                           ("mxfp4", DType.MXFP4), ("fp4", DType.FP4_E4M3_16),
                           ("nvfp4", DType.NVFP4),
                           ("q8_0", DType.GGUF_Q8_0), ("q2_k", DType.GGUF_Q2_K),
                           ("q3_k", DType.GGUF_Q3_K), ("q4_k", DType.GGUF_Q4_K),
                           ("q5_k", DType.GGUF_Q5_K), ("q6_k", DType.GGUF_Q6_K)):
            self.assertIs(DType.for_spec(name), wire)
        with self.assertRaises(ValueError):
            DType.for_spec("iq4_nl")

    def test_fp8_specs_dispatch_on_their_scale_geometry(self):
        """Stock checkpoints ship tile scales, not one fp32 per 128 flat
        elements — the wire dtype has to know which layout it carries."""
        for spec, wire in (
                (QuantSpec("fp8"), DType.FP8_E4M3_B128),
                (QuantSpec("fp8", scale_bytes=4.0, scale_block=128),
                 DType.FP8_E4M3_B128),
                (QuantSpec("fp8", scale_bytes=4.0, scale_block=128 * 128),
                 DType.FP8_E4M3_T128),
                (QuantSpec("fp8", scale_bytes=1.0, scale_block=128 * 128),
                 DType.FP8_E4M3_T128_UE8M0),
                (QuantSpec("fp8", scale_bytes=1.0, scale_block=32 * 32),
                 DType.FP8_E4M3_T32_UE8M0)):
            self.assertIs(DType.for_spec(spec), wire)
        with self.assertRaises(ValueError):
            DType.for_spec(QuantSpec("fp8", scale_bytes=1.0,
                                     scale_block=64 * 64))
        self.assertEqual(DType.FP8_E4M3_T32_UE8M0.fp8_tile, (32, 32))
        self.assertEqual(DType.FP8_E4M3_T128.fp8_scale_dtype, "<f4")
        self.assertIsNone(DType.FP32.fp8_tile)

    def test_packed_dtypes_report_their_codec_and_rate(self):
        self.assertTrue(DType.GGUF_Q6_K.packed)
        self.assertFalse(DType.FP8_E4M3_B128.packed)
        self.assertEqual(DType.GGUF_Q6_K.codec, "q6_k")
        self.assertAlmostEqual(DType.GGUF_Q4_K.itemsize, 0.5625)
        self.assertAlmostEqual(DType.MXFP4.itemsize, 0.53125)


class TestPackedShardLoading(unittest.TestCase):
    """A packed shard through a real node: LOAD_SHARD then EXPERT_BATCH."""

    def setUp(self):
        self.store = ShardStore()
        self.node = NodeServer(self.store, unit_id="ps5-quant",
                               host="127.0.0.1", port=0)
        self.port = self.node.start()
        self.addCleanup(self.node.stop)
        conn = NodeConnection("ps5-quant", "127.0.0.1", self.port)
        conn.connect()
        self.conn = conn
        self.addCleanup(conn.close)

    def _load(self, dtype, body):
        header = ShardHeader(layer=1, first_expert=0, n_experts=1,
                             hidden_size=HIDDEN,
                             intermediate_size=INTERMEDIATE, dtype=dtype)
        return self.conn.request(
            Frame(MsgType.LOAD_SHARD, 0, header.encode() + body,
                  dtype=dtype))

    def test_a_q4_k_shard_stays_packed_until_it_is_run(self):
        sc = np.array([10, 11, 12, 13, 14, 15, 16, 17], np.uint8)
        m = np.array([2, 3, 4, 5, 6, 7, 8, 9], np.uint8)
        d, dmin = np.float16(1.25), np.float16(0.0625)
        rng = np.random.default_rng(7)
        q = rng.integers(0, 16, size=(3, 256), dtype=np.uint8)

        f = np.arange(256)
        matrices = []
        body = b""
        for mi in range(3):
            packed = np.zeros((256, 128), np.uint8)
            byte = (f // 64) * 32 + f % 32
            nib = (f % 64) // 32
            for h in (0, 1):
                mask = nib == h
                packed[:, byte[mask]] |= (
                    q[mi][mask] << (4 * h)).astype(np.uint8)
            scales = _pack_k4_scales(sc + mi, m + mi)
            block = np.concatenate([_f16(d), _f16(dmin), np.tile(scales, 1)])
            rows = np.concatenate(
                [np.tile(block, (256, 1)), packed], axis=1)
            body += rows.tobytes()
            row = (float(d) * (sc + mi)[f // 32] * q[mi]
                   - float(dmin) * (m + mi)[f // 32]).astype(np.float32)
            matrices.append(np.tile(row, (256, 1)))

        reply = self._load(DType.GGUF_Q4_K, body)
        self.assertEqual(reply.msg_type, MsgType.LOAD_ACK)
        held = self.store.get(1, 0)
        self.assertEqual(held.format, "q4_k")
        self.assertTrue(held.quantised)
        self.assertEqual(held.nbytes, len(body))
        self.assertIsNone(held.scales)
        plain = held.dequantised()
        np.testing.assert_allclose(plain.gate, matrices[0])
        np.testing.assert_allclose(plain.up, matrices[1])
        np.testing.assert_allclose(plain.down, matrices[2])

        x = rng.standard_normal(HIDDEN).astype(np.float32)
        reply = self.conn.request(
            Frame(MsgType.EXPERT_BATCH, 0,
                  ExpertBatchPayload((0,), (0.5,), x).encode(), layer=1))
        rows = ExpertRowsPayload.decode(reply.payload).rows
        gate, up, down = matrices
        hidden = x @ gate.T
        hidden = hidden * (1.0 / (1.0 + np.exp(-hidden)))
        hidden = hidden * (x @ up.T)
        np.testing.assert_allclose(
            rows[0], np.float32(0.5) * (hidden @ down.T), rtol=1e-5)

    def test_an_indivisible_row_width_is_refused(self):
        header = ShardHeader(layer=2, first_expert=0, n_experts=1,
                             hidden_size=100, intermediate_size=100,
                             dtype=DType.GGUF_Q2_K)
        with self.assertRaises(Gen9Error):
            self.conn.request(
                Frame(MsgType.LOAD_SHARD, 0,
                      header.encode() + b"\x00" * 8192))
        self.assertFalse(self.store.holds(2, 0))

    def test_dense_fp16_and_bf16_shards_stay_at_wire_width(self):
        """fp16/bf16 carry no scales, but the plan budgeted two bytes per
        parameter — decoding to fp32 at load would double the residency."""
        rng = np.random.default_rng(5)
        matrices = rng.standard_normal((3, INTERMEDIATE, HIDDEN)
                                       ).astype(np.float32) * 0.1

        body16 = matrices.astype(np.float16).reshape(-1).view(np.uint8)
        reply = self._load(DType.FP16, body16.tobytes())
        self.assertEqual(reply.msg_type, MsgType.LOAD_ACK)
        held = self.store.get(1, 0)
        self.assertFalse(held.quantised)
        self.assertEqual(held.format, "fp16")
        self.assertEqual(held.gate.dtype, np.float16)
        self.assertEqual(held.nbytes, len(body16.tobytes()))
        np.testing.assert_allclose(
            held.dequantised().gate, matrices[0], atol=1e-3)

        u16 = (matrices.reshape(-1).view(np.uint32) >> 16).astype(np.uint16)
        reply = self._load(DType.BF16, u16.tobytes())
        self.assertEqual(reply.msg_type, MsgType.LOAD_ACK)
        held = self.store.get(1, 0)
        self.assertEqual(held.format, "bf16")
        self.assertEqual(held.nbytes, len(u16.tobytes()))
        np.testing.assert_allclose(
            held.dequantised().up, matrices[1], atol=0.02)

    def test_an_fp8_tile_shard_decodes_with_its_tile_geometry(self):
        """V4.1's ue8m0-per-32x32-tile scales: the wire dtype carries the
        geometry, the loader upcasts the exponents, and dequantisation is
        per tile, not per flat 128."""
        rng = np.random.default_rng(11)
        codes = rng.integers(0, 128, size=(3, INTERMEDIATE, HIDDEN),
                             dtype=np.uint8)
        ue = rng.integers(118, 128, size=(3, INTERMEDIATE // 32,
                                          HIDDEN // 32), dtype=np.uint8)
        body = codes.reshape(-1).tobytes() + ue.reshape(-1).tobytes()
        reply = self._load(DType.FP8_E4M3_T32_UE8M0, body)
        self.assertEqual(reply.msg_type, MsgType.LOAD_ACK)
        held = self.store.get(1, 0)
        self.assertTrue(held.quantised)
        self.assertEqual(held.format, "fp8-e4m3-tile")
        self.assertEqual(held.scale_tile, (32, 32))
        scales = np.ldexp(
            np.ones(ue.shape, dtype=np.float32),
            ue.astype(np.int16) - 127)
        plain = held.dequantised()
        for mi, got in enumerate((plain.gate, plain.up, plain.down)):
            expect = (fp8.decode(codes[mi])
                      * np.repeat(np.repeat(scales[mi], 32, axis=0),
                                  32, axis=1))
            np.testing.assert_allclose(got, expect)

    def test_an_nvfp4_shard_applies_its_tensor_scale(self):
        """The tensor-level fp32 rides behind each matrix's blocks; dropping
        it would misvalue every weight."""
        d = np.full(4, 0x38, np.uint8)                  # ue4m3 scale = 1.0
        v = np.tile(np.arange(16, dtype=np.uint8), 4)
        qs = np.zeros(32, np.uint8)
        for j in range(4):
            for i in range(8):
                qs[j * 8 + i] = v[j * 16 + i] | (v[j * 16 + 8 + i] << 4)
        block = np.concatenate([d, qs])
        row = np.tile(block, HIDDEN // 64)
        matrices = np.tile(row, (3, INTERMEDIATE, 1))
        tensor_scales = np.array([0.5, 0.25, 2.0], dtype=np.float32)
        body = b"".join(
            matrices[mi].tobytes()
            + tensor_scales[mi].tobytes() for mi in range(3))

        reply = self._load(DType.NVFP4, body)
        self.assertEqual(reply.msg_type, MsgType.LOAD_ACK)
        held = self.store.get(1, 0)
        self.assertEqual(held.format, "nvfp4")
        np.testing.assert_array_equal(held.tensor_scale, tensor_scales)
        plain = held.dequantised()
        decoded = _E2M1_X2[v] / 2                       # scale-1.0 blocks
        for mi, got in enumerate((plain.gate, plain.up, plain.down)):
            expect = np.tile(decoded, (INTERMEDIATE, HIDDEN // 64))
            np.testing.assert_allclose(got, expect * tensor_scales[mi])


if __name__ == "__main__":
    unittest.main()
