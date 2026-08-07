"""Wire-level tests for the subcluster batch frames.

These are codec tests: the activation must travel once per subcluster request,
every count/length must be bounded, and anything malformed or unsupported must
be rejected with a clear ``ProtocolError`` rather than allocated for. The
expert-worker frames must keep decoding exactly as before.
"""

import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps3_cluster import batch as B  # noqa: E402
from ps3_cluster import protocol as P  # noqa: E402


def entries(n):
    return [B.BatchEntry(expert=e, gate=0.5 + e) for e in range(n)]


class TestBatchRequest(unittest.TestCase):
    def test_roundtrip(self):
        x = np.arange(8, dtype=np.float32)
        frame = B.encode_batch_request(7, 99, x, entries(3), deadline_ms=250)
        msg = B.decode_batch_request(frame[4:])
        self.assertEqual(msg["msg_type"], P.MSG_BREQ)
        self.assertEqual(msg["layer"], 7)
        self.assertEqual(msg["expert"], B.NO_EXPERT)
        self.assertEqual(msg["token_id"], 99)
        self.assertEqual(msg["deadline_ms"], 250)
        np.testing.assert_array_equal(msg["array"], x)
        self.assertEqual([e.expert for e in msg["entries"]], [0, 1, 2])
        np.testing.assert_allclose([e.gate for e in msg["entries"]],
                                   [0.5, 1.5, 2.5])

    def test_activation_travels_once_regardless_of_k(self):
        x = np.zeros(1024, dtype=np.float32)  # 4 KiB of activation
        one = B.encode_batch_request(0, 0, x, entries(1))
        eight = B.encode_batch_request(0, 0, x, entries(8))
        # Seven more experts cost seven 8-byte entries, not seven activations.
        self.assertEqual(len(eight) - len(one), 7 * 8)
        self.assertLess(len(eight), len(one) + x.nbytes)

    def test_decode_any_dispatches_on_type(self):
        x = np.ones(4, dtype=np.float32)
        self.assertEqual(
            B.decode_batch(B.encode_batch_request(1, 2, x, entries(2))[4:]
                           )["msg_type"], P.MSG_BREQ)
        self.assertEqual(
            B.decode_batch(B.encode_batch_response(1, 2, x, 2)[4:]
                           )["msg_type"], P.MSG_BRSP)
        self.assertEqual(
            B.decode_batch(B.encode_batch_error(1, 2, B.ERR_UNKNOWN)[4:]
                           )["msg_type"], P.MSG_BERR)

    def test_expert_frame_is_not_a_subcluster_frame(self):
        frame = P.encode(P.MSG_REQ, 1, 2, 3, np.ones(2, np.float32))
        with self.assertRaises(P.ProtocolError):
            B.decode_batch(frame[4:])

    def test_no_entries_rejected(self):
        with self.assertRaises(P.ProtocolError):
            B.encode_batch_request(0, 0, np.ones(2, np.float32), [])

    def test_too_many_entries_rejected_on_encode(self):
        with self.assertRaises(P.ProtocolError):
            B.encode_batch_request(0, 0, np.ones(2, np.float32),
                                   entries(B.MAX_BATCH_ENTRIES + 1))

    def test_declared_count_over_limit_rejected_on_decode(self):
        """A bogus count must be refused before anything is allocated."""
        body = B.encode_batch_request(0, 0, np.ones(2, np.float32),
                                      entries(2))[4:]
        head = len(body) - 2 * 8 - 8   # start of the trailer
        forged = (body[:head]
                  + struct.pack("!HHI", B.MAX_BATCH_ENTRIES + 1, 0, 0)
                  + body[head + 8:])
        with self.assertRaises(P.ProtocolError) as ctx:
            B.decode_batch_request(forged)
        self.assertIn("exceeds", str(ctx.exception))

    def test_unsupported_flags_rejected(self):
        body = B.encode_batch_request(0, 0, np.ones(2, np.float32),
                                      entries(1))[4:]
        head = len(body) - 8 - 8
        forged = (body[:head] + struct.pack("!HHI", 1, 0x0001, 0)
                  + body[head + 8:])
        with self.assertRaises(P.ProtocolError) as ctx:
            B.decode_batch_request(forged)
        self.assertIn("flags", str(ctx.exception))

    def test_truncated_trailer_rejected(self):
        body = B.encode_batch_request(0, 0, np.ones(2, np.float32),
                                      entries(3))[4:]
        with self.assertRaises(P.ProtocolError):
            B.decode_batch_request(body[:-3])

    def test_duplicate_expert_rejected(self):
        body = B.encode_batch_request(
            0, 0, np.ones(2, np.float32),
            [B.BatchEntry(5, 0.5), B.BatchEntry(5, 0.25)])[4:]
        with self.assertRaises(P.ProtocolError) as ctx:
            B.decode_batch_request(body)
        self.assertIn("twice", str(ctx.exception))

    def test_reserved_byte_must_be_zero(self):
        body = bytearray(B.encode_batch_request(0, 0, np.ones(2, np.float32),
                                                entries(1))[4:])
        body[-5] = 1  # the entry's reserved byte
        with self.assertRaises(P.ProtocolError):
            B.decode_batch_request(bytes(body))

    def test_deadline_out_of_range_rejected(self):
        with self.assertRaises(P.ProtocolError):
            B.encode_batch_request(0, 0, np.ones(2, np.float32), entries(1),
                                   deadline_ms=B.MAX_DEADLINE_MS + 1)

    def test_version_mismatch_rejected(self):
        body = bytearray(B.encode_batch_request(0, 0, np.ones(2, np.float32),
                                                entries(1))[4:])
        body[4] = P.VERSION + 1
        with self.assertRaises(P.ProtocolError) as ctx:
            B.decode_batch(bytes(body))
        self.assertIn("version", str(ctx.exception))

    def test_read_frame_refuses_an_absurd_length(self):
        def recv(n):
            return struct.pack("!I", P.MAX_FRAME_BYTES + 1)[:n]
        with self.assertRaises(P.ProtocolError):
            P.read_frame(recv)


class TestBatchResponse(unittest.TestCase):
    def test_roundtrip(self):
        partial = np.array([1.5, -2.25], dtype=np.float32)
        msg = B.decode_batch_response(
            B.encode_batch_response(4, 11, partial, 3)[4:])
        np.testing.assert_array_equal(msg["array"], partial)
        self.assertEqual(msg["n_reduced"], 3)
        self.assertEqual(msg["token_id"], 11)

    def test_zero_reduced_rejected(self):
        with self.assertRaises(P.ProtocolError):
            B.encode_batch_response(0, 0, np.ones(2, np.float32), 0)

    def test_trailing_garbage_rejected(self):
        body = B.encode_batch_response(0, 0, np.ones(2, np.float32), 1)[4:]
        with self.assertRaises(P.ProtocolError):
            B.decode_batch_response(body + b"\x00\x00")


class TestBatchError(unittest.TestCase):
    def test_roundtrip_with_failures(self):
        failures = [B.BatchFailure(3, B.ERR_NODE_TIMEOUT, "ps3-L000-E0003"),
                    B.BatchFailure(4, B.ERR_NODE_UNREACHABLE, "ps3-L000-E0004")]
        msg = B.decode_batch_error(
            B.encode_batch_error(2, 77, B.ERR_NODE_TIMEOUT, failures,
                                 "console did not answer")[4:])
        self.assertEqual(msg["code"], B.ERR_NODE_TIMEOUT)
        self.assertEqual(msg["failures"], failures)
        self.assertEqual(msg["detail"], "console did not answer")

    def test_roundtrip_without_failures(self):
        msg = B.decode_batch_error(
            B.encode_batch_error(0, 1, B.ERR_BAD_REQUEST, (), "nope")[4:])
        self.assertEqual(msg["failures"], [])
        self.assertEqual(msg["detail"], "nope")

    def test_long_strings_are_clipped_not_rejected(self):
        long_id = "n" * (B.MAX_STRING_BYTES + 100)
        msg = B.decode_batch_error(
            B.encode_batch_error(0, 1, B.ERR_UNKNOWN,
                                 [B.BatchFailure(1, B.ERR_UNKNOWN, long_id)],
                                 long_id)[4:])
        self.assertEqual(len(msg["failures"][0].node_id), B.MAX_STRING_BYTES)
        self.assertEqual(len(msg["detail"]), B.MAX_STRING_BYTES)

    def test_truncated_failure_list_rejected(self):
        body = B.encode_batch_error(
            0, 1, B.ERR_UNKNOWN,
            [B.BatchFailure(1, B.ERR_UNKNOWN, "ps3-x")])[4:]
        with self.assertRaises(P.ProtocolError):
            B.decode_batch_error(body[:-4])


class TestExpertFrameCompatibility(unittest.TestCase):
    def test_worker_frames_carry_no_trailer(self):
        for msg_type in (P.MSG_REQ, P.MSG_RSP, P.MSG_ERR, P.MSG_PING,
                         P.MSG_PONG):
            frame = P.encode(msg_type, 1, 2, 3, np.ones(3, np.float32))
            self.assertEqual(P.decode(frame[4:])["trailer"], b"")

    def test_batch_frame_decodes_as_a_plain_p3xc_frame(self):
        """A subcluster frame is still a well-formed P3XC frame."""
        x = np.arange(4, dtype=np.float32)
        msg = P.decode(B.encode_batch_request(1, 2, x, entries(2))[4:])
        np.testing.assert_array_equal(msg["array"], x)
        self.assertEqual(len(msg["trailer"]), 8 + 2 * 8)


if __name__ == "__main__":
    unittest.main()
