import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps3_cluster import protocol as P


class TestProtocol(unittest.TestCase):
    def test_roundtrip_f32(self):
        x = np.arange(12, dtype=np.float32).reshape(3, 4)
        msg = P.decode(P.encode(P.MSG_REQ, 5, 7, 42, x)[4:])
        self.assertEqual(msg["layer"], 5)
        self.assertEqual(msg["expert"], 7)
        self.assertEqual(msg["token_id"], 42)
        self.assertEqual(msg["msg_type"], P.MSG_REQ)
        np.testing.assert_array_equal(msg["array"], x)

    def test_roundtrip_f16(self):
        x = (np.random.rand(8).astype(np.float16))
        msg = P.decode(P.encode(P.MSG_RSP, 0, 0, 0, x)[4:])
        np.testing.assert_array_equal(msg["array"], x)

    def test_packed_u8_mxfp4(self):
        x = np.random.randint(0, 256, size=(16,), dtype=np.uint8)
        msg = P.decode(P.encode(P.MSG_REQ, 1, 2, 3, x)[4:])
        np.testing.assert_array_equal(msg["array"], x)

    def test_wire_is_big_endian(self):
        # A known float must serialise big-endian on the wire regardless of host.
        x = np.array([1.0], dtype=np.float32)  # 0x3F800000
        frame = P.encode(P.MSG_REQ, 0, 0, 0, x)
        payload = frame[-4:]
        self.assertEqual(payload, b"\x3f\x80\x00\x00")

    def test_length_prefix(self):
        x = np.zeros(5, dtype=np.float32)
        frame = P.encode(P.MSG_REQ, 0, 0, 0, x)
        import struct
        (length,) = struct.unpack("!I", frame[:4])
        self.assertEqual(length, len(frame) - 4)

    def test_read_frame_from_stream(self):
        x = np.arange(6, dtype=np.float32)
        frame = P.encode(P.MSG_RSP, 2, 3, 9, x)
        buf = {"data": frame}

        def recv(n):
            out = buf["data"][:n]
            buf["data"] = buf["data"][n:]
            return out

        msg = P.read_frame(recv)
        np.testing.assert_array_equal(msg["array"], x)
        self.assertEqual(msg["expert"], 3)

    def test_bad_magic(self):
        with self.assertRaises(P.ProtocolError):
            P.decode(b"XXXX" + b"\x00" * 20)


if __name__ == "__main__":
    unittest.main()
