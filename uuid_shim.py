
"""Python 3.13 fallback for uuid_utils when Windows blocks its native extension."""

import os
import sys
import time
import types
import uuid
import threading

_lock = threading.Lock()
_last_ms = -1
_counter = 0
_MASK_74 = (1 << 74) - 1


def uuid7(timestamp=None, nanos=None):
    """Generate an RFC 9562 UUIDv7."""

    global _last_ms, _counter

    if timestamp is not None:
        ns = int(timestamp) * 1_000_000_000 + int(nanos or 0)
        ms = ns // 1_000_000
        rand = int.from_bytes(os.urandom(10), "big") & _MASK_74
    else:
        ms = time.time_ns() // 1_000_000

        with _lock:
            if ms > _last_ms:
                _last_ms = ms
                _counter = int.from_bytes(os.urandom(10), "big") & _MASK_74
            else:
                _counter += 1
                if _counter > _MASK_74:
                    _last_ms += 1
                    _counter = 0
                ms = _last_ms

            rand = _counter

    rand_a = rand >> 62
    rand_b = rand & ((1 << 62) - 1)

    value = (
        ((ms & ((1 << 48) - 1)) << 80)
        | (0x7 << 76)
        | (rand_a << 64)
        | (0b10 << 62)
        | rand_b
    )

    return uuid.UUID(int=value)


def install():
    """Use the real package if available; otherwise provide the UUIDv7 fallback."""

    try:
        import uuid_utils.compat
        return
    except (ImportError, OSError):
        pass

    root = types.ModuleType("uuid_utils")
    compat = types.ModuleType("uuid_utils.compat")

    root.__path__ = []
    root.compat = compat
    compat.uuid7 = uuid7

    root.__shim__ = True
    compat.__shim__ = True

    sys.modules["uuid_utils"] = root
    sys.modules["uuid_utils.compat"] = compat


install()
