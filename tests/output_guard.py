"""Exclusive opens of label output files, for the multi-rank labelling tests.

:func:`install` wraps ``builtins.open`` and ``os.open`` in this process. Every open
of a ``*-{labels,nodes,distances}.npy`` or ``*-{...}.parts/part-RRRR.npy`` file, for
reading or writing, takes a non-blocking ``flock(LOCK_EX)`` on the new descriptor
and fails if another process holds the file open the same way. No clocks are
involved: two ranks with one output file open at the same time fail. The lock
lives as long as the descriptor (and any memory map made from it). Locks are
taken only while :data:`active` is true (around the labelling): one process may
open a file twice, as ``np.load(mmap_mode=...)`` does, and that is no overlap
between ranks. Every guarded path is recorded in :data:`opened`, always.
"""
import builtins
import fcntl
import os
import re

OUTPUT = re.compile(r"-(labels|nodes|distances)(?:\.npy|\.parts/part-\d+\.npy)$")
opened = []
active = True


class Overlap(RuntimeError):
    pass


def _lock(fd, path):
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise Overlap(f"{path} is open in another process") from None


def _watched(path):
    return isinstance(path, (str, bytes, os.PathLike)) and OUTPUT.search(os.fsdecode(path)) is not None


def install():
    real_open, real_os_open = builtins.open, os.open

    def guarded_open(file, *args, **kwargs):
        fh = real_open(file, *args, **kwargs)
        if _watched(file):
            opened.append(os.fsdecode(file))
            try:
                if active:
                    _lock(fh.fileno(), file)
            except BaseException:
                fh.close()
                raise
        return fh

    def guarded_os_open(path, flags, *args, **kwargs):
        fd = real_os_open(path, flags, *args, **kwargs)
        if _watched(path):
            opened.append(os.fsdecode(path))
            try:
                if active:
                    _lock(fd, path)
            except BaseException:
                os.close(fd)
                raise
        return fd

    builtins.open, os.open = guarded_open, guarded_os_open


def locking_during(fn):
    """``fn`` with locks taken only while it runs."""
    global active
    active = False

    def wrapped(*args, **kwargs):
        global active
        active = True
        try:
            return fn(*args, **kwargs)
        finally:
            active = False
    return wrapped


def is_final(path):
    """A single-file .npy output, not a sharded .parts/part-RRRR.npy."""
    s = os.fsdecode(path)
    return OUTPUT.search(s) is not None and ".parts/" not in s
