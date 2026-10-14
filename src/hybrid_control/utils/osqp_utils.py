"""Utilities for working with OSQP across versions."""
import contextlib
import os


@contextlib.contextmanager
def suppress_c_stdout():
    """Redirect C-level stdout to /dev/null for the duration of the block.

    OSQP 1.0+ prints polish diagnostic messages directly to the C FILE* stdout,
    bypassing Python's sys.stdout and the verbose=False setting.  Redirecting at
    the file-descriptor level suppresses them without affecting Python output.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    os.dup2(devnull_fd, 1)
    try:
        yield
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(devnull_fd)
