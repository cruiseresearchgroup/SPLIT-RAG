from __future__ import annotations
import logging, time
from contextlib import contextmanager

def get_logger(name: str = "splitrag") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(name)s: %(message)s",
                                datefmt="%H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger

@contextmanager
def timer(msg: str, logger: logging.Logger | None = None):
    logger = logger or get_logger()
    t0 = time.time()
    yield
    dt = time.time() - t0
    logger.info(f"{msg} took {dt:.2f}s")
