"""logging.py

Setup of the logger
"""

import os
from typing import Any
import logging
from datetime import datetime
from mpi4py import MPI

USE_LOGGING = True


class MPIFormatter(logging.Formatter):
    """Custom formatter that includes MPI rank"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.rank: int = MPI.COMM_WORLD.Get_rank()
        self.size: int = MPI.COMM_WORLD.Get_size()
        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        """Add mpi information"""
        record.rank = self.rank
        record.size = self.size
        return super().format(record)


def setup_logger(name: str = "pmmoto", log_dir: str = "logs") -> logging.Logger:
    """Configure logging for both serial and parallel runs

    Args:
        name: Logger name (default: "pmmoto")
        log_dir: Directory to store log files (default: "logs")

    Returns:
        logging.Logger: Configured logger instance

    """
    # Disable the root logger to prevent duplicate messages
    logging.getLogger().handlers.clear()

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Get MPI rank
    rank = MPI.COMM_WORLD.Get_rank()

    # Create handlers
    formatter: logging.Formatter = MPIFormatter(
        fmt="%(asctime)s [Rank %(rank)d/%(size)d] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler - one file per rank
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(
        os.path.join(log_dir, f"{name}_{timestamp}_rank{rank}.log")
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler - only for rank 0
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.propagate = False

    logger.setLevel(logging.INFO)

    return logger


# Add a module-level logger instance
_logger = None


def get_logger() -> logging.Logger:
    """Get or create the logger instance"""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger
