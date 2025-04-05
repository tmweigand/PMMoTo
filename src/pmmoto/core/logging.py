import logging
import os
from datetime import datetime
from mpi4py import MPI


class MPIFormatter(logging.Formatter):
    """Custom formatter that includes MPI rank"""

    def __init__(self, *args, **kwargs):
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.size = MPI.COMM_WORLD.Get_size()
        super().__init__(*args, **kwargs)

    def format(self, record):
        record.rank = self.rank
        record.size = self.size
        return super().format(record)


def setup_logger(name="pmmoto", log_dir="logs"):
    """
    Configure logging for both serial and parallel runs

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
    size = MPI.COMM_WORLD.Get_size()

    # Create handlers
    formatter = MPIFormatter(
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
    else:
        # Disable propagation for non-rank-0 processes
        logger.propagate = False

    logger.setLevel(logging.INFO)

    return logger


# Initialize the logger and make it accessible
# logger = setup_logger()
