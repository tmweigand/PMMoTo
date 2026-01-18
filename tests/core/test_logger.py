"""test_logger.py"""

import pmmoto
import logging


def test_logger(tmp_path):
    """Basic logger setup and singleton reuse"""
    # reset cached logger
    pmmoto.core.logging._logger = None

    logger = pmmoto.core.logging.setup_logger(name="pmmoto", log_dir=tmp_path)
    assert isinstance(logger, logging.Logger)
    assert logger.handlers  # at least one handler
    logger.info("hello")

    # get_logger should return the same instance
    assert pmmoto.core.logging.get_logger() is logger


def test_setup_logger_clears_duplicate_handlers(tmp_path):
    # reset cached logger
    pmmoto.core.logging._logger = None

    logger = pmmoto.core.logging.setup_logger(name="pmmoto", log_dir=tmp_path)
    initial_count = len(logger.handlers)
    assert initial_count >= 1

    # add an extra handler to simulate prior attachments
    extra = logging.NullHandler()
    logger.addHandler(extra)
    assert len(logger.handlers) == initial_count + 1

    # calling setup_logger again should clear old handlers (including extra)
    logger = pmmoto.core.logging.setup_logger(name="pmmoto", log_dir=tmp_path)
    assert len(logger.handlers) == initial_count
    assert extra not in logger.handlers
