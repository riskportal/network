"""
tests/test_log
~~~~~~~~~~~~~~
"""

import io
import logging

import pytest

from risk._log import log_header, logger, params


@pytest.fixture
def log_capture():
    """
    Fixture to capture log output for assertions.

    Returns:
        (log_stream, original_handlers) for inspection and cleanup.
    """
    # Capture log output
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    # Return the stream and handler
    yield log_stream
    logger.removeHandler(handler)


def test_log_header_output(log_capture):
    """
    Test that log_header outputs a correctly formatted section header.

    Args:
        log_capture: Captures logger output.
    """
    # Deliberately set log level to DEBUG to ensure header is logged in GitHub Actions
    logger.setLevel(logging.DEBUG)
    log_header("Unit Test Header")
    logger.handlers[0].flush()  # Ensure buffer is flushed
    contents = log_capture.getvalue()
    assert "Unit Test Header" in contents
    assert any(sym in contents for sym in ("=", "-"))


def test_logger_debug_output(log_capture):
    """
    Test that logger.debug writes the expected message to the log.

    Args:
        log_capture: Captures logger output.
    """
    logger.setLevel(logging.DEBUG)
    logger.debug("Test debug message")
    contents = log_capture.getvalue()
    assert "Test debug message" in contents


def test_params_log_annotation():
    """Test that params.log_annotation logs the correct information."""
    logger.setLevel(logging.DEBUG)
    params.log_annotation(
        filetype="CSV",
        filepath="mock/path/to/file.csv",
        min_nodes_per_term=3,
        max_nodes_per_term=5,
    )
    assert params.annotation["filetype"] == "CSV"
    assert params.annotation["filepath"] == "mock/path/to/file.csv"
    assert params.annotation["min_nodes_per_term"] == 3
    assert params.annotation["max_nodes_per_term"] == 5
