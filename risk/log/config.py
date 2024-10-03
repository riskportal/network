"""
risk/log/config
~~~~~~~~~~~~~~~
"""

import logging

# Create a global logger
logger = logging.getLogger("risk_logger")
logger.setLevel(logging.DEBUG)  # Start with the highest level; adjust as needed
# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Set the handler level (can be controlled later)
# Create a formatter and set it for the handler
formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)

# Add the handler to the logger if it doesn't already exist
if not logger.hasHandlers():
    logger.addHandler(console_handler)


def set_global_verbosity(verbose):
    """Set the global verbosity level for the logger.

    Args:
        verbose (bool): Whether to display all log messages (True) or only error messages (False).

    Returns:
        None
    """
    if verbose:
        logger.setLevel(logging.DEBUG)  # Show all messages
        console_handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)  # Show only error messages
        console_handler.setLevel(logging.ERROR)


def log_header(input_string: str) -> None:
    """Log the input string as a header with a line of dashes above and below it.

    Args:
        input_string (str): The string to be printed as a header.
    """
    border = "-" * len(input_string)
    logger.info(border)
    logger.info(input_string)
    logger.info(border)
