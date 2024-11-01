"""
risk/log/console
~~~~~~~~~~~~~~~~
"""

import logging


def in_jupyter():
    """Check if the code is running in a Jupyter notebook environment.

    Returns:
        bool: True if running in a Jupyter notebook or QtConsole, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # Jupyter Notebook or QtConsole
            return True
        elif shell == "TerminalInteractiveShell":  # Terminal running IPython
            return False
    except NameError:
        return False  # Not in Jupyter


# Define the MockLogger class to replicate logging behavior with print statements in Jupyter
class MockLogger:
    """MockLogger: A lightweight logger replacement using print statements in Jupyter.

    The MockLogger class replicates the behavior of a standard logger using print statements
    to display messages. This is primarily used in a Jupyter environment to show outputs
    directly in the notebook. The class supports logging levels such as `info`, `debug`,
    `warning`, and `error`, while the `verbose` attribute controls whether to display non-error messages.
    """

    def __init__(self, verbose: bool = True):
        """Initialize the MockLogger with verbosity settings.

        Args:
            verbose (bool): If True, display all log messages (info, debug, warning).
                If False, only display error messages. Defaults to True.
        """
        self.verbose = verbose

    def info(self, message: str) -> None:
        """Display an informational message.

        Args:
            message (str): The informational message to be printed.
        """
        if self.verbose:
            print(message)

    def debug(self, message: str) -> None:
        """Display a debug message.

        Args:
            message (str): The debug message to be printed.
        """
        if self.verbose:
            print(message)

    def warning(self, message: str) -> None:
        """Display a warning message.

        Args:
            message (str): The warning message to be printed.
        """
        print(message)

    def error(self, message: str) -> None:
        """Display an error message.

        Args:
            message (str): The error message to be printed.
        """
        print(message)

    def setLevel(self, level: int) -> None:
        """Adjust verbosity based on the logging level.

        Args:
            level (int): Logging level to control message display.
                - logging.DEBUG sets verbose to True (show all messages).
                - logging.WARNING sets verbose to False (show only warning, error, and critical messages).
        """
        if level == logging.DEBUG:
            self.verbose = True  # Show all messages
        elif level == logging.WARNING:
            self.verbose = False  # Suppress all except warning, error, and critical messages


# Set up logger based on environment
if not in_jupyter():
    # Set up logger normally for .py files or terminal environments
    logger = logging.getLogger("risk_logger")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    if not logger.hasHandlers():
        logger.addHandler(console_handler)
else:
    # If in Jupyter, use the MockLogger
    logger = MockLogger()


def set_global_verbosity(verbose):
    """Set the global verbosity level for the logger.

    Args:
        verbose (bool): Whether to display all log messages (True) or only error messages (False).

    Returns:
        None
    """
    if not isinstance(logger, MockLogger):
        # For the regular logger, adjust logging levels
        if verbose:
            logger.setLevel(logging.DEBUG)  # Show all messages
            console_handler.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)  # Show only warning, error, and critical messages
            console_handler.setLevel(logging.WARNING)
    else:
        # For the MockLogger, set verbosity directly
        logger.setLevel(logging.DEBUG if verbose else logging.WARNING)


def log_header(input_string: str) -> None:
    """Log the input string as a header with a line of dashes above and below it.

    Args:
        input_string (str): The string to be printed as a header.
    """
    border = "-" * len(input_string)
    logger.info(border)
    logger.info(input_string)
    logger.info(border)
