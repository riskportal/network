"""
risk/log/console
~~~~~~~~~~~~~~~~
"""


def print_header(input_string: str) -> None:
    """Print the input string as a header with a line of dashes above and below it.

    Args:
        input_string (str): The string to be printed as a header.
    """
    border = "-" * len(input_string)
    print(border)
    print(input_string)
    print(border)
