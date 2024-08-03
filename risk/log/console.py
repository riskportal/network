def print_header(input_string):
    """
    Prints the input string as a header and plots a line of dashes below it of the same length.

    Args:
        input_string: The string to be printed as a header.
    """
    border = "-" * len(input_string)
    print(border)
    print(input_string)
    print(border)
