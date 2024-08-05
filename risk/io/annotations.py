"""
risk/network/io/annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This file contains the code for the RISK class and command-line access.
"""

import json

from risk.annotations import load_annotations
from risk.log import params, print_header


class AnnotationsIO:
    def __init__(self):
        pass

    def load_json_annotations(self, filepath, network):
        """Load annotations from a JSON file and convert them to a DataFrame.

        Args:
            annotations_filepath (str): Path to the JSON annotations file.

        Returns:
            dict: A dictionary containing ordered nodes, ordered annotations, and the annotations matrix.
        """
        filetype = "JSON"
        params.log_annotations(filepath=filepath, filetype=filetype)
        _log_loading(filetype, filepath=filepath)
        # Open and read the JSON file
        with open(filepath, "r") as file:
            annotations_input = json.load(file)

        # Convert the JSON data to a DataFrame
        return load_annotations(network, annotations_input)


def _log_loading(filetype, filepath=None):
    """Log information about the network file being loaded.

    Args:
        filetype (str): The type of the file being loaded (e.g., 'Cytoscape').
        filepath (str, optional): The path to the file being loaded.
    """
    print_header("Loading annotations")
    print(f"Filetype: {filetype}")
    if filepath:
        print(f"Filepath: {filepath}")
