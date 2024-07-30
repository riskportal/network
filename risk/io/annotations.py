"""
risk/network/io/annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This file contains the code for the RISK class and command-line access.
"""

import json

from risk.annotations import load_annotations


class AnnotationsIO:
    def __init__(self):
        """
        Initialize the NetworkAnnotations class.

        Args:
            network (NetworkX graph): The network graph with nodes having IDs matching the annotations file.
        """
        pass

    def load_json_annotations(self, filepath, network):
        """
        Load annotations from a JSON file and convert them to a DataFrame.

        Args:
            annotations_filepath (str): Path to the JSON annotations file.

        Returns:
            dict: A dictionary containing ordered nodes, ordered annotations, and the annotations matrix.
        """
        # Open and read the JSON file
        with open(filepath, "r") as file:
            annotations_input = json.load(file)

        # Convert the JSON data to a DataFrame
        return load_annotations(network, annotations_input)
