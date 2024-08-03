import csv
import json

import numpy as np

from .console import print_header


class Params:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.network = {}
        self.annotations = {}
        self.neighborhoods = {}
        self.graph = {}
        self.plotter = {}

    def log_network(self, **kwargs):
        self.network = {**self.network, **kwargs}

    def log_annotations(self, **kwargs):
        self.annotations = {**self.annotations, **kwargs}

    def log_neighborhoods(self, **kwargs):
        self.neighborhoods = {**self.neighborhoods, **kwargs}

    def log_graph(self, **kwargs):
        self.graph = {**self.graph, **kwargs}

    def log_plotter(self, **kwargs):
        self.plotter = {**self.plotter, **kwargs}

    def save_as_csv(self, filepath):
        try:
            # Load the parameter dictionary
            params = self.load()
            # Find the union of all inner dictionary keys to use as columns
            columns = set()
            for nested_dict in params.values():
                columns.update(nested_dict.keys())
            columns = list(columns)
            # Open the file in write mode
            with open(filepath, "w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["id"] + columns)
                # Write the header
                writer.writeheader()
                # Write the rows
                for key, nested_dict in params.items():
                    row = {"id": key}
                    row.update(nested_dict)
                    writer.writerow(row)

            print(f"Parameters successfully exported to filepath: {filepath}")
        except Exception as e:
            print(f"An error occurred while exporting the parameter: {e}")

    def save_as_json(self, filepath):
        try:
            with open(filepath, "w") as json_file:
                json.dump(self.load(), json_file, indent=4)
            print(f"Parameters successfully exported to filepath: {filepath}")
        except Exception as e:
            print(f"An error occurred while exporting the parameter: {e}")

    def save_as_txt(self, filepath):
        try:
            # Load the parameter dictionary
            params = self.load()
            # Open the file in write mode
            with open(filepath, "w") as txt_file:
                for key, nested_dict in params.items():
                    # Write the key
                    txt_file.write(f"{key}:\n")
                    # Write the nested dictionary values, one per line
                    for nested_key, nested_value in nested_dict.items():
                        txt_file.write(f"  {nested_key}: {nested_value}\n")
                    # Add a blank line between different keys
                    txt_file.write("\n")

            print(f"Parameters successfully exported to filepath: {filepath}")
        except Exception as e:
            print(f"An error occurred while exporting the parameter: {e}")

    def load(self):
        print_header("Loading parameters")
        return _convert_ndarray_to_list(
            {
                "network": self.network,
                "annotations": self.annotations,
                "neighborhoods": self.neighborhoods,
                "graph": self.graph,
                "plotter": self.plotter,
            }
        )


def _convert_ndarray_to_list(d):
    """
    Recursively convert all np.ndarray values in the dictionary to lists.

    Args:
        d (dict): The dictionary to process.

    Returns:
        dict: The processed dictionary with np.ndarray values converted to lists.
    """
    if isinstance(d, dict):
        return {k: _convert_ndarray_to_list(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_convert_ndarray_to_list(v) for v in d]
    elif isinstance(d, np.ndarray):
        return d.tolist()
    else:
        return d
