"""
risk/annotations/io
~~~~~~~~~~~~~~~~~~~

This file contains the code for the RISK class and command-line access.
"""

import json
from typing import Any, Dict

import networkx as nx
import pandas as pd

from risk.annotations.annotations import load_annotations
from risk.log import params, print_header


class AnnotationsIO:
    """Handles the loading and exporting of annotations in various file formats.

    The AnnotationsIO class provides methods to load annotations from different file types (JSON, CSV, Excel, etc.)
    and to export parameter data to various formats like JSON, CSV, and text files.
    """

    def __init__(self):
        pass

    def load_json_annotations(self, filepath: str, network: nx.Graph) -> Dict[str, Any]:
        """Load annotations from a JSON file and convert them to a DataFrame.

        Args:
            filepath (str): Path to the JSON annotations file.
            network (NetworkX graph): The network to which the annotations are related.

        Returns:
            dict: A dictionary containing ordered nodes, ordered annotations, and the annotations matrix.
        """
        filetype = "JSON"
        params.log_annotations(filepath=filepath, filetype=filetype)
        _log_loading(filetype, filepath=filepath)
        # Open and read the JSON file
        with open(filepath, "r") as file:
            annotations_input = json.load(file)

        # Process the JSON data and return it in the context of the network
        return load_annotations(network, annotations_input)

    def load_csv_annotation(
        self,
        filepath: str,
        network: nx.Graph,
        label_colname: str = "label",
        nodes_colname: str = "nodes",
        delimiter: str = ";",
    ) -> Dict[str, Any]:
        """Load annotations from a CSV file and convert them to a DataFrame.

        Args:
            filepath (str): Path to the CSV annotations file.
            network (NetworkX graph): The network to which the annotations are related.
            label_colname (str): Name of the column containing the labels.
            nodes_colname (str): Name of the column containing the nodes.
            delimiter (str): Delimiter used to parse the nodes column (default is ';').

        Returns:
            pd.DataFrame: DataFrame containing the labels and parsed nodes.
        """
        filetype = "CSV"
        params.log_annotations(filepath=filepath, filetype=filetype)
        _log_loading(filetype, filepath=filepath)
        # Load the CSV file into a dictionary
        annotations_input = _load_matrix_file(filepath, label_colname, nodes_colname, delimiter)
        # Process and return the annotations in the context of the network
        return load_annotations(network, annotations_input)

    def load_excel_annotation(
        self,
        filepath: str,
        network: nx.Graph,
        label_colname: str = "label",
        nodes_colname: str = "nodes",
        sheet_name: str = "Sheet1",
        delimiter: str = ";",
    ) -> Dict[str, Any]:
        """Load annotations from an Excel file and convert them to a dictionary.

        Args:
            filepath (str): Path to the Excel annotations file.
            network (NetworkX graph): The network to which the annotations are related.
            label_colname (str): Name of the column containing the labels.
            nodes_colname (str): Name of the column containing the nodes.
            sheet_name (str): The name of the Excel sheet to load (default is 'Sheet1').
            delimiter (str): Delimiter used to parse the nodes column (default is ';').

        Returns:
            dict: A dictionary where each label is paired with its respective list of nodes.
        """
        filetype = "Excel"
        params.log_annotations(filepath=filepath, filetype=filetype)
        _log_loading(filetype, filepath=filepath)
        # Load the specified sheet from the Excel file
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        # Split the nodes column by the specified delimiter
        df[nodes_colname] = df[nodes_colname].apply(lambda x: x.split(delimiter))
        # Convert the DataFrame to a dictionary pairing labels with their corresponding nodes
        label_node_dict = df.set_index(label_colname)[nodes_colname].to_dict()
        return load_annotations(network, label_node_dict)

    def load_tsv_annotation(
        self,
        filepath: str,
        network: nx.Graph,
        label_colname: str = "label",
        nodes_colname: str = "nodes",
    ) -> Dict[str, Any]:
        """Load annotations from a TSV file and convert them to a DataFrame.

        Args:
            filepath (str): Path to the TSV annotations file.
            network (NetworkX graph): The network to which the annotations are related.
            label_colname (str): Name of the column containing the labels.
            nodes_colname (str): Name of the column containing the nodes.

        Returns:
            pd.DataFrame: DataFrame containing the labels and parsed nodes.
        """
        filetype = "TSV"
        params.log_annotations(filepath=filepath, filetype=filetype)
        _log_loading(filetype, filepath=filepath)
        # Load the TSV file with tab delimiter and convert to dictionary
        annotations_input = _load_matrix_file(
            filepath, label_colname, nodes_colname, delimiter="\t"
        )
        # Process and return the annotations in the context of the network
        return load_annotations(network, annotations_input)


def _load_matrix_file(
    filepath: str, label_colname: str, nodes_colname: str, delimiter: str = ";"
) -> Dict[str, Any]:
    """Load annotations from a CSV or TSV file and convert them to a dictionary.

    Args:
        filepath (str): Path to the annotation file.
        label_colname (str): Name of the column containing the labels.
        nodes_colname (str): Name of the column containing the nodes.
        delimiter (str): Delimiter used to parse the nodes column (default is ';').

    Returns:
        dict: A dictionary where each label is paired with its respective list of nodes.
    """
    df = pd.read_csv(filepath)
    # Split the nodes column by the delimiter
    df[nodes_colname] = df[nodes_colname].apply(lambda x: x.split(delimiter))
    # Create a dictionary pairing labels with their corresponding nodes
    label_node_dict = df.set_index(label_colname)[nodes_colname].to_dict()
    return label_node_dict


def _log_loading(filetype: str, filepath: str = "") -> None:
    """Log information about the network file being loaded.

    Args:
        filetype (str): The type of the file being loaded (e.g., 'Cytoscape').
        filepath (str, optional): The path to the file being loaded.
    """
    print_header("Loading annotations")
    print(f"Filetype: {filetype}")
    if filepath:
        print(f"Filepath: {filepath}")
