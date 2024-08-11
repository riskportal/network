"""
risk/network/io/annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This file contains the code for the RISK class and command-line access.
"""

import json

import pandas as pd

from risk.annotations.annotations import load_annotations
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

    def load_csv_annotation(
        self, filepath, network, label_colname="label", nodes_colname="nodes", delimiter=";"
    ):
        """Load annotations from a CSV file and convert them to a DataFrame.

        Args:
            filepath (str): Path to the CSV annotations file.
            network: The network to which the annotations are related.
            label_colname (str): Name of the column containing the labels.
            nodes_colname (str): Name of the column containing the nodes.
            delimiter (str): Delimiter used to parse the nodes column (default is ';').

        Returns:
            pd.DataFrame: DataFrame containing the labels and parsed nodes.
        """
        filetype = "CSV"
        params.log_annotations(filepath=filepath, filetype=filetype)
        _log_loading(filetype, filepath=filepath)
        # Load the CSV file using the utility function
        annotations_df = _load_matrix_file(filepath, label_colname, nodes_colname, delimiter)
        # Process the DataFrame as needed for your specific use case
        return load_annotations(network, annotations_df)

    def load_excel_annotation(
        self,
        filepath,
        network,
        label_colname="label",
        nodes_colname="nodes",
        sheet_name="Sheet1",
        delimiter=";",
    ):
        """Load annotations from an Excel file and convert them to a DataFrame.

        Args:
            filepath (str): Path to the Excel annotations file.
            network: The network to which the annotations are related.
            label_colname (str): Name of the column containing the labels.
            nodes_colname (str): Name of the column containing the nodes.
            sheet_name (str): The name of the Excel sheet to load (default is 'Sheet1').
            delimiter (str): Delimiter used to parse the nodes column (default is ';').

        Returns:
            pd.DataFrame: DataFrame containing the labels and parsed nodes.
        """
        filetype = "Excel"
        params.log_annotations(filepath=filepath, filetype=filetype)
        _log_loading(filetype, filepath=filepath)
        # Load the Excel file into a DataFrame
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        # Split the nodes column by the delimiter
        df[nodes_colname] = df[nodes_colname].apply(lambda x: x.split(delimiter))
        # Process the DataFrame as needed for your specific use case
        annotations_df = df[[label_colname, nodes_colname]]
        return load_annotations(network, annotations_df)

    def load_tsv_annotation(self, filepath, network, label_colname="label", nodes_colname="nodes"):
        """Load annotations from a TSV file and convert them to a DataFrame.

        Args:
            filepath (str): Path to the TSV annotations file.
            network: The network to which the annotations are related.
            label_colname (str): Name of the column containing the labels.
            nodes_colname (str): Name of the column containing the nodes.

        Returns:
            pd.DataFrame: DataFrame containing the labels and parsed nodes.
        """
        filetype = "TSV"
        params.log_annotations(filepath=filepath, filetype=filetype)
        _log_loading(filetype, filepath=filepath)
        # TSV files use a tab delimiter, so we pass '\t' to the utility function
        annotations_df = _load_matrix_file(filepath, label_colname, nodes_colname, delimiter="\t")
        # Process the DataFrame as needed for your specific use case
        return load_annotations(network, annotations_df)


def _load_matrix_file(filepath, label_colname, nodes_colname, delimiter=";"):
    """Load annotations from a CSV or TSV file and convert them to a DataFrame.

    Args:
        filepath (str): Path to the annotation file.
        label_colname (str): Name of the column containing the labels.
        nodes_colname (str): Name of the column containing the nodes.
        delimiter (str): Delimiter used to parse the nodes column (default is ';').

    Returns:
        pd.DataFrame: DataFrame containing the labels and parsed nodes.
    """
    df = pd.read_csv(filepath)
    # Split the nodes column by the delimiter
    df[nodes_colname] = df[nodes_colname].apply(lambda x: x.split(delimiter))
    return df[[label_colname, nodes_colname]]


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
