"""
risk/_annotation/_io
~~~~~~~~~~~~~~~~~~~~
"""

import json
from typing import Any, Dict

import networkx as nx
import pandas as pd

from .._log import log_header, logger, params
from ._annotation import load_annotation


class AnnotationIO:
    """
    Handles the loading and exporting of annotation in various file formats.

    The AnnotationIO class provides methods to load annotation from different file types (JSON, CSV, Excel, etc.)
    and to export parameter data to various formats like JSON, CSV, and text files.
    """

    def load_annotation_json(
        self,
        network: nx.Graph,
        filepath: str,
        min_nodes_per_term: int = 1,
        max_nodes_per_term: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Load annotation from a JSON file and convert them to a DataFrame.

        Args:
            network (NetworkX graph): The network to which the annotation is related.
            filepath (str): Path to the JSON annotation file.
            min_nodes_per_term (int, optional): The minimum number of network nodes required for each annotation
                term to be included. Defaults to 1.
            max_nodes_per_term (int, optional): The maximum number of network nodes allowed for each annotation
                term to be included. Defaults to 10_000.

        Returns:
            Dict[str, Any]: A dictionary containing ordered nodes, ordered annotations, and the annotation matrix.
        """
        filetype = "JSON"
        # Log the loading of the JSON file
        params.log_annotation(
            filetype=filetype,
            filepath=filepath,
            min_nodes_per_term=min_nodes_per_term,
            max_nodes_per_term=max_nodes_per_term,
        )
        self._log_loading_annotation(filetype, filepath=filepath)

        # Load the JSON file into a dictionary
        with open(filepath, "r", encoding="utf-8") as file:
            annotation_input = json.load(file)

        return load_annotation(network, annotation_input, min_nodes_per_term, max_nodes_per_term)

    def load_annotation_excel(
        self,
        network: nx.Graph,
        filepath: str,
        label_colname: str = "label",
        nodes_colname: str = "nodes",
        sheet_name: str = "Sheet1",
        nodes_delimiter: str = ";",
        min_nodes_per_term: int = 1,
        max_nodes_per_term: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Load annotation from an Excel file and associate them with the network.

        Args:
            network (nx.Graph): The NetworkX graph to which the annotation is related.
            filepath (str): Path to the Excel annotation file.
            label_colname (str): Name of the column containing the labels (e.g., GO terms).
            nodes_colname (str): Name of the column containing the nodes associated with each label.
            sheet_name (str, optional): The name of the Excel sheet to load (default is 'Sheet1').
            nodes_delimiter (str, optional): Delimiter used to separate multiple nodes within the nodes column (default is ';').
            min_nodes_per_term (int, optional): The minimum number of network nodes required for each annotation
                term to be included. Defaults to 1.
            max_nodes_per_term (int, optional): The maximum number of network nodes allowed for each annotation
                term to be included. Defaults to 10_000.

        Returns:
            Dict[str, Any]: A dictionary where each label is paired with its respective list of nodes,
                            linked to the provided network.
        """
        filetype = "Excel"
        # Log the loading of the Excel file
        params.log_annotation(
            filetype=filetype,
            filepath=filepath,
            min_nodes_per_term=min_nodes_per_term,
            max_nodes_per_term=max_nodes_per_term,
        )
        self._log_loading_annotation(filetype, filepath=filepath)

        # Load the specified sheet from the Excel file
        annotation = pd.read_excel(filepath, sheet_name=sheet_name)
        # Split the nodes column by the specified nodes_delimiter
        annotation[nodes_colname] = annotation[nodes_colname].apply(
            lambda x: x.split(nodes_delimiter)
        )
        # Convert the DataFrame to a dictionary pairing labels with their corresponding nodes
        annotation_input = annotation.set_index(label_colname)[nodes_colname].to_dict()

        return load_annotation(network, annotation_input, min_nodes_per_term, max_nodes_per_term)

    def load_annotation_csv(
        self,
        network: nx.Graph,
        filepath: str,
        label_colname: str = "label",
        nodes_colname: str = "nodes",
        nodes_delimiter: str = ";",
        min_nodes_per_term: int = 1,
        max_nodes_per_term: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Load annotation from a CSV file and associate them with the network.

        Args:
            network (nx.Graph): The NetworkX graph to which the annotation is related.
            filepath (str): Path to the CSV annotation file.
            label_colname (str): Name of the column containing the labels (e.g., GO terms).
            nodes_colname (str): Name of the column containing the nodes associated with each label.
            nodes_delimiter (str, optional): Delimiter used to separate multiple nodes within the nodes column (default is ';').
            min_nodes_per_term (int, optional): The minimum number of network nodes required for each annotation
                term to be included. Defaults to 1.
            max_nodes_per_term (int, optional): The maximum number of network nodes allowed for each annotation
                term to be included. Defaults to 10_000.

        Returns:
            Dict[str, Any]: A dictionary where each label is paired with its respective list of nodes,
                            linked to the provided network.
        """
        filetype = "CSV"
        # Log the loading of the CSV file
        params.log_annotation(
            filetype=filetype,
            filepath=filepath,
            min_nodes_per_term=min_nodes_per_term,
            max_nodes_per_term=max_nodes_per_term,
        )
        self._log_loading_annotation(filetype, filepath=filepath)

        # Load the CSV file into a dictionary
        annotation_input = self._load_matrix_file(
            filepath, label_colname, nodes_colname, delimiter=",", nodes_delimiter=nodes_delimiter
        )

        return load_annotation(network, annotation_input, min_nodes_per_term, max_nodes_per_term)

    def load_annotation_tsv(
        self,
        network: nx.Graph,
        filepath: str,
        label_colname: str = "label",
        nodes_colname: str = "nodes",
        nodes_delimiter: str = ";",
        min_nodes_per_term: int = 1,
        max_nodes_per_term: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Load annotation from a TSV file and associate them with the network.

        Args:
            network (nx.Graph): The NetworkX graph to which the annotation is related.
            filepath (str): Path to the TSV annotation file.
            label_colname (str): Name of the column containing the labels (e.g., GO terms).
            nodes_colname (str): Name of the column containing the nodes associated with each label.
            nodes_delimiter (str, optional): Delimiter used to separate multiple nodes within the nodes column (default is ';').
            min_nodes_per_term (int, optional): The minimum number of network nodes required for each annotation
                term to be included. Defaults to 1.
            max_nodes_per_term (int, optional): The maximum number of network nodes allowed for each annotation
                term to be included. Defaults to 10_000.

        Returns:
            Dict[str, Any]: A dictionary where each label is paired with its respective list of nodes,
                            linked to the provided network.
        """
        filetype = "TSV"
        # Log the loading of the TSV file
        params.log_annotation(
            filetype=filetype,
            filepath=filepath,
            min_nodes_per_term=min_nodes_per_term,
            max_nodes_per_term=max_nodes_per_term,
        )
        self._log_loading_annotation(filetype, filepath=filepath)

        # Load the TSV file into a dictionary
        annotation_input = self._load_matrix_file(
            filepath, label_colname, nodes_colname, delimiter="\t", nodes_delimiter=nodes_delimiter
        )

        return load_annotation(network, annotation_input, min_nodes_per_term, max_nodes_per_term)

    def load_annotation_dict(
        self,
        network: nx.Graph,
        content: Dict[str, Any],
        min_nodes_per_term: int = 1,
        max_nodes_per_term: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Load annotation from a provided dictionary and convert them to a dictionary annotation.

        Args:
            network (NetworkX graph): The network to which the annotation is related.
            content (Dict[str, Any]): The annotation dictionary to load.
            min_nodes_per_term (int, optional): The minimum number of network nodes required for each annotation
                term to be included. Defaults to 1.
            max_nodes_per_term (int, optional): The maximum number of network nodes allowed for each annotation
                term to be included. Defaults to 10_000.

        Returns:
            Dict[str, Any]: A dictionary containing ordered nodes, ordered annotations, and the annotation matrix.

        Raises:
            TypeError: If the content is not a dictionary.
        """
        # Ensure the input content is a dictionary
        if not isinstance(content, dict):
            raise TypeError(
                f"Expected 'content' to be a dictionary, but got {type(content).__name__} instead."
            )

        filetype = "Dictionary"
        # Log the loading of the annotation from the dictionary
        params.log_annotation(
            filepath="In-memory dictionary",
            filetype=filetype,
            min_nodes_per_term=min_nodes_per_term,
            max_nodes_per_term=max_nodes_per_term,
        )
        self._log_loading_annotation(filetype, "In-memory dictionary")

        # Load the annotation as a dictionary from the provided dictionary
        return load_annotation(network, content, min_nodes_per_term, max_nodes_per_term)

    def _load_matrix_file(
        self,
        filepath: str,
        label_colname: str,
        nodes_colname: str,
        delimiter: str = ",",
        nodes_delimiter: str = ";",
    ) -> Dict[str, Any]:
        """
        Load annotation from a CSV or TSV file and convert them to a dictionary.

        Args:
            filepath (str): Path to the annotation file.
            label_colname (str): Name of the column containing the labels (e.g., GO terms).
            nodes_colname (str): Name of the column containing the nodes associated with each label.
            delimiter (str, optional): Delimiter used to separate columns in the file (default is ',').
            nodes_delimiter (str, optional): Delimiter used to separate multiple nodes within the nodes column (default is ';').

        Returns:
            Dict[str, Any]: A dictionary where each label is paired with its respective list of nodes.
        """
        # Load the CSV or TSV file into a DataFrame
        annotation = pd.read_csv(filepath, delimiter=delimiter)
        # Split the nodes column by the nodes_delimiter to handle multiple nodes per label
        annotation[nodes_colname] = annotation[nodes_colname].apply(
            lambda x: x.split(nodes_delimiter)
        )
        # Create a dictionary pairing labels with their corresponding list of nodes
        label_node_dict = annotation.set_index(label_colname)[nodes_colname].to_dict()
        return label_node_dict

    def _log_loading_annotation(self, filetype: str, filepath: str = "") -> None:
        """
        Log the loading of annotation files.

        Args:
            filetype (str): The type of the file being loaded (e.g., 'Cytoscape').
            filepath (str, optional): The path to the file being loaded.
        """
        log_header("Loading annotation")
        logger.debug(f"Filetype: {filetype}")
        if filepath:
            logger.debug(f"Filepath: {filepath}")
