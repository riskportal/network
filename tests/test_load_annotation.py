"""
tests/test_load_annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import json

import pytest
from scipy.sparse import csr_matrix, vstack


def test_missing_annotation_file(risk_obj, dummy_network):
    """
    Test loading an annotation file that does not exist.

    Args:
        risk_obj: The RISK object instance used for loading annotation.
        dummy_network: The network object to which annotation will be applied.
    """
    annotation_file = "nonexistent_file.csv"
    with pytest.raises(FileNotFoundError):
        risk_obj.load_annotation_csv(
            filepath=annotation_file,
            network=dummy_network,
            min_nodes_per_term=1,
            max_nodes_per_term=1000,
        )


def test_load_annotation_csv(risk_obj, cytoscape_network, data_path):
    """
    Test loading a CSV annotation file and associating it with a network.

    Args:
        risk_obj: The RISK object instance used for loading annotation.
        cytoscape_network: The network object to which annotation will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "csv" / "annotation" / "go_biological_process.csv"
    annotation = risk_obj.load_annotation_csv(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
        max_nodes_per_term=1000,
    )

    assert annotation is not None
    assert len(annotation) > 0  # Check that annotation is loaded.


def test_csv_annotation_structure(risk_obj, cytoscape_network, data_path):
    """
    Test that loaded CSV annotation have the correct structure.

    Args:
        risk_obj: The RISK object instance used for loading annotation.
        cytoscape_network: The network object to which annotation will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "csv" / "annotation" / "go_biological_process.csv"
    annotation = risk_obj.load_annotation_csv(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
        max_nodes_per_term=1000,
    )

    assert isinstance(annotation, dict), "Annotation should be a dictionary"
    assert "matrix" in annotation, "Key 'matrix' missing in annotation"
    assert "ordered_annotation" in annotation, "Key 'ordered_annotation' missing in annotation"
    assert "ordered_nodes" in annotation, "Key 'ordered_nodes' missing in annotation"
    assert isinstance(annotation["matrix"], csr_matrix), "'matrix' should be a sparse matrix"
    assert isinstance(
        annotation["ordered_annotation"], tuple
    ), "'ordered_annotation' should be a tuple"
    assert isinstance(annotation["ordered_nodes"], tuple), "'ordered_nodes' should be a tuple"


def test_load_annotation_dict(risk_obj, dummy_network, dummy_annotation_dict):
    """
    Test loading annotation from a dictionary and associating them with a network.

    Args:
        risk_obj: The RISK object instance used for loading annotation.
        dummy_network: The network object to which annotation will be applied.
        dummy_annotation_dict: A dictionary containing annotation.
    """
    annotation = risk_obj.load_annotation_dict(
        content=dummy_annotation_dict,
        network=dummy_network,
        min_nodes_per_term=1,
        max_nodes_per_term=1000,
    )

    assert annotation is not None
    assert len(annotation) > 0  # Check that annotation is loaded.


def test_dict_annotation_structure(risk_obj, dummy_network, dummy_annotation_dict):
    """
    Test the structure of dictionary-loaded annotation.

    Args:
        risk_obj: The RISK object instance used for loading annotation.
        dummy_network: The network object to which annotation will be applied.
        dummy_annotation_dict: A dictionary containing annotation.
    """
    annotation = risk_obj.load_annotation_dict(
        content=dummy_annotation_dict,
        network=dummy_network,
        min_nodes_per_term=1,
        max_nodes_per_term=1000,
    )

    assert isinstance(annotation, dict), "Annotation should be a dictionary"
    assert "matrix" in annotation, "Key 'matrix' missing in annotation"
    assert "ordered_annotation" in annotation, "Key 'ordered_annotation' missing in annotation"
    assert "ordered_nodes" in annotation, "Key 'ordered_nodes' missing in annotation"
    assert isinstance(annotation["matrix"], csr_matrix), "'matrix' should be a sparse matrix"
    assert isinstance(
        annotation["ordered_annotation"], tuple
    ), "'ordered_annotation' should be a tuple"
    assert isinstance(annotation["ordered_nodes"], tuple), "'ordered_nodes' should be a tuple"


def test_load_annotation_json(risk_obj, cytoscape_network, data_path):
    """
    Test loading a JSON annotation file and associating it with a network.

    Args:
        risk_obj: The RISK object instance used for loading annotation.
        cytoscape_network: The network object to which annotation will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "json" / "annotation" / "go_biological_process.json"
    annotation = risk_obj.load_annotation_json(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
        max_nodes_per_term=1000,
    )

    assert annotation is not None
    assert len(annotation) > 0  # Check that annotation is loaded.


def test_json_annotation_structure(risk_obj, cytoscape_network, data_path):
    """
    Test the structure of JSON-loaded annotation.

    Args:
        risk_obj: The RISK object instance used for loading annotation.
        cytoscape_network: The network object to which annotation will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "json" / "annotation" / "go_biological_process.json"
    annotation = risk_obj.load_annotation_json(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
        max_nodes_per_term=1000,
    )

    assert isinstance(annotation, dict), "Annotation should be a dictionary"
    assert "matrix" in annotation, "Key 'matrix' missing in annotation"
    assert "ordered_annotation" in annotation, "Key 'ordered_annotation' missing in annotation"
    assert "ordered_nodes" in annotation, "Key 'ordered_nodes' missing in annotation"
    assert isinstance(annotation["matrix"], csr_matrix), "'matrix' should be a sparse matrix"
    assert isinstance(
        annotation["ordered_annotation"], tuple
    ), "'ordered_annotation' should be a tuple"
    assert isinstance(annotation["ordered_nodes"], tuple), "'ordered_nodes' should be a tuple"


def test_load_annotation_tsv(risk_obj, cytoscape_network, data_path):
    """
    Test loading a TSV annotation file and associating it with a network.

    Args:
        risk_obj: The RISK object instance used for loading annotation.
        cytoscape_network: The network object to which annotation will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "tsv" / "annotation" / "go_biological_process.tsv"
    annotation = risk_obj.load_annotation_tsv(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
        max_nodes_per_term=1000,
    )

    assert annotation is not None
    assert len(annotation) > 0  # Check that annotation is loaded.


def test_tsv_annotation_structure(risk_obj, cytoscape_network, data_path):
    """
    Test the structure of TSV-loaded annotation.

    Args:
        risk_obj: The RISK object instance used for loading annotation.
        cytoscape_network: The network object to which annotation will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "tsv" / "annotation" / "go_biological_process.tsv"
    annotation = risk_obj.load_annotation_tsv(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
        max_nodes_per_term=1000,
    )

    assert isinstance(annotation, dict), "Annotation should be a dictionary"
    assert "matrix" in annotation, "Key 'matrix' missing in annotation"
    assert "ordered_annotation" in annotation, "Key 'ordered_annotation' missing in annotation"
    assert "ordered_nodes" in annotation, "Key 'ordered_nodes' missing in annotation"
    assert isinstance(annotation["matrix"], csr_matrix), "'matrix' should be a sparse matrix"
    assert isinstance(
        annotation["ordered_annotation"], tuple
    ), "'ordered_annotation' should be a tuple"
    assert isinstance(annotation["ordered_nodes"], tuple), "'ordered_nodes' should be a tuple"


def test_load_annotation_excel(risk_obj, cytoscape_network, data_path):
    """
    Test loading an Excel annotation file and associating it with a network.

    Args:
        risk_obj: The RISK object instance used for loading annotation.
        cytoscape_network: The network object to which annotation will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "excel" / "annotation" / "go_biological_process.xlsx"
    annotation = risk_obj.load_annotation_excel(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
        max_nodes_per_term=1000,
    )

    assert annotation is not None
    assert len(annotation) > 0  # Check that annotation is loaded.


def test_excel_annotation_structure(risk_obj, cytoscape_network, data_path):
    """
    Test the structure of Excel-loaded annotation.

    Args:
        risk_obj: The RISK object instance used for loading annotation.
        cytoscape_network: The network object to which annotation will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "excel" / "annotation" / "go_biological_process.xlsx"
    annotation = risk_obj.load_annotation_excel(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
        max_nodes_per_term=1000,
    )

    assert isinstance(annotation, dict), "Annotation should be a dictionary"
    assert "matrix" in annotation, "Key 'matrix' missing in annotation"
    assert "ordered_annotation" in annotation, "Key 'ordered_annotation' missing in annotation"
    assert "ordered_nodes" in annotation, "Key 'ordered_nodes' missing in annotation"
    assert isinstance(annotation["matrix"], csr_matrix), "'matrix' should be a sparse matrix"
    assert isinstance(
        annotation["ordered_annotation"], tuple
    ), "'ordered_annotation' should be a tuple"
    assert isinstance(annotation["ordered_nodes"], tuple), "'ordered_nodes' should be a tuple"


def test_combined_annotation(risk_obj, cytoscape_network, data_path):
    """
    Test combining annotations from multiple sources.

    Args:
        risk_obj: The RISK object instance used for loading annotation.
        cytoscape_network: The network object to which annotation will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    csv_file = data_path / "csv" / "annotation" / "go_biological_process.csv"
    json_file = data_path / "json" / "annotation" / "go_biological_process.json"
    csv_annotation = risk_obj.load_annotation_csv(
        filepath=str(csv_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
        max_nodes_per_term=1000,
    )
    json_annotation = risk_obj.load_annotation_json(
        filepath=str(json_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
        max_nodes_per_term=1000,
    )
    # Combine the components of the annotations
    combined_annotation = {
        "matrix": vstack(
            (csv_annotation["matrix"], json_annotation["matrix"])
        ),  # Use vstack for sparse matrices
        "ordered_annotation": csv_annotation["ordered_annotation"]
        + json_annotation["ordered_annotation"],
        "ordered_nodes": csv_annotation["ordered_nodes"] + json_annotation["ordered_nodes"],
    }

    # Validate the combined annotation
    assert (
        combined_annotation["matrix"].shape[0]
        == csv_annotation["matrix"].shape[0] + json_annotation["matrix"].shape[0]
    )
    assert len(combined_annotation["ordered_annotation"]) == len(
        csv_annotation["ordered_annotation"]
    ) + len(json_annotation["ordered_annotation"])
    assert len(combined_annotation["ordered_nodes"]) == len(csv_annotation["ordered_nodes"]) + len(
        json_annotation["ordered_nodes"]
    )


def test_min_max_nodes_per_term(risk_obj, cytoscape_network, data_path):
    """
    Test that loaded annotation respects min and max node limits per term.

    Args:
        risk_obj: The RISK object instance used for loading annotation.
        cytoscape_network: The network object to which annotation will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "json" / "annotation" / "go_biological_process.json"
    min_nodes = 2
    max_nodes = 100
    # Load annotation with filtering
    annotation = risk_obj.load_annotation_json(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=min_nodes,
        max_nodes_per_term=max_nodes,
    )
    # Extract the mapping of term to genes from the raw JSON input
    with open(annotation_file, "r") as f:
        raw_dict = json.load(f)

    filtered_terms = annotation["ordered_annotation"]
    for term in filtered_terms:
        gene_count = len(raw_dict[term])
        assert gene_count >= min_nodes, f"Term {term} has too few genes: {gene_count}"
        assert gene_count <= max_nodes, f"Term {term} has too many genes: {gene_count}"
