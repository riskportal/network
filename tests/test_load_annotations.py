"""
tests/test_load_annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import pytest
from scipy.sparse import csr_matrix, vstack

# Ensure dummy fixtures are imported by referencing them in test signatures below.


def test_missing_annotation_file(risk_obj, dummy_network):
    """Test loading an annotation file that does not exist.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        dummy_network: The network object to which annotations will be applied.
    """
    annotation_file = "nonexistent_file.csv"
    with pytest.raises(FileNotFoundError):
        risk_obj.load_csv_annotation(
            filepath=annotation_file,
            network=dummy_network,
            min_nodes_per_term=1,
        )


def test_load_csv_annotation(risk_obj, cytoscape_network, data_path):
    """Test loading a CSV annotation file and associating it with a network.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        cytoscape_network: The network object to which annotations will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "csv" / "annotations" / "go_biological_process.csv"
    annotations = risk_obj.load_csv_annotation(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
    )

    assert annotations is not None
    assert len(annotations) > 0  # Check that annotations are loaded.


def test_csv_annotation_structure(risk_obj, cytoscape_network, data_path):
    """Test that loaded CSV annotations have the correct structure.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        cytoscape_network: The network object to which annotations will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "csv" / "annotations" / "go_biological_process.csv"
    annotations = risk_obj.load_csv_annotation(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
    )

    assert isinstance(annotations, dict), "Annotations should be a dictionary"
    assert "matrix" in annotations, "Key 'matrix' missing in annotations"
    assert "ordered_annotations" in annotations, "Key 'ordered_annotations' missing in annotations"
    assert "ordered_nodes" in annotations, "Key 'ordered_nodes' missing in annotations"
    assert isinstance(annotations["matrix"], csr_matrix), "'matrix' should be a sparse matrix"
    assert isinstance(
        annotations["ordered_annotations"], tuple
    ), "'ordered_annotations' should be a tuple"
    assert isinstance(annotations["ordered_nodes"], tuple), "'ordered_nodes' should be a tuple"


def test_load_dict_annotation(risk_obj, dummy_network, dummy_annotation_dict):
    """Test loading annotations from a dictionary and associating them with a network.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        dummy_network: The network object to which annotations will be applied.
        dummy_annotation_dict: A dictionary containing annotations.
    """
    annotations = risk_obj.load_dict_annotation(
        content=dummy_annotation_dict,
        network=dummy_network,
        min_nodes_per_term=1,
    )

    assert annotations is not None
    assert len(annotations) > 0  # Check that annotations are loaded.


def test_dict_annotation_structure(risk_obj, dummy_network, dummy_annotation_dict):
    """Test the structure of dictionary-loaded annotations.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        dummy_network: The network object to which annotations will be applied.
        dummy_annotation_dict: A dictionary containing annotations.
    """
    annotations = risk_obj.load_dict_annotation(
        content=dummy_annotation_dict,
        network=dummy_network,
        min_nodes_per_term=1,
    )

    assert isinstance(annotations, dict), "Annotations should be a dictionary"
    assert "matrix" in annotations, "Key 'matrix' missing in annotations"
    assert "ordered_annotations" in annotations, "Key 'ordered_annotations' missing in annotations"
    assert "ordered_nodes" in annotations, "Key 'ordered_nodes' missing in annotations"
    assert isinstance(annotations["matrix"], csr_matrix), "'matrix' should be a sparse matrix"
    assert isinstance(
        annotations["ordered_annotations"], tuple
    ), "'ordered_annotations' should be a tuple"
    assert isinstance(annotations["ordered_nodes"], tuple), "'ordered_nodes' should be a tuple"


def test_load_json_annotation(risk_obj, cytoscape_network, data_path):
    """Test loading a JSON annotation file and associating it with a network.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        cytoscape_network: The network object to which annotations will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "json" / "annotations" / "go_biological_process.json"
    annotations = risk_obj.load_json_annotation(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
    )

    assert annotations is not None
    assert len(annotations) > 0  # Check that annotations are loaded.


def test_json_annotation_structure(risk_obj, cytoscape_network, data_path):
    """Test the structure of JSON-loaded annotations.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        cytoscape_network: The network object to which annotations will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "json" / "annotations" / "go_biological_process.json"
    annotations = risk_obj.load_json_annotation(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
    )

    assert isinstance(annotations, dict), "Annotations should be a dictionary"
    assert "matrix" in annotations, "Key 'matrix' missing in annotations"
    assert "ordered_annotations" in annotations, "Key 'ordered_annotations' missing in annotations"
    assert "ordered_nodes" in annotations, "Key 'ordered_nodes' missing in annotations"
    assert isinstance(annotations["matrix"], csr_matrix), "'matrix' should be a sparse matrix"
    assert isinstance(
        annotations["ordered_annotations"], tuple
    ), "'ordered_annotations' should be a tuple"
    assert isinstance(annotations["ordered_nodes"], tuple), "'ordered_nodes' should be a tuple"


def test_load_tsv_annotation(risk_obj, cytoscape_network, data_path):
    """Test loading a TSV annotation file and associating it with a network.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        cytoscape_network: The network object to which annotations will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "tsv" / "annotations" / "go_biological_process.tsv"
    annotations = risk_obj.load_tsv_annotation(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
    )

    assert annotations is not None
    assert len(annotations) > 0  # Check that annotations are loaded.


def test_tsv_annotation_structure(risk_obj, cytoscape_network, data_path):
    """Test the structure of TSV-loaded annotations.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        cytoscape_network: The network object to which annotations will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "tsv" / "annotations" / "go_biological_process.tsv"
    annotations = risk_obj.load_tsv_annotation(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
    )

    assert isinstance(annotations, dict), "Annotations should be a dictionary"
    assert "matrix" in annotations, "Key 'matrix' missing in annotations"
    assert "ordered_annotations" in annotations, "Key 'ordered_annotations' missing in annotations"
    assert "ordered_nodes" in annotations, "Key 'ordered_nodes' missing in annotations"
    assert isinstance(annotations["matrix"], csr_matrix), "'matrix' should be a sparse matrix"
    assert isinstance(
        annotations["ordered_annotations"], tuple
    ), "'ordered_annotations' should be a tuple"
    assert isinstance(annotations["ordered_nodes"], tuple), "'ordered_nodes' should be a tuple"


def test_load_excel_annotation(risk_obj, cytoscape_network, data_path):
    """Test loading an Excel annotation file and associating it with a network.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        cytoscape_network: The network object to which annotations will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "excel" / "annotations" / "go_biological_process.xlsx"
    annotations = risk_obj.load_excel_annotation(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
    )

    assert annotations is not None
    assert len(annotations) > 0  # Check that annotations are loaded.


def test_excel_annotation_structure(risk_obj, cytoscape_network, data_path):
    """Test the structure of Excel-loaded annotations.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        cytoscape_network: The network object to which annotations will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    annotation_file = data_path / "excel" / "annotations" / "go_biological_process.xlsx"
    annotations = risk_obj.load_excel_annotation(
        filepath=str(annotation_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
    )

    assert isinstance(annotations, dict), "Annotations should be a dictionary"
    assert "matrix" in annotations, "Key 'matrix' missing in annotations"
    assert "ordered_annotations" in annotations, "Key 'ordered_annotations' missing in annotations"
    assert "ordered_nodes" in annotations, "Key 'ordered_nodes' missing in annotations"
    assert isinstance(annotations["matrix"], csr_matrix), "'matrix' should be a sparse matrix"
    assert isinstance(
        annotations["ordered_annotations"], tuple
    ), "'ordered_annotations' should be a tuple"
    assert isinstance(annotations["ordered_nodes"], tuple), "'ordered_nodes' should be a tuple"


def test_combined_annotations(risk_obj, cytoscape_network, data_path):
    """Test combining annotations from multiple sources.

    Args:
        risk_obj: The RISK object instance used for loading annotations.
        cytoscape_network: The network object to which annotations will be applied.
        data_path: The base path to the directory containing the annotation files.
    """
    csv_file = data_path / "csv" / "annotations" / "go_biological_process.csv"
    json_file = data_path / "json" / "annotations" / "go_biological_process.json"
    csv_annotations = risk_obj.load_csv_annotation(
        filepath=str(csv_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
    )
    json_annotations = risk_obj.load_json_annotation(
        filepath=str(json_file),
        network=cytoscape_network,
        min_nodes_per_term=1,
    )
    # Combine the components of the annotations
    combined_annotations = {
        "matrix": vstack(
            (csv_annotations["matrix"], json_annotations["matrix"])
        ),  # Use vstack for sparse matrices
        "ordered_annotations": csv_annotations["ordered_annotations"]
        + json_annotations["ordered_annotations"],
        "ordered_nodes": csv_annotations["ordered_nodes"] + json_annotations["ordered_nodes"],
    }

    # Validate the combined annotations
    assert (
        combined_annotations["matrix"].shape[0]
        == csv_annotations["matrix"].shape[0] + json_annotations["matrix"].shape[0]
    )
    assert len(combined_annotations["ordered_annotations"]) == len(
        csv_annotations["ordered_annotations"]
    ) + len(json_annotations["ordered_annotations"])
    assert len(combined_annotations["ordered_nodes"]) == len(
        csv_annotations["ordered_nodes"]
    ) + len(json_annotations["ordered_nodes"])
