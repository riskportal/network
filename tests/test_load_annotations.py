def test_load_csv_annotation(risk_obj, network, data_path):
    annotation_file = data_path / "csv" / "annotations" / "go_biological_process.csv"
    annotations = risk_obj.load_csv_annotation(filepath=str(annotation_file), network=network)

    assert annotations is not None
    assert len(annotations) > 0  # Check that annotations are loaded


def test_load_json_annotation(risk_obj, network, data_path):
    annotation_file = data_path / "json" / "annotations" / "go_biological_process.json"
    annotations = risk_obj.load_json_annotation(filepath=str(annotation_file), network=network)

    assert annotations is not None
    assert len(annotations) > 0  # Check that annotations are loaded


def test_load_tsv_annotation(risk_obj, network, data_path):
    annotation_file = data_path / "tsv" / "annotations" / "go_biological_process.tsv"
    annotations = risk_obj.load_tsv_annotation(filepath=str(annotation_file), network=network)

    assert annotations is not None
    assert len(annotations) > 0  # Check that annotations are loaded


def test_load_excel_annotation(risk_obj, network, data_path):
    annotation_file = data_path / "excel" / "annotations" / "go_biological_process.xlsx"
    annotations = risk_obj.load_excel_annotation(filepath=str(annotation_file), network=network)

    assert annotations is not None
    assert len(annotations) > 0  # Check that annotations are loaded
