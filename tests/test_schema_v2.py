from neural_labs.catalog import ROOT
from neural_labs.schema import validate_repository


def test_repository_schema_has_no_errors() -> None:
    issues = validate_repository(ROOT)
    errors = [issue for issue in issues if issue.severity == "error"]
    assert errors == []
