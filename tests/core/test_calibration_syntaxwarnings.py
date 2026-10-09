import warnings
from pathlib import Path


def test_calibration_module_compiles_without_escape_sequence_warnings():
    source_path = (
        Path(__file__).parents[2] / "pyhealth" / "metrics" / "calibration.py"
    )
    source = source_path.read_text(encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        warnings.simplefilter("error", DeprecationWarning)
        compile(source, str(source_path), "exec")
