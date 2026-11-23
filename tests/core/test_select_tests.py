"""Tests for tools/select_tests.py, the CI affected-test selector.

Each test builds a tiny repo shaped like PyHealth (a ``pyhealth`` package
with re-exporting ``__init__``s and a ``tests/core`` suite) and checks which
test modules a change selects. Stdlib only, so these run anywhere.
"""

import importlib.util
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "select_tests.py"
_spec = importlib.util.spec_from_file_location("select_tests", _TOOL)
st = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = st  # dataclasses look the module up by name
_spec.loader.exec_module(st)

FILES = {
    "pyhealth/__init__.py": "VERSION = '0'\n",
    "pyhealth/models/__init__.py": """
        \"\"\"Models.\"\"\"
        from .base import BaseModel
        from .rnn import RNN
        from .cnn import CNN
        __all__ = ["BaseModel", "RNN", "CNN"]
    """,
    "pyhealth/models/base.py": "class BaseModel: pass\n",
    "pyhealth/models/rnn.py": "from .base import BaseModel\nclass RNN(BaseModel): pass\n",
    "pyhealth/models/cnn.py": "from pyhealth.models.base import BaseModel\nclass CNN(BaseModel): pass\n",
    "pyhealth/medcode/__init__.py": "from .icd import ICD\nfrom .atc import ATC\n",
    "pyhealth/medcode/icd.py": "class ICD: pass\n",
    "pyhealth/medcode/atc.py": "class ATC: pass\n",
    "pyhealth/tasks/__init__.py": """
        def __getattr__(name):
            if name == "Lazy":
                from .lazy import Lazy
                return Lazy
            raise AttributeError(name)
    """,
    "pyhealth/tasks/lazy.py": "class Lazy: pass\n",
    "pyhealth/datasets/__init__.py": "from .toy import ToyDataset\n",
    "pyhealth/datasets/toy.py": (
        "from pathlib import Path\n"
        "CONFIG = Path(__file__).parent / 'configs' / 'toy.yaml'\n"
        "class ToyDataset: pass\n"
    ),
    "pyhealth/datasets/configs/toy.yaml": "tables: {}\n",
    "tests/__init__.py": "",
    "tests/base.py": "",
    "tests/core/__init__.py": "",
    "tests/core/test_rnn.py": "from pyhealth.models import RNN\n",
    "tests/core/test_cnn.py": "from pyhealth.models import CNN\n",
    "tests/core/test_lazy_inner.py": """
        def test():
            from pyhealth.tasks import Lazy
    """,
    "tests/core/test_attr.py": "import pyhealth.medcode as mc\nx = mc.ICD\n",
    "tests/core/test_bare.py": "import pyhealth.medcode as mc\ngetattr(mc, 'ATC')\n",
    "tests/core/test_toy.py": (
        "from pyhealth.datasets import ToyDataset\n"
        "ROOT = 'test-resources/core/toydemo'\n"
    ),
    "tests/core/helpers.py": "from pyhealth.models.cnn import CNN\n",
    "pyhealth/models/report.py": "def coverage_report(): pass\n",
    "tests/core/test_uses_helper.py": "from tests.core.helpers import CNN\n",
    "tests/test_not_in_suite.py": "from pyhealth.models import RNN\n",
}


class SelectTestsTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        for rel, text in FILES.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(text).lstrip())
        self.graph = st.Graph(root=self.root)

    def select(self, *paths, base_source=None):
        sel = st.select(list(paths), self.graph, base_source)
        return st.ALL if sel.mode == st.ALL else {t.split(".")[-1] for t in sel.tests}

    def test_reexport_resolves_to_defining_module(self):
        # Both tests import through models/__init__, but only one uses RNN.
        self.assertEqual(self.select("pyhealth/models/rnn.py"), {"test_rnn"})

    def test_transitive_dependency(self):
        self.assertEqual(self.select("pyhealth/models/base.py"),
                         {"test_rnn", "test_cnn", "test_uses_helper"})

    def test_init_change_selects_its_importers(self):
        # test_uses_helper imports pyhealth.models.cnn, which runs the
        # package __init__ too.
        self.assertEqual(self.select("pyhealth/models/__init__.py"),
                         {"test_rnn", "test_cnn", "test_uses_helper"})

    def test_init_reexport_only_change_is_narrowed(self):
        new = self.root / "pyhealth/models/__init__.py"
        old = new.read_text().replace("from .cnn import CNN\n", "")
        old = old.replace(', "CNN"', "")
        sel = self.select("pyhealth/models/__init__.py",
                          base_source=lambda p: old if p == "pyhealth/models/__init__.py" else None)
        self.assertEqual(sel, {"test_cnn"})

    def test_init_logic_change_is_not_narrowed(self):
        new = self.root / "pyhealth/models/__init__.py"
        old = new.read_text() + "SETTING = 1\n"
        sel = self.select("pyhealth/models/__init__.py", base_source=lambda p: old)
        self.assertEqual(sel, {"test_rnn", "test_cnn", "test_uses_helper"})

    def test_lazy_getattr_and_function_level_import(self):
        self.assertEqual(self.select("pyhealth/tasks/lazy.py"), {"test_lazy_inner"})

    def test_attribute_access_on_module_alias(self):
        self.assertEqual(self.select("pyhealth/medcode/icd.py"), {"test_attr", "test_bare"})
        # test_bare passes the module to getattr, so it depends on all of it.
        self.assertEqual(self.select("pyhealth/medcode/atc.py"), {"test_bare"})

    def test_changed_test_selects_itself(self):
        self.assertEqual(self.select("tests/core/test_cnn.py"), {"test_cnn"})

    def test_tests_outside_ci_suites_are_never_selected(self):
        self.assertEqual(self.select("tests/test_not_in_suite.py"), set())

    def test_config_maps_to_module_naming_it(self):
        self.assertEqual(self.select("pyhealth/datasets/configs/toy.yaml"), {"test_toy"})

    def test_resource_maps_to_tests_naming_it(self):
        self.assertEqual(self.select("test-resources/core/toydemo/a.csv"), {"test_toy"})

    def test_unreferenced_resource_runs_all(self):
        self.assertEqual(self.select("test-resources/core/orphan/a.csv"), st.ALL)

    def test_deleted_module_selects_importers(self):
        (self.root / "pyhealth/models/cnn.py").unlink()
        self.graph = st.Graph(root=self.root)
        self.assertIn("test_uses_helper", self.select("pyhealth/models/cnn.py"))

    def test_infrastructure_runs_all(self):
        for path in ("pyproject.toml", "tests/base.py", "pyhealth/__init__.py",
                     "tests/core/__init__.py", "tools/select_tests.py",
                     "some/unknown/file.txt"):
            with self.subTest(path=path):
                self.assertEqual(self.select(path), st.ALL)

    def test_docs_select_nothing(self):
        self.assertEqual(self.select("README.rst", "docs/index.rst",
                                     "examples/x.py", "tools/bump_version.py",
                                     "pyhealth/models/README.md"), set())

    def test_tool_stem_matching_an_identifier_selects_nothing(self):
        self.assertEqual(self.select("tools/coverage_report.py"), set())

    def test_any_all_wins(self):
        self.assertEqual(self.select("pyhealth/models/rnn.py", "pixi.lock"), st.ALL)


if __name__ == "__main__":
    unittest.main()
