from importlib.metadata import metadata, version
from pathlib import Path

import clhs


def test_installed_version_matches_module():
    assert version("clhs") == clhs.__version__


def test_installed_metadata():
    package_metadata = metadata("clhs")
    assert package_metadata["Requires-Python"] == ">=3.9"
    requirements = package_metadata.get_all("Requires-Dist")
    for requirement in ("numpy>=1.26.0", "pandas>=2.0.0", "tqdm>=4.66.0"):
        assert requirement in requirements
    assert package_metadata["License-Expression"] == "MIT"


def test_package_source_compiles_without_syntax_warnings():
    for source in Path(clhs.__file__).parent.glob("*.py"):
        compile(source.read_text(encoding="utf-8"), str(source), "exec")
