import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/package_source.py"
spec = importlib.util.spec_from_file_location("package_source", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_source_archive_excludes_untracked_and_refuses_dirty(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    (root / "source.py").write_text("x = 1\n", encoding="utf-8")
    git(root, "add", "source.py")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    (root / "data").mkdir()
    (root / "data/private.csv").write_text("not for upload", encoding="utf-8")
    (root / ".env").write_text("TEST_SECRET=not-real", encoding="utf-8")
    output = tmp_path / "source.tar.gz"
    assert len(module.package(root, output)) == 64
    with tarfile.open(output) as archive:
        assert set(archive.getnames()) == {"source.py", "SOURCE.json"}
        provenance = json.load(archive.extractfile("SOURCE.json"))
        assert len(provenance["git_sha"]) == 40
        assert set(provenance["files"]) == {"source.py"}
    with pytest.raises(FileExistsError):
        module.package(root, output)
    (root / "source.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Tracked changes"):
        module.package(root, tmp_path / "changed.tar.gz")


def test_source_archive_refuses_even_committed_data(tmp_path):
    git(tmp_path, "init")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/private.csv").write_text("raw prices", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
    with pytest.raises(ValueError, match="Unsafe tracked"):
        module.package(tmp_path, tmp_path / "bad.tar.gz")
