"""Package committed tracked source only; never include data, secrets or a venv."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


def package(root: Path, output: Path) -> str:
    root, output = root.resolve(), output.resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    if git("status", "--porcelain", "--untracked-files=no").strip():
        raise ValueError("Tracked changes exist; commit and verify them before packaging")
    commit = git("rev-parse", "HEAD").decode().strip()
    names = [x.decode("utf-8") for x in git("ls-files", "-z").split(b"\0") if x]
    contents = {}
    for name in names:
        parts = PurePosixPath(name).parts
        path = root / name
        forbidden = {"data", "runs", "outputs", "mlruns", ".venv", ".git", "dist"}
        if (PurePosixPath(name).is_absolute() or ".." in parts or not parts
                or parts[0] in forbidden or any(p.startswith(".env") for p in parts)
                or name == "SOURCE.json" or path.is_symlink()):
            raise ValueError(f"Unsafe tracked archive member: {name}")
        contents[name] = path.read_bytes()
    provenance = {"git_sha": commit, "files": {name: hashlib.sha256(data).hexdigest()
                                              for name, data in contents.items()}}
    contents["SOURCE.json"] = (json.dumps(provenance, indent=2, ensure_ascii=False)+"\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw:
        with tarfile.open(fileobj=raw, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
            for name, data in sorted(contents.items()):
                entry = tarfile.TarInfo(name)
                entry.size, entry.mode, entry.mtime = len(data), 0o644, 0
                archive.addfile(entry, io.BytesIO(data))
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist/costrank-v01.tar.gz")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(package(root, Path(args.output)))


if __name__ == "__main__":
    main()
