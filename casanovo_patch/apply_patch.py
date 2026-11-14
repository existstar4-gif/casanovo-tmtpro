"""Install the Casanovo-TMT source changes into an installed casanovo 5.2.0.

    pip install casanovo==5.2.0
    python casanovo_patch/apply_patch.py

The three modified files in casanovo_patch/modified/ replace their counterparts
in the installed package. The installed version must be 5.2.0, and the
SHA-256 of each file is checked after copying (SHA256SUMS.txt).
"""
import hashlib
import importlib.metadata
import importlib.util
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
FILES = ["denovo/model.py", "denovo/model_runner.py", "data/ms_io.py"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    version = importlib.metadata.version("casanovo")
    if version != "5.2.0":
        sys.exit(f"casanovo {version} is installed; this patch is for 5.2.0")
    pkg = pathlib.Path(importlib.util.find_spec("casanovo").origin).parent
    expected = {}
    for line in (HERE / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.lstrip("*").removeprefix("casanovo/")] = digest
    for rel in FILES:
        dst = pkg / rel
        shutil.copyfile(HERE / "modified" / "casanovo" / rel, dst)
        if sha256(dst) != expected[rel]:
            sys.exit(f"checksum mismatch after copying {rel}")
        print(f"patched {dst}")


if __name__ == "__main__":
    main()
