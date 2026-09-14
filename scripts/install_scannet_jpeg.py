#!/usr/bin/env python3
"""Install isolated IJG 9e tools (normally automatic in prepare-benchmark)."""

import argparse
import sys
from pathlib import Path

# Preserve direct use from a checkout before its editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egogeneval.errors import ValidationError  # noqa: E402
from egogeneval.preparation.jpeg_tools import install_jpeg_tools  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="pre-downloaded official jpegsrc.v9e.tar.gz")
    args = parser.parse_args()
    prefix = args.prefix.expanduser().resolve()
    if prefix.exists():
        parser.error("prefix already exists; choose a new isolated directory")
    try:
        binary_dir = install_jpeg_tools(prefix, archive=args.archive)
    except ValidationError as error:
        parser.error(str(error))
    print(f"IJG 9e installed. Pass --scannet-jpeg-tools {binary_dir} to prepare-benchmark.")


if __name__ == "__main__":
    main()
