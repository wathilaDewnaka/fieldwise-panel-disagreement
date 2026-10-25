"""One-time setup: tell pylidc where the LIDC-IDRI DICOM tree is.

pylidc reads a config file from the home directory, which is outside the repo
and therefore outside version control. This script writes it from
`configs/paths.yaml` so the DICOM location is stated in exactly one place.

Usage:
    python -m backend.domains.lidc.setup_pylidc            # uses configs/paths.yaml
    python -m backend.domains.lidc.setup_pylidc /some/dir  # explicit override
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from backend.config import paths, require


def config_path() -> Path:
    """Where pylidc looks: `pylidc.conf` on Windows, `.pylidcrc` elsewhere."""
    home = Path.home()
    return home / ("pylidc.conf" if os.name == "nt" else ".pylidcrc")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create the pylidc configuration file.")
    ap.add_argument("dicom_dir", type=Path, nargs="?", default=None,
                    help="LIDC-IDRI DICOM directory (default: lidc_raw in configs/paths.yaml)")
    args = ap.parse_args()

    dicom_dir = (args.dicom_dir.expanduser().resolve() if args.dicom_dir
                 else paths().lidc_raw)
    require(dicom_dir, "LIDC-IDRI DICOM directory")

    # No leading whitespace: configparser reads an indented line as a
    # continuation of the previous value, which silently corrupts `path`.
    out = config_path()
    out.write_text(f"[dicom]\npath = {dicom_dir}\nwarn = True\n", encoding="utf-8")

    print(f"Wrote pylidc config to: {out}")
    print(f"DICOM path: {dicom_dir}")


if __name__ == "__main__":
    main()
