"""Replay the frozen ScanNet JPEG decoder/encoder without changing system Pillow."""

import io
import re
import subprocess
from pathlib import Path

from PIL import Image

from ..errors import ValidationError


def reencode_scannet(payload: bytes, tools: Path, size: tuple[int, int]) -> bytes:
    def run(name: str, arguments: list[str], data: bytes) -> bytes:
        binary = tools / name
        if not binary.is_file():
            raise ValidationError(f"missing {binary}; run scripts/install_scannet_jpeg.py")
        result = subprocess.run([str(binary), "-verbose", *arguments], input=data, capture_output=True)
        diagnostic = result.stderr.decode("utf-8", errors="replace")
        if result.returncode or not re.search(r"version\s+9e\b", diagnostic):
            raise ValidationError(f"{name} must be IJG 9e; exit={result.returncode}; {diagnostic[:400]}")
        return result.stdout

    ppm = run("djpeg", ["-pnm"], payload)
    with Image.open(io.BytesIO(ppm)) as decoded:
        if decoded.size != size or decoded.mode != "RGB":
            raise ValidationError("ScanNet sensor RGB dimensions/mode differ from native header")
    return run("cjpeg", ["-quality", "75"], ppm)
