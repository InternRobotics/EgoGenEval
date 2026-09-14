"""Install checksum-pinned IJG tools into an isolated, reusable cache."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from ..errors import ValidationError
from ..io import read_json, sha256_file, write_json

URL = "https://www.ijg.org/files/jpegsrc.v9e.tar.gz"
SHA256 = "4077d6a6a75aeb01884f708919d25934c93305e49f7e3f36db9129320e6f4f3d"


def default_cache() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "egogeneval"


def validate_jpeg_tools(directory: Path, *, cached: bool = False) -> Path:
    directory = directory.expanduser().resolve()
    sample = b"P6\n1 1\n255\n\x00\x00\x00"
    for name in ("cjpeg", "djpeg"):
        binary = directory / name
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValidationError(f"missing executable IJG 9e tool: {binary}")
        # IJG 9e has no -version switch. A tiny encode/decode with -verbose
        # checks both the version banner and that the cached tools execute.
        result = subprocess.run([str(binary), "-verbose"], input=sample, capture_output=True)
        banner = f"Independent JPEG Group's {name.upper()}, version 9e".encode()
        if result.returncode or banner not in result.stderr:
            raise ValidationError(f"expected working IJG 9e tools, got a different JPEG implementation: {binary}")
        sample = result.stdout
    if cached:
        identity_path = directory.parent / "build_identity.json"
        try:
            identity = read_json(identity_path)
        except (OSError, ValueError) as error:
            raise ValidationError(f"invalid cached IJG installation: {identity_path}") from error
        if identity.get("source_sha256") != SHA256 or any(
            identity.get("binary_sha256", {}).get(name) != sha256_file(directory / name)
            for name in ("cjpeg", "djpeg")
        ):
            raise ValidationError(f"modified cached IJG tools: {directory}; choose a clean --cache-dir")
    return directory


def ensure_jpeg_tools(cache: Path, *, directory: Path | None = None, archive: Path | None = None) -> Path:
    """Reuse a verified installation or build official IJG 9e without root access."""
    if directory is not None:
        return validate_jpeg_tools(directory)
    return install_jpeg_tools(cache.expanduser().resolve() / "ijg-9e", archive=archive)


def install_jpeg_tools(prefix: Path, *, archive: Path | None = None) -> Path:
    """Build at an explicit prefix; also used by the standalone installer."""
    prefix = prefix.expanduser().resolve()
    cache = prefix.parent
    if prefix.exists():
        return validate_jpeg_tools(prefix / "bin", cached=True)
    if shutil.which("make") is None or not (shutil.which("cc") or shutil.which("gcc")):
        raise ValidationError("ScanNet preparation needs a C compiler and make to install IJG 9e; "
                              "install them or pass --scannet-jpeg-tools /path/to/ijg-9e/bin")
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ijg-9e-", dir=cache) as temporary:
        work = Path(temporary)
        try:
            if archive is not None:
                content = archive.expanduser().read_bytes()
            else:
                with urlopen(URL, timeout=60) as response:
                    content = response.read(2 * 1024 * 1024)
        except (OSError, URLError) as error:
            raise ValidationError(f"could not download IJG 9e: {error}; "
                                  "use --jpeg-archive with the official jpegsrc.v9e.tar.gz for offline builds") from error
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValidationError("IJG source archive SHA-256 mismatch")
        archive_path = work / "jpegsrc.v9e.tar.gz"
        archive_path.write_bytes(content)
        with tarfile.open(archive_path) as bundle:
            for member in bundle.getmembers():
                target = (work / member.name).resolve()
                if not target.is_relative_to(work) or not (member.isfile() or member.isdir()):
                    raise ValidationError(f"unexpected IJG archive member: {member.name}")
            if hasattr(tarfile, "data_filter"):
                bundle.extractall(work, filter="data")
            else:
                bundle.extractall(work)
        install = work / "installed"
        log_path = work / "build.log"
        try:
            with log_path.open("w") as log:
                for command in (
                    ["./configure", "--disable-shared", "--enable-static", f"--prefix={install}"],
                    ["make", "-j4"], ["make", "install"],
                ):
                    subprocess.run(command, cwd=work / "jpeg-9e", stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            # Preserve only a diagnostic on failure, never a partially usable installation.
            failed_log = cache / "ijg-9e-build-failed.log"
            shutil.copyfile(log_path, failed_log)
            raise ValidationError(f"IJG build failed; inspect {failed_log}") from error
        validate_jpeg_tools(install / "bin")
        write_json(install / "build_identity.json", {
            "version": "9e", "source_url": URL, "source_sha256": SHA256,
            "recipe": "djpeg -pnm | cjpeg -quality 75",
            "binary_sha256": {name: sha256_file(install / "bin" / name) for name in ("cjpeg", "djpeg")},
        })
        try:
            install.rename(prefix)
        except OSError:
            # Another preparation process may have completed the same cached build.
            if not prefix.is_dir():
                raise
            validate_jpeg_tools(prefix / "bin", cached=True)
    return prefix / "bin"
