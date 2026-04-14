#!/usr/bin/env python3
"""Download wamr release assets and repackage them as Python wheels."""

# /// script
# requires-python = ">=3.12"
# dependencies = ["requests"]
# ///

import hashlib
import io
import stat
import sys
import tarfile
import zipfile
from base64 import urlsafe_b64encode
from pathlib import Path

import requests  # type: ignore[import-untyped]

IMPORT_NAME = "wamr_cli"
DIST_NAME = "wamr_bin"
WAMR_REPO = "cataggar/wamr"

TOOLS = [
    "wamr",
]

PLATFORMS = {
    "linux-x64": {
        "tag": "manylinux_2_17_x86_64.manylinux2014_x86_64",
        "ext": "",
    },
    "linux-arm64": {
        "tag": "manylinux_2_17_aarch64.manylinux2014_aarch64",
        "ext": "",
    },
    "linux-musl-x64": {
        "tag": "musllinux_1_1_x86_64",
        "ext": "",
    },
    "linux-musl-arm64": {
        "tag": "musllinux_1_1_aarch64",
        "ext": "",
    },
    "macos-arm64": {
        "tag": "macosx_11_0_arm64",
        "ext": "",
    },
    "macos-x64": {
        "tag": "macosx_10_9_x86_64",
        "ext": "",
    },
    "windows-x64": {
        "tag": "win_amd64",
        "ext": ".exe",
    },
    "windows-arm64": {
        "tag": "win_arm64",
        "ext": ".exe",
    },
}

_EXEC_ATTR = (
    stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
) << 16
_FILE_ATTR = (stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH) << 16


def sha256_digest(data: bytes) -> str:
    """Return url-safe base64 sha256 digest (no padding)."""
    return urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def download_asset(release_version: str, platform_key: str) -> bytes:
    """Download a wamr release asset."""
    asset_name = f"wamr-{release_version}-{platform_key}.tar.gz"
    url = f"https://github.com/{WAMR_REPO}/releases/download/v{release_version}/{asset_name}"
    print(f"  Downloading {asset_name} ...")
    resp = requests.get(url, allow_redirects=True, timeout=300)
    resp.raise_for_status()
    return resp.content


def build_wheel(
    version: str, platform_key: str, platform_tag: str, ext: str,
    dist_dir: Path, release_version: str | None = None,
) -> Path:
    """Build a single platform wheel with native binaries in data/scripts/."""
    data = download_asset(release_version or version, platform_key)

    # Extract tool binaries from the tarball (exclude wamrc — distributed separately)
    tool_binaries: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile() or "/bin/" not in m.name:
                continue
            basename = m.name.rsplit("/", 1)[-1]
            if basename.startswith("wamrc"):
                continue
            for tool in TOOLS:
                if basename == f"{tool}{ext}":
                    f = tf.extractfile(m)
                    if f is not None:
                        tool_binaries[basename] = f.read()
                    break

    if not tool_binaries:
        raise RuntimeError(f"No tool binaries found in archive for {platform_key}")

    data_scripts_dir = f"{DIST_NAME}-{version}.data/scripts"
    dist_info_dir = f"{DIST_NAME}-{version}.dist-info"

    entries: list[tuple[str, bytes, bool]] = []

    # Python package with find helpers (for programmatic use)
    init_py = Path(__file__).resolve().parent.parent / "python" / IMPORT_NAME / "__init__.py"
    entries.append((f"{IMPORT_NAME}/__init__.py", init_py.read_bytes(), False))

    # Native binaries go in data/scripts/ — pip copies them directly to bin/Scripts
    for name, binary_data in sorted(tool_binaries.items()):
        entries.append((f"{data_scripts_dir}/{name}", binary_data, True))

    readme_path = Path(__file__).resolve().parent.parent / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    metadata = (
        f"Metadata-Version: 2.4\n"
        f"Name: wamr-bin\n"
        f"Version: {version}\n"
        f"Summary: WebAssembly Micro Runtime — native CLI tools\n"
        f"Home-page: https://github.com/cataggar/wamr\n"
        f"License: Apache-2.0\n"
        f"Requires-Python: >=3.9\n"
        f"Description-Content-Type: text/markdown\n"
        f"\n"
        f"{readme_text}"
    )
    entries.append((f"{dist_info_dir}/METADATA", metadata.encode(), False))

    wheel_meta = (
        f"Wheel-Version: 1.0\n"
        f"Generator: build_wheels.py\n"
        f"Root-Is-Purelib: false\n"
        f"Tag: py3-none-{platform_tag}\n"
    )
    entries.append((f"{dist_info_dir}/WHEEL", wheel_meta.encode(), False))

    # No entry_points.txt — binaries are installed directly via data/scripts

    # Build RECORD
    records: list[str] = []
    for arcname, file_data, _ in entries:
        digest = sha256_digest(file_data)
        records.append(f"{arcname},sha256={digest},{len(file_data)}")
    records.append(f"{dist_info_dir}/RECORD,,")
    record_data = ("\n".join(records) + "\n").encode()
    entries.append((f"{dist_info_dir}/RECORD", record_data, False))

    # Write wheel zip
    wheel_name = f"{DIST_NAME}-{version}-py3-none-{platform_tag}.whl"
    wheel_path = dist_dir / wheel_name
    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as whl:
        for arcname, file_data, executable in entries:
            zi = zipfile.ZipInfo(arcname)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = _EXEC_ATTR if executable else _FILE_ATTR
            whl.writestr(zi, file_data)

    print(f"  Built {wheel_name} ({wheel_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return wheel_path


def to_pep440(version: str) -> str:
    """Convert a semver-style version to PEP 440. e.g. 0.1.0-dev.1 -> 0.1.0.dev1"""
    import re

    m = re.match(r"^(\d+\.\d+\.\d+)-dev\.(\d+)$", version)
    if m:
        return f"{m.group(1)}.dev{m.group(2)}"
    return version


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <version>")
        print(f"Example: {sys.argv[0]} 0.1.0")
        sys.exit(1)

    raw_version = sys.argv[1]
    version = to_pep440(raw_version)
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)

    print(f"Building wheels for wamr v{raw_version} (PyPI: {version})\n")

    wheels: list[Path] = []
    for platform_key, info in PLATFORMS.items():
        print(f"[{platform_key}]")
        wheel = build_wheel(
            version, platform_key, info["tag"], info["ext"],
            dist_dir, raw_version,
        )
        wheels.append(wheel)
        print()

    print(f"Done! {len(wheels)} wheels in {dist_dir}/")
    for w in wheels:
        print(f"  {w.name}")


if __name__ == "__main__":
    main()
