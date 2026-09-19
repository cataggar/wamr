"""Shared release archive handling for the Python wheel builders."""

import io
import re
import stat
import tarfile
import zipfile


def _normalize_member(name: str, windows: bool) -> tuple[str, bool]:
    if not name or "\0" in name:
        raise ValueError(f"invalid archive member: {name!r}")
    normalized = name.replace("\\", "/") if windows else name
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"absolute archive member: {name!r}")

    directory = normalized.endswith("/")
    parts = []
    for part in normalized.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(f"traversal archive member: {name!r}")
        parts.append(part)
    normalized = "/".join(parts)
    if not normalized or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"invalid archive member: {name!r}")
    return normalized, directory


def read_archive_files(data: bytes, platform_key: str) -> dict[str, bytes]:
    """Return normalized regular files from a release archive after validation."""
    files: dict[str, bytes] = {}
    seen: set[str] = set()

    def add(name: str, directory: bool, regular: bool, read) -> None:
        normalized, trailing_directory = _normalize_member(
            name, platform_key.startswith("windows-")
        )
        if normalized in seen:
            raise ValueError(f"duplicate archive member: {normalized}")
        seen.add(normalized)
        if trailing_directory and not directory:
            raise ValueError(f"non-directory archive member has trailing separator: {name!r}")
        if directory:
            return
        if not regular:
            raise ValueError(f"non-file archive member: {name!r}")
        files[normalized] = read()

    if platform_key.startswith("windows-"):
        with zipfile.ZipFile(io.BytesIO(data)) as package:
            for member in package.infolist():
                mode = member.external_attr >> 16
                kind = stat.S_IFMT(mode)
                directory = member.is_dir() or kind == stat.S_IFDIR or (
                    member.create_system == 0 and bool(member.external_attr & 0x10)
                )
                add(
                    member.filename,
                    directory and kind in (0, stat.S_IFDIR),
                    not directory and kind in (0, stat.S_IFREG),
                    lambda member=member: package.read(member),
                )
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as package:
            for member in package.getmembers():
                add(
                    member.name,
                    member.isdir(),
                    member.isfile(),
                    lambda member=member: package.extractfile(member).read(),
                )
    return files
