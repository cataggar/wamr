#!/usr/bin/env python3
"""Verify release archives and produce a source-bound release inventory."""

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import struct
import subprocess
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGETS = {
    "linux-x64": ("linux", "x86_64"),
    "linux-arm64": ("linux", "aarch64"),
    "macos-arm64": ("macos", "aarch64"),
    "windows-x64": ("windows", "x86_64"),
    "macos-x64": ("macos", "x86_64"),
    "linux-musl-x64": ("linux", "x86_64"),
    "linux-musl-arm64": ("linux", "aarch64"),
    "linux-riscv64": ("linux", "riscv64"),
    "windows-arm64": ("windows", "aarch64"),
}
REQUIRED_NATIVE = {"linux-x64", "linux-arm64", "macos-arm64", "windows-x64"}
DECIMAL_OUTPUT = (
    b'{"checksum":13856768990818897060,"cases":'
    b'[[1385676899,138567689,9],[2147483648,214748364,8],'
    b'[3000000008,300000000,8]]}\n'
)


def to_pep440(version: str) -> str:
    match = re.fullmatch(r"(\d+\.\d+\.\d+)-dev\.(\d+)", version)
    return f"{match[1]}.dev{match[2]}" if match else version


def release_version(raw: str) -> str:
    version = raw.removeprefix("v")
    number = r"(?:0|[1-9][0-9]*)"
    if not re.fullmatch(
        rf"{number}\.{number}\.{number}"
        r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
        r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?",
        version,
    ):
        raise ValueError(f"invalid release version: {raw!r}")
    prerelease = version.partition("+")[0].partition("-")[2]
    if any(part.isdigit() and len(part) > 1 and part.startswith("0")
           for part in prerelease.split(".")):
        raise ValueError(f"noncanonical prerelease version: {raw!r}")
    return version


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_name(version: str, name: str) -> str:
    return f"wamr-{version}-{name}"


def archive_name(version: str, name: str) -> str:
    suffix = ".zip" if TARGETS[name][0] == "windows" else ".tar.gz"
    return package_name(version, name) + suffix


def check_binary(data: bytes, system: str, arch: str) -> None:
    if system == "linux":
        valid = (
            len(data) >= 64 and data[:6] == b"\x7fELF\x02\x01"
            and struct.unpack_from("<H", data, 16)[0] in (2, 3)
            and struct.unpack_from("<H", data, 18)[0]
            == {"x86_64": 62, "aarch64": 183, "riscv64": 243}[arch]
        )
    elif system == "macos":
        valid = (
            len(data) >= 32 and data[:4] == b"\xcf\xfa\xed\xfe"
            and struct.unpack_from("<I", data, 4)[0]
            == {"x86_64": 0x1000007, "aarch64": 0x100000C}[arch]
            and struct.unpack_from("<I", data, 12)[0] == 2
        )
    else:
        offset = struct.unpack_from("<I", data, 60)[0] if len(data) >= 64 else 0
        valid = (
            offset >= 64 and len(data) >= offset + 26 and data[:2] == b"MZ"
            and data[offset:offset + 4] == b"PE\0\0"
            and struct.unpack_from("<H", data, offset + 4)[0]
            == {"x86_64": 0x8664, "aarch64": 0xAA64}[arch]
            and struct.unpack_from("<H", data, offset + 24)[0] == 0x20B
        )
    if not valid:
        raise ValueError(f"expected a {system}/{arch} executable")


def inspect_package(archive: Path, version: str, name: str, repo: Path = REPO):
    system, arch = TARGETS[name]
    root = package_name(version, name)
    ext = ".exe" if system == "windows" else ""
    expected = {f"{root}/LICENSE", f"{root}/README.md"}
    expected.update(f"{root}/bin/{tool}{ext}" for tool in ("wamr", "wamrc"))
    directories = {root, f"{root}/bin"}
    files = {}
    seen = set()

    def add(member_name, is_dir, is_file, mode, read):
        key = member_name.rstrip("/") if is_dir else member_name
        if key in seen:
            raise ValueError(f"duplicate archive member: {key}")
        seen.add(key)
        if is_dir and key in directories:
            return
        if not is_file or key not in expected:
            raise ValueError(f"unexpected archive member: {key}")
        data = read()
        if not data:
            raise ValueError(f"empty archive member: {key}")
        if "/bin/" in key:
            check_binary(data, system, arch)
            if system != "windows" and not mode & 0o111:
                raise ValueError(f"non-executable archive member: {key}")
        elif data.replace(b"\r\n", b"\n") != (
            repo / key.rsplit("/", 1)[1]
        ).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"packaged documentation differs from source: {key}")
        files[key] = (data, mode)

    if system == "windows":
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                mode = member.external_attr >> 16
                add(member.filename, member.is_dir(),
                    not member.is_dir() and stat.S_IFMT(mode) in (0, stat.S_IFREG),
                    mode, lambda member=member: package.read(member))
    else:
        with tarfile.open(archive, "r:gz") as package:
            for member in package.getmembers():
                add(member.name, member.isdir(), member.isfile(), member.mode,
                    lambda member=member: package.extractfile(member).read())
    if set(files) != expected:
        raise ValueError(f"missing archive members: {sorted(expected - set(files))}")
    return files


def run(command: list[str], expected: bytes | None = None) -> bytes:
    result = subprocess.run(command, capture_output=True, timeout=120)
    if result.returncode or (
        expected is not None and (result.stdout != expected or result.stderr)
    ):
        raise ValueError(
            f"command failed: {command!r}; exit={result.returncode}; "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}"
        )
    return result.stdout


def host_target() -> tuple[str, str]:
    system = {"Darwin": "macos", "Windows": "windows", "Linux": "linux"}
    machine = platform.machine().lower()
    return system.get(platform.system(), platform.system().lower()), {
        "amd64": "x86_64", "arm64": "aarch64",
    }.get(machine, machine)


def windows_signatures(bin_dir: Path):
    # Static PowerShell with paths supplied through the environment, not code.
    script = r"""
    $ErrorActionPreference = 'Stop'
    $result = @()
    foreach ($tool in @('wamr.exe', 'wamrc.exe')) {
      $path = Join-Path $env:RELEASE_SIGNATURE_BIN $tool
      $sig = Get-AuthenticodeSignature -LiteralPath $path
      if ($sig.Status -ne 'Valid' -or $null -eq $sig.TimeStamperCertificate) {
        throw "Invalid or untimestamped signature: $tool ($($sig.Status))"
      }
      $result += @{
        name = $tool
        sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
        status = $sig.Status.ToString()
        signer = $sig.SignerCertificate.Thumbprint
        timestamp_signer = $sig.TimeStamperCertificate.Thumbprint
      }
    }
    ConvertTo-Json -InputObject $result -Depth 4 -Compress
    """
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        env={**os.environ, "RELEASE_SIGNATURE_BIN": str(bin_dir)},
        capture_output=True, timeout=120,
    )
    if result.returncode:
        raise ValueError(f"signature verification failed: {result.stderr!r}")
    return json.loads(result.stdout)


def verify_package(archive: Path, version: str, name: str, source_sha: str,
                   work_dir: Path, require_smoke: bool, require_signature: bool,
                   repo: Path = REPO) -> dict:
    files = inspect_package(archive, version, name, repo)
    root = package_name(version, name)
    extracted = work_dir.resolve() / root
    if extracted.exists():
        raise ValueError(f"extraction directory already exists: {extracted}")
    for member, (data, mode) in files.items():
        path = work_dir.resolve() / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if TARGETS[name][0] != "windows":
            path.chmod(mode & 0o777)
    system, arch = TARGETS[name]
    host = host_target()
    native = host == (system, arch)
    if require_smoke and not native:
        raise ValueError(f"required native smoke cannot run: host={host}, target={name}")
    smoke = {"status": "not-run", "reason": "host-target-mismatch"}
    bin_dir = extracted / "bin"
    signatures = []
    if require_signature:
        if system != "windows" or host[0] != "windows":
            raise ValueError("Windows signature verification requires a Windows host/target")
        signatures = windows_signatures(bin_dir)
    if native:
        if arch not in ("x86_64", "aarch64"):
            raise ValueError(f"native AOT smoke unsupported on {arch}")
        ext = ".exe" if system == "windows" else ""
        wamr, wamrc = (str(bin_dir / f"{tool}{ext}") for tool in ("wamr", "wamrc"))
        versions = {}
        for tool, path in (("wamr", wamr), ("wamrc", wamrc)):
            versions[f"{tool} version"] = run(
                [path, "version"], f"{tool} {version}\n".encode()
            ).decode("utf-8")
        versions["wamr --version"] = run(
            [wamr, "--version"], f"wamr {version}\n".encode()
        ).decode("utf-8")
        cases = []
        for fixture, expected in (
            ("tests/coldstart/noop.wasm", b""),
            ("tests/regressions/1008-aot-magic-u32/printf-decimal.wasm", DECIMAL_OUTPUT),
        ):
            output = work_dir.resolve() / (Path(fixture).stem + ".cwasm")
            run([wamrc, "compile", str(repo / fixture), "--target", arch, "-o", str(output)])
            run([wamr, "run", str(output)], expected)
            cases.append({"fixture": fixture, "fixture_sha256": sha256(repo / fixture),
                          "stdout_sha256": hashlib.sha256(expected).hexdigest()})
        smoke = {"status": "passed", "versions": versions, "cases": cases}
    return {
        "schema_version": 1, "version": version, "pep440": to_pep440(version),
        "source_sha": source_sha, "platform": name, "host": list(host),
        "archive": archive.name, "archive_sha256": sha256(archive),
        "files": {key: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                  for key, (data, _) in sorted(files.items())},
        "native_smoke": smoke, "signatures": signatures,
    }


def release_inventory(directory: Path, version: str, source_sha: str,
                      repo: Path = REPO) -> dict:
    inventory = []
    platforms = {}
    expected_names = {f"wamr-{version}.tar.xz"}
    for name in TARGETS:
        stem = package_name(version, name)
        expected_names.update((archive_name(version, name), f"{stem}.verification.json",
                               f"{stem}.sbom.spdx.json"))
    generated = {"release-manifest.json", "SHA256SUMS", "RELEASE_NOTES.md"}
    unexpected = {path.name for path in directory.iterdir()} - expected_names - generated
    if unexpected:
        raise ValueError(f"unexpected release assets: {sorted(unexpected)}")
    for name in TARGETS:
        stem = package_name(version, name)
        archive = directory / archive_name(version, name)
        report_path = directory / f"{stem}.verification.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        files = inspect_package(archive, version, name, repo)
        expected_files = {
            key: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            for key, (data, _) in files.items()
        }
        expected = {
            "schema_version": 1, "version": version, "pep440": to_pep440(version),
            "source_sha": source_sha, "platform": name, "archive": archive.name,
            "archive_sha256": sha256(archive), "files": expected_files,
        }
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError(f"archive/report identity mismatch: {name}")
        smoke = report["native_smoke"]
        if smoke["status"] == "passed":
            if report["host"] != list(TARGETS[name]):
                raise ValueError(f"non-native execution claim: {name}")
            if smoke.get("versions") != {
                "wamr version": f"wamr {version}\n",
                "wamr --version": f"wamr {version}\n",
                "wamrc version": f"wamrc {version}\n",
            }:
                raise ValueError(f"missing or mismatched CLI versions: {name}")
            expected_cases = [
                {"fixture": fixture, "fixture_sha256": sha256(repo / fixture),
                 "stdout_sha256": hashlib.sha256(output).hexdigest()}
                for fixture, output in (
                    ("tests/coldstart/noop.wasm", b""),
                    ("tests/regressions/1008-aot-magic-u32/printf-decimal.wasm", DECIMAL_OUTPUT),
                )
            ]
            if smoke.get("cases") != expected_cases:
                raise ValueError(f"missing or mismatched native smoke evidence: {name}")
        elif (name in REQUIRED_NATIVE or smoke != {
            "status": "not-run", "reason": "host-target-mismatch",
        } or report["host"] == list(TARGETS[name])):
            raise ValueError(f"missing native execution: {name}")
        if TARGETS[name][0] == "windows":
            signatures = report["signatures"]
            if len(signatures) != 2 or {item["name"] for item in signatures} != {
                "wamr.exe", "wamrc.exe",
            }:
                raise ValueError(f"missing Windows signatures: {name}")
            for signature in signatures:
                binary = expected_files[f"{stem}/bin/{signature['name']}"]
                if (signature["status"] != "Valid" or not signature["signer"]
                    or not signature["timestamp_signer"]
                    or signature["sha256"] != binary["sha256"]):
                    raise ValueError(f"invalid Windows signature evidence: {name}")
        sbom = directory / f"{stem}.sbom.spdx.json"
        document = json.loads(sbom.read_text(encoding="utf-8"))
        if (document.get("spdxVersion") != "SPDX-2.3"
            or document.get("SPDXID") != "SPDXRef-DOCUMENT"
            or not document.get("documentNamespace")):
            raise ValueError(f"invalid SPDX document: {name}")
        inventory.extend((archive, report_path, sbom))
        platforms[name] = report
    source = directory / f"wamr-{version}.tar.xz"
    with tarfile.open(source, "r:xz") as package:
        if package.pax_headers.get("comment") != source_sha:
            raise ValueError("source archive commit differs from candidate")
        for filename in ("LICENSE", "README.md", "build.zig"):
            member = package.getmember(f"wamr-{version}/{filename}")
            if not member.isfile() or package.extractfile(member).read() != (repo / filename).read_bytes():
                raise ValueError(f"source archive content mismatch: {filename}")
    inventory.append(source)
    notes = directory / "RELEASE_NOTES.md"
    draft = repo / "docs" / "releases" / f"v{version}.md"
    text = draft.read_text(encoding="utf-8") if draft.is_file() else f"# wamr {version}\n"
    text += (
        f"\nSource: `{source_sha}`\n\n"
        f"Install: `ghr install cataggar/wamr@v{version}`\n\n"
        "See [INSTALL.md](https://github.com/cataggar/wamr/blob/"
        f"{source_sha}/INSTALL.md) for other installation options.\n"
    )
    notes.write_text(text, encoding="utf-8")
    inventory.append(notes)
    manifest = {
        "schema_version": 1, "version": version, "pep440": to_pep440(version),
        "source_sha": source_sha, "platforms": platforms,
        "workflow": {"event": os.environ.get("GITHUB_EVENT_NAME"),
                     "run_id": os.environ.get("GITHUB_RUN_ID"),
                     "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT")},
        "assets": [{"name": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
                   for path in sorted(inventory)],
    }
    manifest_path = directory / "release-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    inventory.append(manifest_path)
    (directory / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(inventory)),
        encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    version_parser = commands.add_parser("version")
    version_parser.add_argument("version")
    verify = commands.add_parser("verify")
    verify.add_argument("--platform", choices=TARGETS, required=True)
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--work-dir", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--require-smoke", action="store_true")
    verify.add_argument("--require-signature", action="store_true")
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--directory", type=Path, required=True)
    for command in (verify, manifest):
        command.add_argument("--version", required=True)
        command.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    version = release_version(args.version)
    if args.command == "version":
        print(f"version={version}\npep440={to_pep440(version)}")
        return
    actual_sha = run(["git", "-C", str(REPO), "rev-parse", "HEAD"]).decode().strip()
    if args.source_sha != actual_sha:
        raise ValueError(f"source SHA differs from checkout: {args.source_sha} != {actual_sha}")
    if args.command == "verify":
        report = verify_package(
            args.archive, version, args.platform, args.source_sha, args.work_dir,
            args.require_smoke, args.require_signature,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        release_inventory(args.directory, version, args.source_sha)


if __name__ == "__main__":
    main()
