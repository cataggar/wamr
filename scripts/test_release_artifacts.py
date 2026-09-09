#!/usr/bin/env python3

import copy
import hashlib
import io
import json
import runpy
import stat
import struct
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_artifacts as release


VERSION = "3.0.0-dev.14"
SHA = "a" * 40


def binary(system, arch):
    data = bytearray(160)
    if system == "linux":
        data[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<HH", data, 16, 2, {
            "x86_64": 62, "aarch64": 183, "riscv64": 243,
        }[arch])
    elif system == "macos":
        data[:4] = b"\xcf\xfa\xed\xfe"
        struct.pack_into("<I", data, 4, {
            "x86_64": 0x1000007, "aarch64": 0x100000C,
        }[arch])
        struct.pack_into("<I", data, 12, 2)
    else:
        data[:2] = b"MZ"
        struct.pack_into("<I", data, 60, 64)
        data[64:68] = b"PE\0\0"
        struct.pack_into("<H", data, 68, {
            "x86_64": 0x8664, "aarch64": 0xAA64,
        }[arch])
        struct.pack_into("<H", data, 88, 0x20B)
    return bytes(data)


class ReleaseArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.assets = self.root / "assets"
        self.repo.mkdir()
        self.assets.mkdir()
        for name, data in {
            "LICENSE": b"license\n", "README.md": b"readme\n",
            "build.zig": b"build\n", "tests/coldstart/noop.wasm": b"noop",
            "tests/regressions/1008-aot-magic-u32/printf-decimal.wasm": b"decimal",
        }.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def entries(self, name):
        system, arch = release.TARGETS[name]
        root = release.package_name(VERSION, name)
        ext = ".exe" if system == "windows" else ""
        return [
            (f"{root}/LICENSE", b"license\n", 0o644),
            (f"{root}/README.md", b"readme\n", 0o644),
            (f"{root}/bin/wamr{ext}", binary(system, arch), 0o755),
            (f"{root}/bin/wamrc{ext}", binary(system, arch), 0o755),
        ]

    def archive(self, name, entries=None):
        path = self.assets / release.archive_name(VERSION, name)
        entries = self.entries(name) if entries is None else entries
        if path.suffix == ".zip":
            with zipfile.ZipFile(path, "w") as package:
                for filename, data, mode in entries:
                    info = zipfile.ZipInfo(filename)
                    info.external_attr = (stat.S_IFREG | mode) << 16
                    package.writestr(info, data)
        else:
            with tarfile.open(path, "w:gz") as package:
                for filename, data, mode in entries:
                    info = tarfile.TarInfo(filename)
                    info.size = len(data)
                    info.mode = mode
                    package.addfile(info, io.BytesIO(data))
        return path

    def source_archive(self, source_sha=SHA):
        with tarfile.open(
            self.assets / f"wamr-{VERSION}.tar.xz", "w:xz",
            format=tarfile.PAX_FORMAT, pax_headers={"comment": source_sha},
        ) as package:
            for name in ("LICENSE", "README.md", "build.zig"):
                package.add(self.repo / name, arcname=f"wamr-{VERSION}/{name}")

    def report(self, name, archive):
        system, arch = release.TARGETS[name]
        signatures = [
            {"name": f"{tool}.exe", "status": "Valid", "signer": "AB",
             "timestamp_signer": "CD",
             "sha256": hashlib.sha256(binary(system, arch)).hexdigest()}
            for tool in ("wamr", "wamrc")
        ]
        host = (system, arch) if name in release.REQUIRED_NATIVE else ("windows", "x86_64")
        if system != "windows" and name not in release.REQUIRED_NATIVE:
            host = ("other", "other")
        with mock.patch.object(release, "host_target", return_value=host), \
             mock.patch.object(release, "windows_signatures", return_value=signatures), \
             mock.patch.object(release, "run", side_effect=lambda command, expected=None: expected or b"") as run:
            report = release.verify_package(
                archive, VERSION, name, SHA, self.root / name,
                name in release.REQUIRED_NATIVE, system == "windows", self.repo,
            )
        if name in release.REQUIRED_NATIVE:
            self.assertEqual(len(run.call_args_list), 7)
            for call in run.call_args_list:
                self.assertIn(str(self.root / name), call.args[0][0])
        return report

    def cohort(self):
        for name in release.TARGETS:
            archive = self.archive(name)
            report = self.report(name, archive)
            stem = release.package_name(VERSION, name)
            (self.assets / f"{stem}.verification.json").write_text(json.dumps(report))
            (self.assets / f"{stem}.sbom.spdx.json").write_text(json.dumps({
                "spdxVersion": "SPDX-2.3", "SPDXID": "SPDXRef-DOCUMENT",
                "documentNamespace": f"https://example.invalid/{name}",
            }))
        self.source_archive()

    def test_versions_and_pep440(self):
        self.assertEqual(release.release_version("v" + VERSION), VERSION)
        self.assertEqual(release.to_pep440(VERSION), "3.0.0.dev14")
        for version in ("3.0.0", "3.0.0-rc.1", "3.0.0+build.4"):
            self.assertEqual(release.release_version(version), version)
            self.assertEqual(release.to_pep440(version), version)
        for version in ("", "main", "../3.0.0", "3.0.0\n", "3.0.0;echo",
                        "03.0.0", "3.0.0-dev.01", "3.0.0/evil"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                release.release_version(version)

    def test_wheel_builders_share_version_conversion(self):
        with mock.patch.dict(sys.modules, {"requests": mock.Mock()}):
            for name in ("build_wheels.py", "build_wamrc_wheels.py"):
                namespace = runpy.run_path(str(release.REPO / "scripts" / name))
                self.assertIs(namespace["to_pep440"], release.to_pep440)
                self.assertEqual(namespace["to_pep440"](VERSION), "3.0.0.dev14")

    def test_all_published_binary_formats(self):
        for name in release.TARGETS:
            with self.subTest(name=name):
                files = release.inspect_package(self.archive(name), VERSION, name, self.repo)
                self.assertEqual(len(files), 4)

    def test_wrong_machine_empty_and_nonexecutable_tools(self):
        for mode, data in ((0o644, binary("linux", "x86_64")),
                           (0o755, b""), (0o755, binary("linux", "aarch64"))):
            entries = self.entries("linux-x64")
            entries[2] = (entries[2][0], data, mode)
            with self.subTest(mode=mode, data=data), self.assertRaises(ValueError):
                release.inspect_package(self.archive("linux-x64", entries),
                                        VERSION, "linux-x64", self.repo)

    def test_missing_duplicate_unexpected_and_traversal_members(self):
        entries = self.entries("linux-x64")
        for malformed in (entries[:-1], entries + [entries[0]],
                          entries + [("../outside", b"bad", 0o644)],
                          entries + [("/absolute", b"bad", 0o644)],
                          entries + [("other/root", b"bad", 0o644)]):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                release.inspect_package(self.archive("linux-x64", malformed),
                                        VERSION, "linux-x64", self.repo)

    def test_links_are_rejected(self):
        archive = self.archive("linux-x64")
        with tarfile.open(archive, "w:gz") as package:
            info = tarfile.TarInfo(self.entries("linux-x64")[2][0])
            info.type = tarfile.SYMTYPE
            info.linkname = "/outside"
            package.addfile(info)
        with self.assertRaisesRegex(ValueError, "unexpected archive member"):
            release.inspect_package(archive, VERSION, "linux-x64", self.repo)
        archive = self.archive("windows-x64")
        with zipfile.ZipFile(archive, "w") as package:
            info = zipfile.ZipInfo(self.entries("windows-x64")[2][0])
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            package.writestr(info, "/outside")
        with self.assertRaisesRegex(ValueError, "unexpected archive member"):
            release.inspect_package(archive, VERSION, "windows-x64", self.repo)

    def test_documentation_only_allows_line_ending_variation(self):
        entries = self.entries("windows-x64")
        entries[0] = (entries[0][0], b"license\r\n", 0o644)
        release.inspect_package(self.archive("windows-x64", entries), VERSION,
                                "windows-x64", self.repo)
        entries[0] = (entries[0][0], b"different license\n", 0o644)
        with self.assertRaisesRegex(ValueError, "documentation"):
            release.inspect_package(self.archive("windows-x64", entries), VERSION,
                                    "windows-x64", self.repo)

    def test_windows_zip_separators_and_directories_are_host_independent(self):
        name = "windows-x64"
        archive = self.assets / release.archive_name(VERSION, name)
        root = release.package_name(VERSION, name)
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                with zipfile.ZipFile(archive, "w") as package:
                    for filename in (root + "\\", root + "\\bin\\"):
                        info = zipfile.ZipInfo(filename)
                        info.filename = filename
                        info.create_system = 0
                        info.external_attr = 0x10
                        package.writestr(info, b"")
                    for filename, data, _ in self.entries(name):
                        raw = filename.replace("/", "\\", 1) if mixed else filename.replace("/", "\\")
                        info = zipfile.ZipInfo(raw)
                        info.filename = raw
                        info.create_system = 0
                        info.external_attr = 0x20
                        package.writestr(info, data)
                files = release.inspect_package(archive, VERSION, name, self.repo)
                self.assertEqual(set(files), {entry[0] for entry in self.entries(name)})

    def test_windows_zip_normalization_rejects_collisions_and_traversal(self):
        name = "windows-x64"
        root = release.package_name(VERSION, name)
        for filename in (root + "\\LICENSE", "..\\outside", root + "\\..\\outside",
                         "\\\\server\\share", "C:\\outside"):
            with self.subTest(filename=filename):
                archive = self.archive(name)
                with zipfile.ZipFile(archive, "a") as package:
                    info = zipfile.ZipInfo(filename)
                    info.filename = filename
                    info.create_system = 0
                    info.external_attr = 0x20
                    package.writestr(info, b"bad")
                with self.assertRaisesRegex(ValueError, "duplicate|unexpected"):
                    release.inspect_package(archive, VERSION, name, self.repo)

    def test_windows_zip_directory_symlink_is_rejected(self):
        name = "windows-x64"
        archive = self.archive(name)
        filename = release.package_name(VERSION, name) + "\\bin\\"
        with zipfile.ZipFile(archive, "a") as package:
            info = zipfile.ZipInfo(filename)
            info.filename = filename
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            package.writestr(info, "/outside")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            release.inspect_package(archive, VERSION, name, self.repo)

    def test_required_native_smoke_cannot_be_silently_skipped(self):
        archive = self.archive("linux-arm64")
        with mock.patch.object(release, "host_target", return_value=("linux", "x86_64")), \
             self.assertRaisesRegex(ValueError, "required native smoke"):
            release.verify_package(archive, VERSION, "linux-arm64", SHA,
                                   self.root / "extract", True, False, self.repo)

    def test_existing_extraction_directory_is_not_overwritten(self):
        archive = self.archive("linux-x64")
        extracted = self.root / "extract" / release.package_name(VERSION, "linux-x64")
        extracted.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            release.verify_package(archive, VERSION, "linux-x64", SHA,
                                   self.root / "extract", False, False, self.repo)

    def test_runtime_mismatches_fail(self):
        for code, stdout, stderr in ((1, b"", b""), (0, b"wrong", b""),
                                    (0, b"expected", b"warning")):
            result = mock.Mock(returncode=code, stdout=stdout, stderr=stderr)
            with mock.patch.object(release.subprocess, "run", return_value=result), \
                 self.assertRaisesRegex(ValueError, "command failed"):
                release.run(["wamr", "run", "guest.cwasm"], b"expected")

    def test_complete_inventory_hashes_every_retained_asset(self):
        self.cohort()
        manifest = release.release_inventory(self.assets, VERSION, SHA, self.repo)
        self.assertEqual(set(manifest["platforms"]), set(release.TARGETS))
        self.assertEqual(manifest["pep440"], "3.0.0.dev14")
        for line in (self.assets / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ")
            self.assertEqual(digest, release.sha256(self.assets / name))
        self.assertIn("release-manifest.json", (self.assets / "SHA256SUMS").read_text())
        self.assertIn(SHA, (self.assets / "RELEASE_NOTES.md").read_text())

    def test_inventory_rejects_stale_and_incomplete_reports(self):
        self.cohort()
        path = self.assets / f"wamr-{VERSION}-linux-x64.verification.json"
        original = json.loads(path.read_text())
        for field, value in (
            ("source_sha", "b" * 40), ("version", "3.0.0"),
            ("archive_sha256", "0" * 64), ("files", {}),
            ("host", ["linux", "aarch64"]),
            ("native_smoke", {"status": "not-run", "reason": "host-target-mismatch"}),
            ("native_smoke", {"status": "passed", "cases": []}),
        ):
            report = copy.deepcopy(original)
            report[field] = value
            path.write_text(json.dumps(report))
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                release.release_inventory(self.assets, VERSION, SHA, self.repo)
        path.write_text(json.dumps(original))
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            release.release_inventory(self.assets, VERSION, SHA, self.repo)

    def test_inventory_requires_timestamped_signatures_for_both_windows_tools(self):
        self.cohort()
        path = self.assets / f"wamr-{VERSION}-windows-arm64.verification.json"
        original = json.loads(path.read_text())
        for field, value in (("status", "NotSigned"), ("timestamp_signer", ""),
                             ("sha256", "0" * 64)):
            report = copy.deepcopy(original)
            report["signatures"][0][field] = value
            path.write_text(json.dumps(report))
            with self.subTest(field=field), self.assertRaises(ValueError):
                release.release_inventory(self.assets, VERSION, SHA, self.repo)
        original["signatures"] = []
        path.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "missing Windows signatures"):
            release.release_inventory(self.assets, VERSION, SHA, self.repo)

    def test_inventory_requires_spdx_and_exact_source_commit(self):
        self.cohort()
        self.source_archive("b" * 40)
        with self.assertRaisesRegex(ValueError, "source archive commit"):
            release.release_inventory(self.assets, VERSION, SHA, self.repo)
        self.source_archive()
        (self.assets / f"wamr-{VERSION}-linux-x64.sbom.spdx.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "invalid SPDX"):
            release.release_inventory(self.assets, VERSION, SHA, self.repo)

    def test_inventory_rejects_unverified_extra_assets(self):
        self.cohort()
        (self.assets / "wamr-unverified.tar.gz").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "unexpected release assets"):
            release.release_inventory(self.assets, VERSION, SHA, self.repo)

    def test_inventory_requires_actual_cli_version_output(self):
        self.cohort()
        path = self.assets / f"wamr-{VERSION}-linux-x64.verification.json"
        report = json.loads(path.read_text())
        report["native_smoke"]["versions"]["wamrc version"] = "wamrc dev\n"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "CLI versions"):
            release.release_inventory(self.assets, VERSION, SHA, self.repo)

    def test_workflow_publication_boundary_and_matrix(self):
        workflow = (release.REPO / ".github/workflows/release.yml").read_text()
        self.assertIn("workflow_dispatch:", workflow)
        publish = workflow.split("- name: Create release\n", 1)[1]
        self.assertIn(
            "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')",
            publish,
        )
        self.assertIn("make_latest:", publish)
        self.assertIn("upload-release-assets: false", workflow)
        for name in release.TARGETS:
            self.assertIn(f"name: {name}\n", workflow)
        for filename in ("pypi.yml", "pypi-wamrc.yml"):
            follower = (release.REPO / ".github/workflows" / filename).read_text()
            build = follower.split("  publish:", 1)[0]
            self.assertIn("github.event.workflow_run.event == 'push'", build)
            self.assertIn("startsWith(github.event.workflow_run.head_branch, 'v')", build)


if __name__ == "__main__":
    unittest.main()
