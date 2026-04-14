"""wamr-bin — WebAssembly Micro Runtime CLI tools.

Provides a helper to locate the wamr binary installed via
the data/scripts/ wheel layout. The binary is placed directly
in the scripts directory by pip and does not require Python at runtime.
"""

from __future__ import annotations

import os
import sys
import sysconfig


def find_wamr_bin() -> str:
    """Return the path to the wamr binary.

    Searches the scripts directories where pip installs data/scripts/ files.
    """
    ext = ".exe" if sys.platform == "win32" else ""
    exe = f"wamr{ext}"

    targets = [
        sysconfig.get_path("scripts"),
        sysconfig.get_path("scripts", vars={"base": sys.base_prefix}),
    ]

    # User scheme
    if sys.version_info >= (3, 10):
        user_scheme = sysconfig.get_preferred_scheme("user")
    elif os.name == "nt":
        user_scheme = "nt_user"
    else:
        user_scheme = "posix_user"
    targets.append(sysconfig.get_path("scripts", scheme=user_scheme))

    seen: list[str] = []
    for target in targets:
        if not target or target in seen:
            continue
        seen.append(target)
        path = os.path.join(target, exe)
        if os.path.isfile(path):
            return path

    locations = "\n".join(f" - {t}" for t in seen)
    raise FileNotFoundError(
        f"Could not find {exe} in:\n{locations}\n"
    )
