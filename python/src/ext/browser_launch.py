"""Open authenticated local pages without putting credentials in process argv."""

from __future__ import annotations

import asyncio
import html
import os
import tempfile
from collections.abc import Callable
from pathlib import Path


async def open_private_browser(url: str, opener: Callable[[str], bool]) -> None:
    # Keep the file until shutdown: browser launchers may return before it is read.
    with tempfile.TemporaryDirectory(prefix="agent-rt-browser-") as directory:
        os.chmod(directory, 0o700)
        path = Path(directory) / "launch.html"
        escaped = html.escape(url, quote=True)
        with os.fdopen(
            os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
        ) as handle:
            handle.write(
                '<!doctype html><meta name="referrer" content="no-referrer">'
                f'<meta http-equiv="refresh" content="0;url={escaped}">'
                f'<a href="{escaped}">Open Agent RT</a>'
            )
        if not await asyncio.to_thread(opener, path.as_uri()):
            return
        await asyncio.Event().wait()
