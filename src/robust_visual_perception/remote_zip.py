"""Read members of a large remote zip file without downloading all of it.

``RangeFile`` is a seekable, read-only file object backed by HTTP range
requests, so ``zipfile.ZipFile`` can read the archive directory and single
members remotely. ``extract_prefix`` downloads the byte span that holds all
members under one path prefix in a single request, then extracts them.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

READ_AHEAD = 64 * 1024


class RangeFile:
    def __init__(
        self,
        url: str,
        size: int | None = None,
        opener: Callable[..., Any] = urllib.request.urlopen,
        timeout_s: float = 120.0,
    ) -> None:
        self.url = url
        self.opener = opener
        self.timeout_s = timeout_s
        if size is None:
            head = urllib.request.Request(url, method="HEAD")
            with opener(head, timeout=timeout_s) as response:
                size = int(response.headers["Content-Length"])
        self.size = size
        self.pos = 0
        self._buf_start = 0
        self._buf = b""
        self._span: tuple[int, int, Path] | None = None

    # file-object protocol used by zipfile
    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = self._from_span(n) or self._from_buffer(n) or self._fetch(n)
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        pass

    # internals
    def _request(self, start: int, end: int):
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end - 1}"}
        )
        return self.opener(request, timeout=self.timeout_s)

    def _from_span(self, n: int) -> bytes | None:
        if self._span is None:
            return None
        start, end, path = self._span
        if not (start <= self.pos and self.pos + n <= end):
            return None
        with path.open("rb") as f:
            f.seek(self.pos - start)
            return f.read(n)

    def _from_buffer(self, n: int) -> bytes | None:
        offset = self.pos - self._buf_start
        if offset >= 0 and offset + n <= len(self._buf):
            return self._buf[offset : offset + n]
        return None

    def _fetch(self, n: int) -> bytes:
        end = min(self.size, self.pos + max(n, READ_AHEAD))
        with self._request(self.pos, end) as response:
            self._buf = response.read()
        self._buf_start = self.pos
        return self._buf[:n]

    def prefetch(self, start: int, end: int, path: Path) -> None:
        """Download bytes [start, end) to ``path`` and serve reads from it."""
        part = path.with_name(path.name + ".part")
        try:
            with self._request(start, end) as response, part.open("wb") as out:
                shutil.copyfileobj(response, out, 1 << 20)
            if part.stat().st_size != end - start:
                raise OSError(f"short read for span {start}-{end}")
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)
        self._span = (start, end, path)

    def drop_prefetch(self) -> None:
        if self._span is not None:
            self._span[2].unlink(missing_ok=True)
            self._span = None


def _member_span(
    zf: zipfile.ZipFile, members: list[zipfile.ZipInfo]
) -> tuple[int, int]:
    offsets = sorted(i.header_offset for i in zf.infolist())
    start = min(i.header_offset for i in members)
    last = max(i.header_offset for i in members)
    later = [o for o in offsets if o > last]
    end = later[0] if later else zf.start_dir
    return start, end


def extract_prefix(
    zf: zipfile.ZipFile,
    rf: RangeFile,
    prefix: str,
    dest: Path,
    cache_dir: Path,
) -> int:
    """Extract every member under ``prefix`` into ``dest`` (prefix stripped).

    Returns the number of files written, or 0 if ``dest`` already exists.
    Files go to ``dest.part/`` first, so an interrupted run never leaves a
    directory that looks complete.
    """
    dest = Path(dest)
    if dest.exists():
        return 0
    members = [
        i for i in zf.infolist() if i.filename.startswith(prefix) and not i.is_dir()
    ]
    if not members:
        raise FileNotFoundError(f"no members under {prefix!r}")
    part = dest.with_name(dest.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    start, end = _member_span(zf, members)
    try:
        rf.prefetch(start, end, Path(cache_dir) / f"{dest.name}.span")
        for info in members:
            target = part / info.filename[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
        part.replace(dest)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise
    finally:
        rf.drop_prefetch()
    return len(members)
