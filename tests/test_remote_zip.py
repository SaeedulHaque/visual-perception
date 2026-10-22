import io
import zipfile

import pytest

from robust_visual_perception.remote_zip import RangeFile, extract_prefix


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data/a/seqinfo.ini", "x")
        for i in range(5):
            zf.writestr(f"data/b/img1/{i:06d}.jpg", bytes([i]) * 1000)
        zf.writestr("data/c/other.txt", "y" * 500)
    return buf.getvalue()


class FakeServer:
    """Serves byte ranges of ``data`` like an HTTP server with Range support."""

    def __init__(self, data: bytes, fail_after: int | None = None):
        self.data = data
        self.requests = 0
        self.fail_after = fail_after

    def __call__(self, request, timeout=None):
        self.requests += 1
        if self.fail_after is not None and self.requests > self.fail_after:
            raise OSError("connection reset")
        spec = request.get_header("Range")
        start, end = (int(v) for v in spec.removeprefix("bytes=").split("-"))
        return io.BytesIO(self.data[start : end + 1])


def test_range_file_reads_like_a_local_file():
    data = _zip_bytes()
    rf = RangeFile("http://x/z.zip", size=len(data), opener=FakeServer(data))
    rf.seek(10)
    assert rf.read(20) == data[10:30]
    rf.seek(-5, 2)
    assert rf.read() == data[-5:]
    assert rf.tell() == len(data)


def test_zipfile_can_list_and_read_remote_members():
    data = _zip_bytes()
    rf = RangeFile("http://x/z.zip", size=len(data), opener=FakeServer(data))
    zf = zipfile.ZipFile(rf)
    assert "data/c/other.txt" in zf.namelist()
    assert zf.read("data/a/seqinfo.ini") == b"x"


def test_extract_prefix_fetches_one_span(tmp_path):
    data = _zip_bytes()
    server = FakeServer(data)
    rf = RangeFile("http://x/z.zip", size=len(data), opener=server)
    zf = zipfile.ZipFile(rf)
    before = server.requests
    dest = tmp_path / "img1"
    n = extract_prefix(zf, rf, "data/b/img1/", dest, cache_dir=tmp_path)
    assert n == 5
    assert server.requests - before == 1  # one ranged request for the span
    assert (dest / "000003.jpg").read_bytes() == bytes([3]) * 1000
    assert not (tmp_path / "img1.part").exists()


def test_interrupted_extract_leaves_no_destination(tmp_path):
    data = _zip_bytes()
    server = FakeServer(data)
    rf = RangeFile("http://x/z.zip", size=len(data), opener=server)
    zf = zipfile.ZipFile(rf)
    server.fail_after = server.requests  # next request fails
    dest = tmp_path / "img1"
    with pytest.raises(OSError):
        extract_prefix(zf, rf, "data/b/img1/", dest, cache_dir=tmp_path)
    assert not dest.exists()


def test_extract_prefix_skips_existing_destination(tmp_path):
    data = _zip_bytes()
    server = FakeServer(data)
    rf = RangeFile("http://x/z.zip", size=len(data), opener=server)
    zf = zipfile.ZipFile(rf)
    dest = tmp_path / "img1"
    dest.mkdir()
    before = server.requests
    assert extract_prefix(zf, rf, "data/b/img1/", dest, cache_dir=tmp_path) == 0
    assert server.requests == before
