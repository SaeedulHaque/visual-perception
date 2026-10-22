import io

import pytest

from robust_visual_perception.coco_subset import (
    SubsetImage,
    download_file,
    read_manifest,
    select_subset,
    write_manifest,
)

PERSON = 1
CAR = 3


def _instances() -> dict:
    images = [
        {"id": i, "file_name": f"{i:012d}.jpg", "coco_url": f"http://x/{i}.jpg"}
        for i in range(1, 11)
    ]
    anns = [
        # images 1-4: non-crowd persons (image 1 has two)
        {"image_id": 1, "category_id": PERSON, "iscrowd": 0},
        {"image_id": 1, "category_id": PERSON, "iscrowd": 0},
        {"image_id": 2, "category_id": PERSON, "iscrowd": 0},
        {"image_id": 3, "category_id": PERSON, "iscrowd": 0},
        {"image_id": 4, "category_id": PERSON, "iscrowd": 0},
        {"image_id": 4, "category_id": PERSON, "iscrowd": 1},
        # image 5: crowd-only person region, belongs to neither pool
        {"image_id": 5, "category_id": PERSON, "iscrowd": 1},
        # images 6-7: cars only; 8-10: no annotations
        {"image_id": 6, "category_id": CAR, "iscrowd": 0},
        {"image_id": 7, "category_id": CAR, "iscrowd": 0},
    ]
    return {
        "images": images,
        "annotations": anns,
        "categories": [{"id": PERSON, "name": "person"}, {"id": CAR, "name": "car"}],
    }


def test_pools_follow_the_person_rules():
    subset = select_subset(_instances(), n_person=4, n_negative=5, seed=0)
    person_ids = {s.image_id for s in subset if s.n_person > 0}
    negative_ids = {s.image_id for s in subset if s.n_person == 0}
    assert person_ids == {1, 2, 3, 4}
    assert negative_ids == {6, 7, 8, 9, 10}
    assert 5 not in person_ids | negative_ids
    counts = {s.image_id: s.n_person for s in subset}
    assert counts[1] == 2
    assert counts[4] == 1  # crowd annotation not counted


def test_requesting_too_many_raises():
    with pytest.raises(ValueError):
        select_subset(_instances(), n_person=5, n_negative=1)
    with pytest.raises(ValueError):
        select_subset(_instances(), n_person=1, n_negative=6)


def test_smaller_subset_is_prefix_of_larger():
    small = select_subset(_instances(), n_person=2, n_negative=1, seed=0)
    large = select_subset(_instances(), n_person=3, n_negative=2, seed=0)
    small_p = [s for s in small if s.n_person > 0]
    large_p = [s for s in large if s.n_person > 0]
    assert small_p == large_p[:2]
    small_n = [s for s in small if s.n_person == 0]
    large_n = [s for s in large if s.n_person == 0]
    assert small_n == large_n[:1]


def test_seed_changes_selection():
    picks = {
        tuple(s.image_id for s in select_subset(_instances(), 2, 2, seed=seed))
        for seed in range(10)
    }
    assert len(picks) > 1


def test_manifest_round_trip(tmp_path):
    images = [
        SubsetImage(139, "000000000139.jpg", "http://x/139.jpg", 2),
        SubsetImage(285, "000000000285.jpg", "http://x/285.jpg", 0),
    ]
    path = tmp_path / "m.csv"
    write_manifest(path, images)
    assert read_manifest(path) == images
    assert path.read_text().splitlines()[0] == "image_id,file_name,coco_url,n_person"


class _Failing:
    def __init__(self):
        self.calls = 0

    def __call__(self, url, timeout):
        self.calls += 1
        raise OSError("connection reset")


def test_failed_download_leaves_no_files(tmp_path):
    dest = tmp_path / "img.jpg"
    opener = _Failing()
    with pytest.raises(OSError):
        download_file("http://x/img.jpg", dest, retries=3, opener=opener, backoff_s=0)
    assert opener.calls == 3
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_writes_file_and_skips_existing(tmp_path):
    dest = tmp_path / "img.jpg"
    calls = []

    def opener(url, timeout):
        calls.append(url)
        return io.BytesIO(b"jpeg-bytes")

    download_file("http://x/img.jpg", dest, opener=opener)
    assert dest.read_bytes() == b"jpeg-bytes"
    download_file("http://x/img.jpg", dest, opener=opener)
    assert len(calls) == 1
