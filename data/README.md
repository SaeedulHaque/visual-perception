# Data

Nothing in `data/raw/` is committed. Manifests in `data/manifests/` are
committed, so the exact subsets can be rebuilt.

## COCO val2017 subset

```
uv run python scripts/fetch_coco_subset.py --n-person 14 --n-negative 6 --name smoke
uv run python scripts/fetch_coco_subset.py --n-person 500 --n-negative 250 --name main
```

| File | SHA-256 |
|---|---|
| `annotations_trainval2017.zip` | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |
| `instances_val2017.json` | `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` |

Check with `sha256sum data/raw/coco/*.zip data/raw/coco/*.json`.

COCO annotations are licensed CC BY 4.0; images keep their Flickr licenses
(see https://cocodataset.org/#termsofuse).

## MOT17

```
uv run python scripts/fetch_mot17.py
```

Reads the remote `MOT17.zip` (5.9 GB) with HTTP range requests and
downloads only GT for the seven training sequences and the images of the
three selected sequences (FRCNN copy), about 270 MB. Committed outputs:

- `manifests/mot17_sequences.csv`: per-sequence GT statistics, role
  (tuning / evaluation / unused) and the ROI used.
- `manifests/mot17_roi_search.csv`: the GT-only ROI grid search.

MOT17 is licensed CC BY-NC-SA 3.0 (https://motchallenge.net/data/MOT17/).
