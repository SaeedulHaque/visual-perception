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

Added in Phase 2.
