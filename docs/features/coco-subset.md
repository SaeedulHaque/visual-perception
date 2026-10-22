# COCO subset

Module: `src/robust_visual_perception/coco_subset.py`
Script: `scripts/fetch_coco_subset.py`
Tests: `tests/test_coco_subset.py`

## Why COCO val2017

The detector was trained on COCO train2017, so val2017 is data it has not
seen, with the same 80 classes and human-checked boxes. That gives ground
truth for precision, recall and mAP without labelling anything myself.

## Selection rules

- **Person image:** at least one `person` annotation with `iscrowd=0`.
- **Negative image:** no `person` annotation at all.
- **Excluded:** images whose only people are crowd regions (`iscrowd=1`).
  A crowd region marks "many people here, not individually boxed"; it is
  ambiguous as ground truth for a yes/no STOP decision, so these images go
  in neither pool.

Each pool is sorted by image id, shuffled with `numpy.random.default_rng(seed)`
(seed 0), and the first n images are taken. Because the shuffle does not
depend on n, a smaller subset is always a prefix of a larger one: the
20-image smoke subset (14 person + 6 negative) is contained in the main
subset (500 + 250).

## Manifest

`data/manifests/coco_val2017_<name>.csv`, committed to the repository:

```
image_id,file_name,coco_url,n_person
241668,000000241668.jpg,http://images.cocodataset.org/val2017/000000241668.jpg,2
```

`n_person` counts non-crowd person boxes. The manifest fixes which images
and in which order, so anyone can rebuild the exact subset. The images
themselves are not committed.

## Fetching

```
uv run python scripts/fetch_coco_subset.py --n-person 14 --n-negative 6 --name smoke
```

1. Downloads `annotations_trainval2017.zip` (about 240 MB) once into
   `data/raw/coco/` and extracts `instances_val2017.json`.
2. Selects the subset and writes the manifest.
3. Downloads only the selected images to `data/raw/coco/val2017/`, eight in
   parallel, instead of the 1 GB image archive.

Downloads go to `<file>.part` and are renamed only when complete, so an
interrupted download never leaves a truncated file that looks valid.
Existing files are skipped, so the script can be re-run safely.

## Integrity

The COCO image server only answers plain HTTP. To detect a corrupted or
altered annotation file, the SHA-256 checksums are recorded in
`data/README.md`.
