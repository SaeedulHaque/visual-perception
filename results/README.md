# Results

Each run writes `results/<run_name>/` (see
`docs/features/config-and-outputs.md`). Only final runs referenced in the
top-level README are committed; smoke runs stay local.

| Folder | Phase | Contents |
|---|---|---|
| `image_main/` | 2 | COCO, 750 images × 10 conditions × 3 input sizes |
| `sequence_main/` | 2 | MOT17, 3 sequences × 10 conditions, filter sweep |
| `sequence_faults/` | 2 | MOT17 with 2 % missing and 1 % error frames |
| `sim/` | 3 | TurtleBot3 simulation: per-run node logs and GT (`static_r1..3`, `scripted_r1..3`), two bag replays (`sim_replay`, `sim_replay_check`), and `summary/` |

Large stored-detection arrays (`*.npz`) and simulation bags are not
committed.
