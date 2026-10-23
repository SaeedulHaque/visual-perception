# Simulation world and ground truth

Modules: `sim_geometry.py`, `sim_world.py`, `sim_export.py`, `sim_measure.py`
Scripts: `make_sim_world.py`, `measure_sim_person.py`, `make_sim_gt.py`,
`check_sim_gt.py`
Configs: `configs/sim/world.yaml`, `configs/sim/person.yaml`
World: `ros2_ws/src/rvp_ros/worlds/inspection.sdf` (generated)
Tests: `test_sim_geometry.py`, `test_sim_world.py`, `test_sim_export.py`,
`test_sim_measure.py`

## The world

A flat ground plane, one directional light, four static boxes, the
TurtleBot3 Waffle Pi at the origin facing +x, and one walking person (the
Gazebo Fuel actor `Mingfei/actor`). The person walks a closed loop, defined
once in `configs/sim/world.yaml`:

```
            y
            ▲   (3, 3) ────────────────► (10, 3)
            │     ▲                           │
 robot ►    │     │ near crossing             │ far crossing
 (0, 0) ────┼─────┼─────────────────────── x  │ (visible, small:
 facing +x  │     │ (3 m ahead: STOP)         ▼  should stay CLEAR)
            │   (3,-3) ◄──────────────── (10,-3)
```

Walking speed 1 m/s; 1 s turning on the spot at each corner. One loop takes
30 s, so a 120 s run has four near and four far crossings. The boxes stand
behind the far crossing or beside the robot, never between the camera and
the person, so the person is never occluded.

The world file is generated (`scripts/make_sim_world.py`) and a test checks
that the committed file matches the config. The robot is part of the world
file rather than spawned afterwards, so the world can load paused with
everything in place and every run starts from the same state at sim time 0.

## Where the ground truth comes from

Nothing in the ground truth comes from the detector.

| Quantity | Source |
|---|---|
| person position at time t | the scripted loop (`ActorPath.position(t)`) |
| robot pose at time t | Gazebo's PosePublisher on the robot model: the true pose, stamped with sim time, interpolated to the image stamp |
| camera pose | robot pose combined with the camera mount offset from the TB3 model file, (0.076, 0, 0.093) m |
| camera intrinsics | `camera_info` in the bag: fx = fy = 530.47, cx = 320, cy = 240, 640×480 |
| person size | measured once from rendered frames (below) |

Gazebo does not publish the actor's pose, so its position comes from the
same loop definition that generates the world's waypoints. The waypoints
are 0.25 s apart and evenly spaced along each straight leg, so Gazebo's
interpolation between them follows the straight path. The first runs
confirmed that the animation follows sim time: GT boxes and detections move
across the image together.

`/odom` is recorded but never used for GT. Wheel odometry drifts; the
simulator pose does not.

## Person volume

The person is a vertical cylinder at the actor's position.
`scripts/measure_sim_person.py` measures it from a static run without the
detector:

1. Background: the median of 26 frames where the loop puts the person more
   than 5° outside the field of view.
2. Measurement: every frame on the far crossing within 1 m of the optical
   axis, where the whole body is visible (240 frames).
3. The changed pixels (difference to the background > 25 levels, opened
   with a 3×3 kernel, largest component) give the silhouette's top, bottom
   and width, back-projected at the person's known distance.

Result (`configs/sim/person.yaml`, medians): `z_min -0.065 m`,
`z_max 1.600 m`, half-width `0.318 m`. The top edge was the same pixel row
in all 240 frames. The negative `z_min` is not a body below the ground: the
front foot is closer to the camera than the body's centre, and the
back-projection uses the centre distance.

## From pose to GT-STOP

`person_box` projects 16 points around the cylinder's base and top through
the camera model and takes their image bounding box, clipped to the image
like detections. If any point is behind the camera, there is no box. The
box then goes through the same ROI rule as detections:

**Sim ROI** `x 0.30–0.70, y 0–1, height ≥ 0.25`. The x band and the height
are the design's (section 6.3). The design's vertical band (`y ≥ 0.30`) is
dropped. The camera is about 0.1 m above the ground, so a person's box
centre always lies in the upper part of the image, and the band would never
fire. This followed from the geometry and was fixed before any run.

With this rule, the person is GT-STOP on the near crossing and not on the
far one. In `static_r1`, the GT box is 276–277 px tall when the person is
in front of the robot on the near crossing, and 92 px on the far crossing,
under the 120 px minimum (`results/sim/static_r1/gt_frames.csv`).

## Export layout

A bag becomes a folder in the MOT17 layout, so the Phase 2 sequence runner
replays it unchanged:

```
data/raw/sim/sequences/<run>/
  img1/000001.png ...   frames in image-stamp order (docker/sim_run.sh, in the container)
  stamps.csv            frame number -> image stamp (ns)
  poses.csv             true robot pose samples
  camera.json           intrinsics
  export.json           frame count, fps, missing frames
  seqinfo.ini           written by scripts/make_sim_gt.py (host)
  gt/gt.txt             MOT17 columns; visibility 1 (no occlusion)
  frames.csv            per frame: stamp, GT box, gt_stop
```

Each frame's GT is computed from its own stamp, so a frame missing from a
bag would not shift the alignment of the others. `export.json` reports
missing frames, judged from the median frame interval.
