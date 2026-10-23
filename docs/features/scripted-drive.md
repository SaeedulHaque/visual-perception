# Scripted drive and command guard

Nodes: `rvp_ros/scripted_drive.py`, `rvp_ros/cmd_guard.py`
Core module: `drive_profile.py` (test: `test_drive_profile.py`)
Profile: `ros2_ws/src/rvp_ros/config/drive_profile.yaml`

## Motion never depends on perception

```
drive_profile.yaml ──► scripted_drive ──┐
                                         ├──► /rvp/cmd_vel ──► cmd_guard ──► /cmd_vel ──► robot
teleop_twist_keyboard (teleop mode) ────┘                     clamp, timeout
```

Neither command source subscribes to anything from perception. The
perception node has no velocity publisher at all.

## Scripted drive

The profile is a list of `{duration_s, linear_x, angular_z}` segments.
`DriveProfile.command_at(t)` returns the segment's command at `t` seconds
after the start, and zero before the start and after the last segment. The
node evaluates it at 20 Hz on **sim time** (`use_sim_time`). The trajectory
therefore does not depend on how fast the simulation runs, and nothing
moves while the world is paused.

The profile (111.3 s, inside the 120 s run):

| t (s) | Motion |
|---|---|
| 0–5 | wait |
| 5–20 | forward at 0.10 m/s (1.5 m, towards the near crossing) |
| 20–30 | wait, watching the path |
| 30–33.1 | turn left at 0.5 rad/s (about 90°) |
| 33.1–43.1 | wait, looking along the side of the loop |
| 43.1–46.3 | turn back |
| 46.3–61.3 | reverse at 0.10 m/s to the start |
| 61.3–71.3 | wait |
| 71.3–91.3 | slow arc to the left (0.08 m/s, 0.10 rad/s) |
| 91.3–111.3 | retrace the arc (both signs reversed) |

## Safeguards

| Safeguard | Where |
|---|---|
| speed limits 0.15 m/s and 0.5 rad/s | `clamp`, applied in the drive and again in the guard |
| zero after the profile ends | `command_at` |
| zero on shutdown | both nodes publish zero three times on the way out |
| command timeout 0.5 s | `CommandGuard`: if no command arrives for 0.5 s (steady clock), the guard sends zero, so the robot stops if the drive or teleop dies |
| run ends if a node dies | the launch file shuts the whole run down when any node exits early |

The guard runs in every mode, including the static run, where it simply
keeps sending zero.

The guard's timeout is on the wall clock on purpose. If the simulation
slows down, sim-time commands arrive further apart in wall time. At 20 Hz
on sim time and a real-time factor near 1, they are about 50 ms apart, far
inside the 0.5 s timeout.
