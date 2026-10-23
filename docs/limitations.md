# Limitations

What the results of this project can and cannot support. Each point is
specific to how the experiments were run.

## Degradations

- **Synthetic and isolated.** Blur, low light and noise are applied one at a
  time to already-processed JPEGs. A real dark scene is also noisy and often
  motion-blurred, and real noise depends on intensity (shot noise) and is
  applied before demosaicing and compression. The results describe
  sensitivity to each perturbation, not performance in real low light.
- **Low light is a pure gain.** Scaling pixel values darkens the image but
  keeps every edge and the relative contrast, with no added noise. That may
  be why low light hurts least here; a real dark camera image is not this
  clean.
- **Blur is isotropic Gaussian.** Real motion blur is directional.

## Image experiment

- **Subset mAP.** mAP on 750 images is not comparable to published
  full-val2017 numbers, and the subset over-represents images with people.
- **One operating point.** Precision and recall are at confidence 0.5 only.
  The AP columns cover the full threshold range.
- **Small negatives set.** The false-STOP rate rests on 250 images; one
  image changes it by 0.004.

## Latency

- **Interleaved measurement.** Each image is run at all three input sizes
  and ten conditions in turn. The `clean` condition is always the first
  call a detector makes on a new image, and its mean latency is
  consistently a little higher. This points to a measurement-order effect
  (caches, memory), not to image content. A dedicated benchmark that runs
  each setting in its own block is planned for Phase 4.
- **Not an isolated machine.** The laptop ran a desktop session and CPU
  frequency scaling was not fixed. The committed image run was made with the
  machine otherwise idle; an earlier run made while light work (editing,
  small tests, downloads) was going on gave identical accuracy and mean
  latencies about 2 % higher. During the sequence run, light editing went
  on. Treat latencies as indicative for this machine.
- **Detector call only.** Image decoding and degradation are timed
  separately (`load_ms`, `degrade_ms`) and are not part of the latency
  figures. End-to-end latency in a robot pipeline would be higher.

## Sequence experiment

- **Image-space proxy.** GT-STOP and the decision use box position and size
  in the image. A large box is a near person only on average; camera height
  and lens differ between sequences.
- **ROI chosen on GT.** The design ROI qualified one sequence, so the ROI was
  adjusted with a fixed GT-only procedure (see [mot17.md](features/mot17.md)).
  It was not tuned on model output, but it is still a choice that shapes
  which events exist.
- **Three sequences.** One tuning and two evaluation sequences, with 6 to 9
  events each. Delay statistics rest on few events per sequence; differences
  of a few hundredths of a second are not meaningful.
- **Filter sweep on one sequence.** The adopted setting is chosen on
  MOT17-09 only (static camera), then applied unchanged to the moving-camera
  evaluation sequences.
- **Frame rates differ** (14, 25, 30 fps). Delays are reported in frames and
  seconds; the filter works in frames, so the same setting means a longer
  delay in seconds on the 14 fps sequence.

## Simulation (Phase 3)

- **A clean, easy scene.** One person, one animated mesh, flat ground,
  even lighting, no occlusion, no other people, and a camera whose images
  barely change between frames. The detector finds this person reliably,
  so the per-frame signal rarely flickers. The simulation shows that the
  pipeline works on a robot camera and how it behaves under load; it says
  little about how the filter copes with a hard scene. The MOT17 results
  are the evidence for that.
- **Sim-to-real gap.** A rendered person is not a real one. The detector
  was trained on photographs, and its behaviour on real robot images
  (motion blur, exposure changes, cluttered backgrounds) is not measured
  here.
- **Few events.** Four near crossings per static run and two per scripted
  run. Onset delays rest on very few events per run, and differences of a
  frame or two (33 ms each) are within the measurement's resolution.
- **GT person volume.** The person is a cylinder whose height and width
  were measured from silhouettes on one crossing. Swinging arms and legs
  make the real silhouette wider or narrower from frame to frame; GT-STOP
  depends only on the box centre and height, which the cylinder fits well.
- **Path, not pose, for the person.** The person's GT position comes from
  the scripted loop, not from the simulator (which does not publish it).
  It relies on Gazebo following the dense waypoints closely, which the
  GT check (IoU with detections) supports but does not prove to the
  centimetre.
- **A loaded machine.** Gazebo, the bridge, the recorder and the node share
  the laptop's CPU. Inference in the node is slower than in the idle
  offline runs, and the node skips frames when a callback takes longer than
  the 33 ms frame interval. Live metrics are on the frames the node
  processed; the replay covers every frame.
- **Two clocks.** The image's age is measured on the sim clock, processing
  on the wall clock. With a real-time factor near 1 their sum approximates
  the delay to the published state, but it is not one measured interval.
- **Teleop demo.** The keyboard run is a single, non-repeatable
  demonstration and is not part of the repeated results. It ran with the
  Gazebo window open, and the node processed only 63 % of the frames
  (about 91 % headless), so its numbers are not comparable with the
  measured runs.
- **A person too close to see.** In the teleop run I drove the camera up
  to the person's feet. With only a leg in view the detector stayed under
  the 0.5 threshold and the frame state was CLEAR. The repeated runs never
  bring the person that close (largest GT box 301 of 480 px), so they do
  not measure this case.

## General

- One model (YOLO26n), one run per configuration, no confidence intervals.
  Noise is seeded, so reruns give identical noise; the remaining run-to-run
  variation is latency only.
- No training or fine-tuning, so the results say nothing about how much
  augmentation would recover.
