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

## General

- One model (YOLO26n), one run per configuration, no confidence intervals.
  Noise is seeded, so reruns give identical noise; the remaining run-to-run
  variation is latency only.
- No training or fine-tuning, so the results say nothing about how much
  augmentation would recover.
