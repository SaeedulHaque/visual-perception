# Temporal persistence filter

Module: `src/robust_visual_perception/temporal_filter.py`
Tests: `tests/test_temporal_filter.py`

## The problem

A per-frame detector flickers. A person can be detected at 0.55 in one
frame, 0.48 in the next and 0.61 after that. A safety state computed per
frame then toggles STOP / CLEAR / STOP, which in a robot means jerky
stopping and starting. Under degradation, flicker gets worse because
confidences drop towards the threshold.

## The filter

An N-of-K rule with a separate release rule (hysteresis):

- **Assert STOP** when at least `n_on` of the last `k_window` valid frames
  were STOP.
- **Release to CLEAR** only after `m_off` consecutive CLEAR frames.
- Defaults (fixed in the design): `n_on=3`, `k_window=5`, `m_off=5`.

Asserting is "fairly sure there is a person"; releasing is "sure the person
is gone". Making release harder than assertion is the conservative choice
for a safety signal.

## The cost

Nothing is free. With `n_on=3`, a real event is reported at least two
frames late. With `m_off=5`, STOP lasts at least five frames after the
person leaves. An event shorter than `n_on` frames inside the window can be
missed entirely. The sequence experiment (Phase 2) measures exactly these:
transitions per minute (stability) against onset delay, release delay and
missed events (cost).

## Missing data

An `UNKNOWN` input (missing frame or detector error):

1. is passed straight through as `UNKNOWN`, so the filter never reports
   `CLEAR` for a frame it did not see;
2. clears the window and the release counter, so evidence from before the
   gap is not mixed with evidence after it;
3. keeps the last asserted state. If STOP was active before the gap, it
   needs `m_off` fresh CLEAR frames afterwards before it releases.

## Worked examples (n_on=3, k_window=5, m_off=5)

| Input | Output |
|---|---|
| `S S S` | `C C S` |
| `S C S C S` | `C C C C S` |
| `S S S C C C C C` | `C C S S S S S C` |
| `S S S U C C C C C` | `C C S U S S S S C` |
| `S S U S` | `C C U C` |

These are the exact cases in the tests.

## Details

- On release, the window is cleared as well. Otherwise, with a window longer
  than `m_off`, old STOP frames still inside it would re-assert STOP on the
  next single STOP frame.
- `reset()` returns to the initial state (CLEAR, empty window). The runner
  resets the filter between sequences and conditions.
- The filter starts CLEAR, so an event already present at the start of a
  stream is reported after `n_on` frames, like any other onset.

## Usage

```python
filt = PersistenceFilter(n_on=3, k_window=5, m_off=5)
for frame in frames:
    filtered = filt.update(baseline_decision(result).state)
```
